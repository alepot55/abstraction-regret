"""Triton ``worklist`` technique — the work-efficient kernel (<=64 states).

Proof that Triton *can* express the active-set algorithm, unlike Gluon: iterate only
the set bits via ``libdevice.ffs`` inside a data-dependent ``while``, with a
frontier-based epsilon closure. That matters for the thesis, because it separates
expressiveness from efficiency: Triton expresses the right algorithm and still pays a
large constant against the CUDA worklist, so the regret is the execution paradigm and
not a missing algorithm.

The working set is one scalar int64, hence the 64-state ceiling; the CUDA backend's
``worklist_global`` is the unbounded path.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from ...core.nfa import ANY_SYMBOL, NFA
from ...core.packing import WORD_BITS, accept_bitmask
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result, batch_results
from ._common import DeviceCSR, stage_batch, timed_launch

TRITON_WORKLIST_MAX_STATES = WORD_BITS


@triton.jit
def _worklist_kernel(
    sym_row_ptr,
    sym_targets,
    sym_symbols,
    eps_row_ptr,
    eps_targets,
    accept_word,
    input_data,
    input_offsets,
    num_strings,
    out_flags,
    out_lens,
    start_state,
    ANY_ID: tl.constexpr,
    USES_ANY: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_strings:
        in_lo = tl.load(input_offsets + pid)
        input_len = tl.load(input_offsets + pid + 1) - in_lo
        one = tl.full((), 1, tl.int64)
        zero = tl.full((), 0, tl.int64)

        cur = one << start_state
        # frontier epsilon-closure on cur
        frontier = cur
        while frontier != zero:
            newb = zero
            bits = frontier
            while bits != zero:
                s = libdevice.ffs(bits) - 1
                bits = bits & (bits - 1)
                for k in range(tl.load(eps_row_ptr + s), tl.load(eps_row_ptr + s + 1)):
                    newb = newb | (one << tl.load(eps_targets + k))
            newb = newb & (cur ^ (zero - one))  # newb &= ~cur  (xor -1 == bitwise not)
            cur = cur | newb
            frontier = newb

        out_f = 0
        out_l = 0
        done = 0
        if (cur & accept_word) != zero:
            out_f = 1
            done = 1

        for pos in range(input_len):
            if done == 0:
                sym = tl.load(input_data + in_lo + pos)
                nxt = zero
                bits = cur
                while bits != zero:
                    s = libdevice.ffs(bits) - 1
                    bits = bits & (bits - 1)
                    for k in range(tl.load(sym_row_ptr + s), tl.load(sym_row_ptr + s + 1)):
                        tsym = tl.load(sym_symbols + k)
                        hit = tsym == sym
                        if USES_ANY:
                            hit = hit or (tsym == ANY_ID)
                        if hit:
                            nxt = nxt | (one << tl.load(sym_targets + k))
                # frontier epsilon-closure on nxt
                frontier = nxt
                while frontier != zero:
                    newb = zero
                    bits = frontier
                    while bits != zero:
                        s = libdevice.ffs(bits) - 1
                        bits = bits & (bits - 1)
                        for k in range(tl.load(eps_row_ptr + s), tl.load(eps_row_ptr + s + 1)):
                            newb = newb | (one << tl.load(eps_targets + k))
                    newb = newb & (nxt ^ (zero - one))  # newb &= ~nxt
                    nxt = nxt | newb
                    frontier = newb
                cur = nxt
                if (cur & accept_word) != zero:
                    out_f = 1
                    out_l = pos + 1
                    done = 1

        tl.store(out_flags + pid, out_f)
        tl.store(out_lens + pid, out_l)


class TritonWorklistExecutor:
    """Work-efficient Triton worklist (<=64 states, scalar int64 working set)."""

    def __init__(self, nfa: NFA, technique: str = "worklist") -> None:
        if nfa.num_states > TRITON_WORKLIST_MAX_STATES:
            raise ValueError(
                f"triton/worklist supports <={TRITON_WORKLIST_MAX_STATES} states "
                f"(got {nfa.num_states})"
            )
        self.nfa = nfa
        self.technique = technique
        self._dev = torch.device("cuda")
        self._csr = DeviceCSR(nfa, self._dev)
        self._accept_word = accept_bitmask(nfa)

    def run(self, input_bytes: bytes) -> Result:
        return self.run_batch([input_bytes])[0]

    def run_batch(self, inputs: list[bytes]) -> list[Result]:
        if not inputs:
            return []
        dev = self._dev
        n = len(inputs)
        data, offsets, transfer_ms = stage_batch(inputs, dev)
        flags = torch.zeros(n, dtype=torch.int32, device=dev)
        lens = torch.zeros(n, dtype=torch.int32, device=dev)

        kernel_ms = timed_launch(
            lambda: _worklist_kernel[(n,)](
                *self._csr.args,
                self._accept_word,
                data,
                offsets,
                n,
                flags,
                lens,
                int(self.nfa.start_state),
                ANY_ID=int(ANY_SYMBOL),
                USES_ANY=bool(self.nfa.uses_any_symbol),
            )
        )
        return batch_results(flags.cpu().numpy(), lens.cpu().numpy(), kernel_ms, transfer_ms)


@register(Kind.NFA, Backend.TRITON, "worklist")
def _make(automaton: Automaton, technique: str) -> TritonWorklistExecutor:
    assert isinstance(automaton, NFA)
    return TritonWorklistExecutor(automaton, technique)
