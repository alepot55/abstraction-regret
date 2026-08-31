"""Triton ``dense`` technique — one int8 slot per state.

The faithful, correctness-first port of the reference simulator, kept deliberately
as the *abstraction-regret* baseline: 4 bytes per state where one bit suffices
(a 31x working-set blow-up at 500 states). :mod:`~gpufsm.backends.triton.bitpacked`
is the same algorithm on the packed layout, so the pair isolates the byte->bit axis.

Single program (``grid=(1,)``); ``multistream`` is the batched counterpart.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...core.nfa import ANY_SYMBOL, NFA
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result
from ._common import DeviceCSR, stage_input, timed_launch


@triton.jit
def _dense_kernel(
    sym_row_ptr,
    sym_targets,
    sym_symbols,
    eps_row_ptr,
    eps_targets,
    accept,
    input_symbols,
    out_flag,
    out_len,
    cur,
    nxt,
    input_len,
    start_state,
    NUM_STATES: tl.constexpr,
    ANY_ID: tl.constexpr,
    USES_ANY: tl.constexpr,
):
    # Single-program NFA simulation (latch-first-match). One int8 slot/state.
    for i in range(NUM_STATES):
        tl.store(cur + i, 0)
    tl.store(cur + start_state, 1)

    # Epsilon closure: NUM_STATES passes guarantee convergence.
    for _ in range(NUM_STATES):
        for s in range(NUM_STATES):
            if tl.load(cur + s) == 1:
                lo = tl.load(eps_row_ptr + s)
                hi = tl.load(eps_row_ptr + s + 1)
                for k in range(lo, hi):
                    tl.store(cur + tl.load(eps_targets + k), 1)

    # latch-first-match: ``done`` freezes the verdict at the first accepting
    # state. Triton forbids ``return`` inside loops, so the per-position body
    # is guarded by ``done == 0`` and the loop runs to completion regardless.
    out_f = 0
    out_l = 0
    done = 0
    for s in range(NUM_STATES):
        if (tl.load(cur + s) == 1) and (tl.load(accept + s) == 1):
            done = 1
    if done == 1:
        out_f = 1
        out_l = 0

    for pos in range(input_len):
        if done == 0:
            sym = tl.load(input_symbols + pos)
            for i in range(NUM_STATES):
                tl.store(nxt + i, 0)
            for s in range(NUM_STATES):
                if tl.load(cur + s) == 1:
                    lo = tl.load(sym_row_ptr + s)
                    hi = tl.load(sym_row_ptr + s + 1)
                    for k in range(lo, hi):
                        tsym = tl.load(sym_symbols + k)
                        hit = tsym == sym
                        if USES_ANY:
                            hit = hit or (tsym == ANY_ID)
                        if hit:
                            tl.store(nxt + tl.load(sym_targets + k), 1)
            # epsilon closure on nxt
            for _ in range(NUM_STATES):
                for s in range(NUM_STATES):
                    if tl.load(nxt + s) == 1:
                        lo = tl.load(eps_row_ptr + s)
                        hi = tl.load(eps_row_ptr + s + 1)
                        for k in range(lo, hi):
                            tl.store(nxt + tl.load(eps_targets + k), 1)
            for i in range(NUM_STATES):
                tl.store(cur + i, tl.load(nxt + i))
            m = 0
            for s in range(NUM_STATES):
                if (tl.load(cur + s) == 1) and (tl.load(accept + s) == 1):
                    m = 1
            if m == 1:
                out_f = 1
                out_l = pos + 1
                done = 1

    tl.store(out_flag, out_f)
    tl.store(out_len, out_l)


class TritonDenseExecutor:
    """One int8 slot per state, single program. The abstraction-regret baseline."""

    def __init__(self, nfa: NFA, technique: str = "dense") -> None:
        self.nfa = nfa
        self.technique = technique
        self._dev = torch.device("cuda")
        self._csr = DeviceCSR(nfa, self._dev)
        self._accept = torch.as_tensor(nfa.accept.astype("int8"), device=self._dev)
        self._cur = torch.zeros(nfa.num_states, dtype=torch.int8, device=self._dev)
        self._nxt = torch.zeros(nfa.num_states, dtype=torch.int8, device=self._dev)

    def run(self, input_bytes: bytes) -> Result:
        dev = self._dev
        inp, input_len, transfer_ms = stage_input(input_bytes, dev)
        flag = torch.zeros(1, dtype=torch.int32, device=dev)
        mlen = torch.zeros(1, dtype=torch.int32, device=dev)

        kernel_ms = timed_launch(
            lambda: _dense_kernel[(1,)](
                *self._csr.args,
                self._accept,
                inp,
                flag,
                mlen,
                self._cur,
                self._nxt,
                input_len,
                int(self.nfa.start_state),
                NUM_STATES=int(self.nfa.num_states),
                ANY_ID=int(ANY_SYMBOL),
                USES_ANY=bool(self.nfa.uses_any_symbol),
            )
        )
        return Result(
            accepted=bool(flag.item()),
            match_len=int(mlen.item()),
            kernel_ms=kernel_ms,
            total_ms=kernel_ms + transfer_ms,
            transfer_ms=transfer_ms,
        )


@register(Kind.NFA, Backend.TRITON, "dense", default=True)
def _make(automaton: Automaton, technique: str) -> TritonDenseExecutor:
    assert isinstance(automaton, NFA)
    return TritonDenseExecutor(automaton, technique)
