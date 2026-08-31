"""Warp ``multistream`` technique — the Python *thread-SIMT* probe.

One Warp thread per input string, with the active state-set held in a single
register-resident ``uint64`` (hence the 64-state ceiling; a multi-word Warp bitset
is future work, Warp lacks ergonomic per-thread arrays). Mirrors the CUDA
``multistream`` kernel and the :mod:`gpufsm.core.bitmap` spec.

This is the load-bearing comparison point of the thesis: same Python-level
productivity as Triton, but a thread model that *can* express the data-dependent
per-state control flow an NFA needs.
"""

from __future__ import annotations

import time

import warp as wp

from ...core.nfa import ANY_SYMBOL, NFA
from ...core.packing import WORD_BITS, accept_bitmask, pack_inputs
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result, batch_results
from ._common import DEVICE, dev_i32, ensure_initialized, timed_launch

WARP_MAX_STATES = WORD_BITS


@wp.kernel
def _multistream_kernel(
    sym_row_ptr: wp.array(dtype=wp.int32),
    sym_targets: wp.array(dtype=wp.int32),
    sym_symbols: wp.array(dtype=wp.int32),
    eps_row_ptr: wp.array(dtype=wp.int32),
    eps_targets: wp.array(dtype=wp.int32),
    accept_word: wp.uint64,
    input_data: wp.array(dtype=wp.int32),
    input_offsets: wp.array(dtype=wp.int32),
    num_states: wp.int32,
    start_state: wp.int32,
    uses_any: wp.int32,
    any_id: wp.int32,
    out_flags: wp.array(dtype=wp.int32),
    out_lens: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    lo = input_offsets[i]
    input_len = input_offsets[i + 1] - lo
    one = wp.uint64(1)
    zero = wp.uint64(0)

    cur = one << wp.uint64(start_state)
    # epsilon closure (num_states passes guarantee convergence)
    for _it in range(num_states):
        for s in range(num_states):
            if (cur & (one << wp.uint64(s))) != zero:
                for k in range(eps_row_ptr[s], eps_row_ptr[s + 1]):
                    cur = cur | (one << wp.uint64(eps_targets[k]))

    # int(...) is intentional: it declares a *mutable* wp.int32 local. A bare
    # literal (0) makes Warp miscompile the later conditional reassignments.
    out_f = int(0)
    out_l = int(0)
    done = int(0)
    if (cur & accept_word) != zero:
        out_f = 1
        done = 1

    pos = int(0)
    while pos < input_len and done == 0:
        sym = input_data[lo + pos]
        nxt = zero
        for s in range(num_states):
            if (cur & (one << wp.uint64(s))) != zero:
                for k in range(sym_row_ptr[s], sym_row_ptr[s + 1]):
                    tsym = sym_symbols[k]
                    hit = int(0)
                    if tsym == sym:
                        hit = 1
                    if uses_any == 1:
                        if tsym == any_id:
                            hit = 1
                    if hit == 1:
                        nxt = nxt | (one << wp.uint64(sym_targets[k]))
        for _it2 in range(num_states):
            for s in range(num_states):
                if (nxt & (one << wp.uint64(s))) != zero:
                    for k in range(eps_row_ptr[s], eps_row_ptr[s + 1]):
                        nxt = nxt | (one << wp.uint64(eps_targets[k]))
        cur = nxt
        if (cur & accept_word) != zero:
            out_f = 1
            out_l = pos + 1
            done = 1
        pos = pos + 1

    out_flags[i] = out_f
    out_lens[i] = out_l


class WarpMultistreamExecutor:
    """One Warp thread per string; ``uint64`` register working set (<=64 states)."""

    def __init__(self, nfa: NFA, technique: str = "multistream") -> None:
        if nfa.num_states > WARP_MAX_STATES:
            raise ValueError(
                f"warp/multistream supports <={WARP_MAX_STATES} states "
                f"(got {nfa.num_states}); multi-word Warp bitset is future work"
            )
        ensure_initialized()
        self.nfa = nfa
        self.technique = technique
        self._srp = dev_i32(nfa.sym_row_ptr)
        self._st = dev_i32(nfa.sym_targets)
        self._ss = dev_i32(nfa.sym_symbols)
        self._erp = dev_i32(nfa.eps_row_ptr)
        self._et = dev_i32(nfa.eps_targets)
        self._accept = wp.uint64(accept_bitmask(nfa))

    def run(self, input_bytes: bytes) -> Result:
        return self.run_batch([input_bytes])[0]

    def run_batch(self, inputs: list[bytes]) -> list[Result]:
        if not inputs:
            return []
        n = len(inputs)
        t0 = time.perf_counter()
        data_np, offsets_np = pack_inputs(inputs)
        data = wp.from_numpy(data_np, wp.int32, device=DEVICE)
        off = wp.from_numpy(offsets_np, wp.int32, device=DEVICE)
        out_flags = wp.zeros(n, dtype=wp.int32, device=DEVICE)
        out_lens = wp.zeros(n, dtype=wp.int32, device=DEVICE)
        transfer_ms = (time.perf_counter() - t0) * 1000.0

        kernel_ms = timed_launch(
            lambda: wp.launch(
                _multistream_kernel,
                dim=n,
                inputs=[
                    self._srp,
                    self._st,
                    self._ss,
                    self._erp,
                    self._et,
                    self._accept,
                    data,
                    off,
                    int(self.nfa.num_states),
                    int(self.nfa.start_state),
                    int(self.nfa.uses_any_symbol),
                    int(ANY_SYMBOL),
                    out_flags,
                    out_lens,
                ],
                device=DEVICE,
            )
        )
        return batch_results(out_flags.numpy(), out_lens.numpy(), kernel_ms, transfer_ms)


@register(Kind.NFA, Backend.WARP, "multistream")
def _make(automaton: Automaton, technique: str) -> WarpMultistreamExecutor:
    assert isinstance(automaton, NFA)
    return WarpMultistreamExecutor(automaton, technique)
