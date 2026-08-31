"""Triton ``bitpacked`` technique — the byte->bit ablation axis.

Same CSR algorithm as ``dense``, only the working-set layout changes: the active
state-set is a packed bitmask (1 bit/state, 64-bit words) instead of one int8 slot
per state. Apples-to-apples with ``dense``, and the executable spec it mirrors is
:mod:`gpufsm.core.bitmap`.

The accept mask must be int64: Triton has no unsigned 64-bit type and a Python
literal shift is truncated to int32, silently losing every bit above 31 — which is
what broke NFAs with more than 64 states before the >64-state stress tests existed.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ...core.nfa import ANY_SYMBOL, NFA
from ...core.packing import pack_accept, words_for
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result
from ._common import DeviceCSR, stage_input, timed_launch


@triton.jit
def _bitpacked_kernel(
    sym_row_ptr,
    sym_targets,
    sym_symbols,
    eps_row_ptr,
    eps_targets,
    accept_words,
    input_symbols,
    out_flag,
    out_len,
    cur,
    nxt,
    input_len,
    start_state,
    NUM_STATES: tl.constexpr,
    NWORDS: tl.constexpr,
    ANY_ID: tl.constexpr,
    USES_ANY: tl.constexpr,
):
    # One int64 set-bit for runtime targets (avoids int32 overflow at bit 31+
    # and Python/Triton operator ambiguity on ``1 << <runtime>``).
    one = tl.full((), 1, tl.int64)

    # cur := { start_state }
    for w in range(NWORDS):
        tl.store(cur + w, 0)
    sw = start_state >> 6
    sb = one << (start_state & 63)
    tl.store(cur + sw, tl.load(cur + sw) | sb)

    # Epsilon closure: NUM_STATES passes guarantee convergence.
    for _ in range(NUM_STATES):
        for s in range(NUM_STATES):
            wi = s >> 6
            bit = one << (s & 63)  # int64 mask (int32 literals truncate bits >=32)
            if (tl.load(cur + wi) & bit) != 0:
                lo = tl.load(eps_row_ptr + s)
                hi = tl.load(eps_row_ptr + s + 1)
                for k in range(lo, hi):
                    t = tl.load(eps_targets + k)
                    twi = t >> 6
                    tbit = one << (t & 63)
                    tl.store(cur + twi, tl.load(cur + twi) | tbit)

    # Word-parallel accept test (NWORDS iterations, not NUM_STATES).
    out_f = 0
    out_l = 0
    done = 0
    for w in range(NWORDS):
        if (tl.load(cur + w) & tl.load(accept_words + w)) != 0:
            done = 1
    if done == 1:
        out_f = 1
        out_l = 0

    for pos in range(input_len):
        if done == 0:
            sym = tl.load(input_symbols + pos)
            for w in range(NWORDS):
                tl.store(nxt + w, 0)
            for s in range(NUM_STATES):
                wi = s >> 6
                bit = one << (s & 63)  # int64 mask (int32 literals truncate bits >=32)
                if (tl.load(cur + wi) & bit) != 0:
                    lo = tl.load(sym_row_ptr + s)
                    hi = tl.load(sym_row_ptr + s + 1)
                    for k in range(lo, hi):
                        tsym = tl.load(sym_symbols + k)
                        hit = tsym == sym
                        if USES_ANY:
                            hit = hit or (tsym == ANY_ID)
                        if hit:
                            t = tl.load(sym_targets + k)
                            twi = t >> 6
                            tbit = one << (t & 63)
                            tl.store(nxt + twi, tl.load(nxt + twi) | tbit)
            # epsilon closure on nxt
            for _ in range(NUM_STATES):
                for s in range(NUM_STATES):
                    wi = s >> 6
                    bit = one << (s & 63)  # int64 mask (int32 literals truncate bits >=32)
                    if (tl.load(nxt + wi) & bit) != 0:
                        lo = tl.load(eps_row_ptr + s)
                        hi = tl.load(eps_row_ptr + s + 1)
                        for k in range(lo, hi):
                            t = tl.load(eps_targets + k)
                            twi = t >> 6
                            tbit = one << (t & 63)
                            tl.store(nxt + twi, tl.load(nxt + twi) | tbit)
            for w in range(NWORDS):
                tl.store(cur + w, tl.load(nxt + w))
            m = 0
            for w in range(NWORDS):
                if (tl.load(cur + w) & tl.load(accept_words + w)) != 0:
                    m = 1
            if m == 1:
                out_f = 1
                out_l = pos + 1
                done = 1

    tl.store(out_flag, out_f)
    tl.store(out_len, out_l)


class TritonBitpackedExecutor:
    """Packed 1-bit-per-state working set; same algorithm as ``dense``."""

    def __init__(self, nfa: NFA, technique: str = "bitpacked") -> None:
        self.nfa = nfa
        self.technique = technique
        self._nwords = words_for(nfa.num_states)
        self._dev = torch.device("cuda")
        self._csr = DeviceCSR(nfa, self._dev)
        self._accept_words = torch.as_tensor(
            pack_accept(nfa, dtype=np.int64, nwords=self._nwords), device=self._dev
        )
        self._cur = torch.zeros(self._nwords, dtype=torch.int64, device=self._dev)
        self._nxt = torch.zeros(self._nwords, dtype=torch.int64, device=self._dev)

    def run(self, input_bytes: bytes) -> Result:
        dev = self._dev
        inp, input_len, transfer_ms = stage_input(input_bytes, dev)
        flag = torch.zeros(1, dtype=torch.int32, device=dev)
        mlen = torch.zeros(1, dtype=torch.int32, device=dev)

        kernel_ms = timed_launch(
            lambda: _bitpacked_kernel[(1,)](
                *self._csr.args,
                self._accept_words,
                inp,
                flag,
                mlen,
                self._cur,
                self._nxt,
                input_len,
                int(self.nfa.start_state),
                NUM_STATES=int(self.nfa.num_states),
                NWORDS=int(self._nwords),
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


@register(Kind.NFA, Backend.TRITON, "bitpacked")
def _make(automaton: Automaton, technique: str) -> TritonBitpackedExecutor:
    assert isinstance(automaton, NFA)
    return TritonBitpackedExecutor(automaton, technique)
