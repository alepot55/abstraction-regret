"""Triton ``multistream`` technique — the single->multi-stream ablation axis.

``grid=(num_strings,)``: one program per input string, all strings concurrent, the
whole batch in a single launch. The bit-packed working set of ``bitpacked`` is
sliced per program (``cur``/``nxt`` are ``N * NWORDS`` long).

Multi-stream is standard practice since ~2015 and is **not** claimed as a
contribution: it is the honest baseline the ablation has to beat.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ...core.nfa import ANY_SYMBOL, NFA
from ...core.packing import pack_accept, words_for
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result, batch_results
from ._common import DeviceCSR, stage_batch, timed_launch


@triton.jit
def _multistream_kernel(
    sym_row_ptr,
    sym_targets,
    sym_symbols,
    eps_row_ptr,
    eps_targets,
    accept_words,
    input_data,
    input_offsets,
    num_strings,
    out_flags,
    out_lens,
    cur,
    nxt,
    start_state,
    NUM_STATES: tl.constexpr,
    NWORDS: tl.constexpr,
    ANY_ID: tl.constexpr,
    USES_ANY: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < num_strings:
        base = pid * NWORDS
        in_lo = tl.load(input_offsets + pid)
        input_len = tl.load(input_offsets + pid + 1) - in_lo
        one = tl.full((), 1, tl.int64)

        for w in range(NWORDS):
            tl.store(cur + base + w, 0)
        sw = start_state >> 6
        tl.store(cur + base + sw, tl.load(cur + base + sw) | (one << (start_state & 63)))

        for _ in range(NUM_STATES):
            for s in range(NUM_STATES):
                wi = s >> 6
                bit = one << (s & 63)
                if (tl.load(cur + base + wi) & bit) != 0:
                    lo = tl.load(eps_row_ptr + s)
                    hi = tl.load(eps_row_ptr + s + 1)
                    for k in range(lo, hi):
                        t = tl.load(eps_targets + k)
                        twi = base + (t >> 6)
                        tl.store(cur + twi, tl.load(cur + twi) | (one << (t & 63)))

        out_f = 0
        out_l = 0
        done = 0
        for w in range(NWORDS):
            if (tl.load(cur + base + w) & tl.load(accept_words + w)) != 0:
                done = 1
        if done == 1:
            out_f = 1
            out_l = 0

        for pos in range(input_len):
            if done == 0:
                sym = tl.load(input_data + in_lo + pos)
                for w in range(NWORDS):
                    tl.store(nxt + base + w, 0)
                for s in range(NUM_STATES):
                    wi = s >> 6
                    bit = one << (s & 63)
                    if (tl.load(cur + base + wi) & bit) != 0:
                        lo = tl.load(sym_row_ptr + s)
                        hi = tl.load(sym_row_ptr + s + 1)
                        for k in range(lo, hi):
                            tsym = tl.load(sym_symbols + k)
                            hit = tsym == sym
                            if USES_ANY:
                                hit = hit or (tsym == ANY_ID)
                            if hit:
                                t = tl.load(sym_targets + k)
                                twi = base + (t >> 6)
                                tl.store(nxt + twi, tl.load(nxt + twi) | (one << (t & 63)))
                for _ in range(NUM_STATES):
                    for s in range(NUM_STATES):
                        wi = s >> 6
                        bit = one << (s & 63)
                        if (tl.load(nxt + base + wi) & bit) != 0:
                            lo = tl.load(eps_row_ptr + s)
                            hi = tl.load(eps_row_ptr + s + 1)
                            for k in range(lo, hi):
                                t = tl.load(eps_targets + k)
                                twi = base + (t >> 6)
                                tl.store(nxt + twi, tl.load(nxt + twi) | (one << (t & 63)))
                for w in range(NWORDS):
                    tl.store(cur + base + w, tl.load(nxt + base + w))
                m = 0
                for w in range(NWORDS):
                    if (tl.load(cur + base + w) & tl.load(accept_words + w)) != 0:
                        m = 1
                if m == 1:
                    out_f = 1
                    out_l = pos + 1
                    done = 1

        tl.store(out_flags + pid, out_f)
        tl.store(out_lens + pid, out_l)


class TritonMultistreamExecutor:
    """One program per string, whole batch in a single launch.

    ``run_batch`` is the real path; ``run`` is a batch of one.
    """

    def __init__(self, nfa: NFA, technique: str = "multistream") -> None:
        self.nfa = nfa
        self.technique = technique
        self._nwords = words_for(nfa.num_states)
        self._dev = torch.device("cuda")
        self._csr = DeviceCSR(nfa, self._dev)
        self._accept_words = torch.as_tensor(
            pack_accept(nfa, dtype=np.int64, nwords=self._nwords), device=self._dev
        )

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
        cur = torch.zeros(n * self._nwords, dtype=torch.int64, device=dev)
        nxt = torch.zeros(n * self._nwords, dtype=torch.int64, device=dev)

        kernel_ms = timed_launch(
            lambda: _multistream_kernel[(n,)](
                *self._csr.args,
                self._accept_words,
                data,
                offsets,
                n,
                flags,
                lens,
                cur,
                nxt,
                int(self.nfa.start_state),
                NUM_STATES=int(self.nfa.num_states),
                NWORDS=int(self._nwords),
                ANY_ID=int(ANY_SYMBOL),
                USES_ANY=bool(self.nfa.uses_any_symbol),
            )
        )
        return batch_results(flags.cpu().numpy(), lens.cpu().numpy(), kernel_ms, transfer_ms)


@register(Kind.NFA, Backend.TRITON, "multistream")
def _make(automaton: Automaton, technique: str) -> TritonMultistreamExecutor:
    assert isinstance(automaton, NFA)
    return TritonMultistreamExecutor(automaton, technique)
