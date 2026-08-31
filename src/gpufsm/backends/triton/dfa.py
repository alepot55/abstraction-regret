"""Triton DFA kernel — the memory-bound face of the study.

One program per string walking ``cur = trans[cur * 256 + byte]``: a data-dependent
gather feeding a sequential chain. Where the NFA techniques stress control flow, this
stresses the memory system, so the pair measures abstraction regret on both faces
with the algorithm held fixed.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ...core.dfa import DFA
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result, batch_results
from ._common import stage_batch, timed_launch


@triton.jit
def _dfa_kernel(
    trans, accept, input_data, input_offsets, num_strings, start_state, out_flags, out_lens
):
    pid = tl.program_id(0)
    if pid < num_strings:
        lo = tl.load(input_offsets + pid)
        hi = tl.load(input_offsets + pid + 1)
        cur = start_state
        out_f = 0
        out_l = 0
        done = 0
        if tl.load(accept + cur) != 0:
            out_f = 1
            done = 1
        for pos in range(lo, hi):
            if done == 0:
                sym = tl.load(input_data + pos)
                cur = tl.load(trans + cur * 256 + sym)
                if tl.load(accept + cur) != 0:
                    out_f = 1
                    out_l = pos - lo + 1
                    done = 1
        tl.store(out_flags + pid, out_f)
        tl.store(out_lens + pid, out_l)


class TritonDFAExecutor:
    """One Triton program per string; dense transition table resident on the GPU."""

    def __init__(self, dfa: DFA, technique: str = "gather") -> None:
        self.dfa = dfa
        self.technique = technique
        self._dev = torch.device("cuda")
        self._trans = torch.as_tensor(dfa.trans, device=self._dev)
        self._accept = torch.as_tensor(dfa.accept.astype(np.int32), device=self._dev)

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
            lambda: _dfa_kernel[(n,)](
                self._trans,
                self._accept,
                data,
                offsets,
                n,
                int(self.dfa.start_state),
                flags,
                lens,
            )
        )
        return batch_results(flags.cpu().numpy(), lens.cpu().numpy(), kernel_ms, transfer_ms)


@register(Kind.DFA, Backend.TRITON, "gather")
def _make(automaton: Automaton, technique: str) -> TritonDFAExecutor:
    assert isinstance(automaton, DFA)
    return TritonDFAExecutor(automaton, technique)
