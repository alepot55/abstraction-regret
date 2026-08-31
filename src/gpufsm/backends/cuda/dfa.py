"""CUDA DFA executor — the memory-bound face on the hand-written arm.

Adapter onto the extension's ``run_dfa`` entry point; the kernel is in
``native/dfa.cu``. Paired with the Triton and Warp DFA kernels it is the CUDA
reference point for the memory-bound half of the two-faces result.
"""

from __future__ import annotations

import time

import numpy as np

from ...core.dfa import DFA
from ...core.packing import pack_inputs
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result, batch_results
from . import extension


class CUDADFAExecutor:
    """Whole batch in a single launch against the dense transition table."""

    def __init__(self, dfa: DFA, technique: str = "gather") -> None:
        self.dfa = dfa
        self.technique = technique
        self._trans = np.ascontiguousarray(dfa.trans, dtype=np.int32)
        self._accept = np.ascontiguousarray(dfa.accept, dtype=np.int8)

    def run(self, input_bytes: bytes) -> Result:
        return self.run_batch([input_bytes])[0]

    def run_batch(self, inputs: list[bytes]) -> list[Result]:
        if not inputs:
            return []
        t0 = time.perf_counter()
        data, offsets = pack_inputs(inputs)
        transfer_ms = (time.perf_counter() - t0) * 1000.0
        flags, lens, kernel_ms = extension.load().run_dfa(
            self._trans,
            self._accept,
            np.ascontiguousarray(data, dtype=np.int32),
            np.ascontiguousarray(offsets, dtype=np.int32),
            int(self.dfa.num_states),
            int(self.dfa.start_state),
        )
        return batch_results(flags, lens, float(kernel_ms), transfer_ms)


@register(Kind.DFA, Backend.CUDA, "gather", default=True)
def _make(automaton: Automaton, technique: str) -> CUDADFAExecutor:
    assert isinstance(automaton, DFA)
    return CUDADFAExecutor(automaton, technique)
