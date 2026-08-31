"""CUDA NFA executors — thin adapters onto the compiled ``_cuda`` extension.

All the kernel work lives in ``native/`` (built with ``GPUFSM_BUILD_CUDA=ON``); this
module only marshals the NFA into the arrays the extension expects and turns its
return tuples into :class:`~gpufsm.core.result.Result` objects.

The batched techniques all share one executor: they differ solely in which extension
entry point they call, which is exactly what the ablation is about — same host code,
same algorithm, different memory placement or work strategy.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ...core.nfa import NFA
from ...core.packing import pack_accept, pack_inputs, symbols
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result, batch_results
from . import extension

# technique -> entry point in the compiled extension.
SINGLE_TECHNIQUES = {
    "dense": "run_dense",  # one int8 slot per state (regret baseline)
    "bitpacked": "run_bitpacked",  # packed bitmask (byte->bit axis)
}
BATCH_TECHNIQUES = {
    # single -> multi-stream axis, then the CSR placement / transfer ablations
    "multistream": "run_multistream",
    "multistream_shared": "run_multistream_shared",
    "multistream_async": "run_multistream_async",
    # work-efficient active-set family
    "worklist": "run_worklist",  # register-resident working set (<=512 states)
    "worklist_global": "run_worklist_global",  # global working set, no state cap
    "worklist_warp": "run_worklist_warp",  # one warp per string (block-parallel)
    "worklist_shared": "run_worklist_shared",  # working set in shared memory
    "worklist_compact": "run_worklist_compact",  # compacted active-ID frontier
}


def _csr_args(nfa: NFA) -> tuple[np.ndarray, ...]:
    """The five CSR arrays, contiguous and int32, in the extension's argument order."""
    return (
        np.ascontiguousarray(nfa.sym_row_ptr, dtype=np.int32),
        np.ascontiguousarray(nfa.sym_targets, dtype=np.int32),
        np.ascontiguousarray(nfa.sym_symbols, dtype=np.int32),
        np.ascontiguousarray(nfa.eps_row_ptr, dtype=np.int32),
        np.ascontiguousarray(nfa.eps_targets, dtype=np.int32),
    )


class CUDASingleExecutor:
    """One string per launch: the ``dense`` and ``bitpacked`` techniques."""

    def __init__(self, nfa: NFA, technique: str = "dense") -> None:
        self.nfa = nfa
        self.technique = technique
        self._runner: Any = getattr(extension.load(), SINGLE_TECHNIQUES[technique])
        self._accept = (
            np.ascontiguousarray(nfa.accept, dtype=np.int8)
            if technique == "dense"
            else np.ascontiguousarray(pack_accept(nfa), dtype=np.uint64)
        )

    def run(self, input_bytes: bytes) -> Result:
        nfa = self.nfa
        t0 = time.perf_counter()
        syms = np.ascontiguousarray(symbols(input_bytes), dtype=np.int32)
        transfer_ms = (time.perf_counter() - t0) * 1000.0
        accepted, match_len, kernel_ms = self._runner(
            *_csr_args(nfa),
            self._accept,
            syms,
            int(nfa.num_states),
            int(nfa.start_state),
            int(nfa.uses_any_symbol),
        )
        return Result(
            accepted=bool(accepted),
            match_len=int(match_len),
            kernel_ms=float(kernel_ms),
            total_ms=float(kernel_ms) + transfer_ms,
            transfer_ms=transfer_ms,
        )


class CUDABatchExecutor:
    """Whole batch in a single launch; ``technique`` picks the kernel variant.

    ``run_batch`` is the real path, ``run`` is a batch of one.
    """

    def __init__(self, nfa: NFA, technique: str = "multistream") -> None:
        self.nfa = nfa
        self.technique = technique
        self._accept_words = np.ascontiguousarray(pack_accept(nfa), dtype=np.uint64)
        self._runner: Any = getattr(extension.load(), BATCH_TECHNIQUES[technique])

    def run(self, input_bytes: bytes) -> Result:
        return self.run_batch([input_bytes])[0]

    def run_batch(self, inputs: list[bytes]) -> list[Result]:
        if not inputs:
            return []
        nfa = self.nfa
        t0 = time.perf_counter()
        data, offsets = pack_inputs(inputs)
        transfer_ms = (time.perf_counter() - t0) * 1000.0
        flags, lens, kernel_ms = self._runner(
            *_csr_args(nfa),
            self._accept_words,
            np.ascontiguousarray(data, dtype=np.int32),
            np.ascontiguousarray(offsets, dtype=np.int32),
            int(nfa.num_states),
            int(nfa.start_state),
            int(nfa.uses_any_symbol),
        )
        return batch_results(flags, lens, float(kernel_ms), transfer_ms)


def _make_single(automaton: Automaton, technique: str) -> CUDASingleExecutor:
    assert isinstance(automaton, NFA)
    return CUDASingleExecutor(automaton, technique)


def _make_batch(automaton: Automaton, technique: str) -> CUDABatchExecutor:
    assert isinstance(automaton, NFA)
    return CUDABatchExecutor(automaton, technique)


for _tech in SINGLE_TECHNIQUES:
    register(Kind.NFA, Backend.CUDA, _tech)(_make_single)
for _tech in BATCH_TECHNIQUES:
    register(Kind.NFA, Backend.CUDA, _tech)(_make_batch)
