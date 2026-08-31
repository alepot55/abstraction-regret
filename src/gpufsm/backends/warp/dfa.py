"""Warp DFA kernel — the memory-bound face on the thread-SIMT arm.

One thread per string walking ``cur = trans[cur * 256 + byte]``. Paired with
:mod:`gpufsm.backends.triton.dfa` (same algorithm, tile/SPMD DSL) and the CUDA
kernel, it puts the thread-vs-tile contrast on the memory-bound workload too.
"""

from __future__ import annotations

import time

import numpy as np
import warp as wp

from ...core.dfa import DFA
from ...core.packing import pack_inputs
from ...core.registry import Automaton, Backend, Kind, register
from ...core.result import Result, batch_results
from ._common import DEVICE, ensure_initialized, timed_launch


@wp.kernel
def _dfa_kernel(
    trans: wp.array(dtype=wp.int32),
    accept: wp.array(dtype=wp.int32),
    input_data: wp.array(dtype=wp.int32),
    input_offsets: wp.array(dtype=wp.int32),
    num_strings: wp.int32,
    start_state: wp.int32,
    out_flags: wp.array(dtype=wp.int32),
    out_lens: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    lo = input_offsets[i]
    hi = input_offsets[i + 1]
    cur = start_state
    out_f = int(0)  # int(...) declares a mutable wp.int32 local; bare 0 miscompiles
    out_l = int(0)
    done = int(0)
    if accept[cur] != 0:
        out_f = 1
        done = 1
    pos = lo
    while pos < hi and done == 0:
        cur = trans[cur * 256 + input_data[pos]]
        if accept[cur] != 0:
            out_f = 1
            out_l = pos - lo + 1
            done = 1
        pos = pos + 1
    out_flags[i] = out_f
    out_lens[i] = out_l


class WarpDFAExecutor:
    """One Warp thread per string; dense transition table resident on the GPU."""

    def __init__(self, dfa: DFA, technique: str = "gather") -> None:
        ensure_initialized()
        self.dfa = dfa
        self.technique = technique
        self._trans = wp.from_numpy(
            np.ascontiguousarray(dfa.trans, np.int32), wp.int32, device=DEVICE
        )
        self._accept = wp.from_numpy(dfa.accept.astype(np.int32), wp.int32, device=DEVICE)

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
        flags = wp.zeros(n, dtype=wp.int32, device=DEVICE)
        lens = wp.zeros(n, dtype=wp.int32, device=DEVICE)
        transfer_ms = (time.perf_counter() - t0) * 1000.0

        kernel_ms = timed_launch(
            lambda: wp.launch(
                _dfa_kernel,
                dim=n,
                inputs=[
                    self._trans,
                    self._accept,
                    data,
                    off,
                    n,
                    int(self.dfa.start_state),
                    flags,
                    lens,
                ],
                device=DEVICE,
            )
        )
        return batch_results(flags.numpy(), lens.numpy(), kernel_ms, transfer_ms)


@register(Kind.DFA, Backend.WARP, "gather", default=True)
def _make(automaton: Automaton, technique: str) -> WarpDFAExecutor:
    assert isinstance(automaton, DFA)
    return WarpDFAExecutor(automaton, technique)
