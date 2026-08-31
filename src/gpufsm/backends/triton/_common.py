"""Device-side staging and timing shared by the Triton executors.

Every Triton technique needs the same two things: the NFA's CSR arrays resident on
the GPU, and a CUDA-event-timed launch. Each technique used to open-code both, which
is how four copies of the same eight lines ended up in one file.

Importing this module requires ``torch``; the guard lives in the package
``__init__`` so a machine without it simply registers no Triton techniques.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch

from ...core.nfa import NFA
from ...core.packing import pack_inputs, symbols


class DeviceCSR:
    """The NFA's CSR arrays staged on the GPU, built once per executor."""

    __slots__ = (
        "device",
        "sym_row_ptr",
        "sym_targets",
        "sym_symbols",
        "eps_row_ptr",
        "eps_targets",
    )

    def __init__(self, nfa: NFA, device: torch.device) -> None:
        self.device = device
        self.sym_row_ptr = torch.as_tensor(nfa.sym_row_ptr, device=device)
        self.sym_targets = torch.as_tensor(nfa.sym_targets, device=device)
        self.sym_symbols = torch.as_tensor(nfa.sym_symbols, device=device)
        self.eps_row_ptr = torch.as_tensor(nfa.eps_row_ptr, device=device)
        self.eps_targets = torch.as_tensor(nfa.eps_targets, device=device)

    @property
    def args(self) -> tuple[torch.Tensor, ...]:
        """The five CSR tensors in the order every kernel declares them."""
        return (
            self.sym_row_ptr,
            self.sym_targets,
            self.sym_symbols,
            self.eps_row_ptr,
            self.eps_targets,
        )


def stage_input(input_bytes: bytes, device: torch.device) -> tuple[torch.Tensor, int, float]:
    """Upload one input string. Returns ``(tensor, length, transfer_ms)``."""
    t0 = time.perf_counter()
    syms = symbols(input_bytes)
    tensor = torch.as_tensor(syms, device=device)
    return tensor, int(syms.size), (time.perf_counter() - t0) * 1000.0


def stage_batch(
    inputs: list[bytes], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Upload a batch. Returns ``(data, offsets, transfer_ms)``."""
    t0 = time.perf_counter()
    data_np, offsets_np = pack_inputs(inputs)
    data = torch.as_tensor(data_np, device=device)
    offsets = torch.as_tensor(offsets_np, device=device)
    return data, offsets, (time.perf_counter() - t0) * 1000.0


def timed_launch(launch: Callable[[], None]) -> float:
    """Run ``launch`` between CUDA events; return the elapsed kernel time in ms.

    CUDA events (not wall clock) so the number is the device-side kernel time,
    excluding the host-side launch overhead the transfer timing already accounts for.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    launch()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))
