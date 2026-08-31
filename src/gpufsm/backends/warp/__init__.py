"""NVIDIA Warp backend — the Python *thread-SIMT* arm of the abstraction spectrum.

Warp JIT-compiles Python kernels to CUDA C++ with a thread model (``wp.tid()``), so —
unlike tile/SPMD DSLs (Triton) or tensor-only DSLs (cuTile, CuTe, ThunderKittens) — it
expresses the data-dependent per-state control flow, dynamic loops and bit
manipulation an NFA needs. Same Python-level productivity as Triton, different
execution paradigm: that is the contrast the study is built on.

- :mod:`~gpufsm.backends.warp.nfa` — ``multistream``, one thread per string
- :mod:`~gpufsm.backends.warp.dfa` — ``gather``, the memory-bound face
"""

from __future__ import annotations

import importlib

from ...core.registry import Backend, register_availability

_TECHNIQUE_MODULES = (
    "gpufsm.backends.warp.nfa",
    "gpufsm.backends.warp.dfa",
)


def warp_available() -> bool:
    """True when warp imports and reports at least one CUDA device."""
    try:
        import warp as wp

        return bool(wp.get_cuda_device_count() > 0)
    except Exception:
        return False


if warp_available():  # pragma: no cover - requires GPU + warp
    for _module in _TECHNIQUE_MODULES:
        importlib.import_module(_module)

register_availability(Backend.WARP, warp_available)
