"""CUDA backend — the hand-written, low-level arm of the study (the 1.0x reference).

The kernels live in ``native/`` and are compiled into the ``_cuda`` pybind11 extension
by the CMake build when ``GPUFSM_BUILD_CUDA=ON``; on CPU-only installs the extension
is simply absent and nothing here registers.

- :mod:`~gpufsm.backends.cuda.extension` — locates the compiled module
- :mod:`~gpufsm.backends.cuda.nfa` — the ``dense``/``bitpacked``/multi-stream/worklist family
- :mod:`~gpufsm.backends.cuda.dfa` — the memory-bound DFA gather
"""

from __future__ import annotations

import importlib

from ...core.registry import Backend, register_availability
from . import extension

_TECHNIQUE_MODULES = (
    "gpufsm.backends.cuda.nfa",
    "gpufsm.backends.cuda.dfa",
)

if extension.available():  # pragma: no cover - requires the compiled extension
    for _module in _TECHNIQUE_MODULES:
        importlib.import_module(_module)

register_availability(Backend.CUDA, extension.available)
