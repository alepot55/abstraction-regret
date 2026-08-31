"""Triton backend — the high-level tile/SPMD DSL arm of the study.

One module per technique, each holding its ``@triton.jit`` kernel next to the
executor that launches it and the ``@register`` line that publishes it:

- :mod:`~gpufsm.backends.triton.dense` — int8 slot per state (regret baseline)
- :mod:`~gpufsm.backends.triton.bitpacked` — packed bitmask (byte->bit axis)
- :mod:`~gpufsm.backends.triton.multistream` — one program per string (batched)
- :mod:`~gpufsm.backends.triton.worklist` — work-efficient active set (<=64 states)
- :mod:`~gpufsm.backends.triton.dfa` — the memory-bound DFA gather

The availability guard is here and only here: importing a technique module requires
``torch`` and ``triton``, so on a machine without them nothing registers and the rest
of the package carries on. The probe is registered either way, so ``gpufsm env``
reports ``triton`` as unavailable rather than silently omitting it.
"""

from __future__ import annotations

import importlib

from ...core.registry import Backend, register_availability, set_unavailable_reason

_TECHNIQUE_MODULES = (
    "gpufsm.backends.triton.dense",
    "gpufsm.backends.triton.bitpacked",
    "gpufsm.backends.triton.multistream",
    "gpufsm.backends.triton.worklist",
    "gpufsm.backends.triton.dfa",
)


def triton_available() -> bool:
    """True when torch + triton import and a CUDA device is present."""
    try:
        import torch
        import triton  # noqa: F401

        return bool(torch.cuda.is_available())
    except Exception as exc:
        set_unavailable_reason(Backend.TRITON, f"{type(exc).__name__}: {exc}")
        return False


if triton_available():  # pragma: no cover - requires GPU
    for _module in _TECHNIQUE_MODULES:
        importlib.import_module(_module)

register_availability(Backend.TRITON, triton_available)
