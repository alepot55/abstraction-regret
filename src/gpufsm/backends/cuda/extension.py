"""Locating the compiled ``_cuda`` extension — the one place that knows it may be absent.

The extension is optional: with no CUDA toolkit at build time the package still
installs and runs on CPU/Triton. Everything that needs it goes through :func:`load`,
so the import-failure handling is written once rather than at every call site.
"""

from __future__ import annotations

import importlib
from typing import Any

_MODULE = "gpufsm.backends.cuda._cuda"

_cached: Any | None = None


def load() -> Any:
    """Import the extension, caching it. Raises if it was never built."""
    global _cached
    if _cached is None:
        _cached = importlib.import_module(_MODULE)
    return _cached


def available() -> bool:
    """True when the extension imports — i.e. it was built and its deps resolve."""
    try:
        load()
        return True
    except Exception:
        return False
