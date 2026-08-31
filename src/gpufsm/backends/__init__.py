"""Backend registration.

Importing this package is what populates :mod:`gpufsm.core.registry`. The CPU
reference is always available; each GPU backend is a subpackage that probes its own
dependencies and registers techniques only if they are present, so the core stays
installable and testable on CPU-only machines.

Every backend registers its availability probe unconditionally, so ``gpufsm env``
can report a backend as *unavailable* instead of pretending it does not exist.
"""

from __future__ import annotations

import importlib

from . import cpu  # noqa: F401  (always available)

IMPORT_ERRORS: dict[str, str] = {}
"""Why a backend subpackage failed to import, keyed by backend name.

An optional dependency that is simply absent is expected and uninteresting. A backend that
is *installed* and still fails -- a CUDA extension built against the wrong Python, a Triton
version whose API moved -- is a broken measurement arm, and swallowing that silently is how
a sweep comes to report three backends where there should be four with nothing saying so.
The exception is recorded here and ``gpufsm env`` prints it.
"""

# Import by name so a missing optional dependency is a no-op rather than a hard failure.
# Each subpackage keeps its own guard; this loop only tolerates the case where the
# subpackage itself cannot be imported at all -- and records why.
for _backend in (
    "gpufsm.backends.triton",
    "gpufsm.backends.cuda",
    "gpufsm.backends.warp",
):
    try:
        importlib.import_module(_backend)
    except Exception as _exc:  # pragma: no cover - depends on environment
        IMPORT_ERRORS[_backend.rsplit(".", 1)[-1]] = f"{type(_exc).__name__}: {_exc}"
