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

# Import by name so a missing optional dependency is a no-op rather than a hard
# failure. Each subpackage keeps its own guard; this loop only tolerates the case
# where the subpackage itself cannot be imported at all.
for _backend in (
    "gpufsm.backends.triton",
    "gpufsm.backends.cuda",
    "gpufsm.backends.warp",
):
    try:
        importlib.import_module(_backend)
    except Exception:  # pragma: no cover - depends on environment
        pass
