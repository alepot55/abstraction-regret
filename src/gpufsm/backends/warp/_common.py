"""Warp device staging, timing and the one-time runtime init.

``wp.init()`` used to run as a side effect of importing the backend, i.e. of importing
``gpufsm`` at all: on a machine with warp installed, a plain ``gpufsm list`` paid the
Warp runtime startup. It is now idempotent and deferred to the first executor
construction, so importing the package stays free.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import warp as wp

DEVICE = "cuda"

_initialized = False


def ensure_initialized() -> None:
    """Initialize the Warp runtime once, quietly. Safe to call from anywhere."""
    global _initialized
    if _initialized:
        return
    wp.config.quiet = True  # suppress the init/compile banner on the CLI
    wp.init()
    _initialized = True


def dev_i32(a: np.ndarray) -> wp.array:
    """Upload a numpy array to the device as contiguous int32."""
    return wp.from_numpy(np.ascontiguousarray(a, np.int32), wp.int32, device=DEVICE)


def timed_launch(launch: Callable[[], None]) -> float:
    """Time ``launch`` in ms, synchronizing on both sides so the number is the kernel.

    Warp has no CUDA-event wrapper as convenient as torch's, so this is wall clock
    around a drained queue rather than device-side events; the leading synchronize is
    what keeps prior work out of the measurement.
    """
    wp.synchronize()
    t0 = time.perf_counter()
    launch()
    wp.synchronize()
    return (time.perf_counter() - t0) * 1000.0
