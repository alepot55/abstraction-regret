"""Timing statistics for GPU measurements.

Kernel timings are not gaussian (Hoefler & Belli, SC'15): a few slow samples from
clock/thermal transients skew the mean while leaving the median alone. Everything
here reports **median plus a bootstrap CI95** rather than mean ± std, which is what
the paper's numbers are built on.

Pure Python + numpy: no torch, so it imports and is tested on CPU.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

DEFAULT_WARMUP = 3
DEFAULT_SAMPLES = 9


def bootstrap_ci95(samples: list[float], iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI95 **of the median**, resampled ``iters`` times.

    Deterministic given ``seed``, so a re-run of a committed CSV reproduces the
    interval and not merely the point estimate.
    """
    arr = np.asarray(samples, dtype=float)
    if arr.size < 2:
        point = float(arr[0]) if arr.size else 0.0
        return point, point
    rng = np.random.default_rng(seed)
    medians = np.median(rng.choice(arr, size=(iters, arr.size), replace=True), axis=1)
    return float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


def gbps(total_bytes: int, ms: float) -> float:
    """Throughput in Gbit/s for ``total_bytes`` processed in ``ms`` milliseconds."""
    if ms <= 0:
        return 0.0
    return (total_bytes * 8.0) / (ms * 1e-3) / 1e9


def repeat(
    measure: Callable[[], float],
    warmup: int = DEFAULT_WARMUP,
    samples: int = DEFAULT_SAMPLES,
) -> list[float]:
    """Run ``measure`` ``warmup`` times untimed, then collect ``samples`` timings.

    Non-positive readings are dropped, not recorded as zero: a batched executor
    reports the whole batch's kernel time on the first result and 0.0 on the rest,
    and averaging those zeros in would silently divide the timing by the batch size.
    """
    for _ in range(max(0, warmup)):
        measure()
    out = []
    for _ in range(samples):
        ms = measure()
        if ms > 0:
            out.append(float(ms))
    return out


def summarize(samples: list[float], total_bytes: int, seed: int = 0) -> dict[str, float]:
    """Median / CI95 / throughput for one measurement point, ready for a CSV row."""
    if not samples:
        return {"median_ms": 0.0, "ci95_lo_ms": 0.0, "ci95_hi_ms": 0.0, "gbps": 0.0, "n": 0}
    median = float(np.median(samples))
    lo, hi = bootstrap_ci95(samples, seed=seed)
    return {
        "median_ms": median,
        "ci95_lo_ms": lo,
        "ci95_hi_ms": hi,
        "gbps": gbps(total_bytes, median),
        "n": len(samples),
    }
