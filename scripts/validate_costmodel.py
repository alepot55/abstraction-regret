"""Pressure-test the two-parameter cost model: is it predictive or just an overfit?

The model is time_per_symbol = a*traffic + b*num_states^2 (gpufsm.costmodel). With 4 points
per backend, fitting and "predicting" the same 4 points is near-tautological. Two honest tests:

1. HOLDOUT: fit on the small-n points, predict the largest unseen n. A predictive model has
   small holdout error.
2. LEAVE-ONE-OUT: refit dropping each point; a robust fit has a stable compute constant b.

Finding (RTX 4070): the model is predictive + stable for the thread-model backend (CUDA,
~3% holdout, b stable) but NOT for the tile/SPMD backend (Triton, ~45% holdout, b swings 2x) —
Triton's large fixed launch overhead at small n is misattributed to the n^2 term by the
2-parameter model. Consequence: the *measured* throughput ratio is the primary regret metric;
the fitted-b ratio is corroborating and should not be over-claimed for Triton.

Usage:  python scripts/validate_costmodel.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("paper/data/costmodel_rtx4070.csv")


def _fit(sub: pd.DataFrame) -> tuple[float, float]:
    a_mat = np.c_[sub.traffic_bytes_per_sym.to_numpy(float), sub.num_states.to_numpy(float) ** 2]
    rhs = 8e-9 / sub.throughput_gbps.to_numpy(float)  # seconds per symbol
    coef, *_ = np.linalg.lstsq(a_mat, rhs, rcond=None)
    return max(float(coef[0]), 1e-18), max(float(coef[1]), 0.0)


def _pred_gbps(a: float, b: float, traffic: float, n: float) -> float:
    t = a * traffic + b * n * n
    return (8.0 / (t * 1e9)) if t > 0 else float("inf")


def main() -> int:
    cm = pd.read_csv(DATA)
    print("HOLDOUT — fit on n<=128, predict largest n (unseen):")
    print(f"  {'backend':8}{'pred':>10}{'measured':>10}{'error':>9}")
    for be in ["cuda", "triton"]:
        sub = cm[cm.backend == be]
        train, test = sub[sub.num_states <= 128], sub[sub.num_states == sub.num_states.max()]
        if len(train) < 2 or len(test) == 0:
            continue
        a, b = _fit(train)
        n = float(test.num_states.iloc[0])
        pred = _pred_gbps(a, b, float(test.traffic_bytes_per_sym.iloc[0]), n)
        meas = float(test.throughput_gbps.iloc[0])
        print(f"  {be:8}{pred:10.4f}{meas:10.4f}{abs(pred - meas) / meas * 100:8.1f}%")

    print("\nLEAVE-ONE-OUT — compute constant b, dropping each point (stability):")
    for be in ["cuda", "triton"]:
        sub = cm[cm.backend == be]
        all_b = _fit(sub)[1]
        loo = [_fit(sub[sub.num_states != n])[1] for n in sub.num_states.unique()]
        spread = max(loo) / min(loo) if min(loo) > 0 else float("inf")
        print(
            f"  {be:8} b={all_b:.3e}  LOO spread={spread:.2f}x  "
            f"({'stable' if spread < 1.3 else 'UNSTABLE'})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
