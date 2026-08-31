"""Does the DFA regret survive when every backend walks the whole input?

The DFA sweep uses `random_dfa(n, accept_prob=0.02)`, on which every string latches after
about 50 of its 1024 bytes. CUDA and Warp stop there; the Triton kernels run the loop to
completion with the body predicated off. Throughput credits all three with the full 1024
bytes, so the comparison contains a trip-count asymmetry of roughly 20x that has nothing to
do with the abstraction.

This is the control. It measures the same three backends twice:

  latching     accept_prob=0.02 -- the paper's configuration
  non-latching accept_prob=0.00 -- no string ever accepts, so no kernel can exit early and
                                   all three walk exactly the same 1024 bytes

If the tile/SPMD regret is real, it survives the second regime. If it collapses, the DFA
half of the result was measuring the early exit.

Prints the L2 capacity it read from the device, because the regime boundary is only
meaningful against a cache size that was measured rather than assumed.

    python scripts/dfa_latch_control.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

from gpufsm.api import run_batch
from gpufsm.bench.csvio import environment, write_rows
from gpufsm.core.dfa import random_dfa
from gpufsm.reference import simulate_dfa

STATE_GRID = [1024, 4096, 16384, 65536]
N_STRINGS = 2048
STR_LEN = 1024
WARMUP, RUNS = 3, 7
BACKENDS = ("cuda", "warp", "triton")
FIELDS = [
    "regime",
    "accept_prob",
    "backend",
    "num_states",
    "table_mb",
    "throughput_gbps",
    "mean_bytes_examined",
    "accept_rate",
    "gpu",
    "l2_mb",
]


def _batch(seed: int) -> tuple[list[bytes], int]:
    import random

    rng = random.Random(seed)
    batch = [bytes(rng.randint(0, 255) for _ in range(STR_LEN)) for _ in range(N_STRINGS)]
    return batch, sum(len(b) for b in batch)


def _oracle_profile(dfa, batch: list[bytes], sample: int = 200) -> tuple[float, float]:
    """(accept rate, mean bytes examined) — how much input a latching kernel would skip."""
    verdicts = [simulate_dfa(dfa, b) for b in batch[:sample]]
    rate = sum(1 for a, _ in verdicts if a) / len(verdicts)
    seen = statistics.mean(ln if a else STR_LEN for a, ln in verdicts)
    return rate, seen


def _gbps(dfa, batch: list[bytes], total_bytes: int, backend: str) -> float:
    for _ in range(WARMUP):
        run_batch(dfa, batch, backend=backend)
    samples = []
    for _ in range(RUNS):
        res = run_batch(dfa, batch, backend=backend)
        ms = res[0].kernel_ms
        if ms > 0:
            samples.append(total_bytes * 8.0 / (ms * 1e-3) / 1e9)
    return statistics.median(samples) if samples else 0.0


def main() -> int:
    env = environment()
    if env["gpu"] == "(none)":
        print("SKIP: no CUDA device")
        return 0
    l2 = env.get("l2_mb", "?")
    print(f"gpu: {env['gpu']}   L2: {l2} MB   triton: {env.get('triton', '?')}")
    print(f"batch: {N_STRINGS} x {STR_LEN} B\n")

    batch, total_bytes = _batch(seed=99)
    rows: list[dict[str, object]] = []

    for regime, accept_prob in (("latching", 0.02), ("non-latching", 0.0)):
        print(f"== {regime} (accept_prob={accept_prob}) ==")
        print(
            f"{'states':>8} {'MB':>6} {'bytes seen':>11} " + "".join(f"{b:>12}" for b in BACKENDS)
        )
        for n in STATE_GRID:
            dfa = random_dfa(n, accept_prob=accept_prob, seed=n)
            rate, seen = _oracle_profile(dfa, batch)
            speeds = {}
            for backend in BACKENDS:
                # Gate on the oracle before believing any number from this kernel.
                got = run_batch(dfa, batch[:32], backend=backend)
                ref = [simulate_dfa(dfa, b) for b in batch[:32]]
                if [(r.accepted, r.match_len) for r in got] != ref:
                    print(f"  ORACLE MISMATCH on {backend} at n={n} — aborting")
                    return 1
                speeds[backend] = _gbps(dfa, batch, total_bytes, backend)
            table_mb = n / 1024.0
            print(
                f"{n:>8} {table_mb:>6.1f} {seen:>11.1f} "
                + "".join(f"{speeds[b]:>12.1f}" for b in BACKENDS)
            )
            for backend in BACKENDS:
                rows.append(
                    {
                        "regime": regime,
                        "accept_prob": accept_prob,
                        "backend": backend,
                        "num_states": n,
                        "table_mb": round(table_mb, 2),
                        "throughput_gbps": round(speeds[backend], 2),
                        "mean_bytes_examined": round(seen, 1),
                        "accept_rate": round(rate, 3),
                        "gpu": env["gpu"],
                        "l2_mb": l2,
                    }
                )
        print()

    def ratio(regime: str, n: int) -> float:
        get = {
            r["backend"]: r["throughput_gbps"]
            for r in rows
            if r["regime"] == regime and r["num_states"] == n
        }
        t = get.get("triton", 0.0)
        return (get.get("cuda", 0.0) / t) if t else 0.0

    print("CUDA/Triton regret by regime:")
    for n in STATE_GRID:
        lat, non = ratio("latching", n), ratio("non-latching", n)
        print(f"  n={n:>6}: latching {lat:6.2f}x   non-latching {non:6.2f}x")

    out = Path("paper/data/dfa_latch_control.csv")
    print(f"\nwrote {write_rows(out, rows, FIELDS)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
