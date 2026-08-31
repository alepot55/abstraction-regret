"""Finer DFA throughput sweep across table sizes — exposes the L2 memory-bound knee.

The DFA dense-table walk is the memory-bound face of the two-faces thesis: one random
``trans[s*256 + byte]`` lookup per input byte. As ``num_states`` grows the table
(``num_states`` KB, since 256x int32 = 1 KB/state) crosses the GPU's L2 and throughput
should drop for the thread-model backends (CUDA, Warp) while the tile/SPMD Triton kernel
stays flat (scalar-gather-bound, never reaching the memory regime).

Sweeps a fine grid of table sizes for cuda/warp/triton, validates each against the CPU
oracle first, then reports median-of-N batch-kernel throughput. Writes the canonical
``paper/data/dfa_regret_rtx4070.csv`` consumed by ``paper/figures.py``.

Usage:  python scripts/sweep_dfa.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

from gpufsm.api import run_batch
from gpufsm.bench.csvio import environment, guard_device, write_rows
from gpufsm.bench.oracle import OracleMismatch, require
from gpufsm.core.dfa import random_dfa

# 1 KB of transition table per state (256 x int32), so num_states is also the table size
# in KB. The grid spans 1 MB to 100 MB so that it straddles the L2 capacity of any current
# device. The capacity itself is read from the device at run time (environment()['l2_mb'])
# rather than hardcoded: it is a claim about the hardware, and it belongs in the CSV.
STATE_GRID = [1024, 2048, 4096, 6144, 8192, 16384, 32768, 50000, 100000]
N_STRINGS = 4096
STR_LEN = 1024  # 4 MB of input per batch -> stable timing, table-walk dominates
WARMUP = 3
RUNS = 9
BACKENDS = ("cuda", "warp", "triton")
FIELDS = ["backend", "num_states", "table_kb", "throughput_gbps", "gpu", "note"]


def _throughput_gbps(total_bytes: int, kernel_ms: float) -> float:
    if kernel_ms <= 0:
        return float("nan")
    return (total_bytes * 8.0) / (kernel_ms * 1e-3) / 1e9


def _validate(dfa, rng: random.Random) -> bool:
    """Oracle check on a small batch, separate from the timing batch.

    Uses the shared gate rather than a local comparison: one implementation of "agrees with
    the reference" is what keeps the failure message and the sample size identical across
    every driver.
    """
    batch = [bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 48))) for _ in range(32)]
    for be in BACKENDS:
        try:
            require(dfa, batch, backend=be)
        except OracleMismatch as exc:
            print(f"  VALIDATION FAIL backend={be}: {exc}")
            return False
    return True


def _measure(dfa, batch: list[bytes], backend: str) -> float:
    total = sum(len(b) for b in batch)
    for _ in range(WARMUP):
        run_batch(dfa, batch, backend=backend)
    samples = []
    for _ in range(RUNS):
        res = run_batch(dfa, batch, backend=backend)
        samples.append(_throughput_gbps(total, res[0].kernel_ms))
    return statistics.median(samples)


SEEDS = (0, 1, 2)  # median over 3 random DFAs/size: the knee must be seed-robust, not noise


def main() -> int:
    out = Path("paper/data/dfa_regret_rtx4070.csv")
    guard_device(out)
    env = environment()
    gpu = env["gpu"]
    l2_mb = env.get("l2_mb", "")
    l2_kb = int(float(l2_mb) * 1024) if l2_mb else 0

    rng = random.Random(0)
    # one fixed timing batch (random bytes) reused across sizes for comparability
    timing_batch = [bytes(rng.randint(0, 255) for _ in range(STR_LEN)) for _ in range(N_STRINGS)]
    rows: list[dict[str, object]] = []

    for n in STATE_GRID:
        # validate one seed; then measure each backend's throughput as the median over SEEDS
        # (different random DFAs) — so a reported knee reflects the table size, not one DFA.
        if not _validate(random_dfa(n, accept_prob=0.02, seed=n), random.Random(n)):
            print(f"n={n}: validation failed, skipping")
            continue
        table_kb = n  # 256 * int32 = 1 KB/state
        if l2_kb:
            cache = f"fits L2 ({l2_mb} MB)" if table_kb <= l2_kb else f"exceeds L2 ({l2_mb} MB)"
        else:
            cache = "L2 capacity unknown"
        line = f"n={n:6d} ({table_kb / 1024:.1f} MB, {cache}): "
        for be in BACKENDS:
            per_seed = [
                _measure(random_dfa(n, accept_prob=0.02, seed=s * 1000 + n), timing_batch, be)
                for s in SEEDS
            ]
            tp = statistics.median(per_seed)
            rows.append(
                {
                    "backend": be,
                    "num_states": n,
                    "table_kb": table_kb,
                    "throughput_gbps": round(tp, 1),
                    "gpu": gpu,
                    "note": cache,
                }
            )
            line += f"{be}={tp:6.1f}  "
        print(line)

    print(f"\nwrote {write_rows(out, rows, FIELDS)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
