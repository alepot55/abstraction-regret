"""Shared-memory vs global-memory working-set for the warp-cooperative worklist.

worklist_shared stages the per-string working set (cur/nxt/frontier/newb) in dynamic shared
memory instead of global; worklist_warp keeps it in global. This isolates the effect of
working-set *residency/layout* on the work-efficient kernel. Capped at ~1536 states (the
working set must fit 48 KB). Writes paper/data/worklist_shared_rtx4070.csv.

Both techniques are gated against the CPU oracle before anything is timed. Comparing the
two GPU kernels only to each other would pass whenever they are wrong in the same way,
which is the failure mode a shared-memory port is most likely to have.

Usage:  python scripts/bench_worklist_shared.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

from gpufsm import run_batch
from gpufsm.bench import SPARSE_WORKLIST, random_nfa
from gpufsm.bench.csvio import environment, guard_device, write_rows
from gpufsm.bench.oracle import require
from gpufsm.core.nfa import NFA


def _sparse_nfa(n: int, seed: int) -> tuple[NFA, list[int]]:
    """The sparse-worklist family plus its alphabet, which the callers need for inputs."""
    return random_nfa(n, seed, SPARSE_WORKLIST), [ord(c) for c in SPARSE_WORKLIST.alphabet]


SIZES = [256, 512, 1024, 1536]  # all fit 48 KB shared (4*words*8 bytes)
N_STRINGS = 256
STR_LEN = 256
WARMUP = 3
RUNS = 7
TECHNIQUES = ("worklist_warp", "worklist_shared")
FIELDS = ["num_states", "words", "warp_gbps", "shared_gbps", "shared_over_warp", "gpu"]


def _median_ms(nfa: NFA, batch: list[bytes], tech: str) -> float:
    for _ in range(WARMUP):
        run_batch(nfa, batch, backend="cuda", technique=tech)
    return statistics.median(
        run_batch(nfa, batch, backend="cuda", technique=tech)[0].kernel_ms for _ in range(RUNS)
    )


def main() -> int:
    out = Path("paper/data/worklist_shared_rtx4070.csv")
    guard_device(out)
    gpu = environment()["gpu"]

    rng = random.Random(0)
    rows: list[dict[str, object]] = []
    for n in SIZES:
        nfa, alpha = _sparse_nfa(n, seed=n)
        batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(N_STRINGS)]
        for tech in TECHNIQUES:
            require(nfa, batch, backend="cuda", technique=tech)
        tw = _median_ms(nfa, batch, "worklist_warp")
        ts = _median_ms(nfa, batch, "worklist_shared")
        words = (n + 63) // 64
        bits = N_STRINGS * STR_LEN * 8
        w_gbps, s_gbps = bits / (tw * 1e-3) / 1e9, bits / (ts * 1e-3) / 1e9
        rows.append(
            {
                "num_states": n,
                "words": words,
                "warp_gbps": round(w_gbps, 3),
                "shared_gbps": round(s_gbps, 3),
                "shared_over_warp": round(s_gbps / w_gbps, 2),
                "gpu": gpu,
            }
        )
        print(
            f"n={n:5d} words={words:3d}: warp={w_gbps:6.2f} shared={s_gbps:6.2f} "
            f"shared/warp={s_gbps / w_gbps:.2f}x"
        )

    print(f"\nwrote {write_rows(out, rows, FIELDS)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
