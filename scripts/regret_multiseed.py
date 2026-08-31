"""Statistical hardening of the headline regret: multi-seed Triton/Warp vs CUDA.

The 2x2 / abstraction-regret numbers are the paper's centerpiece, so they must be robust to the
specific random NFA, not a single-seed artifact. This measures the multistream throughput regret
(CUDA / DSL) across several seeds and sizes and reports the median + min-max spread. Writes
paper/data/regret_multiseed_rtx4070.csv. (All three backends run at <=64 states; Warp's single-word
kernel caps there. Warp init may intermittently throw a sticky CUDA-716 -> just rerun.)

Usage:  python scripts/regret_multiseed.py
"""

from __future__ import annotations

import random
import statistics
import sys
from pathlib import Path

from gpufsm import NFABuilder, run_batch
from gpufsm.bench.csvio import environment, guard_device, write_rows

SIZES = [32, 48, 64]
SEEDS = range(5)
N_STRINGS, SLEN = 2048, 256
WARMUP, RUNS = 2, 7
FIELDS = [
    "num_states",
    "triton_regret_med",
    "triton_min",
    "triton_max",
    "warp_regret_med",
    "warp_min",
    "warp_max",
    "seeds",
    "gpu",
]


def _mk(n: int, seed: int):
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.1)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)):
            b.add_transition(s, ord(rng.choice("abcde")), rng.randrange(n))
    return b.build()


def main() -> int:
    out = Path("paper/data/regret_multiseed_rtx4070.csv")
    guard_device(out)
    gpu = environment()["gpu"]

    batch = [bytes(b"abcde"[i % 5] for i in range(SLEN)) for _ in range(N_STRINGS)]
    bits = N_STRINGS * SLEN * 8

    def gbps(nfa, be, te):
        for _ in range(WARMUP):
            run_batch(nfa, batch, be, te)
        ms = statistics.median(run_batch(nfa, batch, be, te)[0].kernel_ms for _ in range(RUNS))
        return bits / (ms * 1e-3) / 1e9

    rows: list[dict[str, object]] = []
    print(f"{'n':>4}{'T_regret med[min-max]':>24}{'W_regret med[min-max]':>24}")
    for n in SIZES:
        tr, wr = [], []
        for seed in SEEDS:
            nfa = _mk(n, 1000 + seed * 7 + n)
            c = gbps(nfa, "cuda", "multistream")
            tr.append(c / gbps(nfa, "triton", "multistream"))
            wr.append(c / gbps(nfa, "warp", "multistream"))
        rows.append(
            {
                "num_states": n,
                "triton_regret_med": round(statistics.median(tr), 2),
                "triton_min": round(min(tr), 2),
                "triton_max": round(max(tr), 2),
                "warp_regret_med": round(statistics.median(wr), 2),
                "warp_min": round(min(wr), 2),
                "warp_max": round(max(wr), 2),
                "seeds": len(list(SEEDS)),
                "gpu": gpu,
            }
        )
        print(
            f"{n:4d}   {statistics.median(tr):5.2f} [{min(tr):.2f}-{max(tr):.2f}]"
            f"        {statistics.median(wr):5.2f} [{min(wr):.2f}-{max(wr):.2f}]"
        )

    print(f"\nwrote {write_rows(out, rows, FIELDS)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
