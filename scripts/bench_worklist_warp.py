"""Block-parallel (warp-per-string) vs 1-thread/string worklist — with batch sensitivity.

The single-thread `worklist_global` processes one string's many state-words serially; the
`worklist_warp` kernel spreads them across 32 lanes. The speedup is **batch-dependent**: at
small batch `worklist_global` cannot fill the GPU (few strings = few threads), so warp wins
hugely; at a GPU-saturating batch `worklist_global` has abundant string-level parallelism and
the fair, conservative speedup is smaller (but warp still wins — each global thread still does
~32x more serial per-word work). We therefore report BOTH: a batch-sensitivity sweep on one
automaton, and per-automaton speedups at a saturating batch (the honest headline).

Correctness is gated on the CPU oracle before any timing (both kernels validated bit-for-bit
in the test-suite). Writes paper/data/worklist_warp_rtx4070.csv (saturating-batch, per automaton)
and paper/data/worklist_warp_batch_rtx4070.csv (batch sensitivity).

Usage:  python scripts/bench_worklist_warp.py [--out CSV] [--out-batch CSV]
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

from gpufsm import run_batch
from gpufsm.bench import SPARSE_WORKLIST, random_nfa
from gpufsm.bench.csvio import environment, guard_device, write_rows
from gpufsm.bench.oracle import require
from gpufsm.core.nfa import NFA


def _sparse_nfa(n: int, seed: int):
    """The sparse-worklist family plus its alphabet, which the callers need for inputs."""
    return random_nfa(n, seed, SPARSE_WORKLIST), [ord(c) for c in SPARSE_WORKLIST.alphabet]


SYNTH_SIZES = [512, 2048, 8192]
SAT_STRINGS = 4096  # GPU-saturating batch (>= ~46 SMs * warps): the fair comparison
BATCH_GRID = [64, 256, 1024, 4096, 16384]  # batch-sensitivity sweep
STR_LEN = 256
WARMUP = 3
RUNS = 7
FIELDS = [
    "automaton",
    "num_states",
    "words",
    "n_strings",
    "global_gbps",
    "warp_gbps",
    "speedup",
    "gpu",
]
BATCH_FIELDS = ["num_states", "n_strings", "global_gbps", "warp_gbps", "speedup", "gpu"]


def _median_ms(nfa, batch, tech):
    for _ in range(WARMUP):
        run_batch(nfa, batch, backend="cuda", technique=tech)
    return statistics.median(
        run_batch(nfa, batch, backend="cuda", technique=tech)[0].kernel_ms for _ in range(RUNS)
    )


def _speedup(nfa: NFA, batch: list[bytes]) -> tuple[float, float, None]:
    """(global_gbps, warp_gbps, None) after gating both kernels against the CPU oracle.

    Checking the two kernels only against each other would pass whenever they are wrong in
    the same way, which is exactly what a shared eps-closure helper makes likely. The gate
    is the reference simulator, on a prefix of the batch.
    """
    for tech in ("worklist_global", "worklist_warp"):
        require(nfa, batch, backend="cuda", technique=tech)
    tg = _median_ms(nfa, batch, "worklist_global")
    tw = _median_ms(nfa, batch, "worklist_warp")
    bits = len(batch) * STR_LEN * 8
    return bits / (tg * 1e-3) / 1e9, bits / (tw * 1e-3) / 1e9, None


def main() -> int:
    # This is the one driver that writes two files, so it cannot use `driver_out`; the
    # first flag is spelled the same way on purpose.
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--out", default="paper/data/worklist_warp_rtx4070.csv")
    parser.add_argument("--out-batch", default="paper/data/worklist_warp_batch_rtx4070.csv")
    args = parser.parse_args()
    out, outb = Path(args.out), Path(args.out_batch)
    guard_device(out)
    guard_device(outb)
    gpu = environment()["gpu"]

    rng = random.Random(0)

    # (1) batch sensitivity on one fixed automaton — documents the confound.
    batch_rows: list[dict[str, object]] = []
    nfa, alpha = _sparse_nfa(8192, seed=8192)
    print("batch-sensitivity (8192-state synthetic, 128 words):")
    for nstr in BATCH_GRID:
        batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(nstr)]
        g, w, _ = _speedup(nfa, batch)
        batch_rows.append(
            {
                "num_states": 8192,
                "n_strings": nstr,
                "global_gbps": round(g, 4),
                "warp_gbps": round(w, 4),
                "speedup": round(w / g, 1),
                "gpu": gpu,
            }
        )
        print(f"  n_strings={nstr:6d}: global={g:8.4f} warp={w:8.4f} speedup={w / g:6.1f}x")

    # (2) per-automaton at a SATURATING batch — the honest, conservative headline.
    rows: list[dict[str, object]] = []
    print(f"\nsaturating batch ({SAT_STRINGS} strings) — fair speedup:")
    for n in SYNTH_SIZES:
        nfa, alpha = _sparse_nfa(n, seed=n)
        batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(SAT_STRINGS)]
        g, w, _ = _speedup(nfa, batch)
        rows.append(
            {
                "automaton": "synthetic",
                "num_states": n,
                "words": (n + 63) // 64,
                "n_strings": SAT_STRINGS,
                "global_gbps": round(g, 4),
                "warp_gbps": round(w, 4),
                "speedup": round(w / g, 1),
                "gpu": gpu,
            }
        )
        print(f"  synthetic n={n:6d}: global={g:8.4f} warp={w:8.4f} speedup={w / g:6.1f}x")

    try:
        from gpufsm.io.anml import load_anml
        from gpufsm.io.datasets import DATASETS, ensure

        for key in ["levenshtein", "fermi", "brill"]:
            nfa = load_anml(ensure(DATASETS[key], "data/anmlzoo"))
            alpha = sorted({int(s) for s in nfa.sym_symbols if 0 <= int(s) <= 255}) or [97]
            batch = [bytes(rng.choice(alpha) for _ in range(STR_LEN)) for _ in range(SAT_STRINGS)]
            g, w, _ = _speedup(nfa, batch)
            rows.append(
                {
                    "automaton": key,
                    "num_states": nfa.num_states,
                    "words": (nfa.num_states + 63) // 64,
                    "n_strings": SAT_STRINGS,
                    "global_gbps": round(g, 4),
                    "warp_gbps": round(w, 4),
                    "speedup": round(w / g, 1),
                    "gpu": gpu,
                }
            )
            print(
                f"  {key:11s} n={nfa.num_states:6d}: global={g:8.4f} warp={w:8.4f} "
                f"speedup={w / g:6.1f}x"
            )
    except Exception as e:
        print(f"(real-automata pass skipped: {type(e).__name__}: {e})")

    write_rows(out, rows, FIELDS)
    write_rows(outb, batch_rows, BATCH_FIELDS)
    print(f"\nwrote {out} ({len(rows)} rows) and {outb} ({len(batch_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
