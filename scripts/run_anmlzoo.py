"""Real ANMLZoo automata: validate GPU == the reference oracle, then measure throughput.

Random NFAs make the regret measurement controllable, but they are not evidence that the
engine is correct at production scale. This driver closes that gap on the six pinned
ANMLZoo families (2.8k to 48k states, up to 6.3M symbol transitions): each is fetched with
its SHA-256 verified, parsed with all-input/start-of-data semantics, checked bit-for-bit
against the CPU oracle, and only then timed.

It measures the two scalable CUDA worklist techniques against each other --
``worklist_global`` (one thread per string) and ``worklist_warp`` (one warp per string,
lanes partitioning the state words) -- because the block-parallel speedup on *real*
automata is the claim; on random NFAs the active set is too uniform to exercise it.

    python scripts/run_anmlzoo.py                    # all six, oracle-gated, writes the CSV
    python scripts/run_anmlzoo.py --only levenshtein # one family, no CSV
    python scripts/run_anmlzoo.py --no-csv           # measure without writing

The oracle gate runs on a small prefix of the batch, not the whole of it: the CPU
simulator is O(active set x transitions) per byte, and 2048 strings against a 48k-state
automaton would dominate the runtime of a check whose purpose is to catch a wrong kernel,
which the first few dozen strings already do.
"""

from __future__ import annotations

import argparse
import random
import sys
import time

from gpufsm.bench.csvio import gpu_slug, print_environment, write_rows
from gpufsm.bench.oracle import require
from gpufsm.bench.timing import repeat, summarize
from gpufsm.core.nfa import NFA
from gpufsm.io.anml import load_anml
from gpufsm.io.datasets import DATASETS, ensure

CACHE_DIR = "data/anmlzoo"
TECHNIQUES = ("worklist_global", "worklist_warp")
FIELDS = [
    "automaton",
    "num_states",
    "transitions",
    "global_gbps",
    "warp_gbps",
    "speedup",
    "n_strings",
    "gpu",
]


def build_batch(nfa: NFA, n_strings: int, max_len: int, seed: int) -> tuple[list[bytes], int]:
    """A deterministic input batch drawn from the automaton's own alphabet.

    Bytes outside the alphabet would die in the first step and measure nothing but the
    launch overhead, so the batch is drawn from the symbols the automaton actually reads.
    """
    alphabet = sorted({int(s) for s in nfa.sym_symbols if 0 <= int(s) <= 255}) or [97]
    rng = random.Random(seed)
    batch = [
        bytes(rng.choice(alphabet) for _ in range(rng.randint(0, max_len)))
        for _ in range(n_strings)
    ]
    return batch, sum(len(d) for d in batch)


def measure(nfa: NFA, batch: list[bytes], total_bytes: int, technique: str) -> float:
    """Throughput in Gbps for one technique, median over repeated batch launches."""
    from gpufsm.api import run_batch

    def once() -> float:
        results = run_batch(nfa, batch, backend="cuda", technique=technique)
        return results[0].kernel_ms if results else 0.0

    samples = repeat(once)
    return summarize(samples, total_bytes)["gbps"]


def run_one(key: str, n_strings: int, max_len: int, seed: int, gate: int) -> dict[str, object]:
    """Load, oracle-gate and time one ANMLZoo family. Returns its CSV row."""
    path = ensure(DATASETS[key], CACHE_DIR)  # download + SHA-256 verify (cached)
    t0 = time.perf_counter()
    nfa = load_anml(path)
    print(
        f"{key}: states={nfa.num_states} sym_trans={nfa.num_sym_transitions} "
        f"eps={nfa.num_eps_transitions} accept={int(nfa.accept.sum())} "
        f"load={time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    batch, total_bytes = build_batch(nfa, n_strings, max_len, seed)
    for technique in TECHNIQUES:
        require(nfa, batch, backend="cuda", technique=technique, limit=gate)
    print(f"  oracle: both techniques match the reference on {gate} strings", flush=True)

    speeds = {t: measure(nfa, batch, total_bytes, t) for t in TECHNIQUES}
    speedup = (
        speeds["worklist_warp"] / speeds["worklist_global"] if speeds["worklist_global"] else 0.0
    )
    print(
        f"  global={speeds['worklist_global']:.3f} Gbps  "
        f"warp={speeds['worklist_warp']:.3f} Gbps  speedup={speedup:.1f}x",
        flush=True,
    )
    return {
        "automaton": key,
        "num_states": nfa.num_states,
        "transitions": nfa.num_sym_transitions,
        "global_gbps": round(speeds["worklist_global"], 3),
        "warp_gbps": round(speeds["worklist_warp"], 3),
        "speedup": round(speedup, 1),
        "n_strings": n_strings,
        "gpu": gpu_slug(),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=sorted(DATASETS), help="run a single family")
    ap.add_argument("--n-strings", type=int, default=2048, help="batch size (default: 2048)")
    ap.add_argument("--max-len", type=int, default=40, help="max input length in bytes")
    ap.add_argument("--seed", type=int, default=0, help="batch RNG seed")
    ap.add_argument("--gate", type=int, default=64, help="strings checked against the oracle")
    ap.add_argument("--out", default=None, help="CSV path (default: derived from the GPU name)")
    ap.add_argument("--no-csv", action="store_true", help="measure without writing a CSV")
    args = ap.parse_args(argv)

    print_environment()
    if gpu_slug() == "nocuda":
        print("SKIP: no CUDA device visible", file=sys.stderr)
        return 0

    keys = [args.only] if args.only else list(DATASETS)
    rows = [run_one(k, args.n_strings, args.max_len, args.seed, args.gate) for k in keys]

    if args.no_csv or args.only:
        return 0
    out = args.out or f"paper/data/real_automata_throughput_{gpu_slug()}.csv"
    print(f"\nwrote {write_rows(out, rows, FIELDS)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
