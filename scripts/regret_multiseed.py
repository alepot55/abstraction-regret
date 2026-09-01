"""Statistical hardening of the headline regret: multi-seed Triton/Warp vs CUDA.

The 2x2 / abstraction-regret numbers are the paper's centerpiece, so they must be robust to the
specific random NFA, not a single-seed artifact. This measures the multistream throughput regret
(CUDA / DSL) across several seeds and sizes and reports the median + min-max spread. Writes
paper/data/regret_multiseed_rtx4070.csv. (All three backends run at <=64 states; Warp's single-word
kernel caps there. Warp init may intermittently throw a sticky CUDA-716 -> just rerun.)

Uses the shared harness for both the automata and the inputs, deliberately. It used to build
its own NFAs and to time 2048 copies of one periodic "abcdeabcde..." string: every lane then
walked an identical trajectory, so branch divergence -- the mechanism the regret is about --
was zero by construction, in the one measurement whose job is to show the headline is robust.
The committed CSV predates this fix and needs re-running; see paper/data/README.md.

Usage:  python scripts/regret_multiseed.py
"""

from __future__ import annotations

import statistics
import sys

from gpufsm import run_batch
from gpufsm.bench import DENSE, random_batch, random_nfa
from gpufsm.bench.csvio import driver_out, environment, guard_device, write_rows
from gpufsm.bench.oracle import require
from gpufsm.core.nfa import NFA
from gpufsm.reference import simulate

_SUMMARY = (__doc__ or "").split("\n\n")[0]
"""First paragraph of the module docstring: the --help description."""

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


def _mk(n: int, seed: int) -> NFA | None:
    """The canonical DENSE family, or None when the draw is degenerate.

    ``None`` means the automaton's start closure already accepts, so every kernel returns at
    position 0 without reading a byte and the "throughput" is launch overhead divided by a
    batch size. Four of the fifteen (size, seed) points in the committed CSV are like that,
    which is where its 3.34x maximum comes from -- a number that says nothing about Warp.
    A throughput measurement has to consume input to mean anything.

    The generator itself is the shared one; this file used to carry its own transcription,
    which is exactly the drift ``gpufsm.bench.generators`` exists to prevent.
    """
    nfa = random_nfa(n, seed, DENSE)
    accepted, match_len = simulate(nfa, b"")
    return None if (accepted and match_len == 0) else nfa


def main() -> int:
    out = driver_out("paper/data/regret_multiseed_rtx4070.csv", _SUMMARY)
    guard_device(out)
    gpu = environment()["gpu"]

    batch, _total = random_batch(N_STRINGS, SLEN)
    bits = N_STRINGS * SLEN * 8

    def gbps(nfa, be, te):
        require(nfa, batch, backend=be, technique=te)
        for _ in range(WARMUP):
            run_batch(nfa, batch, be, te)
        ms = statistics.median(run_batch(nfa, batch, be, te)[0].kernel_ms for _ in range(RUNS))
        return bits / (ms * 1e-3) / 1e9

    rows: list[dict[str, object]] = []
    print(f"{'n':>4}{'T_regret med[min-max]':>24}{'W_regret med[min-max]':>24}")
    for n in SIZES:
        tr, wr, skipped = [], [], 0
        for seed in SEEDS:
            nfa = _mk(n, 1000 + seed * 7 + n)
            if nfa is None:  # degenerate draw: accepts before reading a byte
                skipped += 1
                continue
            c = gbps(nfa, "cuda", "multistream")
            tr.append(c / gbps(nfa, "triton", "multistream"))
            wr.append(c / gbps(nfa, "warp", "multistream"))
        if not tr:
            print(f"n={n}: every seed drew a degenerate automaton — widen SEEDS")
            return 1
        if skipped:
            print(f"n={n}: skipped {skipped} degenerate seed(s)")
        rows.append(
            {
                "num_states": n,
                "triton_regret_med": round(statistics.median(tr), 2),
                "triton_min": round(min(tr), 2),
                "triton_max": round(max(tr), 2),
                "warp_regret_med": round(statistics.median(wr), 2),
                "warp_min": round(min(wr), 2),
                "warp_max": round(max(wr), 2),
                "seeds": len(tr),
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
