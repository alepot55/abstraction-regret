"""Cross-architecture re-run of the two headline results, on a rented second GPU.

The paper's external-validity claim is that the regret follows the **execution paradigm**
and not the architecture. Testing it means re-running two measurements somewhere other
than the RTX 4070 they were taken on:

1. **NFA regret 2x2** — Triton (tile/SPMD) and Warp (thread-SIMT) against CUDA at equal
   algorithm. The prediction is that the absolute factors rescale with the architecture
   while the *structure* holds: Triton pays, Warp does not.
2. **DFA knee** — throughput against transition-table size. The prediction is that CUDA
   peaks while the table is L2-resident and drops past it, and that Triton stays flat
   because it never reaches the memory-bound regime at all.

Throughputs are **not** comparable across GPUs; the shape and the ordering are what the
claim rests on. Both profiles gate every backend against the CPU oracle before reporting.

    python scripts/second_gpu.py --profile quick   # DFA knee only
    python scripts/second_gpu.py --profile rich    # both

The committed A100 rows in ``paper/data/`` were condensed by hand from this script's
output into the plotting schema; see ``paper/data/README.md`` for the mapping.
"""

from __future__ import annotations

import argparse
import sys

from gpufsm import Backend, random_dfa, run_batch
from gpufsm.bench import DENSE, random_batch, random_byte_batch, random_nfa
from gpufsm.bench.csvio import environment, write_rows
from gpufsm.bench.oracle import matches
from gpufsm.bench.timing import repeat, summarize

NFA_SIZES = (32, 48, 64)
NFA_SEEDS = (0, 1, 2)
NFA_STRINGS, NFA_LEN = 2048, 256

DFA_SIZES = (2048, 8192, 16384, 32768, 40960, 49152, 81920, 131072)
DFA_STRINGS, DFA_LEN = 4096, 256

GPU_BACKENDS = (Backend.CUDA, Backend.TRITON, Backend.WARP)

NFA_FIELDS = ["num_states", "seed", "backend", "technique", "gbps", "median_ms", "regret_vs_cuda"]
DFA_FIELDS = ["num_states", "table_mb", "backend", "gbps", "median_ms"]


def _throughput(automaton, batch, total_bytes, backend, technique=None) -> dict[str, float]:
    samples = repeat(
        lambda: run_batch(automaton, batch, backend=backend, technique=technique)[0].kernel_ms
    )
    return summarize(samples, total_bytes)


def _usable(automaton, probe: list[bytes], technique: str | None) -> list[Backend]:
    """The GPU backends that both run and agree with the CPU oracle on ``probe``."""
    out = []
    for backend in GPU_BACKENDS:
        try:
            if matches(automaton, probe, backend, technique):
                out.append(backend)
            else:
                print(f"  {backend.value}: ORACLE MISMATCH — excluded", file=sys.stderr)
        except Exception as exc:
            print(f"  {backend.value}: unavailable ({type(exc).__name__})", file=sys.stderr)
    return out


def nfa_regret() -> list[dict[str, object]]:
    """Triton/Warp throughput against CUDA at equal algorithm, per state count."""
    batch, total = random_batch(NFA_STRINGS, NFA_LEN, seed=0)
    probe = random_nfa(48, seed=7, shape=DENSE)
    print("NFA backends (oracle-gated):", file=sys.stderr)
    usable = _usable(probe, batch[:32], "multistream")
    if Backend.CUDA not in usable:
        print("SKIP nfa: CUDA is the 1.0x reference and is not usable here", file=sys.stderr)
        return []

    rows: list[dict[str, object]] = []
    for n in NFA_SIZES:
        for seed in NFA_SEEDS:
            nfa = random_nfa(n, seed=1000 + seed * 7 + n, shape=DENSE)
            cuda = _throughput(nfa, batch, total, Backend.CUDA, "multistream")
            for backend in usable:
                stats = _throughput(nfa, batch, total, backend, "multistream")
                regret = cuda["gbps"] / stats["gbps"] if stats["gbps"] else 0.0
                rows.append(
                    {
                        "num_states": n,
                        "seed": seed,
                        "backend": backend.value,
                        "technique": "multistream",
                        "gbps": round(stats["gbps"], 3),
                        "median_ms": round(stats["median_ms"], 5),
                        "regret_vs_cuda": round(regret, 3),
                    }
                )
    return rows


def dfa_knee() -> list[dict[str, object]]:
    """Throughput against table size — the L2 knee, or the absence of one."""
    batch, total = random_byte_batch(DFA_STRINGS, DFA_LEN, seed=99)
    probe_dfa = random_dfa(8192, accept_prob=0.02, seed=8192)
    print("DFA backends (oracle-gated):", file=sys.stderr)
    usable = _usable(probe_dfa, batch[:32], None)

    rows: list[dict[str, object]] = []
    for n in DFA_SIZES:
        dfa = random_dfa(n, accept_prob=0.02, seed=n)
        for backend in usable:
            stats = _throughput(dfa, batch, total, backend)
            rows.append(
                {
                    "num_states": n,
                    "table_mb": round(dfa.table_bytes / 1e6, 2),
                    "backend": backend.value,
                    "gbps": round(stats["gbps"], 3),
                    "median_ms": round(stats["median_ms"], 5),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--profile", choices=("quick", "rich"), default="rich")
    ap.add_argument("--outdir", default="paper/data/cross_arch")
    args = ap.parse_args(argv)

    env = environment()
    if env["gpu"] == "(none)":
        print("SKIP: no CUDA device")
        return 0
    for key, value in env.items():
        print(f"{key:9s}: {value}")
    slug = env["gpu"].lower().replace(" ", "_")

    if args.profile == "rich":
        rows = nfa_regret()
        if rows:
            path = write_rows(f"{args.outdir}/second_gpu_nfa_{slug}.csv", rows, NFA_FIELDS)
            print(f"\nwrote {path} ({len(rows)} rows)")

    rows = dfa_knee()
    if rows:
        path = write_rows(f"{args.outdir}/second_gpu_dfa_{slug}.csv", rows, DFA_FIELDS)
        print(f"wrote {path} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
