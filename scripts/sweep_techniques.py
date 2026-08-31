"""Rigorous multi-technique throughput sweep -> versioned CSV (median + CI95).

Measures the parallel multi-stream-family techniques (the throughput-meaningful ones)
across automaton sizes, on a fixed input batch, with non-Gaussian-appropriate
statistics (median + percentile-bootstrap 95% CI over per-run batch-kernel times).
Captures GPU + library versions for reproducibility. Output feeds the paper figures.

The single-program dense/bitpacked kernels are excluded: they are one latency-bound
GPU thread (~0 throughput) and are reported separately as the naive baseline.

Run on a GPU box:  python scripts/sweep_techniques.py [out.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from gpufsm.api import run_batch
from gpufsm.bench import random_batch, random_nfa
from gpufsm.bench.csvio import guard_device, write_rows
from gpufsm.bench.oracle import require
from gpufsm.bench.timing import bootstrap_ci95
from gpufsm.core.registry import Backend, available_backends, list_techniques

FIELDS = [
    "gpu",
    "backend",
    "technique",
    "num_states",
    "n_strings",
    "slen",
    "median_ms",
    "ci95_lo_ms",
    "ci95_hi_ms",
    "throughput_gbps",
    "samples",
    "torch",
    "triton",
    "warp",
    "cuda",
]

_MULTISTREAM = {
    "multistream",
    "multistream_shared",
    "multistream_async",
    "worklist",
}
_SIZES = [32, 64, 128, 256, 500]
_N_STRINGS = 2048
_SLEN = 256
_SAMPLES = 9
_WARMUP = 3


def env_info() -> dict[str, str]:
    info = {"gpu": "?", "torch": "?", "triton": "?", "warp": "?", "cuda": "?"}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda or "?"
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    for mod, key in (("triton", "triton"), ("warp", "warp")):
        try:
            info[key] = __import__(mod).__version__
        except Exception:
            pass
    return info


def measure(nfa, backend, technique, batch, total_bytes) -> dict | None:
    # Nothing reports a throughput before it agrees with the CPU reference. This driver
    # produces the CSV behind the headline 2x2, and it was the one measurement path that
    # never called the gate.
    require(nfa, batch, backend=backend, technique=technique)
    for _ in range(_WARMUP):
        run_batch(nfa, batch, backend=backend, technique=technique)
    samples = []
    for _ in range(_SAMPLES):
        km = run_batch(nfa, batch, backend=backend, technique=technique)[0].kernel_ms
        if km > 0:
            samples.append(km)
    if not samples:
        return None
    median = float(np.median(samples))
    lo, hi = bootstrap_ci95(samples)
    gbps = (total_bytes * 8.0) / (median * 1e-3) / 1e9
    return {
        "median_ms": round(median, 5),
        "ci95_lo_ms": round(lo, 5),
        "ci95_hi_ms": round(hi, 5),
        "throughput_gbps": round(gbps, 5),
        "samples": len(samples),
    }


def main() -> int:
    out_path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/data/sweep_techniques.csv")
    guard_device(out_path)
    env = env_info()
    backends = [
        b for b in available_backends() if b in (Backend.TRITON, Backend.CUDA, Backend.WARP)
    ]
    if not backends:
        print("no GPU backend — run on a GPU box")
        return 0
    batch, total_bytes = random_batch(_N_STRINGS, _SLEN)
    rows: list[dict[str, object]] = []
    for be in backends:
        for te in list_techniques(be):
            if te not in _MULTISTREAM:
                continue
            for n in _SIZES:
                if be is Backend.WARP and n > 64:
                    continue
                nfa = random_nfa(n, seed=1000 + n)
                try:
                    m = measure(nfa, be, te, batch, total_bytes)
                except (ValueError, RuntimeError) as e:
                    msg = str(e)
                    print(f"{be.value:7}/{te:18} n={n:4d}  SKIP ({type(e).__name__}: {msg[:120]})")
                    # A CUDA "misaligned address"/error-716-class fault is STICKY: it poisons
                    # the process's CUDA context, so every later measurement is meaningless.
                    # (Observed intermittently from Warp's init on this Warp/CUDA combo.) Abort
                    # the sweep with a clear rerun hint rather than emit a half/empty CSV.
                    if "misaligned" in msg or "716" in msg or "an illegal" in msg:
                        print(
                            "ABORT: CUDA context poisoned (sticky error) — rerun the sweep "
                            "(intermittent, often from Warp init). Partial results discarded."
                        )
                        return 1
                    continue
                if m is None:
                    continue
                row = {
                    "gpu": env["gpu"],
                    "backend": be.value,
                    "technique": te,
                    "num_states": n,
                    "n_strings": _N_STRINGS,
                    "slen": _SLEN,
                    **m,
                    "torch": env["torch"],
                    "triton": env["triton"],
                    "warp": env["warp"],
                    "cuda": env["cuda"],
                }
                rows.append(row)
                ci = f"[{m['ci95_lo_ms']:.4f},{m['ci95_hi_ms']:.4f}]"
                print(
                    f"{be.value:7}/{te:18} n={n:4d}  median={m['median_ms']:9.4f} ms  "
                    f"CI95={ci}  {m['throughput_gbps']:8.4f} Gbps"
                )
    if not rows:
        print("no rows measured")
        return 1
    print(f"\nwrote {write_rows(out_path, rows, FIELDS)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
