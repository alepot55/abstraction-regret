"""Benchmark sweeps and reproducible CSV output (mean/std/CI95)."""

from __future__ import annotations

import csv
from pathlib import Path

from ..api import benchmark
from ..core.nfa import NFA
from ..core.registry import Backend, available_backends, list_techniques
from ..core.result import BenchmarkStats

CSV_FIELDS = ["backend", "technique", "accepted", "match_len", "n", "mean_ms", "std_ms", "ci95_ms"]


def sweep(
    nfa: NFA,
    input_bytes: bytes,
    backends: list[Backend] | None = None,
    repeats: int = 10,
    warmup: int = 3,
) -> list[BenchmarkStats]:
    """Benchmark every (backend, technique) that is available."""
    backends = backends or available_backends()
    out: list[BenchmarkStats] = []
    for backend in backends:
        for technique in list_techniques(backend):
            out.append(
                benchmark(
                    nfa,
                    input_bytes,
                    backend=backend,
                    technique=technique,
                    repeats=repeats,
                    warmup=warmup,
                )
            )
    return out


def write_csv(stats: list[BenchmarkStats], path: str | Path) -> Path:
    """Write benchmark stats to ``path`` as CSV; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for s in stats:
            writer.writerow(s.as_row())
    return path
