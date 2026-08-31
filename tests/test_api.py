"""Public API: run / benchmark on the CPU backend, and cross-backend agreement."""

from __future__ import annotations

import pytest

from gpufsm import Backend, available_backends, benchmark, run, simulate
from gpufsm.examples import EXAMPLES


def test_cpu_backend_available():
    assert Backend.CPU in available_backends()


def test_run_matches_reference_on_all_available_backends():
    for name in EXAMPLES:
        nfa, inputs = EXAMPLES[name]()
        for data, _ in inputs:
            ref_accepted, ref_len = simulate(nfa, data)
            for b in available_backends():
                res = run(nfa, data, backend=b)
                assert res.accepted == ref_accepted, f"{b.value}/{name} {data!r}"
                assert res.match_len == ref_len, f"{b.value}/{name} {data!r}"


def test_benchmark_stats_shape():
    nfa, _ = EXAMPLES["ab_star_c_plus_d"]()
    stats = benchmark(nfa, b"abcd" * 64, backend=Backend.CPU, repeats=5, warmup=2)
    assert stats.n == 5
    assert len(stats.raw_ms) == 5
    assert stats.mean_ms >= 0.0
    assert stats.ci95_ms >= 0.0
    row = stats.as_row()
    assert row["backend"] == "cpu"
    assert row["n"] == 5


def test_benchmark_rejects_zero_repeats():
    nfa, _ = EXAMPLES["ab_star_c_plus_d"]()
    with pytest.raises(ValueError):
        benchmark(nfa, b"abcd", repeats=0)
