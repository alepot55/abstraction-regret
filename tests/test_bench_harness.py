"""The shared measurement harness, tested on CPU.

Timing statistics, CSV schema enforcement and the oracle gate used to be inline in
each measurement script, where nothing could test them. They decide what the paper's
numbers mean, so they are worth pinning even though they never touch a kernel.
"""

from __future__ import annotations

import csv

import pytest

from gpufsm import Backend, run_batch
from gpufsm.bench.csvio import environment, gpu_slug, write_rows
from gpufsm.bench.generators import random_batch, random_nfa
from gpufsm.bench.oracle import OracleMismatch, compare, matches, require
from gpufsm.bench.timing import bootstrap_ci95, gbps, repeat, summarize


# ------------------------------------------------------------------------ timing
def test_bootstrap_ci95_brackets_the_median():
    samples = [1.0, 1.1, 0.9, 1.05, 0.95, 1.2, 0.8, 1.0, 1.02]
    lo, hi = bootstrap_ci95(samples)
    assert lo <= sorted(samples)[len(samples) // 2] <= hi


def test_bootstrap_ci95_is_deterministic():
    samples = [3.0, 1.0, 2.0, 5.0, 4.0]
    assert bootstrap_ci95(samples, seed=42) == bootstrap_ci95(samples, seed=42)


def test_bootstrap_ci95_degenerates_gracefully():
    assert bootstrap_ci95([]) == (0.0, 0.0)
    assert bootstrap_ci95([2.5]) == (2.5, 2.5)


def test_gbps_conversion():
    # 1 GB in 8 ms -> 1e9 bytes * 8 bits / 8e-3 s = 1000 Gbit/s
    assert gbps(1_000_000_000, 8.0) == pytest.approx(1000.0)
    assert gbps(1000, 0.0) == 0.0


def test_repeat_runs_warmup_untimed_and_drops_non_positive():
    calls = []

    def measure() -> float:
        calls.append(len(calls))
        # batched executors report 0.0 on every result but the first
        return 0.0 if len(calls) % 2 == 0 else 1.5

    samples = repeat(measure, warmup=2, samples=6)
    assert len(calls) == 8  # 2 warmup + 6 timed
    assert all(s == 1.5 for s in samples)
    assert 0.0 not in samples


def test_summarize_reports_median_ci_and_throughput():
    out = summarize([1.0, 2.0, 3.0], total_bytes=1_000_000, seed=0)
    assert out["median_ms"] == pytest.approx(2.0)
    assert out["n"] == 3
    assert out["ci95_lo_ms"] <= out["median_ms"] <= out["ci95_hi_ms"]
    assert out["gbps"] == pytest.approx(gbps(1_000_000, 2.0))


def test_summarize_of_nothing_is_zeroed_not_an_exception():
    assert summarize([], total_bytes=10) == {
        "median_ms": 0.0,
        "ci95_lo_ms": 0.0,
        "ci95_hi_ms": 0.0,
        "gbps": 0.0,
        "n": 0,
    }


# ------------------------------------------------------------------------- csvio
def test_write_rows_writes_the_declared_schema(tmp_path):
    path = write_rows(
        tmp_path / "sub" / "out.csv",
        [{"a": 1, "b": "x"}, {"a": 2}],
        fields=["a", "b"],
    )
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows == [{"a": "1", "b": "x"}, {"a": "2", "b": ""}]


def test_write_rows_rejects_a_column_outside_the_schema(tmp_path):
    """A mistyped column name must fail loudly, not vanish from the CSV."""
    with pytest.raises(ValueError, match=r"fields not in the schema: \['gpbs'\]"):
        write_rows(tmp_path / "out.csv", [{"gbps": 1.0}, {"gpbs": 2.0}], fields=["gbps"])


def test_environment_reports_something_for_the_gpu():
    env = environment()
    assert env["python"]
    assert env["gpu"]  # "(none)" on a CPU-only machine, never missing
    assert gpu_slug()


# ------------------------------------------------------------------------ oracle
def test_oracle_accepts_a_backend_that_agrees_with_itself():
    nfa = random_nfa(24, seed=5)
    batch, _ = random_batch(16, 32, seed=1)
    assert matches(nfa, batch, Backend.CPU, "bitmap")
    require(nfa, batch, Backend.CPU, "bitmap")  # must not raise


def test_oracle_reports_the_first_disagreement(monkeypatch):
    nfa = random_nfa(24, seed=5)
    batch, _ = random_batch(8, 32, seed=1)

    real = run_batch

    def lying_run_batch(automaton, inputs, backend=Backend.CPU, technique=None):
        out = real(automaton, inputs, backend=backend, technique=technique)
        if technique == "bitmap":
            out[3].match_len += 1  # corrupt exactly one verdict
            out[3].accepted = not out[3].accepted
        return out

    monkeypatch.setattr("gpufsm.bench.oracle.run_batch", lying_run_batch)

    bad = compare(nfa, batch, Backend.CPU, "bitmap")
    assert [i for i, _, _ in bad] == [3]
    assert not matches(nfa, batch, Backend.CPU, "bitmap")
    with pytest.raises(OracleMismatch, match="first at index 3"):
        require(nfa, batch, Backend.CPU, "bitmap")
