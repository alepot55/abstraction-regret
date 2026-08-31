"""CLI smoke tests and CSV output."""

from __future__ import annotations

import csv

from gpufsm.bench import sweep, write_csv
from gpufsm.cli import main
from gpufsm.examples import EXAMPLES


def test_cli_env_list_verify():
    assert main(["env"]) == 0
    assert main(["list"]) == 0
    assert main(["verify"]) == 0  # all backends agree with reference on examples


def test_cli_bench_and_sweep():
    assert main(["bench", "--size", "256", "--repeats", "3", "--warmup", "1"]) == 0
    assert main(["sweep", "--size", "256", "--repeats", "3", "--warmup", "1"]) == 0


def test_sweep_writes_csv(tmp_path):
    nfa, _ = EXAMPLES["ab_star_c_plus_d"]()
    stats = sweep(nfa, b"abcd" * 16, repeats=3, warmup=1)
    # CPU exposes two techniques (reference, bitmap); both must appear.
    techs = {s.technique for s in stats}
    assert {"reference", "bitmap"} <= techs

    out = write_csv(stats, tmp_path / "bench.csv")
    assert out.exists()
    rows = list(csv.DictReader(out.open()))
    assert len(rows) == len(stats)
    assert {"backend", "technique", "mean_ms", "ci95_ms"} <= set(rows[0])
