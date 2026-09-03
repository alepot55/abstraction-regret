"""The committed result CSVs are part of the contract, so they are tested like code.

Figures and every published number are regenerated from ``paper/data/``, which means a CSV
whose schema has drifted from the driver that writes it, or from the figure that reads it,
breaks the artifact without breaking a single import. These tests pin three things:

* the exact header of every committed CSV, so a driver cannot silently change a schema and
  leave the committed data describing something else;
* that every value parses as the type its column implies, so a truncated or hand-edited row
  is caught here rather than as a confusing matplotlib error;
* that every column ``paper/figures.py`` reads is present in the file it reads it from.

Pinning the headers here duplicates the ``FIELDS`` list in each driver on purpose. That is
what makes it a pin: the test fails when the two disagree, which is precisely the event
worth hearing about.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "paper" / "data"

# The exact header every committed CSV must have.
SCHEMAS: dict[str, list[str]] = {
    "sweep_techniques.csv": [
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
    ],
    "costmodel_rtx4070.csv": [
        "backend",
        "technique",
        "num_states",
        "traffic_bytes_per_sym",
        "throughput_gbps",
        "gpu",
        "note",
    ],
    "dfa_regret_rtx4070.csv": [
        "backend",
        "num_states",
        "table_kb",
        "throughput_gbps",
        "gpu",
        "note",
    ],
    "scalar_ablation_rtx4070.csv": ["n_strings", "tile_gbps", "scalar_gbps", "cliff", "gpu"],
    "regret_multiseed_rtx4070.csv": [
        "num_states",
        "triton_regret_med",
        "triton_min",
        "triton_max",
        "warp_regret_med",
        "warp_min",
        "warp_max",
        "seeds",
        "gpu",
    ],
    "worklist_warp_rtx4070.csv": [
        "automaton",
        "num_states",
        "words",
        "n_strings",
        "global_gbps",
        "warp_gbps",
        "speedup",
        "gpu",
    ],
    "worklist_warp_batch_rtx4070.csv": [
        "num_states",
        "n_strings",
        "global_gbps",
        "warp_gbps",
        "speedup",
        "gpu",
    ],
    "worklist_shared_rtx4070.csv": [
        "num_states",
        "words",
        "warp_gbps",
        "shared_gbps",
        "shared_over_warp",
        "gpu",
    ],
    "real_automata_throughput_rtx4070.csv": [
        "automaton",
        "num_states",
        "transitions",
        "global_gbps",
        "warp_gbps",
        "speedup",
        "n_strings",
        "gpu",
    ],
    "nsight_rtx4070.csv": [
        "kernel",
        "num_states",
        "n_strings",
        "sm_throughput_pct",
        "dram_throughput_pct",
        "l2_hit_pct",
        "achieved_occupancy_pct",
    ],
    "regret_a100.csv": ["num_states", "triton_regret", "warp_regret"],
    "cross_arch/dfa_latch_a100.csv": [
        "regime",
        "accept_prob",
        "backend",
        "num_states",
        "table_mb",
        "throughput_gbps",
        "mean_bytes_examined",
        "accept_rate",
        "gpu",
        "l2_mb",
    ],
    "dfa_knee_a100.csv": [
        "backend",
        "num_states",
        "table_mb",
        "throughput_gbps",
        "gpu",
        "l2_mb",
    ],
    "dfa_knee_rich_a100.csv": ["backend", "num_states", "table_mb", "throughput_gbps"],
}

# Columns that must parse as a number wherever they appear. Anything not listed is free text
# (a backend name, a GPU name, a note).
NUMERIC = {
    "num_states",
    "n_strings",
    "slen",
    "accept_prob",
    "mean_bytes_examined",
    "accept_rate",
    "median_ms",
    "ci95_lo_ms",
    "ci95_hi_ms",
    "throughput_gbps",
    "samples",
    "traffic_bytes_per_sym",
    "table_kb",
    "table_mb",
    "l2_mb",
    "tile_gbps",
    "scalar_gbps",
    "cliff",
    "triton_regret",
    "warp_regret",
    "triton_regret_med",
    "triton_min",
    "triton_max",
    "warp_regret_med",
    "warp_min",
    "warp_max",
    "seeds",
    "words",
    "global_gbps",
    "warp_gbps",
    "shared_gbps",
    "shared_over_warp",
    "speedup",
    "transitions",
    "sm_throughput_pct",
    "dram_throughput_pct",
    "l2_hit_pct",
    "achieved_occupancy_pct",
}

# What paper/figures.py reads, and out of which file. If a figure grows a column, this is
# where the missing data shows up -- not in a KeyError halfway through a plot.
# Columns that are allowed to be blank: free-text annotations a driver only fills in for
# the rows where it has something to say.
OPTIONAL = {"note"}

FIGURE_INPUTS = {
    "sweep_techniques.csv": {"backend", "technique", "num_states", "throughput_gbps"},
    "costmodel_rtx4070.csv": {"backend", "num_states", "traffic_bytes_per_sym", "throughput_gbps"},
    "dfa_regret_rtx4070.csv": {"backend", "num_states", "table_kb", "throughput_gbps"},
}


def _rows(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (DATA / name).open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_committed_csv_has_its_pinned_header(name: str) -> None:
    header, _ = _rows(name)
    assert header == SCHEMAS[name], f"{name} header drifted from the pinned schema"


@pytest.mark.parametrize("name", sorted(SCHEMAS))
def test_committed_csv_is_non_empty_and_parses(name: str) -> None:
    _, rows = _rows(name)
    assert rows, f"{name} has a header and no data"
    for i, row in enumerate(rows):
        for column, value in row.items():
            if column not in OPTIONAL:
                assert value != "", f"{name} row {i}: column {column!r} is empty"
            if column in NUMERIC and value != "":
                float(value)  # raises ValueError, which is the assertion


def test_every_committed_csv_is_pinned() -> None:
    """A new CSV must be added to SCHEMAS, or it ships with nothing checking it.

    The glob is recursive on purpose: the first cross-architecture result landed in
    ``paper/data/cross_arch/`` and a non-recursive glob left the newest and most
    consequential CSV as the one file this guarantee did not cover.
    """
    on_disk = {p.relative_to(DATA).as_posix() for p in DATA.rglob("*.csv")}
    assert on_disk == set(SCHEMAS), f"unpinned: {sorted(on_disk - set(SCHEMAS))}"


@pytest.mark.parametrize("name", sorted(FIGURE_INPUTS))
def test_figure_inputs_are_present(name: str) -> None:
    header, _ = _rows(name)
    missing = FIGURE_INPUTS[name] - set(header)
    assert not missing, f"paper/figures.py reads {sorted(missing)} from {name}, which lacks it"


def test_data_directory_is_documented() -> None:
    """Every committed CSV is named in paper/data/README.md, which records its provenance."""
    readme = (DATA / "README.md").read_text()
    for name in SCHEMAS:
        assert name in readme, f"{name} has no provenance entry in paper/data/README.md"


# --- the drivers' own schemas, read without importing them ----------------------------
#
# Each driver declares the header it writes as a module-level list. Importing the drivers
# to read it is not an option -- they import torch and triton, which the CPU suite must run
# without -- so the list is lifted out of the source with `ast`, which needs no dependency
# and cannot execute a kernel by accident.

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

DRIVER_SCHEMAS = {
    "sweep_techniques.py": ("FIELDS", "sweep_techniques.csv"),
    "calibrate_costmodel.py": ("FIELDS", "costmodel_rtx4070.csv"),
    "sweep_dfa.py": ("FIELDS", "dfa_regret_rtx4070.csv"),
    "ablate_scalar_control.py": ("FIELDS", "scalar_ablation_rtx4070.csv"),
    "regret_multiseed.py": ("FIELDS", "regret_multiseed_rtx4070.csv"),
    "bench_worklist_shared.py": ("FIELDS", "worklist_shared_rtx4070.csv"),
    "bench_worklist_warp.py": ("FIELDS", "worklist_warp_rtx4070.csv"),
    "bench_worklist_warp.py:batch": ("BATCH_FIELDS", "worklist_warp_batch_rtx4070.csv"),
    "run_anmlzoo.py": ("FIELDS", "real_automata_throughput_rtx4070.csv"),
    "dfa_latch_control.py": ("FIELDS", "cross_arch/dfa_latch_a100.csv"),
}


def _literal_list(script: str, name: str) -> list[str]:
    import ast

    tree = ast.parse((SCRIPTS / script).read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return list(ast.literal_eval(node.value))
    raise AssertionError(f"{script} has no module-level {name}")


@pytest.mark.parametrize("key", sorted(DRIVER_SCHEMAS))
def test_driver_schema_matches_its_committed_csv(key: str) -> None:
    """A driver's header and the CSV it overwrites must be the same list, in order."""
    script = key.split(":")[0]
    name, csv_name = DRIVER_SCHEMAS[key]
    assert _literal_list(script, name) == SCHEMAS[csv_name], (
        f"scripts/{script}::{name} would write a different header than {csv_name} has"
    )
