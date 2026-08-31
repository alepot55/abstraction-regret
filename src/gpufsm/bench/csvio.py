"""Versioned CSV output — the only way measurements reach ``paper/data``.

Paper figures are regenerated from committed CSVs, so the CSV *is* the record. Two
rules are enforced here rather than left to each script: the schema is explicit (a
row with an unexpected key is an error, not a silently dropped column), and the
environment that produced the numbers is captured alongside them.
"""

from __future__ import annotations

import csv
import platform
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def write_rows(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: list[str]) -> Path:
    """Write ``rows`` to ``path`` with exactly ``fields`` as the header.

    Raises on a row carrying a key outside ``fields``: a typo in a column name would
    otherwise drop that measurement from the CSV without a word.
    """
    rows = list(rows)
    allowed = set(fields)
    for i, row in enumerate(rows):
        unknown = set(row) - allowed
        if unknown:
            raise ValueError(f"row {i} has fields not in the schema: {sorted(unknown)}")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return path


def environment() -> dict[str, str]:
    """What produced the numbers: GPU, driver stack and interpreter.

    A throughput without the device it was measured on is not reproducible, and the
    committed CSVs are compared across machines (RTX 4070 vs A100) all the time.
    """
    info: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gpu": "(none)",
    }
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda or "(unknown)"
            # L2 capacity is where the DFA transition table stops fitting, so it is the axis
            # the memory-bound face is read against. It used to be a constant in a comment,
            # which is a claim about the hardware that nothing checked; ask the device.
            l2 = getattr(props, "L2_cache_size", None) or getattr(props, "l2_cache_size", None)
            if l2:
                info["l2_mb"] = f"{int(l2) / (1 << 20):.0f}"
    except Exception:  # pragma: no cover - depends on environment
        pass
    try:
        import triton

        info["triton"] = triton.__version__
    except Exception:  # pragma: no cover - depends on environment
        pass
    return info


def gpu_slug() -> str:
    """The GPU name as a filename fragment, e.g. ``nvidia_geforce_rtx_4070``."""
    name = environment()["gpu"]
    return name.lower().replace(" ", "_") if name != "(none)" else "nocuda"


def print_environment(stream: Any = sys.stdout) -> None:
    """Print the environment block measurement scripts put at the top of their output."""
    for key, value in environment().items():
        print(f"{key:9s}: {value}", file=stream)


class WrongDevice(RuntimeError):
    """A driver was about to overwrite a result measured on a different GPU."""


def _normalize(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def same_device(recorded: str, live: str) -> bool:
    """Whether a CSV's recorded device and the live one are the same GPU.

    Names are recorded with different verbosity across files -- ``RTX4070`` in the
    hand-written ones, ``NVIDIA GeForce RTX 4070`` from ``torch.cuda.get_device_name``
    -- so the comparison is containment on the alphanumeric reduction rather than
    equality, which would call the same card two devices.
    """
    a, b = _normalize(recorded), _normalize(live)
    return bool(a) and bool(b) and (a in b or b in a)


def guard_device(path: str | Path, column: str = "gpu") -> None:
    """Refuse to overwrite a committed result that was measured on another GPU.

    The canonical CSVs under ``paper/data/`` carry the device in their *filename*
    (``..._rtx4070.csv``). Re-running a driver elsewhere would rewrite that file in
    place, leaving a result that claims a device it was never measured on -- the kind of
    corruption that is invisible in a diff and fatal in a paper. This reads the device
    the file already records and stops unless it matches the live one.

    A file that does not exist yet, or that records no device, is not second-guessed.
    """
    path = Path(path)
    if not path.exists():
        return
    live = environment()["gpu"]
    if live == "(none)":
        return
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    recorded = {r[column] for r in rows if r.get(column)} if rows else set()
    if not recorded or any(same_device(r, live) for r in recorded):
        return
    raise WrongDevice(
        f"{path} records {sorted(recorded)} but this machine is {live!r}. "
        f"Overwriting it would leave a file claiming a device it was not measured on. "
        f"Pass an explicit --out (or delete the file) to record this run separately."
    )
