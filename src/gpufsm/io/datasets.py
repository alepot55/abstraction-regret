"""Robust, checksummed dataset acquisition.

Replaces the legacy fragile SharePoint download with verifiable fetches: every
dataset declares a SHA-256, downloads are checksum-verified, and a cached copy is
reused. Small fixtures are vendored in the repo; the large ANMLZoo/AutomataZoo
suite is fetched on demand.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    name: str
    url: str
    sha256: str


# Known datasets — SHA-256 pinned from the public ANMLZoo repo (a trusted academic
# mirror, jackwadden/ANMLZoo). Each is a homogeneous ANML automaton loadable via
# gpufsm.io.anml. Add more entries (with their pinned SHA) as needed.
_ANMLZOO = "https://raw.githubusercontent.com/jackwadden/ANMLZoo/master"
DATASETS: dict[str, Dataset] = {
    # Levenshtein edit-distance automaton (k=24, 20x3): 2784 STEs, pure homogeneous,
    # all-input start states. Smallest ANMLZoo .anml; validated GPU==reference on it.
    "levenshtein": Dataset(
        name="levenshtein_24_20x3.1chip.anml",
        url=f"{_ANMLZOO}/Levenshtein/anml/24_20x3.1chip.anml",
        sha256="8d6ec59d7c57a6e41112f90c244b5c393ff71124df8062ab025c8f243f6a7370",
    ),
    # Hamming-distance automaton (93, 20X3): 11346 STEs, pure homogeneous, all-input.
    "hamming": Dataset(
        name="hamming_93_20X3.1chip.anml",
        url=f"{_ANMLZOO}/Hamming/anml/93_20X3.1chip.anml",
        sha256="6005437dac4581223c30c9d039b08e6a6a856e821507b300023665995f91170b",
    ),
    # Brill part-of-speech tagging: 42658 STEs, pure homogeneous (character-class STEs).
    "brill": Dataset(
        name="brill.1chip.anml",
        url=f"{_ANMLZOO}/Brill/anml/brill.1chip.anml",
        sha256="2a914b59da8340b71d79bfaa39e518fc536729c2bcbeaafaedccae71061af885",
    ),
    # Fermi (particle-physics trigger regexes): 40786 STEs, pure homogeneous. A
    # network/regex-style automaton — a different family from the edit-distance ones.
    "fermi": Dataset(
        name="fermi_2400.1chip.anml",
        url=f"{_ANMLZOO}/Fermi/anml/fermi_2400.1chip.anml",
        sha256="2d0df2ee2d6730c58353fe44dfc06092bdd4141ecdfbc1da2c9769ea2df95272",
    ),
    # RandomForest (ML decision forest as an automaton): 33223 STEs, 6.27M symbol
    # transitions — a high transition-density stress for the CSR traffic model.
    "randomforest": Dataset(
        name="rf.1chip.anml",
        url=f"{_ANMLZOO}/RandomForest/anml/rf.1chip.anml",
        sha256="78ee215c1b0668666539e794a37554bda6c443fe7aebcccb657b0ca5570f6bf3",
    ),
    # CoreRings (Synthetic STE-only ring stress automaton): 48005 STEs, one transition
    # per state — the largest pure-STE state count in our suite, minimal density.
    "corerings": Dataset(
        name="CoreRings.anml",
        url=f"{_ANMLZOO}/Synthetic/anml/CoreRings.anml",
        sha256="0f751a223495cca0e5f69151998311f7dfd86827b6c98ebca25786bad707a3a9",
    ),
}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (constant memory)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(path: str | Path, expected_sha256: str) -> bool:
    """True iff ``path`` exists and its SHA-256 matches ``expected_sha256``."""
    p = Path(path)
    return p.is_file() and sha256_file(p) == expected_sha256


def ensure(dataset: Dataset, dest_dir: str | Path) -> Path:
    """Return a checksum-verified local copy of ``dataset``, downloading if needed."""
    if not dataset.sha256:
        raise ValueError(
            f"dataset {dataset.name!r} has no SHA-256 pinned; refusing to download unverified data"
        )
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / dataset.name

    if verify(dest, dataset.sha256):
        return dest

    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(dataset.url, tmp)  # noqa: S310 - pinned, checksum-verified below
    if sha256_file(tmp) != dataset.sha256:
        tmp.unlink(missing_ok=True)
        raise OSError(f"checksum mismatch for {dataset.name!r} downloaded from {dataset.url}")
    tmp.replace(dest)
    return dest
