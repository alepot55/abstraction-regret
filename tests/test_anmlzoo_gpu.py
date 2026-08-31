"""Real ANMLZoo automata: GPU (worklist_global) must match the reference oracle.

Marked `gpu` and gated on network + CUDA availability — skips gracefully (no hard
failure) when the dataset can't be fetched or no GPU/CUDA backend is present, so it is
safe in CPU-only / offline CI but exercises real production-scale automata on a GPU box.
"""

from __future__ import annotations

import random

import pytest

from gpufsm.core.registry import Backend, available_backends, list_techniques

pytestmark = pytest.mark.gpu

_HAS_WORKLIST_GLOBAL = (
    Backend.CUDA in available_backends() and "worklist_global" in list_techniques(Backend.CUDA)
)


@pytest.mark.skipif(not _HAS_WORKLIST_GLOBAL, reason="needs CUDA worklist_global")
@pytest.mark.parametrize(
    "key", ["levenshtein", "hamming", "brill", "fermi", "randomforest", "corerings"]
)
def test_real_anmlzoo_matches_reference(key):
    from gpufsm.api import run_batch
    from gpufsm.io.anml import load_anml
    from gpufsm.io.datasets import DATASETS, ensure
    from gpufsm.reference import simulate

    try:
        path = ensure(DATASETS[key], "data/anmlzoo")
    except Exception as e:  # network/checksum failure -> skip, don't fail CI
        pytest.skip(f"could not fetch ANMLZoo {key!r}: {e}")

    nfa = load_anml(path)
    assert nfa.num_states > 500  # genuinely beyond the register-kernel cap
    alphabet = sorted({int(s) for s in nfa.sym_symbols if 0 <= int(s) <= 255}) or [97]
    rng = random.Random(0)
    batch = [bytes(rng.choice(alphabet) for _ in range(rng.randint(0, 24))) for _ in range(24)]
    refs = [simulate(nfa, d) for d in batch]
    res = run_batch(nfa, batch, backend="cuda", technique="worklist_global")
    assert [(r.accepted, r.match_len) for r in res] == refs
