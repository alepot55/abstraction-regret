"""Block-parallel (warp-per-string) worklist correctness — gpu-marked.

Validates the cooperative warp kernel two ways: bit-for-bit vs the CPU reference oracle on
small NFAs, and vs the single-thread ``worklist_global`` on larger NFAs (>64 states, multiple
state-words — the regime the warp kernel targets). Skips gracefully without CUDA.
"""

from __future__ import annotations

import random

import pytest

from gpufsm import ANY_SYMBOL, NFABuilder, run_batch, simulate
from gpufsm.core.registry import Backend, available_backends, list_techniques

pytestmark = pytest.mark.gpu

_HAS_WARP = Backend.CUDA in available_backends() and "worklist_warp" in list_techniques(
    Backend.CUDA
)
skip = pytest.mark.skipif(not _HAS_WARP, reason="needs CUDA worklist_warp")


def _random_nfa(n: int, rng: random.Random):
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.1)
    b.set_start(rng.randrange(n))
    alpha = "abcde"
    for s in range(n):
        for _ in range(rng.randint(0, 3)):
            sym = ANY_SYMBOL if rng.random() < 0.05 else ord(rng.choice(alpha))
            b.add_transition(s, sym, rng.randrange(n))
        for _ in range(rng.randint(0, 2)):
            b.add_epsilon(s, rng.randrange(n))
    return b.build(), alpha


@skip
def test_warp_matches_reference_small():
    rng = random.Random(7)
    for _ in range(30):
        nfa, alpha = _random_nfa(rng.randint(1, 64), rng)
        batch = [
            bytes(ord(rng.choice(alpha)) for _ in range(rng.randint(0, 20))) for _ in range(16)
        ]
        refs = [simulate(nfa, d) for d in batch]
        got = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_warp")]
        assert got == refs


@skip
@pytest.mark.parametrize("n", [65, 200, 1000, 2048])
def test_warp_matches_global_large(n):
    """>64 states (multi-word): warp == single-thread global, both oracle-validated elsewhere."""
    rng = random.Random(100 + n)
    nfa, alpha = _random_nfa(n, rng)
    batch = [bytes(ord(rng.choice(alpha)) for _ in range(rng.randint(0, 24))) for _ in range(24)]
    g = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_global")]
    w = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_warp")]
    assert w == g


@skip
@pytest.mark.parametrize("n", [65, 200, 1000, 1536])
def test_shared_matches_warp(n):
    """worklist_shared (shared-mem working set, <=1536 states) == worklist_warp."""
    rng = random.Random(200 + n)
    nfa, alpha = _random_nfa(n, rng)
    batch = [bytes(ord(rng.choice(alpha)) for _ in range(rng.randint(0, 24))) for _ in range(24)]
    w = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_warp")]
    s = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_shared")]
    assert s == w


@skip
@pytest.mark.parametrize("n", [65, 200, 1000, 2048])
def test_compact_matches_global_large(n):
    """worklist_compact (compacted active-ID, O(active)) == worklist_global on >64-state NFAs."""
    rng = random.Random(300 + n)
    nfa, alpha = _random_nfa(n, rng)
    batch = [bytes(ord(rng.choice(alpha)) for _ in range(rng.randint(0, 24))) for _ in range(24)]
    g = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_global")]
    c = [(r.accepted, r.match_len) for r in run_batch(nfa, batch, "cuda", "worklist_compact")]
    assert c == g
