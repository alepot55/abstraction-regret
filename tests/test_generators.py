"""The unified generator must reproduce the eleven originals byte-for-byte.

Every committed throughput CSV is only meaningful if its automaton can be rebuilt
exactly. The reference implementations below are verbatim transcriptions of the
pre-refactor generators (see the git history of scripts/sweep_techniques.py,
scripts/bench_worklist_warp.py and tests/test_worklist_warp.py); the tests assert that
the unified shapes draw from the RNG in exactly the same order.

If one of these fails, the CSVs under paper/data/ no longer describe the automata the
code now builds. Do not "fix" it by updating the reference.
"""

from __future__ import annotations

import random

import pytest

from gpufsm.bench.generators import (
    DENSE,
    NO_ACCEPT,
    SPARSE_WORKLIST,
    WITH_WILDCARDS,
    random_batch,
    random_nfa,
)
from gpufsm.core.nfa import ANY_SYMBOL, NFA, NFABuilder

_SEEDS = [0, 1, 7, 1000, 1032, 1500, 20260816]
_SIZES = [1, 2, 32, 48, 64, 128, 500]


# --------------------------------------------------------------- reference impls
def ref_dense(n: int, seed: int) -> NFA:
    """scripts/sweep_techniques.py::random_nfa — the canonical generator."""
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.1)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)):
            b.add_transition(s, ord(rng.choice("abcde")), rng.randrange(n))
    return b.build()


def ref_no_accept(n: int, seed: int) -> NFA:
    """The no-accept generator: the pre-unification ``random_nfa_noaccept``."""
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=False)
    b.set_start(rng.randrange(n))
    for s in range(n):
        for _ in range(rng.randint(1, 3)):
            b.add_transition(s, ord(rng.choice("abcde")), rng.randrange(n))
    return b.build()


def ref_sparse_worklist(n: int, seed: int) -> NFA:
    """scripts/bench_worklist_warp.py::_random_nfa (== bench_worklist_shared.py)."""
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(n):
        b.add_state(accept=rng.random() < 0.02)
    b.set_start(0)
    alpha = [ord(c) for c in "abcde"]
    for s in range(n):
        for _ in range(2):
            b.add_transition(s, rng.choice(alpha), rng.randrange(n))
        if rng.random() < 0.3:
            b.add_epsilon(s, rng.randrange(n))
    return b.build()


def ref_with_wildcards(n: int, seed: int) -> NFA:
    """tests/test_worklist_warp.py::_random_nfa."""
    rng = random.Random(seed)
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
    return b.build()


# --------------------------------------------------------------------- assertions
def assert_same_nfa(a: NFA, b: NFA) -> None:
    assert a.num_states == b.num_states
    assert a.start_state == b.start_state
    assert a.accept.tolist() == b.accept.tolist()
    assert a.sym_row_ptr.tolist() == b.sym_row_ptr.tolist()
    assert a.sym_targets.tolist() == b.sym_targets.tolist()
    assert a.sym_symbols.tolist() == b.sym_symbols.tolist()
    assert a.eps_row_ptr.tolist() == b.eps_row_ptr.tolist()
    assert a.eps_targets.tolist() == b.eps_targets.tolist()


@pytest.mark.parametrize("n", _SIZES)
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize(
    ("shape", "reference"),
    [
        (DENSE, ref_dense),
        (NO_ACCEPT, ref_no_accept),
        (SPARSE_WORKLIST, ref_sparse_worklist),
        (WITH_WILDCARDS, ref_with_wildcards),
    ],
    ids=["dense", "no_accept", "sparse_worklist", "with_wildcards"],
)
def test_shape_reproduces_its_original(shape, reference, n, seed):
    assert_same_nfa(random_nfa(n, seed, shape), reference(n, seed))


def test_fixed_out_degree_does_not_consume_randomness():
    """The trap: random.randint(k, k) still calls getrandbits and shifts the stream."""
    rng = random.Random(0)
    before = rng.getstate()
    assert rng.randint(2, 2) == 2
    assert rng.getstate() != before, "if this ever stops holding, _count can be simplified"


def test_random_nfa_rejects_an_empty_automaton():
    with pytest.raises(ValueError, match="num_states must be >= 1"):
        random_nfa(0, seed=0)


def test_random_batch_matches_the_original_construction():
    """scripts/sweep_techniques.py::make_batch, with its exact numpy draw."""
    import numpy as np

    n_strings, slen = 64, 256
    rng = np.random.default_rng(0)
    flat = rng.integers(ord("a"), ord("a") + 5, size=n_strings * slen, dtype=np.uint8).tobytes()
    expected = [flat[i * slen : (i + 1) * slen] for i in range(n_strings)]

    batch, total = random_batch(n_strings, slen, seed=0)
    assert batch == expected
    assert total == n_strings * slen


def test_random_batch_shapes():
    batch, total = random_batch(4, 8, seed=3)
    assert len(batch) == 4
    assert all(len(s) == 8 for s in batch)
    assert total == 32
    assert set(b"".join(batch)) <= set(b"abcde")
