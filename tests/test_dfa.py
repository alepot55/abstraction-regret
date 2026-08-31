"""DFA core tests: reference simulator + DFA-vs-NFA cross-oracle equivalence (CPU)."""

from __future__ import annotations

import random

from gpufsm.core.dfa import DFABuilder, random_dfa
from gpufsm.core.nfa import NFABuilder
from gpufsm.reference import simulate, simulate_dfa

A, B = ord("a"), ord("b")


def _dfa_starts_with_ab() -> DFABuilder:
    d = DFABuilder()
    s0, s1, s2 = d.add_state(), d.add_state(), d.add_state(accept=True)
    d.set_start(s0)
    d.add_transition(s0, A, s1)
    d.add_transition(s1, B, s2)
    return d.build()


def test_simulate_dfa_latch_first_match():
    dfa = _dfa_starts_with_ab()
    assert simulate_dfa(dfa, b"ab") == (True, 2)
    assert simulate_dfa(dfa, b"abc") == (True, 2)  # latch at first accept
    assert simulate_dfa(dfa, b"ax") == (False, 0)
    assert simulate_dfa(dfa, b"a") == (False, 0)
    assert simulate_dfa(dfa, b"") == (False, 0)


def test_dfa_matches_equivalent_nfa():
    # Same language ("starts with ab") built as an NFA; the two oracles must agree.
    dfa = _dfa_starts_with_ab()
    b = NFABuilder()
    q0, q1, q2 = b.add_state(), b.add_state(), b.add_state(accept=True)
    b.set_start(q0)
    b.add_transition(q0, "a", q1)
    b.add_transition(q1, "b", q2)
    nfa = b.build()
    rng = random.Random(0)
    for _ in range(500):
        data = bytes(rng.choice((A, B, ord("c"))) for _ in range(rng.randint(0, 6)))
        assert simulate_dfa(dfa, data) == simulate(nfa, data), data


def test_random_dfa_shape_and_determinism():
    dfa = random_dfa(64, seed=1)
    assert dfa.trans.size == 64 * 256
    assert dfa.num_states == 64
    # deterministic generation
    assert (random_dfa(64, seed=1).trans == dfa.trans).all()
    # simulate runs and returns a well-formed verdict
    acc, mlen = simulate_dfa(dfa, bytes(range(50)))
    assert isinstance(acc, bool) and mlen >= 0
