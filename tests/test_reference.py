"""The CPU reference oracle must produce the expected verdicts on known cases."""

from __future__ import annotations

import pytest

from gpufsm import simulate
from gpufsm.examples import EXAMPLES


@pytest.mark.parametrize("name", list(EXAMPLES))
def test_examples_match_expected(name):
    nfa, inputs = EXAMPLES[name]()
    for data, expected in inputs:
        accepted, match_len = simulate(nfa, data)
        assert accepted == expected, f"{name}: {data!r} -> {accepted}, expected {expected}"
        if accepted:
            assert 0 <= match_len <= len(data)


def test_zero_length_match():
    from gpufsm import NFABuilder

    b = NFABuilder()
    s0 = b.add_state(accept=True)
    b.set_start(s0)
    nfa = b.build()
    accepted, match_len = simulate(nfa, b"anything")
    assert accepted is True
    assert match_len == 0


def test_epsilon_closure_reaches_accept():
    from gpufsm import NFABuilder

    b = NFABuilder()
    s0 = b.add_state()
    s1 = b.add_state()
    s2 = b.add_state(accept=True)
    b.set_start(s0)
    b.add_transition(s0, "x", s1)
    b.add_epsilon(s1, s2)
    nfa = b.build()
    accepted, match_len = simulate(nfa, b"x")
    assert accepted is True
    assert match_len == 1
