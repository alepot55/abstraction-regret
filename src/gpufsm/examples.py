"""Canonical small NFAs and labelled inputs, reused by the CLI and the tests.

These mirror the hand-built automata from the legacy test fixtures so the
correctness oracle and every backend are exercised on known cases.
"""

from __future__ import annotations

from .core.nfa import NFA, NFABuilder

# (input, expected_accepted) pairs, latch-first-match semantics.
LabeledInputs = list[tuple[bytes, bool]]


def ab_star_c_plus_d() -> tuple[NFA, LabeledInputs]:
    """NFA roughly matching ``(a b* c+)+ d`` (latch-first-match)."""
    b = NFABuilder()
    s0 = b.add_state()
    s1 = b.add_state()
    s2 = b.add_state()
    s3 = b.add_state(accept=True)
    b.set_start(s0)
    b.add_transition(s0, "a", s1)
    b.add_transition(s1, "b", s1)
    b.add_transition(s1, "c", s2)
    b.add_epsilon(s2, s0)
    b.add_transition(s2, "c", s2)
    b.add_transition(s2, "d", s3)
    inputs: LabeledInputs = [
        (b"abcd", True),
        (b"acd", True),
        (b"abcabcabcd", True),
        (b"acacd", True),
        (b"cd", False),
        (b"d", False),
        (b"", False),
        (b"abc", False),
    ]
    return b.build(), inputs


def a_star_b_c_opt_d() -> tuple[NFA, LabeledInputs]:
    """NFA matching ``a* b (c|epsilon) d`` (latch-first-match)."""
    b = NFABuilder()
    s0 = b.add_state()
    s1 = b.add_state()
    s2 = b.add_state()
    s3 = b.add_state(accept=True)
    b.set_start(s0)
    b.add_transition(s0, "a", s0)
    b.add_transition(s0, "b", s1)
    b.add_transition(s1, "c", s2)
    b.add_epsilon(s1, s2)
    b.add_transition(s2, "d", s3)
    inputs: LabeledInputs = [
        (b"abd", True),
        (b"bcd", True),
        (b"bd", True),
        (b"aaaaaaabd", True),
        (b"aaabcd", True),
        (b"ac", False),
        (b"bb", False),
        (b"", False),
    ]
    return b.build(), inputs


EXAMPLES = {
    "ab_star_c_plus_d": ab_star_c_plus_d,
    "a_star_b_c_opt_d": a_star_b_c_opt_d,
}
