"""NFA builder / CSR invariants."""

from __future__ import annotations

import numpy as np

from gpufsm import ANY_SYMBOL, NFABuilder


def test_builder_csr_shapes():
    b = NFABuilder()
    s0 = b.add_state()
    s1 = b.add_state(accept=True)
    b.set_start(s0)
    b.add_transition(s0, "a", s1)
    b.add_transition(s0, ANY_SYMBOL, s1)
    b.add_epsilon(s1, s0)
    nfa = b.build()

    assert nfa.num_states == 2
    assert nfa.start_state == s0
    assert nfa.accept.tolist() == [False, True]
    assert nfa.sym_row_ptr.tolist() == [0, 2, 2]
    assert nfa.eps_row_ptr.tolist() == [0, 0, 1]
    assert nfa.uses_any_symbol is True
    assert nfa.num_sym_transitions == 2
    assert nfa.num_eps_transitions == 1
    assert nfa.sym_row_ptr.dtype == np.int32


def test_char_and_int_symbols_equivalent():
    b = NFABuilder()
    s0 = b.add_state()
    s1 = b.add_state(accept=True)
    b.set_start(s0)
    b.add_transition(s0, "a", s1)
    b.add_transition(s0, ord("a"), s1)
    nfa = b.build()
    assert nfa.sym_symbols.tolist() == [ord("a"), ord("a")]


def test_out_of_range_state_raises():
    b = NFABuilder()
    b.add_state()
    try:
        b.add_transition(0, "a", 5)
    except IndexError:
        return
    raise AssertionError("expected IndexError")
