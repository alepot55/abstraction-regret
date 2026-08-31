"""Regression tests for input validation at the library's boundaries.

Each test here corresponds to a way the library used to accept bad input and produce a
plausible-looking wrong answer instead of an error. They are grouped in one file because
they share a theme rather than a module: an automaton built through a builder is trusted by
every backend afterwards, so the builder is the last place a bad index can be caught before
it reaches a kernel that indexes without bounds.
"""

from __future__ import annotations

import pytest

from gpufsm.bench.oracle import compare, matches, require
from gpufsm.core.dfa import ALPHABET, DFABuilder
from gpufsm.core.nfa import ANY_SYMBOL, NFABuilder
from gpufsm.core.registry import Backend, Kind, get_factory


def _nfa():
    b = NFABuilder()
    s0 = b.add_state()
    s1 = b.add_state(accept=True)
    b.set_start(s0)
    b.add_transition(s0, "a", s1)
    return b.build()


class TestSymbolCoercion:
    """A one-character string must not be able to become the wildcard sentinel."""

    def test_chr_256_is_rejected_not_read_as_any_symbol(self) -> None:
        b = NFABuilder()
        s0 = b.add_state()
        s1 = b.add_state(accept=True)
        b.set_start(s0)
        with pytest.raises(ValueError, match="code point 256"):
            b.add_transition(s0, chr(256), s1)

    @pytest.mark.parametrize("ch", ["Ā", "€", "\U0001f600"])
    def test_non_byte_code_points_are_rejected(self, ch: str) -> None:
        b = NFABuilder()
        s0 = b.add_state()
        b.set_start(s0)
        with pytest.raises(ValueError, match="over bytes"):
            b.add_transition(s0, ch, s0)

    def test_latin1_boundary_is_still_accepted(self) -> None:
        b = NFABuilder()
        s0 = b.add_state()
        s1 = b.add_state(accept=True)
        b.set_start(s0)
        b.add_transition(s0, chr(255), s1)
        assert int(b.build().sym_symbols[0]) == 255

    def test_any_symbol_is_reachable_by_its_name(self) -> None:
        b = NFABuilder()
        s0 = b.add_state()
        s1 = b.add_state(accept=True)
        b.set_start(s0)
        b.add_transition(s0, ANY_SYMBOL, s1)
        assert int(b.build().sym_symbols[0]) == ANY_SYMBOL


class TestDFABuilderBounds:
    """DFABuilder must bounds-check like NFABuilder: the GPU kernels do not."""

    def test_set_start_rejects_out_of_range(self) -> None:
        b = DFABuilder()
        b.add_state()
        with pytest.raises(IndexError):
            b.set_start(7)

    def test_add_transition_rejects_out_of_range_dst(self) -> None:
        b = DFABuilder()
        s0 = b.add_state()
        with pytest.raises(IndexError):
            b.add_transition(s0, ord("a"), 99)

    def test_add_transition_rejects_negative_src(self) -> None:
        """A negative index used to wrap and silently rewire the last state."""
        b = DFABuilder()
        b.add_state()
        b.add_state()
        with pytest.raises(IndexError):
            b.add_transition(-1, ord("a"), 0)

    def test_symbol_range_is_still_checked(self) -> None:
        b = DFABuilder()
        s0 = b.add_state()
        with pytest.raises(ValueError, match="0..255"):
            b.add_transition(s0, ALPHABET, s0)

    def test_a_valid_build_still_works(self) -> None:
        b = DFABuilder()
        s0 = b.add_state()
        s1 = b.add_state(accept=True)
        b.set_start(s0)
        b.add_transition(s0, ord("a"), s1)
        dfa = b.build()
        assert dfa.start_state == s0
        assert bool(dfa.accept[s1])


class TestOracleGateIsNeverVacuous:
    """A gate that checked nothing must not look like a gate that found nothing wrong."""

    def test_empty_input_list_raises(self) -> None:
        with pytest.raises(ValueError, match="vacuously"):
            compare(_nfa(), [], Backend.CPU)

    def test_zero_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="limit must be"):
            require(_nfa(), [b"a"], Backend.CPU, limit=0)

    def test_negative_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="limit must be"):
            matches(_nfa(), [b"a"], Backend.CPU, limit=-1)

    def test_a_real_check_still_passes(self) -> None:
        require(_nfa(), [b"a", b"b"], Backend.CPU)


class TestRegistryErrors:
    """An unavailable backend must say so, not raise a bare lookup failure."""

    def test_missing_backend_message_names_the_reason(self) -> None:
        with pytest.raises(KeyError) as exc:
            get_factory(Kind.NFA, Backend.CUDA, None)
        message = str(exc.value)
        assert "cuda" in message
        assert "gpufsm env" in message or "registers no" in message
