"""The registry is the single extension point — including for DFAs.

The DFA path used to bypass it entirely (a separate ``run_dfa_batch`` with a hand-rolled
``if backend == ...`` chain). These tests pin the property that replaced it: one
registry keyed by ``(Kind, Backend, technique)``, and ``run``/``run_batch`` picking the
kind from the automaton itself.
"""

from __future__ import annotations

import pytest

from gpufsm import (
    Backend,
    DFABuilder,
    Kind,
    NFABuilder,
    list_kinds,
    list_techniques,
    random_dfa,
    run,
    run_batch,
    simulate,
    simulate_dfa,
)


def _nfa():
    b = NFABuilder()
    s0, s1 = b.add_state(), b.add_state(accept=True)
    b.set_start(s0)
    b.add_transition(s0, "a", s1)
    return b.build()


def _dfa():
    b = DFABuilder()
    s0, s1 = b.add_state(), b.add_state(accept=True)
    b.set_start(s0)
    b.add_transition(s0, ord("a"), s1)
    return b.build()


def test_kind_is_derived_from_the_automaton():
    assert Kind.of(_nfa()) is Kind.NFA
    assert Kind.of(_dfa()) is Kind.DFA


def test_kind_rejects_anything_else():
    with pytest.raises(TypeError, match="expected an NFA or a DFA"):
        Kind.of("not an automaton")  # type: ignore[arg-type]


def test_cpu_registers_both_kinds():
    assert list_kinds(Backend.CPU) == [Kind.NFA, Kind.DFA]
    assert list_techniques(Backend.CPU, Kind.NFA) == ["reference", "bitmap"]
    assert list_techniques(Backend.CPU, Kind.DFA) == ["reference"]


def test_run_dispatches_an_nfa_to_the_nfa_oracle():
    nfa = _nfa()
    res = run(nfa, b"a", backend=Backend.CPU)
    assert (res.accepted, res.match_len) == simulate(nfa, b"a")


def test_run_dispatches_a_dfa_to_the_dfa_oracle():
    dfa = _dfa()
    for data in (b"a", b"b", b""):
        res = run(dfa, data, backend=Backend.CPU)
        assert (res.accepted, res.match_len) == simulate_dfa(dfa, data)


def test_run_batch_on_a_dfa_without_a_native_batch_path():
    dfa = random_dfa(32, seed=7)
    batch = [b"abc", b"", b"zzzz"]
    got = [(r.accepted, r.match_len) for r in run_batch(dfa, batch, backend=Backend.CPU)]
    assert got == [simulate_dfa(dfa, b) for b in batch]


def test_unknown_technique_names_the_available_ones():
    with pytest.raises(KeyError, match="available: "):
        run(_dfa(), b"a", backend=Backend.CPU, technique="bitmap")


def test_techniques_do_not_leak_across_kinds():
    """'bitmap' is an NFA technique; asking for it as a DFA one must not resolve."""
    assert "bitmap" not in list_techniques(Backend.CPU, Kind.DFA)


# --- the default technique must be declared, not emergent ---------------------------


def test_every_registered_kind_backend_pair_declares_a_default() -> None:
    """`run(automaton, backend=...)` must resolve to a technique somebody chose.

    Before `register(..., default=True)` existed, `get_factory` took `techs[0]`, i.e. dict
    insertion order, i.e. whichever module imported first. Reordering a tuple of module
    names for readability silently changed what every default-argument call measured.
    """
    from gpufsm.core.registry import _DEFAULTS, _REGISTRY

    pairs = {(kind, backend) for kind, backend, _ in _REGISTRY}
    undeclared = sorted((k.value, b.value) for k, b in pairs if (k, b) not in _DEFAULTS)
    assert not undeclared, f"no default technique declared for: {undeclared}"


def test_declared_default_is_actually_registered() -> None:
    from gpufsm.core.registry import _DEFAULTS, _REGISTRY

    for (kind, backend), technique in _DEFAULTS.items():
        assert (kind, backend, technique) in _REGISTRY, (
            f"{backend.value}/{kind.value} declares default {technique!r}, which is not registered"
        )


def test_get_factory_resolves_none_to_the_declared_default() -> None:
    from gpufsm.core.registry import get_factory

    technique, _ = get_factory(Kind.NFA, Backend.CPU, None)
    assert technique == "reference"
