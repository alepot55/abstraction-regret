"""gpufsm — portable GPU finite-state-machine processing across DSLs.

Two automaton models, one API. NFA simulation is the control-flow-bound face of the
study; DFA simulation is the memory-bound one. Both are reached through
:func:`run` / :func:`run_batch`, dispatched on the automaton's type::

    from gpufsm import NFABuilder, Backend, run, benchmark

    b = NFABuilder()
    s0 = b.add_state(); s1 = b.add_state(accept=True)
    b.set_start(s0); b.add_transition(s0, "a", s1)
    nfa = b.build()

    run(nfa, b"a", backend=Backend.CPU)            # -> Result(accepted=True, match_len=1)
    benchmark(nfa, b"a" * 1000, repeats=10)        # -> BenchmarkStats

    dfa = random_dfa(1024, seed=0)
    run_batch(dfa, [b"abc", b"xyz"], backend=Backend.CUDA)
"""

from __future__ import annotations

from .api import benchmark, run, run_batch
from .core.dfa import DFA, DFABuilder, random_dfa
from .core.nfa import ANY_SYMBOL, NFA, NFABuilder
from .core.registry import (
    Backend,
    Kind,
    available_backends,
    is_available,
    list_kinds,
    list_techniques,
)
from .core.result import BenchmarkStats, Result
from .reference import simulate, simulate_dfa

__version__ = "0.1.0"

__all__ = [
    "ANY_SYMBOL",
    "NFA",
    "NFABuilder",
    "DFA",
    "DFABuilder",
    "random_dfa",
    "Backend",
    "Kind",
    "Result",
    "BenchmarkStats",
    "run",
    "run_batch",
    "benchmark",
    "simulate",
    "simulate_dfa",
    "available_backends",
    "is_available",
    "list_kinds",
    "list_techniques",
]
