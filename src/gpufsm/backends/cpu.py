"""CPU reference backend — always available, wraps the correctness oracles.

Registers a technique per oracle for both automaton kinds, so ``run(dfa, backend=CPU)``
resolves through the same registry as the NFA path and the DFA GPU kernels have an
in-registry oracle to be compared against.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..core.bitmap import simulate_bitmap
from ..core.dfa import DFA
from ..core.nfa import NFA
from ..core.registry import Automaton, Backend, Kind, register, register_availability
from ..core.result import Result
from ..reference import simulate, simulate_dfa

Simulator = Callable[[Automaton, bytes], tuple[bool, int]]

# (kind, technique) -> simulator. The first NFA entry is the default technique.
_SIMULATORS: dict[tuple[Kind, str], Simulator] = {
    (Kind.NFA, "reference"): lambda a, b: simulate(_as_nfa(a), b),
    (Kind.NFA, "bitmap"): lambda a, b: simulate_bitmap(_as_nfa(a), b),
    (Kind.DFA, "reference"): lambda a, b: simulate_dfa(_as_dfa(a), b),
}


def _as_nfa(automaton: Automaton) -> NFA:
    assert isinstance(automaton, NFA)
    return automaton


def _as_dfa(automaton: Automaton) -> DFA:
    assert isinstance(automaton, DFA)
    return automaton


class CPUExecutor:
    def __init__(self, automaton: Automaton, technique: str = "reference") -> None:
        self.automaton = automaton
        self.technique = technique
        self._sim = _SIMULATORS[(Kind.of(automaton), technique)]

    def run(self, input_bytes: bytes) -> Result:
        t0 = time.perf_counter()
        accepted, match_len = self._sim(self.automaton, input_bytes)
        dt = (time.perf_counter() - t0) * 1000.0
        return Result(accepted=accepted, match_len=match_len, kernel_ms=dt, total_ms=dt)


def _make(automaton: Automaton, technique: str) -> CPUExecutor:
    return CPUExecutor(automaton, technique)


for _kind, _tech in _SIMULATORS:
    register(_kind, Backend.CPU, _tech)(_make)


register_availability(Backend.CPU, lambda: True)
