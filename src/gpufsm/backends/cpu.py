"""CPU reference backend — always available, wraps the correctness oracles.

Registers a technique per oracle for both automaton kinds, so ``run(dfa, backend=CPU)``
resolves through the same registry as the NFA path and the DFA GPU kernels have an
in-registry oracle to be compared against.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from ..core.bitmap import BitmapProgram
from ..core.dfa import DFA
from ..core.nfa import NFA
from ..core.registry import Automaton, Backend, Kind, register, register_availability
from ..core.result import Result
from ..reference import simulate, simulate_dfa

Program = Callable[[bytes], tuple[bool, int]]
Compiler = Callable[[Automaton], Program]

# (kind, technique) -> a *compiler*: given the automaton it returns the callable that runs
# an input. Anything that depends only on the automaton is done here, once, at executor
# construction -- which is what the GPU backends do (they upload the CSR and pack the accept
# words in __init__), so a CPU-vs-GPU timing compares the same work. The bitmap technique
# used to recompile its per-state masks on every call, inside the timed region.
_SIMULATORS: dict[tuple[Kind, str], Compiler] = {
    (Kind.NFA, "reference"): lambda a: lambda data: simulate(_as_nfa(a), data),
    (Kind.NFA, "bitmap"): lambda a: BitmapProgram(_as_nfa(a)).run,
    (Kind.DFA, "reference"): lambda a: lambda data: simulate_dfa(_as_dfa(a), data),
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
        self._run = _SIMULATORS[(Kind.of(automaton), technique)](automaton)

    def run(self, input_bytes: bytes) -> Result:
        t0 = time.perf_counter()
        accepted, match_len = self._run(input_bytes)
        dt = (time.perf_counter() - t0) * 1000.0
        return Result(accepted=accepted, match_len=match_len, kernel_ms=dt, total_ms=dt)


def _make(automaton: Automaton, technique: str) -> CPUExecutor:
    return CPUExecutor(automaton, technique)


# 'reference' is the oracle, so it is the default for both kinds.
for _kind, _tech in _SIMULATORS:
    register(_kind, Backend.CPU, _tech, default=(_tech == "reference"))(_make)


register_availability(Backend.CPU, lambda: True)
