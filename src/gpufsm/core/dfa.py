"""DFA representation — the *memory-bound* automata workload.

A DFA complements the NFA story (:mod:`gpufsm.core.nfa`) for the "two faces of
abstraction regret" thesis. Where NFA simulation is control-flow/compute-bound (active-set
traversal + epsilon-closure), DFA simulation is **memory-bound**: a single random lookup
into a dense transition table ``T[state, symbol]`` per input byte. For a large DFA the
table does not fit cache, so throughput is set by the *table layout* and the memory system
— the regime where the memory-centric thesis (and a DSL's ability to express a cache-/
coalescing-friendly layout) bites hardest.

Semantics match the NFA reference: **latch-first-match** (report at the first accepting
state reached, ``match_len`` = bytes consumed). The oracle itself is
:func:`gpufsm.reference.simulate_dfa`, next to the NFA one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ALPHABET = 256


@dataclass(frozen=True)
class DFA:
    """Deterministic FA with a dense transition table (one next-state per symbol).

    ``trans`` is a flat int32 array of length ``num_states * 256``; the next state from
    ``s`` on byte ``c`` is ``trans[s * 256 + c]``. Every (state, symbol) is total (a
    self-looping non-accepting *dead* state encodes "no transition").
    """

    num_states: int
    start_state: int
    accept: np.ndarray  # bool [num_states]
    trans: np.ndarray  # int32 [num_states * 256]

    @property
    def table_bytes(self) -> int:
        return int(self.trans.size) * self.trans.itemsize


def random_dfa(num_states: int, *, accept_prob: float = 0.05, seed: int = 0) -> DFA:
    """A random total DFA over the byte alphabet — for throughput/regret measurement.

    State 0 is the start; a fraction ``accept_prob`` of states (excluding start) accept.
    Transitions are uniform random over states, giving a large dense table that, for big
    ``num_states``, exceeds cache and exposes the memory-bound regime.
    """
    if num_states < 1:
        raise ValueError("num_states must be >= 1")
    rng = np.random.default_rng(seed)
    trans = rng.integers(0, num_states, size=num_states * ALPHABET, dtype=np.int32)
    accept = rng.random(num_states) < accept_prob
    accept[0] = False  # keep start non-accepting so match_len 0 doesn't trivially fire
    return DFA(num_states=num_states, start_state=0, accept=accept, trans=trans)


class DFABuilder:
    """Build a DFA incrementally; unset transitions route to a non-accepting dead state."""

    def __init__(self) -> None:
        self._accept: list[bool] = []
        self._edges: list[dict[int, int]] = []
        self._start = 0

    def add_state(self, accept: bool = False) -> int:
        idx = len(self._accept)
        self._accept.append(bool(accept))
        self._edges.append({})
        return idx

    def _check(self, state: int) -> None:
        """Bounds-check a state index, as :class:`~gpufsm.core.nfa.NFABuilder` does.

        This used to be absent here, and the asymmetry was not harmless: a negative ``src``
        indexed ``self._edges`` from the end and silently rewired a different state, while an
        out-of-range ``dst`` was written straight into the transition table. The CPU oracle
        would then raise on it, but the GPU kernels index ``trans`` without a bound and read
        whatever is there -- a wrong verdict on the device and no verdict at all on the host.
        """
        if not 0 <= state < len(self._accept):
            raise IndexError(f"state {state} out of range (0..{len(self._accept) - 1})")

    def set_start(self, state: int) -> None:
        self._check(state)
        self._start = state

    def add_transition(self, src: int, symbol: int, dst: int) -> None:
        self._check(src)
        self._check(dst)
        if not 0 <= symbol < ALPHABET:
            raise ValueError(f"symbol must be 0..255, got {symbol}")
        self._edges[src][symbol] = dst

    def build(self) -> DFA:
        n = len(self._accept)
        if n == 0:
            raise ValueError("cannot build a DFA with zero states")
        dead = n  # absorbing non-accepting dead state
        total = n + 1
        trans = np.full(total * ALPHABET, dead, dtype=np.int32)
        for s in range(n):
            for c, d in self._edges[s].items():
                trans[s * ALPHABET + c] = d
        trans[dead * ALPHABET : (dead + 1) * ALPHABET] = dead  # dead self-loops
        accept = np.zeros(total, dtype=bool)
        accept[:n] = self._accept
        return DFA(num_states=total, start_state=self._start, accept=accept, trans=trans)
