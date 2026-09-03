"""NFA representation in CSR form, plus an ergonomic builder.

The same CSR layout is consumed by every backend (CPU reference, Triton, CUDA),
so that comparisons are apples-to-apples. Symbols are bytes ``0..255``; the
sentinel :data:`ANY_SYMBOL` (256) is a wildcard transition matching any input
byte. Semantics are *latch-first-match*: a match is reported as soon as any
accepting state becomes active (see :mod:`gpufsm.reference`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ANY_SYMBOL = 256
"""Wildcard symbol id: a transition labelled with it matches any input byte."""


def _coerce_symbol(symbol: int | str) -> int:
    """Accept an int (0..255 or ANY_SYMBOL) or a single-character string.

    The string path is range-checked like the int path. It used to return ``ord(symbol)``
    unvalidated, so ``chr(256)`` -- an ordinary one-character Python string -- became the
    ANY_SYMBOL sentinel and turned the transition into a wildcard that matched every input
    byte. Anyone building an automaton over a non-Latin-1 alphabet hit it silently.
    """
    if isinstance(symbol, str):
        if len(symbol) != 1:
            raise ValueError(f"symbol string must be a single char, got {symbol!r}")
        code = ord(symbol)
        if not 0 <= code <= 255:
            raise ValueError(
                f"symbol {symbol!r} has code point {code}; transitions are over bytes "
                f"(0..255). Encode the text first, or pass ANY_SYMBOL for a wildcard."
            )
        return code
    if symbol == ANY_SYMBOL or 0 <= symbol <= 255:
        return int(symbol)
    raise ValueError(f"symbol must be 0..255 or ANY_SYMBOL, got {symbol}")


@dataclass(frozen=True, eq=False)
class NFA:
    """Immutable NFA in CSR form. Build one via :class:`NFABuilder`.

    ``eq=False`` is deliberate. The generated ``__eq__`` compares the numpy fields with
    ``==``, which yields an array, so ``a == b`` raised ``ValueError: truth value of an
    array ... is ambiguous`` on two identical automata; and ``frozen=True`` then
    synthesised a ``__hash__`` over those same unhashable arrays. Falling back to identity
    makes the type usable as a dict key and comparable without lying. ``frozen`` still
    states the intent -- the arrays are shared with every backend and must not be mutated --
    but it cannot enforce it on their contents.
    """

    num_states: int
    start_state: int
    accept: np.ndarray  # bool [num_states]

    sym_row_ptr: np.ndarray  # int32 [num_states + 1]
    sym_targets: np.ndarray  # int32 [nnz_sym]
    sym_symbols: np.ndarray  # int32 [nnz_sym]  (0..255 or ANY_SYMBOL)

    eps_row_ptr: np.ndarray  # int32 [num_states + 1]
    eps_targets: np.ndarray  # int32 [nnz_eps]

    @property
    def uses_any_symbol(self) -> bool:
        return bool(np.any(self.sym_symbols == ANY_SYMBOL))

    @property
    def num_sym_transitions(self) -> int:
        return int(self.sym_targets.size)

    @property
    def num_eps_transitions(self) -> int:
        return int(self.eps_targets.size)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"NFA(states={self.num_states}, start={self.start_state}, "
            f"accept={int(self.accept.sum())}, "
            f"sym={self.num_sym_transitions}, eps={self.num_eps_transitions})"
        )


class NFABuilder:
    """Mutable builder that finalizes to an immutable CSR :class:`NFA`."""

    def __init__(self) -> None:
        self._accept: list[bool] = []
        self._sym: list[list[tuple[int, int]]] = []  # per state: (symbol, target)
        self._eps: list[list[int]] = []  # per state: targets
        self._start: int = 0

    def add_state(self, accept: bool = False) -> int:
        idx = len(self._accept)
        self._accept.append(bool(accept))
        self._sym.append([])
        self._eps.append([])
        return idx

    def set_start(self, state: int) -> None:
        self._check(state)
        self._start = state

    def set_accept(self, state: int, accept: bool = True) -> None:
        self._check(state)
        self._accept[state] = bool(accept)

    def add_transition(self, src: int, symbol: int | str, dst: int) -> None:
        self._check(src)
        self._check(dst)
        self._sym[src].append((_coerce_symbol(symbol), dst))

    def add_epsilon(self, src: int, dst: int) -> None:
        self._check(src)
        self._check(dst)
        self._eps[src].append(dst)

    def build(self) -> NFA:
        n = len(self._accept)
        if n == 0:
            raise ValueError("cannot build an NFA with zero states")

        sym_row_ptr = np.zeros(n + 1, dtype=np.int32)
        eps_row_ptr = np.zeros(n + 1, dtype=np.int32)
        for s in range(n):
            sym_row_ptr[s + 1] = sym_row_ptr[s] + len(self._sym[s])
            eps_row_ptr[s + 1] = eps_row_ptr[s] + len(self._eps[s])

        sym_targets = np.empty(int(sym_row_ptr[-1]), dtype=np.int32)
        sym_symbols = np.empty(int(sym_row_ptr[-1]), dtype=np.int32)
        eps_targets = np.empty(int(eps_row_ptr[-1]), dtype=np.int32)
        for s in range(n):
            base = int(sym_row_ptr[s])
            for k, (sym, dst) in enumerate(self._sym[s]):
                sym_symbols[base + k] = sym
                sym_targets[base + k] = dst
            ebase = int(eps_row_ptr[s])
            for k, dst in enumerate(self._eps[s]):
                eps_targets[ebase + k] = dst

        return NFA(
            num_states=n,
            start_state=self._start,
            accept=np.array(self._accept, dtype=bool),
            sym_row_ptr=sym_row_ptr,
            sym_targets=sym_targets,
            sym_symbols=sym_symbols,
            eps_row_ptr=eps_row_ptr,
            eps_targets=eps_targets,
        )

    def _check(self, state: int) -> None:
        if not 0 <= state < len(self._accept):
            raise IndexError(f"state {state} out of range (0..{len(self._accept) - 1})")
