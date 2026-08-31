"""Canonical random automata and input batches — one generator, not eleven.

Every measurement script used to carry its own ``random_nfa``. Eleven copies existed
and they had drifted into three incompatible families, which is a reproducibility
hazard rather than a style problem: a committed CSV is only meaningful if the
automaton behind it can be rebuilt exactly, and "the canonical generator" pointed at
one file among eleven.

The shapes below reproduce those families **byte-for-byte**, including the order of
the RNG calls, so ``random_nfa(n, seed, DENSE)`` yields exactly the automaton the
committed numbers were measured on. ``tests/test_generators.py`` pins that against
transcriptions of the original implementations; do not reorder the draws.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.nfa import ANY_SYMBOL, NFA, NFABuilder

DEFAULT_ALPHABET = "abcde"


@dataclass(frozen=True)
class NFAShape:
    """Structural parameters of a random NFA family.

    ``out_degree`` and ``eps_degree`` are inclusive ``(lo, hi)`` bounds drawn with
    ``randint``. ``eps_prob`` adds at most one epsilon edge per state and is mutually
    exclusive with ``eps_degree`` — the two families that need epsilons draw it
    differently and both orders must be preserved.

    ``accept_prob=None`` means "no accepting states **and no draw**", which is not the
    same as ``0.0``: the original no-accept generator never called ``rng.random()`` per
    state, so drawing it anyway would shift every later draw and change the automaton.
    """

    alphabet: str = DEFAULT_ALPHABET
    accept_prob: float | None = 0.1
    out_degree: tuple[int, int] = (1, 3)
    eps_prob: float = 0.0
    eps_degree: tuple[int, int] = (0, 0)
    any_prob: float = 0.0
    random_start: bool = True


DENSE = NFAShape()
"""The canonical family: the generator every committed throughput CSV was measured on."""

NO_ACCEPT = NFAShape(accept_prob=None)
"""DENSE without accept states, so strings never latch early and throughput is sustained."""

SPARSE_WORKLIST = NFAShape(accept_prob=0.02, out_degree=(2, 2), eps_prob=0.3, random_start=False)
"""Sparse active sets with epsilons — the family the worklist benchmarks use."""

WITH_WILDCARDS = NFAShape(out_degree=(0, 3), eps_degree=(0, 2), any_prob=0.05)
"""Wildcards and epsilon chains — the correctness-stress family."""


def _count(rng: random.Random, bounds: tuple[int, int]) -> int:
    """Draw an inclusive count, *without* touching the RNG when it is fixed.

    ``random.randint(k, k)`` still consumes randomness (``_randbelow(1)`` calls
    ``getrandbits``), so routing a fixed count through it would desynchronize the
    stream against the original generators and silently change every automaton.
    """
    lo, hi = bounds
    return lo if lo == hi else rng.randint(lo, hi)


def random_nfa(num_states: int, seed: int, shape: NFAShape = DENSE) -> NFA:
    """Build a random NFA of ``num_states`` states reproducibly from ``seed``."""
    if num_states < 1:
        raise ValueError("num_states must be >= 1")
    rng = random.Random(seed)
    b = NFABuilder()
    for _ in range(num_states):
        accept = False if shape.accept_prob is None else rng.random() < shape.accept_prob
        b.add_state(accept=accept)
    b.set_start(rng.randrange(num_states) if shape.random_start else 0)

    alpha = [ord(c) for c in shape.alphabet]
    for s in range(num_states):
        for _ in range(_count(rng, shape.out_degree)):
            if shape.any_prob and rng.random() < shape.any_prob:
                sym = ANY_SYMBOL
            else:
                sym = rng.choice(alpha)
            b.add_transition(s, sym, rng.randrange(num_states))
        if shape.eps_prob and rng.random() < shape.eps_prob:
            b.add_epsilon(s, rng.randrange(num_states))
        if shape.eps_degree != (0, 0):
            for _ in range(_count(rng, shape.eps_degree)):
                b.add_epsilon(s, rng.randrange(num_states))
    return b.build()


def _draw_symbols(count: int, seed: int, alphabet: str) -> np.ndarray[Any, Any]:
    """``count`` uniform symbols from ``alphabet``, as uint8 — the one draw all batches share."""
    lo = ord(alphabet[0])
    return np.random.default_rng(seed).integers(lo, lo + len(alphabet), size=count, dtype=np.uint8)


def random_batch(
    num_strings: int,
    length: int,
    seed: int = 0,
    alphabet: str = DEFAULT_ALPHABET,
) -> tuple[list[bytes], int]:
    """``(batch, total_bytes)``: ``num_strings`` strings of ``length`` bytes each.

    Built as one numpy draw and sliced, which is what makes a 4096x256 batch cheap
    enough to rebuild per measurement instead of being cached and mutated.
    """
    flat = _draw_symbols(num_strings * length, seed, alphabet).tobytes()
    batch = [flat[i * length : (i + 1) * length] for i in range(num_strings)]
    return batch, num_strings * length


def random_batch_2d(
    num_strings: int,
    length: int,
    seed: int = 0,
    alphabet: str = DEFAULT_ALPHABET,
) -> np.ndarray[Any, Any]:
    """The same draw as :func:`random_batch`, kept as a ``(num_strings, length)`` array.

    The hand-written CUDA and lane-packed Triton kernels take the batch as a 2-D uint8
    array rather than a list of ``bytes``; this is the same random stream, only reshaped,
    so a measurement can switch between the two views without changing its inputs.
    """
    return _draw_symbols(num_strings * length, seed, alphabet).reshape(num_strings, length)


def random_byte_batch(num_strings: int, length: int, seed: int = 0) -> tuple[list[bytes], int]:
    """``random_batch`` over the **full** byte alphabet — the DFA's input distribution."""
    flat = (
        np.random.default_rng(seed)
        .integers(0, 256, size=num_strings * length, dtype=np.uint8)
        .tobytes()
    )
    batch = [flat[i * length : (i + 1) * length] for i in range(num_strings)]
    return batch, num_strings * length


def random_bytes_2d(num_strings: int, length: int, seed: int = 0) -> np.ndarray[Any, Any]:
    """Uniform over the **full** byte alphabet — for the DFA, whose table is 256 wide.

    Distinct from :func:`random_batch_2d`: restricting a DFA's input to five symbols
    would leave 98% of every transition row untouched and turn a memory-bound sweep
    into a cache-resident one.
    """
    return (
        np.random.default_rng(seed)
        .integers(0, 256, size=num_strings * length, dtype=np.uint8)
        .reshape(num_strings, length)
    )
