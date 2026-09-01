"""Host-side array packing shared by every GPU backend.

The Triton, CUDA and Warp executors all have to hand the device the same three
things: the input batch as a flat symbol buffer plus CSR-style offsets, the accept
set as a packed bitmask, and single inputs as an int32 symbol array. Each backend
used to carry its own copy of that code, which is how the copies drifted (int64 vs
uint64 accept words, an empty-batch guard present in some and missing in others).

Everything here is numpy-only, so it imports on a CPU-only machine and is covered
by the ordinary test suite rather than only by GPU runs.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .nfa import NFA

WORD_BITS = 64
"""Bits per word of the packed working set — matches the GPU kernels' word type."""


def words_for(num_states: int) -> int:
    """Number of :data:`WORD_BITS`-wide words needed to hold ``num_states`` bits."""
    return (num_states + WORD_BITS - 1) // WORD_BITS


def symbols(input_bytes: bytes) -> np.ndarray:
    """One input string as the int32 symbol array the kernels index with."""
    return np.frombuffer(input_bytes, dtype=np.uint8).astype(np.int32)


def pack_inputs(inputs: list[bytes]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a batch into one int32 symbol buffer plus CSR-style offsets.

    ``offsets`` has ``len(inputs) + 1`` entries; string ``i`` occupies
    ``data[offsets[i]:offsets[i + 1]]``. An all-empty batch yields an empty buffer
    rather than tripping ``np.frombuffer`` on a zero-length bytes object.

    Raises ``ValueError`` past 2 GiB of total input. The offsets are int32 because that is
    what every kernel indexes with; accumulating them *in* int32 wraps to a negative base
    index silently, which the device then reads out of bounds and reports a wrong verdict
    for. A batch that large is a mistake either way, and an explicit error names it.
    """
    total = sum(len(b) for b in inputs)
    limit = int(np.iinfo(np.int32).max)
    if total > limit:
        raise ValueError(
            f"batch of {total} bytes exceeds the int32 offset range ({limit}); "
            f"split it into smaller batches"
        )
    offsets = np.zeros(len(inputs) + 1, dtype=np.int32)
    running = 0
    for i, b in enumerate(inputs):
        running += len(b)
        offsets[i + 1] = running
    if offsets[-1] > 0:
        data = np.frombuffer(b"".join(inputs), dtype=np.uint8).astype(np.int32)
    else:
        data = np.zeros(0, dtype=np.int32)
    return data, offsets


def pack_accept(
    nfa: NFA, dtype: type = np.uint64, nwords: int | None = None
) -> np.ndarray[Any, Any]:
    """Pack the accept set into ``nwords`` bitmask words (1 bit per state).

    ``dtype`` selects the word type the target kernel expects: the CUDA extension
    takes ``uint64``, Triton takes ``int64`` (``tl`` has no unsigned 64-bit type).
    ``nwords`` defaults to the minimum that fits ``nfa.num_states``.

    The shift operand is cast to ``dtype`` on purpose: numpy would otherwise promote
    a Python int and silently widen (or, for ``uint64``, produce a float).
    """
    if nwords is None:
        nwords = words_for(nfa.num_states)
    words: np.ndarray[Any, Any] = np.zeros(nwords, dtype=dtype)
    one = dtype(1)
    for s in range(nfa.num_states):
        if nfa.accept[s]:
            words[s // WORD_BITS] |= one << dtype(s % WORD_BITS)
    return words


def accept_bitmask(nfa: NFA) -> int:
    """The accept set as a single Python int — for kernels holding it in a register.

    Only meaningful when ``nfa.num_states <= WORD_BITS``; callers that pass it to a
    ``uint64`` kernel parameter must enforce that bound themselves.
    """
    mask = 0
    for s in range(nfa.num_states):
        if nfa.accept[s]:
            mask |= 1 << s
    return mask
