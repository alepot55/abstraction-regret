"""Host-side packing is shared by every GPU backend — so test it on CPU.

Before these helpers were extracted each backend carried its own copy, and the copies
had drifted: differing accept-word dtypes and an empty-batch guard that existed in
some and not others. Pinning the behaviour here means a GPU is no longer needed to
catch a regression in the code every kernel depends on.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpufsm.core.nfa import NFABuilder
from gpufsm.core.packing import (
    WORD_BITS,
    accept_bitmask,
    pack_accept,
    pack_inputs,
    symbols,
    words_for,
)


def _nfa(accepting: set[int], n: int = 8):
    b = NFABuilder()
    for i in range(n):
        b.add_state(accept=i in accepting)
    b.set_start(0)
    return b.build()


@pytest.mark.parametrize(
    ("num_states", "expected"),
    [(1, 1), (63, 1), (64, 1), (65, 2), (128, 2), (129, 3), (500, 8)],
)
def test_words_for_rounds_up(num_states, expected):
    assert words_for(num_states) == expected


def test_symbols_are_int32_byte_values():
    out = symbols(b"\x00A\xff")
    assert out.dtype == np.int32
    assert out.tolist() == [0, 65, 255]


def test_pack_inputs_offsets_slice_each_string():
    inputs = [b"ab", b"", b"xyz"]
    data, offsets = pack_inputs(inputs)
    assert offsets.tolist() == [0, 2, 2, 5]
    for i, s in enumerate(inputs):
        assert bytes(data[offsets[i] : offsets[i + 1]].astype(np.uint8)) == s


def test_pack_inputs_handles_an_all_empty_batch():
    """np.frombuffer(b'') is the edge case only some of the old copies guarded."""
    data, offsets = pack_inputs([b"", b""])
    assert data.size == 0
    assert data.dtype == np.int32
    assert offsets.tolist() == [0, 0, 0]


def test_pack_inputs_handles_an_empty_batch():
    data, offsets = pack_inputs([])
    assert data.size == 0
    assert offsets.tolist() == [0]


def test_accept_bitmask_sets_one_bit_per_accepting_state():
    assert accept_bitmask(_nfa({0, 3, 7})) == 0b10001001


@pytest.mark.parametrize("dtype", [np.uint64, np.int64])
def test_pack_accept_matches_the_scalar_bitmask(dtype):
    nfa = _nfa({1, 5, 6})
    words = pack_accept(nfa, dtype=dtype)
    assert words.dtype == dtype
    assert int(words[0]) == accept_bitmask(nfa)


def test_pack_accept_spans_words_above_the_word_boundary():
    """The bug this guards: a Python-int shift truncates and drops bits >= 32."""
    n = 200
    accepting = {0, 31, 32, 63, 64, 127, 128, 199}
    nfa = _nfa(accepting, n=n)
    words = pack_accept(nfa, dtype=np.uint64)
    assert words.size == words_for(n)
    recovered = {s for s in range(n) if int(words[s // WORD_BITS]) >> (s % WORD_BITS) & 1}
    assert recovered == accepting


def test_pack_accept_honours_an_explicit_word_count():
    words = pack_accept(_nfa({0}), nwords=8)
    assert words.size == 8
    assert int(words[0]) == 1
    assert not words[1:].any()
