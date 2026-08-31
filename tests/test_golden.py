"""Replay the pinned oracle verdicts — the invariant no refactor may break.

``tests/data/golden.json`` records ``(accepted, match_len)`` for a fixed corpus of
serialized automata. Every backend, technique, layout and file move must leave these
verdicts untouched: a failure here means the *semantics* changed, not the structure.

Regenerate only deliberately: ``python -m tests.generate_golden``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpufsm.core.bitmap import simulate_bitmap
from gpufsm.reference import simulate, simulate_dfa

from .generate_golden import GOLDEN_PATH, dfa_from_json, nfa_from_json

_CORPUS = json.loads(Path(GOLDEN_PATH).read_text())


def test_corpus_is_non_trivial():
    """Guard against a truncated/empty golden file silently passing everything."""
    assert len(_CORPUS["nfa_cases"]) >= 300
    assert len(_CORPUS["dfa_cases"]) >= 12
    assert any(c["accepted"] for c in _CORPUS["nfa_cases"])
    assert any(not c["accepted"] for c in _CORPUS["nfa_cases"])


@pytest.mark.parametrize("case", _CORPUS["nfa_cases"], ids=lambda c: c["id"])
def test_reference_matches_golden(case):
    nfa = nfa_from_json(case["nfa"])
    data = bytes.fromhex(case["input_hex"])
    assert simulate(nfa, data) == (case["accepted"], case["match_len"])


@pytest.mark.parametrize("case", _CORPUS["nfa_cases"], ids=lambda c: c["id"])
def test_bitmap_matches_golden(case):
    """The bit-packed spec is the GPU kernels' model: hold it to the same verdicts."""
    nfa = nfa_from_json(case["nfa"])
    data = bytes.fromhex(case["input_hex"])
    assert simulate_bitmap(nfa, data) == (case["accepted"], case["match_len"])


@pytest.mark.parametrize("case", _CORPUS["dfa_cases"], ids=lambda c: c["id"])
def test_dfa_reference_matches_golden(case):
    dfa = dfa_from_json(case["dfa"])
    data = bytes.fromhex(case["input_hex"])
    assert simulate_dfa(dfa, data) == (case["accepted"], case["match_len"])
