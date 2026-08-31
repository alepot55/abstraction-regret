"""ANML loader tests — hand-built start-of-data + all-input fixtures (no downloads)."""

from __future__ import annotations

import pytest

from gpufsm.core.nfa import ANY_SYMBOL
from gpufsm.io.anml import load_anml, parse_symbol_set
from gpufsm.reference import simulate

# Homogeneous ANML for the pattern "ab": s0 (start, matches 'a') -> s1 (matches 'b', report).
_ANML_AB = """<?xml version="1.0"?>
<automata-network id="t">
  <state-transition-element id="s0" symbol-set="[0x61]" start="start-of-data">
    <activate-on-match element="s1"/>
  </state-transition-element>
  <state-transition-element id="s1" symbol-set="[0x62]">
    <report-on-match reportcode="1"/>
  </state-transition-element>
</automata-network>
"""


def test_parse_symbol_set_forms():
    assert parse_symbol_set("*") == {ANY_SYMBOL}
    assert parse_symbol_set("[0x61-0x63]") == {97, 98, 99}
    assert parse_symbol_set("[abc]") == {97, 98, 99}
    assert parse_symbol_set("[0x61]") == {97}
    # negation over 0..255
    neg = parse_symbol_set("[^0x61]")
    assert 97 not in neg and len(neg) == 255 and 98 in neg


def test_load_anml_fixture_semantics(tmp_path):
    p = tmp_path / "ab.anml"
    p.write_text(_ANML_AB)
    nfa = load_anml(p)
    assert simulate(nfa, b"ab") == (True, 2)
    assert simulate(nfa, b"a") == (False, 0)
    assert simulate(nfa, b"ax") == (False, 0)
    assert simulate(nfa, b"b") == (False, 0)
    assert simulate(nfa, b"") == (False, 0)


_ANML_ALLINPUT = """<?xml version="1.0"?>
<automata-network id="t">
  <state-transition-element id="s0" symbol-set="[0x61]" start="all-input">
    <report-on-match reportcode="1"/>
  </state-transition-element>
</automata-network>
"""


def test_load_anml_all_input_semantics(tmp_path):
    # all-input STE matching 'a': reports at the FIRST 'a' anywhere in the input
    # (re-seeded every position), unlike start-of-data which only fires at position 0.
    p = tmp_path / "ai.anml"
    p.write_text(_ANML_ALLINPUT)
    nfa = load_anml(p)
    assert simulate(nfa, b"a") == (True, 1)
    assert simulate(nfa, b"xa") == (True, 2)  # matches mid-stream (all-input)
    assert simulate(nfa, b"xxa") == (True, 3)
    assert simulate(nfa, b"xxx") == (False, 0)
    assert simulate(nfa, b"") == (False, 0)


def test_start_of_data_does_not_match_midstream(tmp_path):
    # Contrast: a start-of-data STE matching 'a' fires only at position 0.
    sod = _ANML_ALLINPUT.replace("all-input", "start-of-data")
    p = tmp_path / "sod.anml"
    p.write_text(sod)
    nfa = load_anml(p)
    assert simulate(nfa, b"a") == (True, 1)
    assert simulate(nfa, b"xa") == (False, 0)  # 'a' not at position 0 -> no match


def test_load_anml_rejects_unsupported_elements(tmp_path):
    # A boolean gate (<or>) changes semantics; the loader must refuse, not ignore it.
    gated = """<?xml version="1.0"?>
<automata-network id="t">
  <state-transition-element id="s0" symbol-set="[0x61]" start="all-input"/>
  <or id="g0"><report-on-match reportcode="1"/></or>
</automata-network>
"""
    p = tmp_path / "gated.anml"
    p.write_text(gated)
    with pytest.raises(ValueError, match="unsupported"):
        load_anml(p)


def test_load_anml_empty_raises(tmp_path):
    p = tmp_path / "empty.anml"
    p.write_text('<?xml version="1.0"?><automata-network id="t"></automata-network>')
    with pytest.raises(ValueError):
        load_anml(p)
