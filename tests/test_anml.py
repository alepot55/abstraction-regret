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


# --- the loader must refuse a file it cannot parse, not guess ------------------------
#
# The module promises that "a partially-understood file never produces a wrong automaton".
# Four deviations used to break that promise silently. None of them occurs in the six pinned
# ANMLZoo families -- verified by loading all six before and after this change and comparing
# the CSR arrays byte for byte -- so these paths only fire on a file that would previously
# have been mis-parsed into a different language.

_TEMPLATE = """<anml><automata-network>{body}</automata-network></anml>"""


def _load(tmp_path, body: str):
    p = tmp_path / "a.anml"
    p.write_text(_TEMPLATE.format(body=body))
    return load_anml(p)


class TestLoaderRefusesMalformedFiles:
    def test_duplicate_ste_id_raises(self, tmp_path) -> None:
        body = (
            '<state-transition-element id="s" symbol-set="a" start="start-of-data"/>'
            '<state-transition-element id="s" symbol-set="b"/>'
        )
        with pytest.raises(ValueError, match="duplicate"):
            _load(tmp_path, body)

    def test_missing_ste_id_raises(self, tmp_path) -> None:
        body = '<state-transition-element symbol-set="a" start="start-of-data"/>'
        with pytest.raises(ValueError, match="without an 'id'"):
            _load(tmp_path, body)

    def test_dangling_activate_target_raises(self, tmp_path) -> None:
        """A lost edge is a different language; it used to be dropped in silence."""
        body = (
            '<state-transition-element id="s" symbol-set="a" start="start-of-data">'
            '<activate-on-match element="NOSUCH"/>'
            "</state-transition-element>"
        )
        with pytest.raises(ValueError, match="activates unknown element"):
            _load(tmp_path, body)

    def test_activate_target_with_port_resolves(self, tmp_path) -> None:
        """ANML writes the target as "id:port"; the id is what identifies the element."""
        body = (
            '<state-transition-element id="s" symbol-set="a" start="start-of-data">'
            '<activate-on-match element="t:in"/>'
            "</state-transition-element>"
            '<state-transition-element id="t" symbol-set="b">'
            "<report-on-match/>"
            "</state-transition-element>"
        )
        nfa = _load(tmp_path, body)
        assert simulate(nfa, b"ab") == (True, 2)

    def test_symbol_set_error_names_the_ste(self, tmp_path) -> None:
        body = '<state-transition-element id="s" symbol-set="[0x4]" start="start-of-data"/>'
        with pytest.raises(ValueError, match="STE 's'"):
            _load(tmp_path, body)


class TestSymbolSetTokenizer:
    @pytest.mark.parametrize("text", ["[0x4]", "[0xZZ]", "[]", "[^]"])
    def test_malformed_symbol_sets_raise(self, text: str) -> None:
        with pytest.raises(ValueError):
            parse_symbol_set(text)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [("[0x41]", {0x41}), ("[\\x41]", {0x41}), ("[0x41-0x43]", {0x41, 0x42, 0x43})],
    )
    def test_both_hex_prefixes_still_parse(self, text: str, expected: set[int]) -> None:
        assert parse_symbol_set(text) == expected
