"""ANML (Automata Network Markup Language) loading.

ANML is Micron's XML format for *homogeneous* automata: the matching symbol-set
lives on the state-transition-element (STE), not on the edge. ``activate-on-match``
edges connect STEs; an STE activates when a predecessor was active **and** the
current input byte is in the STE's symbol-set. ``start-of-data`` STEs are seeded at
position 0; ``report-on-match`` STEs are accepting.

We convert that to gpufsm's *edge-labelled* CSR :class:`~gpufsm.nfa.NFA` by pushing
each STE's symbol-set onto its incoming edges, plus a synthetic start state that
seeds the start-of-data STEs:

    edge u --c--> v   for every activate-on-match u->v and every c in symbolset(v)
    START --c--> s    for every start-of-data STE s and every c in symbolset(s)
    accept            = the report-on-match STEs

Supported symbol-set forms: bracketed classes ``[...]`` with byte ranges
``0xHH-0xHH``, single bytes ``0xHH`` / ``\\xHH``, literal ASCII, and negation
``[^...]`` over 0..255; a bare ``*`` means "any byte" (:data:`gpufsm.nfa.ANY_SYMBOL`).
Unsupported constructs raise rather than silently mis-parsing.

This is a well-defined subset, validated by hand-built fixtures in ``tests/test_anml.py``
and by loading the six pinned ANMLZoo families. The suite's data is fetched on demand
with a pinned checksum (see :mod:`gpufsm.io.datasets`).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

from ..core.nfa import ANY_SYMBOL, NFA, NFABuilder

_ALL_BYTES = range(256)


_HEX_ATOM = re.compile(r"(?:0x|\\x)[0-9a-fA-F]{2}", re.IGNORECASE)
"""Exactly two hex digits after the prefix; anything shorter is a malformed atom."""


def _parse_atom(tok: str) -> int:
    """Parse a single symbol atom: 0xHH, \\xHH, or a one-char literal."""
    t = tok
    if t[:2].lower() in ("0x", "\\x"):
        return int(t[2:], 16)
    if len(t) == 1:
        return ord(t)
    raise ValueError(f"unsupported ANML symbol atom: {tok!r}")


def parse_symbol_set(s: str) -> set[int]:
    """Parse an ANML symbol-set attribute into a set of byte values (0..255).

    Returns ``{ANY_SYMBOL}`` for the wildcard ``*``. Raises on unsupported syntax.
    """
    s = s.strip()
    if s == "*":
        return {ANY_SYMBOL}
    if not (s.startswith("[") and s.endswith("]")):
        return {_parse_atom(s)}
    body = s[1:-1]
    if not body:
        raise ValueError(f"empty ANML symbol-set {s!r}: the STE would be unreachable")
    negate = body.startswith("^")
    if negate:
        # `[^]` is the *negated* empty class, i.e. every byte -- maximally reachable, not
        # unreachable. The emptiness check above therefore runs before the caret is stripped.
        body = body[1:]

    # Tokenize into atoms: 0xHH | \xHH | single char. A hex prefix must be followed by
    # exactly two hex digits -- taking four characters blindly turned a truncated atom at
    # the end of a class into a different, plausible byte value ("[0x4]" parsed as 0x04).
    tokens: list[str] = []
    i = 0
    while i < len(body):
        if body[i : i + 2].lower() in ("0x", "\\x"):
            atom = body[i : i + 4]
            if not _HEX_ATOM.fullmatch(atom):
                raise ValueError(f"malformed hex atom {atom!r} in ANML symbol-set {s!r}")
            tokens.append(atom)
            i += 4
        else:
            tokens.append(body[i])
            i += 1

    out: set[int] = set()
    j = 0
    while j < len(tokens):
        if j + 2 < len(tokens) and tokens[j + 1] == "-":
            lo, hi = _parse_atom(tokens[j]), _parse_atom(tokens[j + 2])
            if hi < lo:
                raise ValueError(f"inverted ANML range in {s!r}")
            out.update(range(lo, hi + 1))
            j += 3
        else:
            out.add(_parse_atom(tokens[j]))
            j += 1

    if negate:
        out = {b for b in _ALL_BYTES if b not in out}
    return out


def _local(tag: str) -> str:
    """Strip an XML namespace from a tag name."""
    return tag.rsplit("}", 1)[-1]


# Elements the homogeneous-STE subset understands. Anything else (counters, boolean
# gates: and/or/nor/inverter, report-on-high, ...) changes the automaton's semantics, so
# we REFUSE rather than silently ignore it (which would yield a wrong automaton).
_ALLOWED_TAGS = frozenset(
    {
        "anml",
        "automata-network",
        "description",
        "state-transition-element",
        "activate-on-match",
        "report-on-match",
    }
)


def load_anml(path: str | Path) -> NFA:
    """Load a (supported-subset) ANML file into an edge-labelled :class:`NFA`.

    Raises if the file uses ANML features outside the homogeneous-STE subset (counters,
    boolean gates, ...), so a partially-understood file never produces a wrong automaton.
    """
    root = ET.parse(path).getroot()  # noqa: S314 - local trusted automata files
    unsupported = sorted({_local(e.tag) for e in root.iter()} - _ALLOWED_TAGS)
    if unsupported:
        raise ValueError(
            f"{path}: unsupported ANML elements {unsupported} (only the homogeneous-STE "
            f"subset is supported; counters/boolean gates change semantics)"
        )
    stes = [e for e in root.iter() if _local(e.tag) == "state-transition-element"]
    if not stes:
        raise ValueError(f"no state-transition-element found in {path}")

    # Validate the ids before allocating anything. Keying a dict on `ste.get("id")` meant a
    # missing id became the key None and two STEs sharing an id collapsed into one state,
    # in both cases leaving an allocated state orphaned and the automaton quietly wrong. The
    # six pinned ANMLZoo families contain none of these, so this only ever fires on a file
    # that would previously have been mis-parsed.
    ids = [ste.get("id") for ste in stes]
    if any(i is None for i in ids):
        n = sum(1 for i in ids if i is None)
        raise ValueError(f"{path}: {n} state-transition-element(s) without an 'id' attribute")
    seen: set[str] = set()
    duplicates = sorted({i for i in ids if i in seen or seen.add(i)})  # type: ignore[func-returns-value,arg-type]
    if duplicates:
        raise ValueError(f"{path}: duplicate state-transition-element ids {duplicates[:5]}")

    symset = {}
    for ste in stes:
        raw = ste.get("symbol-set", "*")
        try:
            symset[ste.get("id")] = parse_symbol_set(raw)
        except ValueError as exc:
            raise ValueError(f"{path}: STE {ste.get('id')!r}: {exc}") from None

    # Three synthetic start states encode ANML's two start modes correctly:
    #   q_root (the NFA start) eps-> q_all and q_first.
    #   q_all has a self-loop on ANY symbol, so it stays active at EVERY position ->
    #     it seeds `all-input` STEs each step (an all-input STE may match anywhere).
    #   q_first has no self-loop, so it is active only at position 0 -> it seeds
    #     `start-of-data` STEs once (they may match only at the start of the input).
    b = NFABuilder()
    q_root = b.add_state()
    q_all = b.add_state()
    q_first = b.add_state()
    b.set_start(q_root)
    b.add_epsilon(q_root, q_all)
    b.add_epsilon(q_root, q_first)
    b.add_transition(q_all, ANY_SYMBOL, q_all)  # persists every position
    ste_state = {ste.get("id"): b.add_state() for ste in stes}

    def add_edges_into(target_id: str, src_state: int) -> None:
        for c in symset[target_id]:
            b.add_transition(src_state, c, ste_state[target_id])

    for ste in stes:
        sid = ste.get("id")
        assert sid is not None  # validated above
        st = ste_state[sid]
        start_attr = (ste.get("start") or "none").lower()
        if start_attr == "all-input":
            add_edges_into(sid, q_all)
        elif start_attr == "start-of-data":
            add_edges_into(sid, q_first)
        for child in ste:
            t = _local(child.tag)
            if t == "activate-on-match":
                # A target that does not resolve is a *lost edge*, i.e. a different
                # language, and the previous `if tgt in ste_state` guard dropped it in
                # silence. ANML also writes the target as "id:port"; the port selects an
                # input port on the element and the id is what identifies it.
                tgt = child.get("element")
                if tgt is None:
                    raise ValueError(f"{path}: STE {sid!r}: activate-on-match without 'element'")
                if tgt not in ste_state and ":" in tgt:
                    tgt = tgt.split(":", 1)[0]
                if tgt not in ste_state:
                    raise ValueError(
                        f"{path}: STE {sid!r} activates unknown element {child.get('element')!r}"
                    )
                add_edges_into(tgt, st)
            elif t == "report-on-match":
                b.set_accept(st, True)
    return b.build()
