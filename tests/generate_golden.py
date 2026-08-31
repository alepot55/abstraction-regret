"""Regenerate ``tests/data/golden.json`` — the refactor safety net.

The golden file pins the *verdicts* of the correctness oracles
(:func:`gpufsm.reference.simulate` for NFAs, :func:`gpufsm.reference.simulate_dfa` for DFAs)
on a fixed corpus. :mod:`tests.test_golden` replays them; any refactor that changes a
single ``(accepted, match_len)`` fails loudly.

The automata are **serialized in full** (CSR arrays / transition tables), not as
generator seeds. That is deliberate: the corpus must stay valid even when the random
generators are moved, unified or reparameterized, which is exactly what the refactor
does. The generator below is self-contained for the same reason.

Run only to establish a new baseline, and only when the change in verdicts is
understood and intended::

    python -m tests.generate_golden            # rewrite the baseline
    python -m tests.generate_golden --check    # would it change? exit 1 if so, write nothing

``--check`` exists because the dangerous move here is the easy one. A refactor that changes
the semantics makes ``test_golden`` fail; regenerating makes it pass again, and the diff that
records what changed lives only in the git history of a minified JSON file. ``--check``
answers "would this rewrite the baseline?" without rewriting it, and CI runs it, so an
unintended semantic change has to be argued for rather than silently absorbed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from gpufsm.core.dfa import DFA, DFABuilder
from gpufsm.core.nfa import ANY_SYMBOL, NFA, NFABuilder
from gpufsm.examples import EXAMPLES
from gpufsm.reference import simulate, simulate_dfa

GOLDEN_PATH = Path(__file__).parent / "data" / "golden.json"
FORMAT_VERSION = 1

_ALPHABET = "abcd"
_FUZZ_CASES = 300
_FUZZ_SEED = 20260816
_DEEP_CASES = 200


def nfa_to_json(nfa: NFA) -> dict[str, Any]:
    return {
        "num_states": int(nfa.num_states),
        "start_state": int(nfa.start_state),
        "accept": [bool(x) for x in nfa.accept],
        "sym_row_ptr": [int(x) for x in nfa.sym_row_ptr],
        "sym_targets": [int(x) for x in nfa.sym_targets],
        "sym_symbols": [int(x) for x in nfa.sym_symbols],
        "eps_row_ptr": [int(x) for x in nfa.eps_row_ptr],
        "eps_targets": [int(x) for x in nfa.eps_targets],
    }


def nfa_from_json(d: dict[str, Any]) -> NFA:
    return NFA(
        num_states=int(d["num_states"]),
        start_state=int(d["start_state"]),
        accept=np.array(d["accept"], dtype=bool),
        sym_row_ptr=np.array(d["sym_row_ptr"], dtype=np.int32),
        sym_targets=np.array(d["sym_targets"], dtype=np.int32),
        sym_symbols=np.array(d["sym_symbols"], dtype=np.int32),
        eps_row_ptr=np.array(d["eps_row_ptr"], dtype=np.int32),
        eps_targets=np.array(d["eps_targets"], dtype=np.int32),
    )


def dfa_to_json(dfa: DFA) -> dict[str, Any]:
    return {
        "num_states": int(dfa.num_states),
        "start_state": int(dfa.start_state),
        "accept": [bool(x) for x in dfa.accept],
        "trans": [int(x) for x in dfa.trans],
    }


def dfa_from_json(d: dict[str, Any]) -> DFA:
    return DFA(
        num_states=int(d["num_states"]),
        start_state=int(d["start_state"]),
        accept=np.array(d["accept"], dtype=bool),
        trans=np.array(d["trans"], dtype=np.int32),
    )


def _fuzz_nfa(rng: random.Random, n_states: int) -> NFA:
    """Self-contained fuzz generator — intentionally NOT imported from the harness."""
    b = NFABuilder()
    for _ in range(n_states):
        b.add_state(accept=rng.random() < 0.2)
    b.set_start(rng.randrange(n_states))
    for s in range(n_states):
        for _ in range(rng.randint(0, 3)):
            sym = ANY_SYMBOL if rng.random() < 0.1 else rng.choice(_ALPHABET)
            b.add_transition(s, sym, rng.randrange(n_states))
        for _ in range(rng.randint(0, 2)):
            b.add_epsilon(s, rng.randrange(n_states))
    return b.build()


def _fuzz_nfa_deep(rng: random.Random, n_states: int) -> NFA:
    """A fuzz NFA whose start configuration does NOT already accept.

    Three quarters of the accepted cases in the original corpus had ``match_len == 0``: the
    start state (or its epsilon closure) was accepting, so the per-symbol loop never ran and
    the corpus barely exercised the one thing it exists to pin -- *latch-first-match*, which
    is a statement about which position the match is reported at. Resampling until the start
    closure is non-accepting makes every accepted case here consume at least one byte.
    """
    for _ in range(64):
        nfa = _fuzz_nfa(rng, n_states)
        accepted, match_len = simulate(nfa, b"")
        if not (accepted and match_len == 0):
            return nfa
    return nfa  # pragma: no cover - 64 rejections in a row is not reachable in practice


def _fuzz_dfa(rng: random.Random, n_states: int) -> DFA:
    b = DFABuilder()
    for i in range(n_states):
        b.add_state(accept=(i > 0 and rng.random() < 0.25))
    b.set_start(0)
    for s in range(n_states):
        for ch in _ALPHABET:
            if rng.random() < 0.8:
                b.add_transition(s, ord(ch), rng.randrange(n_states))
    return b.build()


def build_corpus() -> dict[str, Any]:
    nfa_cases: list[dict[str, Any]] = []

    # 1. The canonical hand-built examples, on their labelled inputs.
    for name, factory in sorted(EXAMPLES.items()):
        nfa, inputs = factory()
        payload = nfa_to_json(nfa)
        for i, (data, expected) in enumerate(inputs):
            accepted, match_len = simulate(nfa, data)
            assert accepted == expected, f"{name}[{i}] disagrees with its own label"
            nfa_cases.append(
                {
                    "id": f"example:{name}:{i}",
                    "nfa": payload,
                    "input_hex": data.hex(),
                    "accepted": accepted,
                    "match_len": match_len,
                }
            )

    # 2. Fuzz corpus: random NFAs incl. epsilon cycles and ANY_SYMBOL wildcards.
    rng = random.Random(_FUZZ_SEED)
    for i in range(_FUZZ_CASES):
        nfa = _fuzz_nfa(rng, rng.randint(1, 12))
        data = bytes(ord(rng.choice(_ALPHABET)) for _ in range(rng.randint(0, 16)))
        accepted, match_len = simulate(nfa, data)
        nfa_cases.append(
            {
                "id": f"fuzz:{i}",
                "nfa": nfa_to_json(nfa),
                "input_hex": data.hex(),
                "accepted": accepted,
                "match_len": match_len,
            }
        )

    # 3. Deep corpus: matches that happen *after* consuming input, which is where
    #    latch-first-match actually says something. Longer inputs than the fuzz section so a
    #    first match has room to land somewhere other than position 0 or 1.
    drng_deep = random.Random(_FUZZ_SEED + 2)
    for i in range(_DEEP_CASES):
        nfa = _fuzz_nfa_deep(drng_deep, drng_deep.randint(2, 14))
        data = bytes(ord(drng_deep.choice(_ALPHABET)) for _ in range(drng_deep.randint(1, 48)))
        accepted, match_len = simulate(nfa, data)
        nfa_cases.append(
            {
                "id": f"deep:{i}",
                "nfa": nfa_to_json(nfa),
                "input_hex": data.hex(),
                "accepted": accepted,
                "match_len": match_len,
            }
        )

    # 4. DFA corpus (the memory-bound face's oracle).
    dfa_cases: list[dict[str, Any]] = []
    drng = random.Random(_FUZZ_SEED + 1)
    for i in range(12):
        dfa = _fuzz_dfa(drng, drng.randint(2, 6))
        payload = dfa_to_json(dfa)
        for j in range(3):
            data = bytes(ord(drng.choice(_ALPHABET)) for _ in range(drng.randint(0, 24)))
            accepted, match_len = simulate_dfa(dfa, data)
            dfa_cases.append(
                {
                    "id": f"dfa:{i}:{j}",
                    "dfa": payload,
                    "input_hex": data.hex(),
                    "accepted": accepted,
                    "match_len": match_len,
                }
            )

    return {
        "format_version": FORMAT_VERSION,
        "note": (
            "Oracle verdicts pinned before the 2026-08-16 refactor, extended with a "
            "deep section whose matches land after at least one consumed byte. Automata "
            "are serialized in full so the corpus survives changes to the generators."
        ),
        "nfa_cases": nfa_cases,
        "dfa_cases": dfa_cases,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--check",
        action="store_true",
        help="report whether the baseline would change, and write nothing",
    )
    args = ap.parse_args(argv)

    corpus = build_corpus()
    serialized = json.dumps(corpus, separators=(",", ":")) + "\n"
    n_nfa, n_dfa = len(corpus["nfa_cases"]), len(corpus["dfa_cases"])

    if args.check:
        if not GOLDEN_PATH.exists():
            print(f"FAIL: {GOLDEN_PATH} does not exist")
            return 1
        if GOLDEN_PATH.read_text() == serialized:
            print(f"ok: {GOLDEN_PATH} matches the oracles ({n_nfa} NFA, {n_dfa} DFA cases)")
            return 0
        print(
            f"FAIL: regenerating would change {GOLDEN_PATH}. The oracle's verdicts moved.\n"
            f"      That is a semantic change: justify it, then rerun without --check."
        )
        return 1

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(serialized)
    print(f"wrote {GOLDEN_PATH} ({n_nfa} NFA cases, {n_dfa} DFA cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
