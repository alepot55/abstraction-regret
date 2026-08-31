"""The bit-packed simulator must be verdict-identical to the reference oracle.

This pins down the executable spec that the GPU bit-packed kernels must mirror,
and fuzzes it against random NFAs so the equivalence is not just anecdotal.
"""

from __future__ import annotations

import random

from gpufsm import ANY_SYMBOL, Backend, NFABuilder, run, simulate
from gpufsm.core.bitmap import simulate_bitmap
from gpufsm.examples import EXAMPLES


def test_bitmap_matches_reference_on_examples():
    for name in EXAMPLES:
        nfa, inputs = EXAMPLES[name]()
        for data, _ in inputs:
            assert simulate_bitmap(nfa, data) == simulate(nfa, data), f"{name} {data!r}"


def _random_nfa(rng: random.Random, n_states: int, alphabet: str) -> NFABuilder:
    b = NFABuilder()
    for _ in range(n_states):
        b.add_state(accept=rng.random() < 0.2)
    b.set_start(rng.randrange(n_states))
    for s in range(n_states):
        for _ in range(rng.randint(0, 3)):
            sym = ANY_SYMBOL if rng.random() < 0.1 else rng.choice(alphabet)
            b.add_transition(s, sym, rng.randrange(n_states))
        for _ in range(rng.randint(0, 2)):
            b.add_epsilon(s, rng.randrange(n_states))
    return b


def test_bitmap_matches_reference_fuzz():
    rng = random.Random(1234)
    alphabet = "abcd"
    for _ in range(300):
        nfa = _random_nfa(rng, rng.randint(1, 12), alphabet).build()
        data = bytes(ord(rng.choice(alphabet)) for _ in range(rng.randint(0, 16)))
        ref = simulate(nfa, data)
        bm = simulate_bitmap(nfa, data)
        assert ref == bm, f"mismatch nfa={nfa!r} data={data!r}: ref={ref} bitmap={bm}"


def test_bitmap_registered_as_cpu_technique():
    from gpufsm import list_techniques

    assert "bitmap" in list_techniques(Backend.CPU)
    nfa, inputs = EXAMPLES["ab_star_c_plus_d"]()
    data = inputs[0][0]
    assert run(nfa, data, backend=Backend.CPU, technique="bitmap").matches(
        run(nfa, data, backend=Backend.CPU, technique="reference")
    )
