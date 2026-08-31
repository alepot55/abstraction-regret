"""GPU backend correctness — skipped unless a GPU backend is actually available.

On a GPU box these assert that Triton/CUDA reproduce the CPU reference oracle
(accepted + match_len) on every example and a fuzz of random NFAs.
"""

from __future__ import annotations

import random

import pytest

from gpufsm import ANY_SYMBOL, NFABuilder, available_backends, run, run_batch, simulate
from gpufsm.core.registry import Backend as _B
from gpufsm.core.registry import list_techniques
from gpufsm.examples import EXAMPLES

_GPU_BACKENDS = [b for b in available_backends() if b in (_B.TRITON, _B.CUDA)]
pytestmark = pytest.mark.gpu

skip_no_gpu = pytest.mark.skipif(not _GPU_BACKENDS, reason="no GPU backend available")


def _cases():
    for backend in _GPU_BACKENDS:
        for technique in list_techniques(backend):
            yield backend, technique


@skip_no_gpu
def test_gpu_matches_reference_on_examples():
    for backend, technique in _cases():
        for name in EXAMPLES:
            nfa, inputs = EXAMPLES[name]()
            for data, _ in inputs:
                ref = simulate(nfa, data)
                res = run(nfa, data, backend=backend, technique=technique)
                assert (res.accepted, res.match_len) == ref, (
                    f"{backend.value}/{technique} {name} {data!r}: "
                    f"got ({res.accepted},{res.match_len}) want {ref}"
                )


@skip_no_gpu
def test_gpu_matches_reference_fuzz():
    rng = random.Random(7)
    alphabet = "abc"
    for backend, technique in _cases():
        for _ in range(40):
            b = NFABuilder()
            n = rng.randint(1, 8)
            for _ in range(n):
                b.add_state(accept=rng.random() < 0.25)
            b.set_start(rng.randrange(n))
            for s in range(n):
                for _ in range(rng.randint(0, 2)):
                    sym = ANY_SYMBOL if rng.random() < 0.1 else rng.choice(alphabet)
                    b.add_transition(s, sym, rng.randrange(n))
                for _ in range(rng.randint(0, 1)):
                    b.add_epsilon(s, rng.randrange(n))
            nfa = b.build()
            data = bytes(ord(rng.choice(alphabet)) for _ in range(rng.randint(0, 10)))
            ref = simulate(nfa, data)
            res = run(nfa, data, backend=backend, technique=technique)
            assert (res.accepted, res.match_len) == ref


@skip_no_gpu
def test_gpu_run_batch_matches_reference():
    """run_batch (native multi-stream + loop fallback) reproduces the oracle per string."""
    rng = random.Random(11)
    alphabet = "abcd"
    for backend, technique in _cases():
        for _ in range(8):
            b = NFABuilder()
            n = rng.randint(1, 64)  # ≤64 so every technique applies (incl. Warp single-word)
            for _ in range(n):
                b.add_state(accept=rng.random() < 0.2)
            b.set_start(rng.randrange(n))
            for s in range(n):
                for _ in range(rng.randint(0, 2)):
                    sym = ANY_SYMBOL if rng.random() < 0.05 else ord(rng.choice(alphabet))
                    b.add_transition(s, sym, rng.randrange(n))
                for _ in range(rng.randint(0, 1)):
                    b.add_epsilon(s, rng.randrange(n))
            nfa = b.build()
            batch = [
                bytes(ord(rng.choice(alphabet)) for _ in range(rng.randint(0, 16)))
                for _ in range(rng.randint(1, 24))
            ]
            refs = [simulate(nfa, d) for d in batch]
            res = run_batch(nfa, batch, backend=backend, technique=technique)
            assert [(r.accepted, r.match_len) for r in res] == refs, (
                f"{backend.value}/{technique} batch mismatch"
            )


# Multi-word (>64 states) batch coverage for backends/techniques that support it
# (CUDA + Triton multistream variants); Warp's single-word kernel is excluded.
_MULTIWORD_CASES = [
    (b, t) for (b, t) in _cases() if b in (_B.CUDA, _B.TRITON) and t.startswith("multistream")
]


@skip_no_gpu
@pytest.mark.skipif(not _MULTIWORD_CASES, reason="no multi-word-capable GPU technique")
def test_gpu_multiword_batch_matches_reference():
    rng = random.Random(23)
    alphabet = "abcde"
    for backend, technique in _MULTIWORD_CASES:
        for n in (65, 130, 300):  # NWORDS 2,3,5
            b = NFABuilder()
            for _ in range(n):
                b.add_state(accept=rng.random() < 0.1)
            b.set_start(rng.randrange(n))
            for s in range(n):
                for _ in range(rng.randint(0, 2)):
                    sym = ANY_SYMBOL if rng.random() < 0.05 else ord(rng.choice(alphabet))
                    b.add_transition(s, sym, rng.randrange(n))
                for _ in range(rng.randint(0, 1)):
                    b.add_epsilon(s, rng.randrange(n))
            nfa = b.build()
            batch = [
                bytes(ord(rng.choice(alphabet)) for _ in range(rng.randint(0, 24)))
                for _ in range(rng.randint(1, 16))
            ]
            refs = [simulate(nfa, d) for d in batch]
            res = run_batch(nfa, batch, backend=backend, technique=technique)
            assert [(r.accepted, r.match_len) for r in res] == refs, (
                f"{backend.value}/{technique} multiword batch mismatch (n={n})"
            )
