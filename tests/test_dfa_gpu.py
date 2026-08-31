"""DFA GPU kernel correctness vs the reference oracle (gpu-marked; skips without a GPU).

Covers every backend that registers a DFA technique, not only CUDA: the DFA face is half the
paper's result and Triton and Warp are two of its three arms.
"""

from __future__ import annotations

import random

import pytest

pytestmark = pytest.mark.gpu


def _cuda_dfa_available() -> bool:
    try:
        import gpufsm.backends.cuda._cuda as c  # noqa: F401

        return hasattr(c, "run_dfa")
    except Exception:
        return False


def _dfa_backends():
    """Every available backend with a registered DFA technique."""
    from gpufsm.core.registry import Backend, Kind, available_backends, list_techniques

    return [
        b for b in available_backends() if b is not Backend.CPU and list_techniques(b, Kind.DFA)
    ]


@pytest.mark.skipif(not _cuda_dfa_available(), reason="needs CUDA _cuda.run_dfa")
def test_cuda_dfa_matches_reference():
    from gpufsm.api import run_batch
    from gpufsm.core.dfa import random_dfa
    from gpufsm.reference import simulate_dfa

    rng = random.Random(0)
    for n in (16, 256, 4096):
        dfa = random_dfa(n, accept_prob=0.02, seed=n)
        batch = [bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 40))) for _ in range(48)]
        refs = [simulate_dfa(dfa, b) for b in batch]
        got = [(r.accepted, r.match_len) for r in run_batch(dfa, batch, backend="cuda")]
        assert got == refs, f"DFA mismatch at n={n}"


@pytest.mark.skipif(not _dfa_backends(), reason="no GPU backend registers a DFA technique")
def test_every_gpu_dfa_backend_matches_reference():
    """The Triton and Warp DFA paths had no test at all; only CUDA did."""
    from gpufsm.api import run_batch
    from gpufsm.core.dfa import random_dfa
    from gpufsm.reference import simulate_dfa

    rng = random.Random(7)
    batch = [bytes(rng.randint(0, 255) for _ in range(rng.randint(0, 64))) for _ in range(48)]
    for backend in _dfa_backends():
        for n in (16, 256, 4096):
            dfa = random_dfa(n, accept_prob=0.02, seed=n)
            expected = [simulate_dfa(dfa, data) for data in batch]
            got = [(r.accepted, r.match_len) for r in run_batch(dfa, batch, backend=backend)]
            assert got == expected, f"{backend.value} DFA disagrees with the oracle at n={n}"
