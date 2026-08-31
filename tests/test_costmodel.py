"""Cost-model byte-counting + calibration tests (deterministic, CPU-only)."""

from __future__ import annotations

import math

import pytest

from gpufsm.core.nfa import NFABuilder
from gpufsm.costmodel import (
    CostModel,
    Measurement,
    Residency,
    calibrate,
    csr_traffic_per_symbol,
    relative_error,
    traffic_per_symbol,
    working_set_bytes,
)


def _chain(n: int):
    """Simple n-state chain 0-'a'->1-'a'->...; last state accepts."""
    b = NFABuilder()
    for _ in range(n):
        b.add_state()
    b.set_accept(n - 1, True)
    b.set_start(0)
    for s in range(n - 1):
        b.add_transition(s, "a", s + 1)
    return b.build()


def test_working_set_bytes_byte_vs_word():
    nfa = _chain(100)
    # dense: one int8 per state
    assert working_set_bytes(nfa, Residency.GLOBAL_BYTE) == 100
    # packed: ceil(100/64)=2 words * 8 bytes
    assert working_set_bytes(nfa, Residency.GLOBAL_WORD) == 2 * 8


def test_byte_to_bit_reduces_working_set_traffic():
    nfa = _chain(256)
    dense = traffic_per_symbol(nfa, "cuda", "dense")
    bit_global = traffic_per_symbol(nfa, "triton", "bitpacked")  # global words
    # Same CSR term; packed working set must move strictly fewer bytes than int8/state.
    assert bit_global < dense


def test_register_residency_zeroes_working_set_traffic():
    nfa = _chain(64)
    # CUDA bitpacked is register-resident -> only CSR traffic remains.
    cuda_bit = traffic_per_symbol(nfa, "cuda", "bitpacked")
    assert cuda_bit == csr_traffic_per_symbol(nfa)


def test_shared_csr_zeroes_csr_traffic_steady_state():
    nfa = _chain(128)
    # multistream_shared: register working set + shared CSR -> ~0 modeled global traffic.
    assert traffic_per_symbol(nfa, "cuda", "multistream_shared") == 0


def test_csr_traffic_grows_with_transitions():
    small = _chain(10)
    big = _chain(200)
    assert csr_traffic_per_symbol(big) > csr_traffic_per_symbol(small)


def test_predict_throughput_positive_and_monotone():
    model = CostModel(eff_bandwidth_bytes_per_s=1e12, compute_s_per_state2=1e-11)
    nfa = _chain(128)
    dense = model.predict_throughput_gbps(nfa, "cuda", "dense")
    bit = model.predict_throughput_gbps(nfa, "cuda", "bitpacked")
    # Less traffic (register bitpacked) -> not slower than dense.
    assert bit >= dense > 0


def test_calibrate_recovers_known_params():
    # Synthesize measurements from a ground-truth model, then check we recover it.
    truth = CostModel(eff_bandwidth_bytes_per_s=2.0e12, compute_s_per_state2=5.0e-12)
    cases = [
        (_chain(64), "cuda", "dense"),
        (_chain(64), "cuda", "bitpacked"),
        (_chain(256), "triton", "bitpacked"),
        (_chain(256), "cuda", "dense"),
    ]
    ms = [
        Measurement(nfa, be, te, truth.predict_throughput_gbps(nfa, be, te))
        for (nfa, be, te) in cases
    ]
    fit = calibrate(ms)
    assert math.isclose(
        fit.eff_bandwidth_bytes_per_s, truth.eff_bandwidth_bytes_per_s, rel_tol=1e-3
    )
    assert math.isclose(fit.compute_s_per_state2, truth.compute_s_per_state2, rel_tol=1e-3)
    # And predictions match the synthetic measurements (near-zero error).
    for m in ms:
        assert relative_error(fit, m) < 1e-3


def test_calibrate_requires_two_points():
    with pytest.raises(ValueError):
        calibrate([Measurement(_chain(8), "cuda", "dense", 10.0)])
