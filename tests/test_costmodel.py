"""Cost-model byte-counting + calibration tests (deterministic, CPU-only)."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from gpufsm.core.nfa import NFABuilder
from gpufsm.costmodel import (
    _RESIDENCY,
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


# --- the cost model must know every technique it could be asked about ----------------
#
# `traffic_per_symbol` used to fall back to GLOBAL_WORD for an undeclared (backend,
# technique), which silently modelled the entire worklist family -- the work-efficient
# kernels the thesis turns on -- as something it had never been told about. It now raises,
# and this pins the declaration against the technique names the backends actually register.
# The names are read out of the backend sources rather than the live registry because the
# CPU suite runs without torch, triton or the CUDA extension, so the registry is nearly
# empty here and would make the test vacuous.

_BACKENDS = Path(__file__).resolve().parent.parent / "src" / "gpufsm" / "backends"


def _declared_nfa_techniques() -> set[tuple[str, str]]:
    """(backend, technique) pairs registered for NFAs, scraped from the sources."""
    found: set[tuple[str, str]] = set()
    for path in _BACKENDS.rglob("*.py"):
        text = path.read_text()
        backend = path.relative_to(_BACKENDS).parts[0].removesuffix(".py")
        # @register(Kind.NFA, Backend.X, "name")
        for be, tech in re.findall(r'register\(\s*Kind\.NFA,\s*Backend\.(\w+),\s*"([^"]+)"', text):
            found.add((be.lower(), tech))
        # the CUDA dicts: SINGLE_TECHNIQUES / BATCH_TECHNIQUES map technique -> entry point
        for block in re.findall(r"(?:SINGLE|BATCH)_TECHNIQUES\s*=\s*\{(.*?)\n\}", text, re.S):
            for tech in re.findall(r'"([a-z_]+)"\s*:\s*"run_', block):
                found.add((backend, tech))
        # the CPU table: (Kind.NFA, "name"): ...
        for tech in re.findall(r'\(Kind\.NFA,\s*"([^"]+)"\)', text):
            found.add(("cpu", tech))
    return found


def test_every_registered_nfa_technique_has_a_declared_residency() -> None:
    missing = sorted(_declared_nfa_techniques() - set(_RESIDENCY))
    assert not missing, f"no residency declared in gpufsm.costmodel for: {missing}"


def test_the_scrape_actually_found_the_families_it_should() -> None:
    """Guard the guard: a regex that matched nothing would make the test above vacuous."""
    found = _declared_nfa_techniques()
    assert ("cuda", "worklist_global") in found
    assert ("triton", "multistream") in found
    assert ("cpu", "reference") in found
    assert len(found) >= 15, f"scrape found only {len(found)} techniques"


def test_an_undeclared_technique_raises_instead_of_guessing() -> None:
    b = NFABuilder()
    s0 = b.add_state(accept=True)
    b.set_start(s0)
    with pytest.raises(KeyError, match="no working-set residency"):
        traffic_per_symbol(b.build(), "cuda", "a_technique_that_does_not_exist")
