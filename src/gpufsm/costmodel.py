"""Predictive memory cost model — operationalizing "abstraction regret".

The thesis: for irregular NFA processing the Triton↔CUDA (and DSL↔DSL) gap is set by
the *memory layout a model can express*, not by algorithm or scheduling. To make that
claim testable rather than rhetorical, we model the **memory traffic moved per input
symbol** under each technique and predict throughput, then validate the predictions
against measured runs (see ``scripts``/sweep CSVs).

The model is deliberately simple (KISS) and analytic — two physically-meaningful
fitted constants, not a black box:

    time_per_symbol = traffic_bytes_per_symbol / eff_bandwidth
                    + num_states**2 * compute_s_per_state2

i.e. a roofline-style sum of a **memory term** (what the layout/technique changes) and
a **compute term** that is *quadratic* in num_states: the faithful constant-algorithm
kernel does an O(n) transition scan plus an O(n^2) epsilon-closure (n convergence
passes x n states) per input symbol. This was confirmed empirically — throughput
scales as 1/n^2; a linear compute term mis-fits (~85% error), the n^2 term fits well.

Consequence the model makes quantitative: while the dense full-scan kernel is
COMPUTE-bound, the memory term is negligible, so memory-layout techniques (shared CSR,
async, even byte->bit) barely move throughput — measured directly, multistream_shared
(modeled traffic = 0) ties multistream (traffic > 0). The "abstraction regret" (memory
layout) only bites once the algorithm is made work-efficient (sparse active-set /
worklist, as in ngAP) so memory becomes the bottleneck. The model thus predicts which
regime a kernel is in and the ceiling on what a memory technique can buy.

Byte-counting is deterministic and unit-tested on CPU; calibration/validation of the
two constants needs measured throughput (GPU), done from the sweep data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .core.nfa import NFA

WORD_BITS = 64
_INT = 4  # bytes per int32 CSR element
_WORD = WORD_BITS // 8  # bytes per 64-bit working-set word


class Residency(str, Enum):
    """Where a technique keeps the active/next state-set working set."""

    GLOBAL_BYTE = "global_byte"  # int8 per state in (global-backed) local memory — `dense`
    GLOBAL_WORD = "global_word"  # packed 64-bit words in global/local memory
    REGISTER = "register"  # packed words held in registers (CUDA ≤512 states, Warp ≤64)


# Map (backend, technique) -> working-set residency. Drives the memory term.
# CUDA/Warp bit-packed kernels keep the word(s) in registers; the Triton bit-packed
# kernel stores them in a global scratch tensor (it cannot express register residency).
#
# Every registered NFA technique must appear here. It used to cover eleven of them and
# fall back to GLOBAL_WORD for the rest, which silently modelled the whole worklist family
# -- the work-efficient kernels the thesis turns on -- as something it had never been told
# about. `tests/test_costmodel.py` pins the coverage against the registry.
_RESIDENCY: dict[tuple[str, str], Residency] = {
    ("cpu", "reference"): Residency.GLOBAL_BYTE,
    ("cpu", "bitmap"): Residency.GLOBAL_WORD,
    ("triton", "dense"): Residency.GLOBAL_BYTE,
    ("triton", "bitpacked"): Residency.GLOBAL_WORD,
    ("triton", "multistream"): Residency.GLOBAL_WORD,
    # Triton's worklist holds the active set in one scalar int64 register, which is why it
    # is capped at 64 states; nothing about the set spills to global memory per symbol.
    ("triton", "worklist"): Residency.REGISTER,
    ("cuda", "dense"): Residency.GLOBAL_BYTE,
    ("cuda", "bitpacked"): Residency.REGISTER,
    ("cuda", "multistream"): Residency.REGISTER,
    ("cuda", "multistream_shared"): Residency.REGISTER,
    ("cuda", "multistream_async"): Residency.REGISTER,
    # Worklist family: `worklist` is the register-resident path (<=512 states); the others
    # deliberately move the working set out of registers, which is the axis they ablate.
    ("cuda", "worklist"): Residency.REGISTER,
    ("cuda", "worklist_global"): Residency.GLOBAL_WORD,
    ("cuda", "worklist_warp"): Residency.GLOBAL_WORD,
    ("cuda", "worklist_shared"): Residency.GLOBAL_WORD,
    ("cuda", "worklist_compact"): Residency.GLOBAL_WORD,
    ("warp", "multistream"): Residency.REGISTER,
}

# The per-symbol step touches the working set a small constant number of times
# (zero the next set, scatter into it, copy back, accept-test). Absorbed as one
# factor; the absolute value is folded into the fitted bandwidth, so only the
# *ratio* between techniques matters for prediction.
_WS_TOUCHES = 4


def working_set_bytes(nfa: NFA, residency: Residency) -> int:
    """Footprint of one state-set vector under a residency choice."""
    if residency is Residency.GLOBAL_BYTE:
        return nfa.num_states  # one int8 slot per state
    nwords = (nfa.num_states + WORD_BITS - 1) // WORD_BITS
    return nwords * _WORD  # packed 64-bit words


def working_set_traffic_per_symbol(nfa: NFA, residency: Residency) -> int:
    """Global-memory traffic for the working set, per input symbol.

    Register-resident working sets move ~0 global bytes per symbol — the whole point
    of byte→bit + global→register: the state vector never touches global memory.
    """
    if residency is Residency.REGISTER:
        return 0
    return _WS_TOUCHES * working_set_bytes(nfa, residency)


def csr_traffic_per_symbol(nfa: NFA, in_shared: bool = False) -> int:
    """CSR transition-table traffic per symbol (worst case: full state scan).

    Reads ``sym_row_ptr`` (num_states+1) and the symbol/target arrays. When the CSR is
    staged in shared memory (``multistream_shared``) the per-symbol *global* traffic is
    ~0 after the one-time block-level load, so this returns 0 for the steady state.
    """
    if in_shared:
        return 0
    nnz_sym = int(nfa.sym_targets.size)
    return ((nfa.num_states + 1) + 2 * nnz_sym) * _INT


def traffic_per_symbol(nfa: NFA, backend: str, technique: str) -> int:
    """Total modeled global-memory bytes moved per input symbol for a technique.

    Raises on a technique whose residency has not been declared. Guessing one produces a
    prediction that looks as authoritative as a modelled one, and the cost model's whole
    job is to say which regime a kernel is in.
    """
    try:
        residency = _RESIDENCY[(backend, technique)]
    except KeyError:
        raise KeyError(
            f"no working-set residency declared for {backend}/{technique}; "
            f"add it to gpufsm.costmodel._RESIDENCY"
        ) from None
    in_shared = technique == "multistream_shared"
    return working_set_traffic_per_symbol(nfa, residency) + csr_traffic_per_symbol(
        nfa, in_shared=in_shared
    )


@dataclass(frozen=True)
class CostModel:
    """Two fitted constants; predicts time/throughput from modeled memory traffic.

    ``eff_bandwidth_bytes_per_s``: effective sustained global bandwidth seen by the
    kernel. ``compute_s_per_state2``: per-symbol compute cost per num_states**2 — the
    O(n^2) epsilon-closure + scan dominates the faithful kernel (identical across
    memory techniques at constant algorithm).
    """

    eff_bandwidth_bytes_per_s: float
    compute_s_per_state2: float

    def time_per_symbol_s(self, nfa: NFA, backend: str, technique: str) -> float:
        mem = traffic_per_symbol(nfa, backend, technique) / self.eff_bandwidth_bytes_per_s
        compute = (nfa.num_states**2) * self.compute_s_per_state2
        return mem + compute

    def predict_throughput_gbps(self, nfa: NFA, backend: str, technique: str) -> float:
        """Predicted input throughput in Gbps (1 symbol = 1 byte of input)."""
        t = self.time_per_symbol_s(nfa, backend, technique)
        if t <= 0:
            return float("inf")
        return (8.0 / t) / 1e9  # bits-per-symbol / time → bits/s → Gbps


@dataclass(frozen=True)
class Measurement:
    """One observed point used to calibrate/validate the model."""

    nfa: NFA
    backend: str
    technique: str
    throughput_gbps: float


def fit(traffic: list[float], n_squared: list[float], seconds_per_symbol: list[float]) -> CostModel:
    """Least-squares fit of ``time = a*traffic + b*n^2`` over raw columns.

    The single implementation of the model's fit. There used to be three -- here, in
    ``scripts/validate_costmodel.py`` and inline in ``paper/figures.py`` -- and they had
    already drifted: the figure generator carried a unit error that put every predicted
    point nine decades off its own y = x line, and kept a clamp this function no longer uses.

    A negative fitted ``a`` means the data carry no memory term at all: the kernel is
    compute-bound and the unconstrained solve is describing noise. Clamping ``a`` to a tiny
    positive number reported an effective bandwidth of 10^9 GB/s, six orders above any real
    device, while leaving ``b`` fitted against a memory term the clamp had just removed. So
    the compute-bound branch refits ``b`` alone and says ``inf`` for the bandwidth, which
    makes the pair the solution of a well-posed problem either way -- what
    :func:`relative_error` assumes when it uses them together.
    """
    import numpy as np

    if len(traffic) < 2:
        raise ValueError("calibration needs >= 2 points")
    a_mat = np.stack([np.asarray(traffic, dtype=float), np.asarray(n_squared, dtype=float)], axis=1)
    b_vec = np.asarray(seconds_per_symbol, dtype=float)
    coef, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    a = float(coef[0])  # 1/bandwidth (s/byte)
    if a <= 0.0:
        coef_b, *_ = np.linalg.lstsq(a_mat[:, 1:2], b_vec, rcond=None)
        return CostModel(
            eff_bandwidth_bytes_per_s=math.inf,
            compute_s_per_state2=max(float(coef_b[0]), 0.0),
        )
    return CostModel(
        eff_bandwidth_bytes_per_s=1.0 / a,
        compute_s_per_state2=max(float(coef[1]), 0.0),
    )


def calibrate(measurements: list[Measurement]) -> CostModel:
    """Fit the model from measured throughputs. See :func:`fit`.

    Each measurement gives ``time_per_symbol = 8e-9/throughput_gbps`` seconds and a row
    ``(traffic, num_states^2)``. Requires >= 2 points spanning techniques.
    """
    traffic, n_squared, times = [], [], []
    for m in measurements:
        if m.throughput_gbps <= 0:
            continue
        traffic.append(float(traffic_per_symbol(m.nfa, m.backend, m.technique)))
        n_squared.append(float(m.nfa.num_states**2))
        times.append(8e-9 / m.throughput_gbps)  # seconds per symbol
    if len(traffic) < 2:
        raise ValueError("need >= 2 valid (positive-throughput) measurements")
    return fit(traffic, n_squared, times)


def relative_error(model: CostModel, m: Measurement) -> float:
    """|predicted - measured| / measured throughput — the validation metric."""
    pred = model.predict_throughput_gbps(m.nfa, m.backend, m.technique)
    if m.throughput_gbps <= 0:
        return float("inf")
    return abs(pred - m.throughput_gbps) / m.throughput_gbps
