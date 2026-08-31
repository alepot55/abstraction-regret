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
_RESIDENCY: dict[tuple[str, str], Residency] = {
    ("cpu", "reference"): Residency.GLOBAL_BYTE,
    ("cpu", "bitmap"): Residency.GLOBAL_WORD,
    ("triton", "dense"): Residency.GLOBAL_BYTE,
    ("triton", "bitpacked"): Residency.GLOBAL_WORD,
    ("triton", "multistream"): Residency.GLOBAL_WORD,
    ("cuda", "dense"): Residency.GLOBAL_BYTE,
    ("cuda", "bitpacked"): Residency.REGISTER,
    ("cuda", "multistream"): Residency.REGISTER,
    ("cuda", "multistream_shared"): Residency.REGISTER,
    ("cuda", "multistream_async"): Residency.REGISTER,
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
    """Total modeled global-memory bytes moved per input symbol for a technique."""
    residency = _RESIDENCY.get((backend, technique), Residency.GLOBAL_WORD)
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


def calibrate(measurements: list[Measurement]) -> CostModel:
    """Fit (eff_bandwidth, compute_s_per_state2) by least squares.

    Each measurement gives ``time_per_symbol = 8e-9/throughput_gbps`` (seconds) and a
    linear equation ``time = a*traffic + b*num_states**2`` with ``a = 1/bandwidth`` and
    ``b = compute_s_per_state2``. We solve for (a, b) over all points, clamp to be
    physical, and return the fitted model. Requires >= 2 points spanning techniques.
    """
    import numpy as np

    if len(measurements) < 2:
        raise ValueError("calibration needs >= 2 measurements")
    rows = []
    rhs = []
    for m in measurements:
        if m.throughput_gbps <= 0:
            continue
        traffic = traffic_per_symbol(m.nfa, m.backend, m.technique)
        rows.append([float(traffic), float(m.nfa.num_states**2)])
        rhs.append(8e-9 / m.throughput_gbps)  # seconds per symbol
    if len(rows) < 2:
        raise ValueError("need >= 2 valid (positive-throughput) measurements")
    a_mat = np.asarray(rows, dtype=float)
    b_vec = np.asarray(rhs, dtype=float)
    coef, *_ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    a = max(float(coef[0]), 1e-18)  # 1/bandwidth (s/byte)
    b = max(float(coef[1]), 0.0)  # s/state^2
    return CostModel(eff_bandwidth_bytes_per_s=1.0 / a, compute_s_per_state2=b)


def relative_error(model: CostModel, m: Measurement) -> float:
    """|predicted - measured| / measured throughput — the validation metric."""
    pred = model.predict_throughput_gbps(m.nfa, m.backend, m.technique)
    if m.throughput_gbps <= 0:
        return float("inf")
    return abs(pred - m.throughput_gbps) / m.throughput_gbps
