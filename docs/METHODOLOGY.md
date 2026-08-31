# Methodology

What is measured, against what, and with which statistic. The code tour is
[`ARCHITECTURE.md`](ARCHITECTURE.md); the ways this could be wrong are [`THREATS.md`](THREATS.md).

## The quantity

**Abstraction regret** is the performance a GPU DSL forecloses, *at a fixed algorithm*, because
it cannot express the memory layout or the control flow the workload needs. It is a ratio
against a hand-written CUDA baseline on the same algorithm, the same data and the same device.

The definition is what forces the protocol: if each DSL were allowed to pick the algorithm that
suits it, the measurement would be of algorithmic redesign, not of what the abstraction
precludes. The limit is one of scope, not of bias, and it is why the kernels are mirrored.

## Two workloads, chosen to be two faces of one cause

- **NFA simulation** — active-set traversal with epsilon closure over a CSR automaton.
  Control-flow-bound: the work per input symbol depends on which states are live.
- **DFA simulation** — a dense `num_states x 256` transition table, one gather per input byte.
  Memory-bound: as the table grows past the device's cache the gather goes to DRAM.

A DSL that pays on only one of these could be blamed on the workload. One that pays on both is
paying for its execution model. That is the whole design of the experiment.

## What is compared

The same automaton, in the same CSR representation, executed by:

| | Abstraction | Paradigm |
|---|---|---|
| CUDA (hand-written, pybind11) | low | thread / SIMT |
| NVIDIA Warp | high (Python) | thread / SIMT |
| Triton | high (Python) | tile / SPMD |
| Gluon | low (explicit layouts) | tile / SPMD |
| CPU reference (`gpufsm.reference`) | — | the correctness oracle |
| CPU bit-packed (`gpufsm.core.bitmap`) | — | the executable spec of the packed GPU kernels |

Gluon is the control that turns an observation into an attribution. It shares Triton's compiler
stack and only *adds* layout and shared-memory control, removing no tuning lever. If the
binding constraint were "Triton was not tuned or laid out well", Gluon could express the kernel.
It cannot (`scripts/gluon_probe.py`), so the constraint is the paradigm.

Within each DSL the *technique* varies — full-scan dense, bit-packed, multi-stream, the
shared-CSR and async-transfer ablations, the work-efficient worklist family — and the best is
reported. Per-language tuning is not excluded; only the algorithm is pinned.

## Correctness comes before any number

Every backend and technique must reproduce the oracle's `(accepted, match_len)` exactly, on
examples, on randomized fuzz automata up to 500 states, and on six real ANMLZoo automata up to
48k states and 6.3M transitions. `gpufsm.bench.oracle.require` raises before anything is timed,
so a fast wrong kernel cannot be reported as a fast kernel.

`scripts/oracle_gate.py` is the comprehensive form: it walks the registry, so it cannot drift
from what is registered, and it treats a missing expected backend as a failure rather than a
skip.

## Metrics

- **Kernel time** (ms) — device-side, the scientific quantity.
- **Transfer time** (ms) — host/device movement, measured and reported separately, never folded
  into kernel time. (One exception, documented: `multistream_async` returns an overlapped
  end-to-end time by construction. See `THREATS.md`.)
- **Throughput** (Gbit/s) — total input bits over the batch kernel time, on a fixed batch.

## Statistics

GPU timings are not Gaussian: a few slow samples from clock and thermal transients move a mean
and leave a median alone (Hoefler and Belli, SC'15). Everything the paper reports is therefore
a **median with a percentile-bootstrap 95% CI**, over 9 samples after 3 warmups
(`gpufsm.bench.timing`). The bootstrap is seeded, so a committed CSV reproduces its interval
and not merely its point estimate.

The headline regret ratios are additionally measured over several random-NFA seeds, reported as
a median with the min-max spread, so that a single unlucky automaton cannot carry the result.
Parallel-scaling comparisons use a GPU-saturating batch, so no baseline is starved of
parallelism and then declared slow.

Two deliberate departures, each stated where it occurs: `gpufsm.api.benchmark` (an interactive
convenience, not a paper path) reports mean and a normal-approximation CI, and
`scripts/calibrate_costmodel.py` uses best-of-ten because a fit wants the machine's capability
with interference removed.

## Bound diagnosis

Kernels are classified compute- versus memory-bound with the **instruction** roofline, which is
the appropriate one for these integer- and transaction-dominated FSM kernels; a FLOP roofline
would understate them. Two independent lines of evidence back the compute-bound finding for
the full-scan kernels: a controlled ablation (`multistream_shared` removes the modelled global
CSR traffic entirely and changes throughput by at most 1.3%) and the `1/n^2` scaling the
eps-closure predicts. Nsight Compute counters corroborate; no claim depends on them, which
matters because counters are admin-gated.

## The cost model

A two-parameter model, `time_per_symbol = a * traffic + b * num_states^2`, fitted **per
backend** (`gpufsm.costmodel`). The per-DSL fit is the point: a single global fit leaves ~80%
error while per-backend fits land near 15%, because the gap between DSLs is a per-DSL constant
that the shared traffic and `n^2` terms cannot absorb. The ratio of the fitted compute constants
is the regret expressed as a model parameter rather than as a measurement.

It is pressure-tested rather than merely fitted: `scripts/validate_costmodel.py` runs a holdout
(fit on small `n`, predict the largest unseen `n`) and leave-one-out refits. The model is
predictive for CUDA and not for Triton, and the paper says so.

## Environment

Hardware and library versions are captured per CSV row by `gpufsm.bench.csvio.environment`.
Reference host: NVIDIA RTX 4070 (sm_89, 46 SMs, 12 GB); software: CUDA toolkit 13.x, Triton
3.5.1, Warp 1.14, PyTorch 2.9 (cu128). Cross-architecture validation on an A100-80GB.
