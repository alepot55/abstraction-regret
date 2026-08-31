# Cost-Model Calibration & the Quantified Abstraction Regret

Generated: 2026-06-26. Hardware: RTX 4070 (sm_89). Data: `paper/data/costmodel_rtx4070.csv`
(regenerate with `python scripts/calibrate_costmodel.py`). Model: `gpufsm.costmodel`.

This is the first quantitative test of the "abstraction regret" thesis on measured GPU
throughput. Throughput = batched multi-stream (4096×256 B), best-of-10 batch kernel time.

## Model

```
time_per_symbol = traffic_bytes_per_symbol / eff_bandwidth   (memory term)
                + num_states**2 * compute_s_per_state2        (compute term)
```

The compute term is **quadratic** because the faithful constant-algorithm kernel does an
O(n) transition scan + an O(n²) epsilon-closure (n convergence passes × n states) per input
symbol. Confirmed empirically: throughput ∝ 1/n² (n=32→64→128→256 ⇒ ÷~4 each step); a linear
compute term mis-fits (~85% error), the n² term fits the compute-bound regime to <1% at
the largest size measured (n=256).

## Finding 1 — the kernels are COMPUTE-bound; memory layout is (here) irrelevant

`multistream_shared` stages the CSR into shared memory → **modeled traffic = 0** — yet its
throughput is identical to `multistream` (global CSR) at every size:

| n | multistream (global CSR) | multistream_shared (traffic 0) | multistream_async |
|---|---|---|---|
| 32 | 1.026 | 1.027 | 0.964 |
| 64 | 0.262 | 0.263 | 0.258 |
| 128 | 0.047 | 0.047 | 0.047 |
| 256 | 0.012 | 0.012 | 0.012 |

So for the dense full-scan algorithm the global-memory traffic is **not** the bottleneck:
the O(n²) compute is. This refines the thesis: **the memory-organization axes (shared CSR,
async transfer, even byte→bit) only bite once the algorithm is made work-efficient**
(sparse active-set / worklist, as in ngAP), moving the kernel into the memory-bound regime.
The cost model predicts exactly this — the memory term is dwarfed by the n² compute term, so
its fitted bandwidth coefficient is negligible (→ "inf GB/s ⇒ compute-bound").

## Finding 2 — the Triton↔CUDA gap is a per-DSL constant: the abstraction regret

A single global fit gives ~80% error because no traffic/n² term can absorb the constant
factor between DSLs. Fitting **per backend** (n² model) fits well and isolates the per-DSL
compute-efficiency constant. The ratio vs the CUDA baseline **is** the abstraction regret on
this kernel:

| Backend | compute (ns/state²) | **regret vs CUDA (fit)** | regret (measured throughput) |
|---|---|---|---|
| **Triton** (tile/SPMD) | 0.103 | **10.1×** | 6–8× |
| **CUDA** (C++ SIMT) | 0.0102 | 1.00× (baseline) | 1.00× |
| **Warp** (Python thread-SIMT) | 0.0065 | **0.63×** (beats hand CUDA) | 0.9× |

(Per-backend predicted-vs-measured throughput error is **<1% at n=256** — CUDA 0.3%, Triton
0.6% — and ~2% (CUDA) / ~13% (Triton) at n=128; the larger error at n=32/64 (20–60%) is the
fixed launch overhead the pure-n² model omits. Warp has only two points, so its fit is exact
by construction and not a quality signal.)

### Validation: predictive for CUDA, NOT for Triton (`scripts/validate_costmodel.py`)

The full-fit residual above is in-sample. Two honest out-of-sample tests:

| backend | holdout error (fit n≤128, predict n=256) | leave-one-out spread of b |
|---|---|---|
| **CUDA** | **2.7%** (predictive) | **1.03×** (stable) |
| **Triton** | **45%** (fails) | **2.46×** (UNSTABLE) |

So the 2-parameter model is **predictive and robust for the thread-model backend (CUDA)** but
**not for the tile/SPMD backend (Triton)**: Triton's large fixed launch overhead at small n is
misattributed to the n² term, so extrapolation and the fitted-b are unstable. This is itself
consistent with the regret thesis (the overhead is part of Triton's regret), but the
consequence is methodological: **the primary regret metric is the directly-measured throughput
ratio (Triton 6–8×, robust); the fitted-b ratio (10.1×) is corroborating and must not be
over-claimed for Triton.** A 3-parameter model (add a constant term) would fit Triton but
cannot be validated with only 4 points (3 params, 3 train points ⇒ exactly determined).

## Interpretation — regret is the execution *paradigm*, not abstraction height

Two **equally high-level Python DSLs** land on opposite ends: **Triton pays 10.1×** (fit; 6–8×
measured throughput) while **Warp beats hand-written CUDA (0.63×)** — same algorithm, same hardware. The difference is
the execution model: Triton's tile/SPMD paradigm is a poor fit for the data-dependent,
per-state, scalar control flow automata need (it can only express the kernel as one strained
program), whereas Warp's thread-SIMT model expresses it naturally and its codegen is
excellent. CUDA sits between (full control, but my hand kernel isn't maximally tuned).

**Thesis, sharpened:** the abstraction regret on irregular automata is set by *whether the
model's execution paradigm can express the workload's control flow + memory layout* — not by
how "high-level" the DSL looks. This complements the Gluon finding
(`docs/DSL_EXPRESSIVENESS.md`): Gluon (tile, layout-explicit) cannot express the kernel at
all, and Triton (tile) expresses it at 10.1× cost, while Warp (thread) is essentially free.

## Caveats / next steps

- These kernels are compute-bound, so they are the **wrong regime** to showcase the memory
  axes. The next engine must be work-efficient (active-set/worklist) so memory becomes the
  bottleneck; only then do byte→bit / shared-CSR / async show their value (and the cost
  model's memory term becomes load-bearing). This is the path to contribution (B) approaching
  ngAP/CUDA.
- Throughputs here are absolute-low (single-thread-per-string, full-scan); they are for
  *relative* DSL/technique comparison, not SOTA throughput claims.
- Validate on a 2nd GPU architecture before the camera-ready (generality).
