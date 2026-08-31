# Kernel-design experiments — what we tried and what moved the needle

A living log of GPU worklist kernel variants and their measured outcomes (RTX 4070, sm_89).
Goal: characterize the work-efficient NFA worklist and find the path to SOTA-absolute
throughput. Skeptical-scientist rule: report negative results honestly; a tried-and-failed
optimization is knowledge. All numbers trace to `paper/data/*.csv` and are reproducible via
the cited scripts.

## Baseline regimes (established earlier)
- **Full-scan kernels are compute-bound (O(n²) eps-closure):** throughput ∝ 1/n²; memory-layout
  axes (`multistream_shared`, `_async`) are inert (`docs/RESULTS_COSTMODEL.md`). The fix is a
  work-efficient kernel.
- **Register worklist** (≤512 states, 1 thread/string): 15–170 Gbps — the fast small-automaton path.
- **`worklist_global`** (1 thread/string, global working set, no state cap): the scalable but
  under-parallelized large-automaton baseline.

## Active-set sparsity (premise measurement)
Real automata are extremely sparse-active (mean active states over a random run):
levenshtein **7** / 2787 (44 words), fermi **269** / 40786 (638 words), brill **1.5** / 42661
(667 words). This motivated testing whether the O(nwords) bitmap scan is wasteful.

## Experiments

### ✅ `worklist_warp` — block-parallel (one warp/string), 32 lanes partition the words
**Outcome: the win.** 3–9× over `worklist_global` on real automata at a GPU-saturating batch
(up to ~12× dense synthetic, up to ~180× at small batch). Source of the win is **occupancy /
parallelism** (Nsight: 17→57% occupancy, SM 1→16%), not memory. The speedup is **batch- and
density-dependent** (`paper/data/worklist_warp{,_batch}_rtx4070.csv`).

### ◐ `worklist_shared` — warp kernel, working set in shared memory (≤1536 states)
**Outcome: inert (0.99–1.10× vs `worklist_warp`).** Nsight explains it: the working set already
lives in L2 at 99.96% hit, so moving it to shared memory changes nothing. Memory-layout
privatization does not help once the kernel is work-efficient
(`paper/data/worklist_shared_rtx4070.csv`).

### ✗ `worklist_compact` — compacted active-ID frontier (O(active)), 1 thread/string
**Outcome: refuted hypothesis — barely beats the bitmap scan (0.8–1.5× vs `worklist_global`),
and far slower than `worklist_warp` (0.1–0.4×).** Measured: levenshtein 0.8× (worse), fermi
1.0–1.2×, brill 1.5× vs the 1-thread bitmap kernel. Why the O(active) advantage doesn't
materialize: skipping an empty 64-bit word in the bitmap scan is one load + branch (cheap), and
the bitmap's sequential/coalesced word access beats the compacted frontier's scattered state-ID
loads + per-transition `visited` test-and-set dedup. So word-scanning was never the bottleneck.

## Nsight characterization (`paper/data/nsight_rtx4070.csv`)
`worklist_warp` is **latency/instruction-bound, not memory-bound**: DRAM ≤2.25%, L2 hit ≥97.6%
even for brill's 17 MB CSR, because all strings share the CSR and only a hot row
subset is touched per batch (stays L2-resident).

## Claim verifications (skeptical re-checks, 2026-06-26)
- **Warp beats hand-CUDA (NFA) on the median, with a wide tail.** warp/cuda multistream =
  1.11–1.13× (regret 0.89–0.90×) on the single-seed sweep. The 5-seed run
  (`paper/data/regret_multiseed_rtx4070.csv`) gives a *median* regret of 0.85 with min 0.78
  and **max 3.34** at n=32. That maximum is not evidence about Warp: four of the fifteen
  (size, seed) points used an automaton accepting before any byte was read, where the
  measured "throughput" is launch overhead. The driver now skips those and the file needs
  re-running (`docs/THREATS.md`). Quote the median
  and the spread together; the earlier "spread ±1% across 3 seeds" here was refuted by the
  5-seed file that superseded it. (The cost-model fit gives 0.63×, but Warp has only two
  points so that fit is exact by construction — `docs/RESULTS_COSTMODEL.md` — and it is not
  evidence.)
- **DFA Triton-flatness = a per-program scalar ceiling**, not memory or parallelism. Triton DFA
  throughput ramps with batch (64→1.7, 256→6.6, 1024→18, 4096→24.7 Gbps) then **saturates at
  ~29 Gbps** by ~4k strings (16384→28.7), while CUDA keeps scaling (16384→428 Gbps). So Triton's
  flat ~29 Gbps regardless of table size is its tile/SPMD per-program scalar-gather ceiling
  (enough programs to fill the GPU, memory idle) — confirming the model-bound reading.
- **Cost model** predictive for CUDA (2.7% holdout) but not Triton (45%, unstable) — see
  `docs/RESULTS_COSTMODEL.md`; measured throughput ratio is the primary metric.
- **CAUSAL primitive ablation (within Triton).** Same language/data/harness/parallelism (one
  program/string), only the access pattern differs: tile-vectorized reduction vs a carried
  data-dependent scalar recurrence. At a saturating batch the tile kernel scales to **1320 Gbps**
  while the scalar pattern hits a hard **~81 Gbps** per-program ceiling — a **16× cliff** from the
  control pattern alone (`scripts/ablate_scalar_control.py`, `paper/data/scalar_ablation_rtx4070.csv`).
  This makes the capability→cost attribution *causal*: the regret appears iff the inexpressible
  scalar/control-flow primitive is required (the scalar ceiling mirrors the DFA Triton-flat result).

## Where this points (path to SOTA-absolute)
Neither memory layout (shared) nor work-reduction (compaction) is the lever; **parallelism**
(warp) is, and it's now in. The remaining gap to ngAP-class absolute throughput is **algorithmic
redundancy across strings/symbols**: memoization (reuse repeated state-set transitions),
non-blocking multi-symbol processing (ngAP), or cross-string CSR-access batching/coalescing.
That — not another data-layout tweak — is the next worthwhile kernel experiment.

## Reproduce
- warp/shared/compact correctness: `pytest tests/test_worklist_warp.py -m gpu`
- warp speedup + batch sensitivity: `python scripts/bench_worklist_warp.py`
- shared vs warp: `python scripts/bench_worklist_shared.py`
- compact vs global vs warp (the 3-way above): run `run_batch(nfa, batch, "cuda", t)` for
  `t in {worklist_global, worklist_warp, worklist_compact}` on the pinned ANMLZoo automata
  (`gpufsm.io.datasets.DATASETS`) at batch 256 and 4096; numbers above are medians of 5.
