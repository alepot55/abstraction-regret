# `paper/data/` — provenance of every committed number

Figures are regenerated from these CSVs and from nothing else, so a CSV whose origin is
not written down is a number nobody can check. This file is that record: for each file,
what wrote it, on which device, and which claim rests on it.

Two conventions hold across all of them:

- **Throughput is Gbit/s of input consumed by the batch kernel**, median over repeated
  launches after warmup. Host/device transfer is measured separately and never folded in.
- **Nothing here was written before the CPU oracle agreed with the kernel.** Every driver
  that produced a file in this directory calls `gpufsm.bench.oracle.require` and aborts on a
  mismatch rather than reporting a throughput (`src/gpufsm/bench/oracle.py`). The two
  exceptions in the table below measure no automaton: `scalar_ablation` compares two raw
  Triton kernels, and `nsight` is a profiler transcript.

## Measured directly by a script in this repository

| File | Written by | Device | Backs |
|---|---|---|---|
| `sweep_techniques.csv` | `scripts/sweep_techniques.py` | RTX 4070 | the throughput table; `fig_throughput_vs_states`, `fig_worklist_speedup`, `fig_memory_ablation`, `fig_abstraction_regret` |
| `costmodel_rtx4070.csv` | `scripts/calibrate_costmodel.py` | RTX 4070 | the two-parameter cost model; `fig_costmodel_fit`. `scripts/validate_costmodel.py` re-reads it for the holdout test |
| `dfa_regret_rtx4070.csv` | `scripts/sweep_dfa.py` | RTX 4070 | the DFA table-size curve; `fig_dfa_memory_bound`. Its `note` column was recomputed against the card's real 36 MB L2 — a derived label, no measured value touched |
| `scalar_ablation_rtx4070.csv` | `scripts/ablate_scalar_control.py` | RTX 4070 | the causal tile-vs-scalar cliff inside Triton. **Upper bound** — see below |
| `regret_multiseed_rtx4070.csv` | `scripts/regret_multiseed.py` | RTX 4070 | that the headline regret is not a single-seed artifact. **Predates a driver fix** — see below |
| `worklist_warp_rtx4070.csv` | `scripts/bench_worklist_warp.py` | RTX 4070 | warp-per-string vs thread-per-string on real automata |
| `worklist_warp_batch_rtx4070.csv` | `scripts/bench_worklist_warp.py` | RTX 4070 | that the same speedup is batch-dependent, which is why the headline uses a saturating batch |
| `worklist_shared_rtx4070.csv` | `scripts/bench_worklist_shared.py` | RTX 4070 | the negative result: moving the working set to shared memory is inert (0.99-1.10x) |
| `real_automata_throughput_rtx4070.csv` | `scripts/run_anmlzoo.py` | RTX 4070 | correctness and throughput on the six pinned ANMLZoo families |

## Assembled by hand from a script's output

These are honest exceptions, recorded as such rather than presented as direct output.

| File | Source | What the hand step did |
|---|---|---|
| `regret_a100.csv` | `scripts/second_gpu.py --profile rich` | pivoted the script's per-seed, per-backend rows into one regret ratio per size, for plotting |
| `dfa_knee_a100.csv` | `scripts/second_gpu.py --profile quick` | kept the CUDA rows and added the `l2_mb` constant for the device |
| `dfa_knee_rich_a100.csv` | `scripts/second_gpu.py --profile rich` | a separate, later run of the same profile with all three backends |
| `nsight_rtx4070.csv` | `ncu` over `scripts/profile_target.py` | transcribed the four Nsight Compute counters per kernel from the profiler report |

Re-running `scripts/second_gpu.py` writes `paper/data/cross_arch/second_gpu_{nfa,dfa}_<gpu>.csv`
in the script's own schema. It does not overwrite the four files above, and it is not
expected to reproduce them byte-for-byte: throughputs are not comparable across machines,
and the cross-architecture claim is about the *shape* (Triton pays the regret and Warp does
not; CUDA has a DFA knee on both cards and Triton has none on either), not the absolute
numbers.

### Two files whose driver changed after they were measured

- `regret_multiseed_rtx4070.csv` was measured with an input batch of 2048 copies of one
  periodic string, which makes branch divergence zero — in the measurement whose purpose is
  to show the headline is robust — and four of its fifteen points used an automaton that
  accepts before any byte is read, where the number is launch overhead rather than
  throughput. Both are fixed in the driver, so re-running will not reproduce these values.
- `scalar_ablation_rtx4070.csv` compares two kernels that differ in arithmetic as well as in
  control pattern, so its 16x cliff is an upper bound on the scalar-control cost.

Both are written up in [`../../docs/THREATS.md`](../../docs/THREATS.md).

### Known gaps, stated rather than hidden

- `regret_a100.csv`, `dfa_knee_rich_a100.csv` and `nsight_rtx4070.csv` carry **no `gpu`
  column**; their device is recorded only in the filename and in this table. Every CSV
  written by a driver today records the device inline via `gpufsm.bench.csvio.environment`.
- `dfa_knee_a100.csv` and `dfa_knee_rich_a100.csv` are **two different runs**, not two views
  of one. The CUDA column in the `rich` file is non-monotonic in table size; the knee claim
  in the paper reads the `quick` run, and the `rich` run is what supports the Triton-flat
  half of the comparison.
- The device name is spelled `RTX4070` in most files and `NVIDIA GeForce RTX 4070` in
  `sweep_techniques.csv`, because the latter records `torch.cuda.get_device_name` verbatim.

## Re-measuring

No driver can silently overwrite a result measured on another machine: each one calls
`gpufsm.bench.csvio.guard_device` on its output path first and refuses when the file's
recorded GPU is not the live one. `run_anmlzoo.py` and `second_gpu.py` go further and derive
the filename from the device. Reproducing the *figures* needs no GPU at all:

```bash
python paper/figures.py
```
