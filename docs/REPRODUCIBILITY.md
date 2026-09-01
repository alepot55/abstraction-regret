# Reproducibility

Every figure and headline number regenerates from committed code plus versioned CSVs. This
file maps each published claim to the exact command that produces it, and states plainly what
the commands do *not* guarantee.

## Environment

CPU-only (reference oracle, bit-packed spec, cost model, figures — no GPU):

```bash
python -m pip install -e ".[dev,paper]"
gpufsm env            # python / numpy / backend availability + versions
```

GPU backends (Triton / Warp are wheels; CUDA is built through CMake):

```bash
pip install -e ".[dev,triton,warp]" --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON
```

Gotchas learned on the reference host (RTX 4070, CUDA 13.x):

- **`GPUFSM_BUILD_CUDA=ON` as an environment variable is not enough.** scikit-build-core reads
  the define from `pyproject.toml`; pass it as
  `--config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON`.
- **A toolkit newer than the driver's maximum CUDA rejects embedded PTX at load**
  (`PTX ... unsupported toolchain`). `CMakeLists.txt` therefore defaults to real SASS only
  (`75/80/86/89-real`). Override `-DCMAKE_CUDA_ARCHITECTURES` for other GPUs, and avoid both
  bare architecture numbers and `native` — they embed PTX and bring the error back.
- On PEP 668 ("externally managed") hosts, use a virtualenv.

Pin the GPU, driver and toolkit versions in anything you report. `gpufsm.bench.csvio.environment`
captures them, and the sweep CSV records them per row.

## Correctness

```bash
pytest -m "not gpu" -q    # the CPU suite: oracle semantics, bit-packed spec, generators,
                          # ANML loader, cost model, CLI, API, committed-data contracts
pytest -m gpu -q          # GPU backends against the oracle (needs a GPU) — see the warning
python scripts/oracle_gate.py --require cuda,triton,warp
                          # every registered triple; a missing backend is a failure, not a skip
gpufsm verify             # quick: every registered NFA technique, on the examples only
```

`gpufsm.reference` is the single oracle (latch-first-match). The bit-packed CPU simulator
(`gpufsm.core.bitmap`) and every GPU technique are checked bit-identical to it.

> **A green `pytest -m gpu` can mean nothing ran.** Every gpu-marked test is wrapped in a
> `skipif` on backend availability, and a skip counts as a pass, so on a box where the CUDA
> extension failed to build the suite exits 0 having verified nothing. It also does not reach
> every registered `(Kind, Backend, technique)` triple, though it now covers the Warp backend
> and every backend that registers a DFA technique. `scripts/oracle_gate.py` walks the registry
> instead, treats an expected-but-absent backend as a failure, and refuses to exit 0 when it
> checked zero triples. Run `gpufsm env` first either way.

## Claims to commands

Every driver that reports a throughput for an automaton gates on the CPU oracle first, and
refuses to overwrite a committed CSV measured on a different GPU
(`gpufsm.bench.csvio.guard_device`). All of them write through the schema-checked
`write_rows`, so a renamed column is an error rather than a silently dropped measurement.

Every driver takes `--out`, which is how you re-measure on a different GPU: the committed
files carry the reference device in their name, so a run elsewhere has to go elsewhere.

```bash
python scripts/sweep_dfa.py --out paper/data/cross_arch/dfa_regret_a100.csv
```

Four scripts have no oracle gate, and none of them reports an automaton throughput:
`gluon_probe.py` is a compile probe, `profile_target.py` issues one launch for `ncu`,
`validate_costmodel.py` only re-reads a committed CSV, and `ablate_scalar_control.py`
compares two raw Triton kernels that are not automata at all.

| Claim | Command | Artifact |
|---|---|---|
| Throughput sweep (median + CI95) over techniques and sizes | `python scripts/sweep_techniques.py` | `paper/data/sweep_techniques.csv` |
| Cost-model calibration and the regret ratios | `python scripts/calibrate_costmodel.py` | `paper/data/costmodel_rtx4070.csv`, `docs/RESULTS_COSTMODEL.md` |
| The cost model is predictive, not overfit | `python scripts/validate_costmodel.py` | holdout + leave-one-out, printed |
| Every figure | `python paper/figures.py` | `paper/figures/fig_*.{pdf,png}` |
| Abstraction regret: Triton 6-8x, Warp 0.85x median (0.78-3.34 over 5 seeds) vs CUDA | sweep + multiseed | `fig_abstraction_regret` |
| The regret is not a single-seed artifact | `python scripts/regret_multiseed.py` | `paper/data/regret_multiseed_rtx4070.csv` — **driver since fixed, file needs re-running** |
| DFA memory-bound L2 knee (CUDA peaks then drops; Triton flat) | `python scripts/sweep_dfa.py` | `paper/data/dfa_regret_rtx4070.csv`, `fig_dfa_memory_bound` |
| Causal: the scalar-control cliff inside Triton | `python scripts/ablate_scalar_control.py` | `paper/data/scalar_ablation_rtx4070.csv` |
| Block-parallel warp worklist vs one thread per string | `python scripts/bench_worklist_warp.py` | `paper/data/worklist_warp{,_batch}_rtx4070.csv` |
| Shared-memory working set is inert (0.99-1.10x) | `python scripts/bench_worklist_shared.py` | `paper/data/worklist_shared_rtx4070.csv` |
| Correct and measured on six real ANMLZoo families | `python scripts/run_anmlzoo.py` | `paper/data/real_automata_throughput_rtx4070.csv` |
| Cross-architecture: the 2x2 and the knee hold on an A100 | `python scripts/second_gpu.py --profile rich` | `paper/data/{regret_a100,dfa_knee*}.csv` |
| Gluon cannot express the kernel (falsifiable) | `python scripts/gluon_probe.py` | exit 0 confirmed, 1 falsified, 2 inconclusive, 3 skipped |
| Nsight counters corroborate the bound diagnosis | `ncu` over `scripts/profile_target.py`, see `docs/PROFILING.md` | `paper/data/nsight_rtx4070.csv` |

Figures depend only on committed CSVs, so the paper rebuilds deterministically. The sweep and
calibration scripts skip unsupported `(backend, technique, size)` cells — Triton and Warp
worklists above 64 states, for instance — with a log line rather than failing.

Read [`THREATS.md`](THREATS.md) alongside this table: two of the claims above have open
confounds that the commands here will reproduce rather than resolve. One of them has an
experiment waiting for a GPU:

```bash
python scripts/dfa_latch_control.py   # does the DFA regret survive a non-latching input?
```

It measures the three backends at `accept_prob=0.02` (the paper's configuration, where every
string latches after ~50 of its 1024 bytes and only the thread-SIMT arms exit early) and at
`accept_prob=0.0` (where nothing latches, so all three walk the same input). If the tile/SPMD
regret survives the second regime it is real; if it collapses, the DFA half was measuring the
early exit. It has not been run.

## What re-running does and does not reproduce

- **Shape reproduces; absolute throughput does not.** Numbers are machine-specific. The claims
  are ratios and orderings, and that is what a re-run should be checked against.
- **Four committed CSVs were condensed by hand** from a driver's output. Which four, and what
  the hand step did, is in [`../paper/data/README.md`](../paper/data/README.md).
- **The A100 driver writes into `paper/data/cross_arch/` in its own schema.** It does not
  reproduce the four hand-condensed A100 files in place, by design.

> **Warp init / CUDA error 716.** On some Warp 1.14 + CUDA 12.9 combinations, Warp's
> initialisation intermittently throws `misaligned address`. The fault is *sticky*: it poisons
> the process CUDA context, so every later measurement in that process is meaningless.
> `sweep_techniques.py` detects it, aborts with a rerun hint and a non-zero exit code rather
> than emit a half-empty CSV. Re-run the sweep.

## Data

- Real ANMLZoo automata are fetched on demand by `gpufsm.io.datasets.ensure`, which **verifies
  SHA-256** and refuses an unverified download. Six pinned pure-STE families from the public
  `jackwadden/ANMLZoo` mirror: `levenshtein` (2787 states), `hamming` (11349), `brill` (42661),
  `fermi` (40786), `randomforest` (33223, 6.27M transitions), `corerings` (48005). They are
  cached under `data/anmlzoo/` and are **not** committed to this repository.
- `gpufsm.io.anml.load_anml` parses them with correct all-input / start-of-data semantics;
  the parser's fixtures are inline in `tests/test_anml.py`.
- `tests/test_anmlzoo_gpu.py` (gpu-marked, skips offline) fetches all six, runs
  `worklist_global` and checks GPU against the reference bit-for-bit.

## Profiling

GPU performance counters are admin-gated by the driver. `docs/PROFILING.md` has the one-time
enable and the `ncu` recipe. **No claim depends on counters**: the compute-bound result is
established by controlled ablation (`multistream_shared` removes the modelled CSR traffic
entirely and moves throughput by at most 1.3%) and by 1/n^2 scaling. The counters corroborate
it. See `docs/THREATS.md` before quoting any of these numbers.
