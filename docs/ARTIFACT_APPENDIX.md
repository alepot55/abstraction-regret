# Artifact Appendix

Artifact-evaluation appendix for *The Two Faces of Abstraction Regret: Control-Flow and
Memory-Layout Limits of GPU DSLs on Irregular Automata* (IEEE HPEC 2026). Companion to
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) (the full claim-to-command map),
[`METHODOLOGY.md`](METHODOLOGY.md) and [`THREATS.md`](THREATS.md).

## Abstract

The artifact is this repository: a registry-based framework with a CPU reference oracle, GPU
backends (CUDA, Triton, NVIDIA Warp), a Gluon expressibility probe, the versioned result CSVs
and the figure generator. It regenerates every figure from committed data with one command and
no GPU, validates every registered kernel bit-for-bit against the oracle, and runs the
falsifiable Triton/Gluon control. The GPU portion runs on any CUDA device; results in the paper
are from an RTX 4070 (sm_89), cross-validated on an A100-80GB.

## Check-list

- **Algorithm:** NFA simulation (active set + epsilon closure) and DFA dense-table walk; a
  work-efficient worklist and a warp-per-string variant.
- **Program:** Python 3.10+ package `gpufsm`; CUDA C++ kernels via pybind11; Triton and Warp kernels.
- **Compilation:** CMake >= 3.18 through scikit-build-core (CUDA, opt-in); Triton and Warp are wheels.
- **Run-time environment:** Linux; CUDA toolkit for the GPU backends. **No admin rights needed** —
  Nsight counters are optional and no claim depends on them.
- **Hardware:** any CUDA GPU for the GPU claims; CPU-only suffices for correctness, the cost
  model and every figure.
- **Metrics:** throughput (Gbit/s), median with percentile-bootstrap CI95; predicted-vs-measured fit error.
- **Output:** CSVs in `paper/data/`, figures in `paper/figures/`, pass/fail from `pytest` and
  `scripts/oracle_gate.py`.
- **Disk / time:** < 100 MB checked out; CPU suite ~2 s; the GPU sweeps a few minutes on one GPU.
- **Public:** yes, MIT. **Archived DOI:** to be minted at the first tagged release.

## Installation

```bash
# CPU-only: correctness, cost model, every figure. No GPU.
python -m pip install -e ".[dev,paper]"

# GPU backends, on a CUDA box:
pip install -e ".[dev,triton,warp]" --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON
gpufsm env      # records python / backend availability + versions; check the backends are there
```

## Workflow

```bash
pytest -m "not gpu" -q                   # the CPU suite, no GPU
python paper/figures.py                  # every figure, from the committed CSVs, no GPU

python scripts/oracle_gate.py --require cuda,triton,warp   # correctness, strictly
python scripts/sweep_techniques.py       # throughput sweep        -> sweep_techniques.csv
python scripts/calibrate_costmodel.py    # cost-model fit          -> costmodel_rtx4070.csv
python scripts/validate_costmodel.py     # holdout + leave-one-out (printed)
python scripts/sweep_dfa.py              # DFA table-size sweep    -> dfa_regret_rtx4070.csv
python scripts/bench_worklist_warp.py    # warp-per-string speedup -> worklist_warp*.csv
python scripts/bench_worklist_shared.py  # shared-memory ablation  -> worklist_shared_*.csv
python scripts/ablate_scalar_control.py  # the causal tile-vs-scalar cliff
python scripts/regret_multiseed.py       # multi-seed robustness
python scripts/run_anmlzoo.py            # six real automata, oracle-gated
python scripts/second_gpu.py --profile rich   # cross-architecture re-run
python scripts/gluon_probe.py            # 0 confirmed, 1 falsified, 2 inconclusive, 3 skipped
```

## Expected results

| Experiment | Expected |
|---|---|
| `pytest -m "not gpu"` | all pass; the 24 gpu-marked tests are deselected |
| `oracle_gate.py` | every available triple `ok`; a required-but-absent backend fails the run |
| `figures.py` | six figures rewritten; `git diff` shows only re-render noise |
| technique sweep | Triton 6-8x slower than CUDA at equal algorithm; Warp near parity |
| DFA sweep | CUDA and Warp rise then fall with table size; Triton flat at 29-32 Gbps |
| cost model | per-backend fit ~15% relative error; predictive for CUDA, not for Triton |
| `gluon_probe.py` | exit 0 with "confirmed - no scalar element load" |

## Evaluating this artifact honestly

[`THREATS.md`](THREATS.md) lists what an evaluator should attack, including two open confounds
found by auditing the code after the paper was written: the Triton kernels do not exit the input
loop early where CUDA and Warp do (immaterial on the NFA face, live on the DFA face), and the
RTX 4070's L2 capacity was asserted rather than measured. Both are reproducible from this
repository, and the second is one command away on any 4070.
