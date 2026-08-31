# abstraction-regret

**Where does a GPU DSL's abstraction actually cost you? Measured on automata, across four DSLs,
at a fixed algorithm.**

This is the research artifact for *The Two Faces of Abstraction Regret: Control-Flow and
Memory-Layout Limits of GPU DSLs on Irregular Automata* (IEEE HPEC 2026). It contains the
framework, the CPU reference oracle every GPU kernel is validated against, the measurement
drivers, the versioned result CSVs, and the figure generator.

The Python package is called `gpufsm`: it simulates finite automata on the GPU under **CUDA**,
**Triton**, **NVIDIA Warp** and **Gluon**, holding the algorithm fixed, and measures what each
abstraction precludes.

## The finding

Arrange the four DSLs on a 2x2 of *abstraction height* against *execution paradigm*, and the
cost tracks the **column**, not the row.

| | thread / SIMT | tile / SPMD |
|---|---|---|
| **low-level** | CUDA — 1x (baseline) | Gluon — cannot express the kernel |
| **high-level** | NVIDIA Warp — 0.8-0.9x | Triton — 6-8x |

Warp's range is measured throughput, not a model fit: 0.90x on the single-seed sweep and 0.85x
median over five random-NFA seeds. The spread runs 0.78 to 3.34, so on one seed Warp is 3.3x
*slower* than CUDA. Quote the median with its spread, not the best point.

Two workloads pin down two faces of the same cause. NFA simulation is control-flow-bound; DFA
simulation is memory-bound. Triton pays on **both**, which is what rules out "it is the
workload" and leaves "it is the execution model". The named missing primitive is scalar,
data-dependent per-element work inside a tile.

Gluon is the control that makes this falsifiable rather than a story: it shares Triton's
compiler stack and only *adds* layout and shared-memory control, so if the kernel were
expressible-but-untuned, Gluon would express it. `scripts/gluon_probe.py` is that experiment,
and it exits 1 the day Gluon compiles the kernel (and 3, not 0, when it could not be run
at all -- a falsifiable claim needs its negative and its "not tested" to differ).

## Install

```bash
pip install -e ".[dev]"                                  # core (CPU) + dev tools, no GPU needed
pip install -e ".[dev,triton,warp]"                      # + the Triton and Warp backends (needs a GPU)
pip install -e "." --config-settings=cmake.define.GPUFSM_BUILD_CUDA=ON   # + the CUDA extension
```

The build is **graceful**: with no CUDA toolkit or GPU the extension is skipped and the package
still installs and runs on CPU. A backend that cannot load reports as unavailable rather than
disappearing, so `gpufsm env` always tells you what you actually have.

## Quickstart

```python
from gpufsm import NFABuilder, Backend, run, run_batch, benchmark, random_dfa

b = NFABuilder()
s0 = b.add_state(); s1 = b.add_state(accept=True)
b.set_start(s0); b.add_transition(s0, "a", s1)
nfa = b.build()

run(nfa, b"a", backend=Backend.CPU)          # Result(accepted=True, match_len=1, ...)
benchmark(nfa, b"a" * 4096, repeats=10)      # BenchmarkStats

dfa = random_dfa(4096, seed=0)               # the memory-bound face, same API
run_batch(dfa, [b"abc", b"xyz"], backend=Backend.CUDA)
```

```bash
gpufsm env        # environment + available backends, per automaton kind
gpufsm list       # every registered (kind, backend, technique)
gpufsm verify     # check the default technique of every backend against the CPU reference
gpufsm bench --backend cpu --size 4096 --repeats 10
```

## Design

Four invariants carry the study. Each is one place in the code, on purpose.

- **One API** — `gpufsm.api`: `run`, `run_batch`, `benchmark`. NFAs and DFAs both go through it;
  the automaton's type selects the kind, so there is no second entry point to keep in sync.
- **One extension point** — `gpufsm.core.registry`: a backend or a technique is one module plus
  one `@register(Kind, Backend, "name")` line. Nothing dispatches on a hand-parsed string.
- **One oracle** — `gpufsm.reference`: a CPU simulator with latch-first-match semantics. Every
  backend reproduces its `(accepted, match_len)` bit-for-bit, and every driver that reports an
  automaton throughput calls `gpufsm.bench.oracle.require` before timing anything.
- **One harness** — `gpufsm.bench`: the random automata, the timing statistics, the CSV schema.
  The drivers under `scripts/` are thin, so two measurements cannot disagree about what a
  "median throughput" is.

The fairness protocol that makes the number an *abstraction* cost rather than an algorithmic
one: every backend consumes the same CSR automaton and implements the same algorithm, with
kernels structurally mirrored from one specification. Across a comparison only the DSL varies;
within a DSL the *technique* varies and the best is reported. Regret is defined at a fixed
algorithm, so letting each DSL pick its own would measure algorithmic redesign instead.

## Layout

| Path | What |
|---|---|
| `src/gpufsm/core/` | automata, CSR representation, bit-packing, the registry, result types |
| `src/gpufsm/reference.py` | the oracle: `simulate` (NFA) and `simulate_dfa` (DFA) |
| `src/gpufsm/api.py` | the public entry points |
| `src/gpufsm/backends/` | one module per technique: `cpu`, `cuda/` (+ `native/*.cu`), `triton/`, `warp/` |
| `src/gpufsm/bench/` | the shared harness: generators, timing, oracle gate, CSV schema |
| `src/gpufsm/io/` | the ANML parser and the checksum-pinned dataset fetcher |
| `tests/` | CPU suite plus GPU-marked tests; `test_golden.py` pins the oracle's verdicts |
| `scripts/` | the measurement drivers, one per published claim |
| `paper/data/` | the committed result CSVs, with their provenance in `paper/data/README.md` |
| `paper/figures.py` | regenerates every figure from those CSVs, with no GPU |
| `docs/` | [architecture](docs/ARCHITECTURE.md), [threats](docs/THREATS.md), methodology, the claim-to-command map, the artifact appendix |

## Reproducing

Figures and the correctness suite need **no GPU**:

```bash
pytest -m "not gpu" -q     # the whole CPU suite: oracle semantics, the bit-packed spec,
                           # the generators, the ANML loader, input validation, and the
                           # contracts on the committed CSVs
python paper/figures.py    # every figure in the paper, from the committed CSVs
```

Reproducing the *measurements* needs a CUDA device. The claim-to-command map is
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and the artifact check-list is
[`docs/ARTIFACT_APPENDIX.md`](docs/ARTIFACT_APPENDIX.md). The one command that matters most:

```bash
python scripts/oracle_gate.py --require cuda,triton,warp
```

It walks the registry and checks every registered technique against the CPU reference. It is
strict on purpose: a `pytest -m gpu` run degrades to a silent pass when a backend failed to
build, because a skip counts as a pass. `oracle_gate.py` treats an expected-but-absent backend
as a failure and refuses to exit 0 having checked nothing.

## What this artifact does not claim

Stated here rather than left for a reader to discover:

- **Absolute throughput is not state of the art.** On real ANMLZoo automata the engine is
  sub-Gbps to a few Gbps. The study measures a *ratio* between DSLs at a fixed algorithm; the
  algorithm itself is deliberately simple so that it can be mirrored across four languages.
- **The regret is measured, not derived.** The hand-written CUDA baseline calibrates the number.
  What is answerable without it is the *direction*: whether an IR can express the needed
  primitive is decidable from the IR alone.
- **Some committed CSVs were condensed by hand** from a driver's output into a plotting schema.
  Which ones, and what the hand step did, is recorded in [`paper/data/README.md`](paper/data/README.md).
- **The Triton kernels run the full input where CUDA and Warp exit at the first match.**
  Immaterial on the NFA face, and a live confound on the DFA face. Written up in
  [`docs/THREATS.md`](docs/THREATS.md), which is the first thing to read before trusting a
  number here.
- **The GPU test suite does not cover every registered technique.** `scripts/oracle_gate.py`
  does; `pytest -m gpu` does not, and the gap is listed in `docs/REPRODUCIBILITY.md`.

## Citing

If you use this work, cite the HPEC 2026 paper; [`CITATION.cff`](CITATION.cff) carries both the
paper and the software record. The manuscript itself is not redistributed here — see the IEEE
Xplore record.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md). The short version: `ruff`, `mypy` and the CPU suite
must be green before a commit, and a new backend or technique is one module plus one `@register`
line.

## License

MIT — see [`LICENSE`](LICENSE). The ANMLZoo automata are fetched on demand from the public
`jackwadden/ANMLZoo` mirror and are not redistributed here; each download is SHA-256 verified.
