# Architecture

A tour of the code, ordered the way a value flows through it, with the reasoning behind the
decisions that are not obvious from the source.

## The shape of the thing

```
    NFABuilder / random_nfa                  bytes
             |                                 |
             v                                 v
        NFA (CSR, frozen)  ------>  api.run / api.run_batch
             |                                 |
             |                       registry.get_factory(kind, backend, technique)
             |                                 |
             |                                 v
             |                            Executor.run  ->  Result(accepted, match_len, timings)
             |                                 |
             +--> reference.simulate ----------+
                        (the oracle every Result is compared against)
```

Four modules carry the whole design, and each exists exactly once on purpose. The recurring
failure mode in a study like this is *drift*: two implementations of "the same" thing that
quietly diverge, after which the comparison measures the divergence instead of the DSL.

## The automaton: one CSR layout for everybody

`gpufsm.core.nfa.NFA` is a frozen dataclass holding six numpy arrays: `accept`, and two CSR
pairs — `sym_row_ptr / sym_targets / sym_symbols` for symbol transitions and
`eps_row_ptr / eps_targets` for epsilon transitions. Symbols are bytes `0..255` plus the
sentinel `ANY_SYMBOL = 256` for a wildcard.

Every backend consumes **this** representation. That is the fairness protocol made
structural: if Triton were handed a different layout than CUDA, the measured gap would
include a data-structure choice, and the study claims to measure only what the abstraction
forecloses at a fixed algorithm.

DFAs (`gpufsm.core.dfa.DFA`) are the other face: a dense `num_states x 256` int32 transition
table, one gather per input byte, `1 KB` of table per state. The size of that table against
the GPU's L2 is the whole memory-bound experiment.

## The oracle: `gpufsm.reference`

Two functions, `simulate` (NFA) and `simulate_dfa` (DFA), both implementing *latch-first-match*:
report as soon as any accepting state is active, returning the length of the matched prefix,
with `0` meaning the start state itself accepts. They are written for clarity, not speed; they
are the definition of correct, and a fast wrong kernel is the failure this study cannot afford.

`Result.matches` compares only `(accepted, match_len)` — never timings — so correctness and
performance never get conflated in a single equality.

## The registry: one extension point

`gpufsm.core.registry` maps `(Kind, Backend, technique) -> factory`. Three consequences:

- `api.run` derives `Kind` from the automaton's type, so NFA and DFA share one entry point and
  there is no second dispatch path to keep in sync.
- Availability is a *probe registered outside the import guard*. A backend whose import failed
  still reports as unavailable rather than vanishing, which is why `gpufsm env` can tell you
  the CUDA extension did not build instead of silently listing three backends where there
  should be four.
- `scripts/oracle_gate.py` enumerates the registry rather than a hand-written list, so it
  cannot drift from what is actually registered.

## Batching, and why one `Result` carries the kernel time

A batched launch has a single kernel time for the whole batch. `core.result.batch_results`
puts it on the **first** `Result` and leaves the rest at zero, so `sum(r.kernel_ms)` is the
launch time for the batch rather than a meaningless multiple of it. Every batched executor —
Triton, CUDA, Warp; NFA and DFA — builds its results through that one function, so the
convention cannot differ between them.

`bench.timing.repeat` drops non-positive readings for the same reason: averaging the zeros
in would divide the measured time by the batch size and report a fictional speedup.

## The kernels

| Family | Where | Idea |
|---|---|---|
| full scan | `dense.cu`, `bitpacked.cu`, `triton/dense.py`, `triton/bitpacked.py` | one slot (byte, then bit) per state, scanned every input symbol |
| multi-stream | `*multistream*` | one program/block per string; the standard batching axis, plus the shared-CSR and async-transfer ablations |
| worklist | `worklist_register.cu`, `worklist_global.cu`, `triton/worklist.py` | iterate only the set bits of the active set, via `ffs`, inside a data-dependent loop |
| DFA gather | `dfa.cu`, `triton/dfa.py`, `warp/dfa.py` | one dense-table lookup per input byte |

The worklist family is where the thesis lives. Triton *can* express it — `libdevice.ffs`
inside a `while`, with a frontier-based epsilon closure — which is what separates
expressiveness from efficiency: Triton expresses the right algorithm and still pays a large
constant, so the cost is the execution paradigm rather than a missing algorithm. Gluon, on
the same compiler stack with *more* layout control, cannot express it at all
(`scripts/gluon_probe.py`).

### CUDA translation units

`backends/cuda/native/` is one `.cu` per kernel family, compiled **without** relocatable
device code. A `__device__` helper is therefore private to its translation unit; anything two
families need lives in `native/include/`. Moving a helper into a `.cu` that another unit calls
is a link error, not a warning, which is why CI compiles every translation unit with `nvcc`
even though it has no GPU to run them on.

Device memory is owned by `DeviceScope`, an RAII holder. The entry points previously collected
pointers in a vector and freed them at the end of the happy path; since `CUDA_CHECK` throws,
any failure after the first allocation skipped the drain and leaked everything. A destructor
cannot be skipped by a throw.

### Things that look removable and are not

- `int(0)` in the Warp kernels declares a mutable `wp.int32` local. A bare literal makes Warp
  miscompile later conditional reassignment. `ruff`'s `UP018` is disabled for that directory,
  and the glob must track any rename — while it pointed at pre-refactor filenames, `--fix`
  silently rewrote the casts and produced wrong results that no CPU-only machine could observe.
- `CMAKE_CUDA_ARCHITECTURES` uses the `-real` suffix. Bare numbers and `native` embed PTX, and
  a toolkit newer than the driver's maximum CUDA rejects embedded PTX at module load even when
  matching SASS is present.
- The RNG draw order in `bench/generators.py` is pinned by `tests/test_generators.py`. Change
  it and the committed CSVs stop describing the automata the code builds.

## The measurement harness

`gpufsm.bench` holds everything a measurement needs that is not a kernel: the canonical random
automata (`generators`), median plus percentile-bootstrap CI95 (`timing`), the correctness gate
(`oracle`), and the schema-checked CSV writer with the environment capture and the
wrong-device guard (`csvio`). The drivers under `scripts/` are thin on purpose — there were
once eleven copies of `random_nfa`, and they had drifted.

Two invariants the harness enforces rather than documents:

- `oracle.require` raises before anything is timed, so a throughput cannot be printed for a
  kernel that disagrees with the reference.
- `csvio.guard_device` refuses to overwrite a committed CSV whose recorded GPU differs from
  the live one. The committed files carry the device in their filename; re-running elsewhere
  used to rewrite them in place, leaving a file that claimed a device it was never measured on.

## Statistics, and the one place two of them coexist

The paper's numbers are **median plus percentile-bootstrap CI95**, because GPU timings are not
Gaussian: a few slow samples from clock and thermal transients move a mean and leave a median
alone. That is what `bench.timing` implements and what the sweep CSV records.

Two deliberate exceptions, both stated where they occur rather than hidden:

- `api.BenchmarkStats` (the convenience API, not a paper path) reports mean, standard
  deviation and a normal-approximation CI. It exists for interactive use.
- `scripts/calibrate_costmodel.py` uses **best-of-ten** per point. A fit wants the machine's
  capability with interference removed; the committed CSV records that choice in its `note`
  column, and the published fit constants come from that statistic.
