# Threats to validity, as they stand in the code

The paper has a threats section. This file is the version that names files and line numbers,
including things found by auditing this repository after the paper was written. Each item says
whether the paper reflects it and whether it is resolved.

A reader who wants to attack these results should start here.

## 1. The Triton kernels do not exit the input loop early; CUDA and Warp do — **declared in the paper, not yet resolved**

The fairness protocol is "same algorithm, kernels structurally mirrored, only the DSL varies".
The loop-exit behaviour is not mirrored:

| | code | behaviour |
|---|---|---|
| CUDA | `native/dfa.cu:25`, `bitpacked.cu:44`, `worklist_*.cu` (`&& !done`) | `break`s at the first accepting state |
| Warp | `warp/dfa.py:44`, `warp/nfa.py:69` | `while pos < hi and done == 0` |
| Triton | `triton/dfa.py:37` and the other four kernels | `for pos in range(lo, hi)` with the body under `if done == 0` |

The verdicts are identical — the guarded body is skipped — so the oracle gate and the GPU
tests stay green. What differs is the number of loop iterations executed. The comment at
`triton/dense.py:57` explains this as "Triton forbids `return` inside loops", which is true
but not the whole constraint: the trip count is the issue, and `while pos < n and done == 0`
is expressible in Triton (`triton/worklist.py` already uses a data-dependent `while`).

**Where it bites.** On the DFA benchmark (`random_dfa(n, accept_prob=0.02)`, 1024-byte random
strings) every string latches, and it latches early. Measured with the CPU oracle over 200
strings:

| states | accept rate | mean bytes examined | of |
|---|---|---|---|
| 1024 | 1.000 | 50.1 | 1024 |
| 6144 | 1.000 | 49.4 | 1024 |
| 16384 | 1.000 | 49.7 | 1024 |
| 100000 | 1.000 | 54.5 | 1024 |

So CUDA and Warp run about 50 iterations per string and Triton runs 1024, while the reported
throughput credits all three with the full 1024 bytes. Two consequences:

- The DFA regret ratio between CUDA and Triton is inflated by an amount this measurement
  cannot separate from the abstraction cost.
- Triton's flat 29-32 Gbps across a 100x range of table sizes has two candidate explanations —
  "scalar-gather-bound, never reaching the memory regime" (the paper's reading) and "dominated
  by ~974 predicated-off iterations per string" — and the current experiment does not
  distinguish them.

**What would settle it.** `scripts/dfa_latch_control.py` is that experiment: it measures the
same three backends twice, at `accept_prob=0.02` (the paper's configuration, where everything
latches) and at `accept_prob=0.0` (where nothing does, so all three walk the same 1024 bytes).
If the tile/SPMD regret survives the second regime it is real; if it collapses, the DFA half was
measuring the early exit. **It needs a GPU and has not been run yet.** Alternatively, mirror the
loop shape across all three backends -- `while pos < n and done == 0` is expressible in Triton,
and `triton/worklist.py` already uses a data-dependent `while`.

**Where it does not bite.** On the NFA sweep (`random_nfa(n, seed=1000+n)`, 256-byte strings)
the oracle gives accept rates of 0.01-0.21 and mean symbols consumed of 202-254 out of 256, an
asymmetry of 1.0-1.3x. The headline NFA regret — the 2x2 that the paper's central claim rests
on — is not materially affected.

## 2. The RTX 4070's L2 capacity was asserted, and the assertion was wrong — **fixed in the paper**

`sweep_dfa.py` used to hardcode `# 6 MB L2 on the RTX 4070` and derive the `fits L2` /
`exceeds L2` annotation from it. Nothing in the repository ever queried the device.

The RTX 4070 is AD104 with **36 MB** of L2. (The AD104 die carries 48 MB; the 4070 SKU enables
36 and the 4070 SUPER uses all 48.) 6 MB is the Ampere-generation number. The repository's own
data was already uncomfortable with the small figure: `docs/PROFILING.md` reports an L2 hit rate
above 97% on brill, whose CSR is far larger than 6 MB, and calls that surprising -- at 36 MB it
is unremarkable.

What *is* measured is the shape: CUDA peaks at a 6 MB table and declines afterwards
(364 -> 338 -> 228 -> 176 -> 163 Gbps). The peak is real. Its attribution to L2 capacity is
the part that was never checked.

The code no longer asserts it. `gpufsm.bench.csvio.environment()` reads the cache size from the
device (`l2_mb`), `sweep_dfa.py` derives its annotation from that, and `paper/figures.py` marks
the *measured* CUDA peak instead of drawing a line at an assumed cache size. Confirm on the
machine itself:

```bash
python -c "import torch; p=torch.cuda.get_device_properties(0); print(p.name, p.L2_cache_size/2**20, 'MB')"
```

The paper has been corrected accordingly. The consequence worth knowing: its
cross-architecture argument used to read "the knee moves by ~6x, tracking the 6.7x larger L2".
The two cards' L2 differ by 1.1x (36 vs 40 MB), so the knee shift is real but is not a
cache-capacity effect, and the paper no longer attributes it to one.

## 3. Two drivers were fixed after their CSV was measured — **re-run needed**

Both files below are still the committed evidence, and both were produced by a version of
their driver that has since been corrected. The drivers in the repository are the fixed ones,
so re-running them will not reproduce these numbers exactly.

**`regret_multiseed_rtx4070.csv`.** Two defects, both fixed in the driver, neither yet
re-measured. It timed 2048 copies of a single periodic `"abcdeabcde..."` string, so every lane
walked an identical trajectory and branch divergence — the mechanism the whole result is
about — was zero by construction; it now uses `gpufsm.bench.random_batch`. And **four of its
fifteen (size, seed) points draw an automaton whose start closure already accepts**, so every
kernel returns at position 0 and the "throughput" is launch overhead divided by a batch size.
That population is where the file's 3.34x maximum comes from, which is why the paper reports
the median alone; the driver now skips such draws. Whether the median survives a divergent
input on non-degenerate automata is exactly the question this file was meant to answer, and it
has not been asked yet.

**`scalar_ablation_rtx4070.csv`.** The tile and scalar kernels differ in arithmetic as well as
in access and control pattern: the scalar one runs 256 serially dependent
`state = (state * 31 + b) % 1000003` steps, and integer modulo has no hardware instruction on
NVIDIA GPUs. Part of the 16x cliff is that arithmetic. A third kernel with the same serial loop
and a cheap recurrence would separate the two; read the cliff as an upper bound until it runs.

## 4. `multistream_async` is timed on a different clock than its comparators — **open, sized**

`native/bitpacked.cu:410` documents its return value as "the overlapped end-to-end device
time" and returns `total_ms`; every other technique returns kernel-only `kernel_ms`. The
Python wrapper (`backends/cuda/nfa.py`) stores whichever it gets into `Result.kernel_ms`, and
the memory-ablation figure then plots the async point against kernel-only points.

The bias runs against async — it is charged for transfer that the others are not — and the
ablation's conclusion is that the memory axes are inert. In the committed cost-model data the
gap is about 6% (1.026 vs 0.964 Gbps at n=32). Read the async column as end-to-end, not as
kernel time.

## 5. Warp is timed with a host clock, the others with CUDA events — **immaterial, sized**

`backends/warp/_common.py` times a launch with `time.perf_counter()` around
`wp.synchronize()`; Triton and CUDA use CUDA events. Host launch overhead is therefore inside
the Warp number and outside the others.

Sized rather than assumed: the Warp points in `sweep_techniques.csv` have median kernel times
of 7.3-28.8 **ms**, so a launch overhead on the order of 10 microseconds is about 0.1% of the
smallest measurement. This is a real asymmetry in the code and an immaterial one in the data.

## 6. Where the correctness gate runs, and where a green test means nothing — **fixed**

`gpufsm.bench.oracle.require` now runs before timing in every driver that reports an automaton
throughput — including `sweep_techniques.py`, `calibrate_costmodel.py` and `regret_multiseed.py`,
the three that produce the headline CSVs and were the ones that never called it. Two others
compared two GPU kernels against *each other* rather than against the reference, a check that
passes whenever both are wrong in the same way; they use the oracle now.

The four scripts with no gate report no automaton throughput: `gluon_probe.py` (a compile
probe), `profile_target.py` (one launch for `ncu`), `validate_costmodel.py` (re-reads a
committed CSV) and `ablate_scalar_control.py` (two raw Triton kernels, no automaton).

`scripts/oracle_gate.py` is the comprehensive check; `pytest -m gpu` is not, because a
gpu-marked test whose backend failed to build skips, and a skip counts as a pass.

## 7. Absolute throughput is far from state of the art — **by design**

On real ANMLZoo automata the engine runs at sub-Gbps to a few Gbps. The study measures a ratio
between DSLs at a fixed algorithm, and the algorithm is deliberately simple so that it can be
mirrored across four languages. Nothing here should be read as a claim about the fastest way
to run an automaton on a GPU.

## 8. Four committed CSVs were condensed by hand — **documented**

The A100 cross-architecture files and the Nsight counters were transcribed from a driver's or
a profiler's output into a plotting schema. Which ones, and what the hand step did, is in
[`../paper/data/README.md`](../paper/data/README.md).
