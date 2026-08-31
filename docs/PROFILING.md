# Profiling notes (Nsight)

## Measured results (RTX 4070) — confirm the compute-bound claim

Nsight Compute counters (collected with `sudo /usr/local/cuda/bin/ncu`, single clean
launch via `scripts/profile_target.py`):

| kernel (n=128, 8192 strings) | SM thrpt % | DRAM thrpt % | L2 hit % | achieved occ % |
|---|---|---|---|---|
| `multistream` (full-scan) | **19.44** | **0.01** | 79.19 | 16.67 |
| `multistream_shared` | 19.60 | 0.01 | **93.02** | 16.67 |

**Interpretation (hardware-level confirmation of the cost-model finding):**
- Full-scan is **compute-bound, not memory-bound**: SM throughput (≈19%) exceeds DRAM
  throughput (0.01%) by ~3 orders of magnitude. The O(n²) eps-closure dominates.
- `multistream_shared` (CSR in shared memory) has **identical SM%, DRAM% and occupancy**;
  only the L2 hit rate rises (79→93%). Since DRAM is not the bottleneck, that locality win
  does not move runtime — exactly why the memory-layout axes are inert in this regime
  (matches `multistream_shared` tying `multistream` in the throughput sweep).
- Absolute SM% is modest because occupancy is ~17% (one program per thread); the *ratio*
  SM≫DRAM is the load-bearing observation.

The work-efficient `worklist` (n=256, 16384 strings) profiles at SM 5.4% / DRAM 6.3% /
L2 95.6% / occ 23.4% — roughly balanced and *under-utilized* (it does far less work), i.e.
latency/occupancy-bound at this scale. At a small batch (512 strings, 2 blocks) occupancy
is only 16.6% with SM 0.23% / DRAM 4.5% — strongly under-utilized. **This motivates the
block-parallel (warp-cooperative) worklist** as future work: with few strings, one
thread/string cannot fill the GPU.

So the compute-bound claim is now **measured**, not only inferred from the ablation.

## Enabling ncu (requires administrator rights on the measurement host)

Per-session: `sudo /usr/local/cuda/bin/ncu ...` (note the absolute path — sudo's PATH lacks
`/usr/local/cuda/bin`). Permanent (non-sudo) profiling needs
`NVreg_RestrictProfilingToAdminUsers=0` + reboot.

## (historical) Counter permission gate

`ncu` (Nsight Compute) is installed (`/usr/local/cuda/bin/ncu`) but collecting hardware
counters fails with:

```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU
Performance Counters on the target device 0.
```

GPU performance counters are admin-gated by the driver. To enable (one-time, needs sudo):

- **Quick (per session):** run ncu under sudo — `sudo ncu ...`.
- **Permanent:** allow non-admin profiling, then reboot:
  ```
  echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/nvidia-prof.conf
  sudo update-initramfs -u && sudo reboot
  ```
  See https://developer.nvidia.com/ERR_NVGPUCTRPERM

Once enabled, profile a single clean kernel launch with:

```
ncu --target-processes all -k regex:worklist --launch-count 1 \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active \
    python scripts/profile_target.py cuda worklist 256
```

`scripts/profile_target.py` issues exactly one `run_batch` (one kernel launch) so the
profile is clean. Compare `cuda/worklist` vs `cuda/multistream` (full-scan):
- compute % >> DRAM % ⇒ compute-bound; the reverse ⇒ memory-bound.

`nsys` (Nsight Systems, timeline tracing) does NOT need counter permissions and can show
the `multistream_async` H2D/kernel/D2H overlap; use it for the sync→async evidence.

## The compute-bound claim does NOT depend on ncu

The central regime claim is already established by *controlled experiment*, not counters:

1. **Ablation control:** `multistream_shared` stages the CSR into shared memory (modeled
   global traffic = 0) yet lands within 1.3% of `multistream` (global CSR) and
   `multistream_async` at every size (`paper/data/sweep_techniques.csv`). If the kernel were
   memory-bound, removing CSR global traffic would help — it does not ⇒ compute-bound.
   Precisely: the bootstrap CI95 intervals are *disjoint* at 4 of the 5 sizes, so
   `multistream_shared` is resolvably faster; the effect is 0.22–1.26%, two orders of
   magnitude below the 6–8× regret it is being contrasted with. The axis is inert in
   magnitude, not statistically indistinguishable, and the earlier wording said the latter.
2. **Scaling:** throughput ∝ 1/n² for the full-scan kernels (the O(n²) eps-closure), fit by
   the cost model to <1% at the largest size measured (n=256) (`docs/RESULTS_COSTMODEL.md`).
3. **Cure:** the work-efficient `worklist` kernel (O(active), no O(n²)) is ≈330×–10⁴× faster
   and reaches 15–170 Gbps — the regime where memory layout *can* matter; profiling it (once
   ncu is unblocked) is the next confirmatory step.

So Nsight counters are **confirmatory, not load-bearing** for the paper's argument.

## Worklist kernels profiled (ncu, single launch; `paper/data/nsight_rtx4070.csv`)

Profiled the work-efficient worklist kernels (8192-state synthetic, 2048 strings; and brill
42661 states for the large-CSR case):

| kernel | states | SM% | DRAM% | L2 hit% | occ% | dur |
|---|---|---|---|---|---|---|
| `worklist_global` (1 thread/string) | 8192 | 1.06 | 0.01 | 100 | 16.6 | 62.3 ms |
| `worklist_warp` (warp/string) | 8192 | 15.98 | 0.13 | 99.96 | 56.8 | 2.64 ms |
| `worklist_warp` | 42661 (17 MB CSR) | 16.30 | 2.25 | 97.6 | 36.6 | — |

**Findings.** (1) The warp kernel fixes the single-thread kernel's catastrophic
under-utilization (occupancy 16.6→56.8%, SM 1.1→16%) — the source of its speedup is
**parallelism/occupancy, not memory**. (2) The worklist is **latency/instruction-bound, not
memory-bound**: DRAM stays $\le$2.25% and L2 hit $\ge$97.6% *even for brill's 17 MB CSR* — because all strings share the CSR and only a hot subset of rows is
touched per batch, staying L2-resident. (3) This is *why* `worklist_shared` (working set in
shared memory) is inert: the working set already lives in L2 at 99.96% hit. ⇒ The path to SOTA
absolute throughput is **algorithmic** (cut instructions/serialization: compacted active-ID
worklist, fewer `atomicOr`/`__syncwarp`, ngAP-style non-blocking multi-symbol), **not** memory
layout. This reinforces the control-flow-bound (not bandwidth-bound) nature of NFA simulation.
(Note: the brill run reported app exit code 6 on teardown after the profiled launch; the
single-launch counters above were captured cleanly and are consistent with the 8192 trend.)
