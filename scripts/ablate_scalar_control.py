"""Causal primitive ablation: the cost of scalar data-dependent control flow IN Triton.

The capability->cost map claims the tile-SPMD regret on automata is *caused* by an inexpressible
primitive: scalar, data-dependent per-element processing (scalar-gather-in-tile + a carried
scalar recurrence). This isolates that cost cleanly: two Triton kernels process the SAME batch
with ONE program per string, differing ONLY in the access/control pattern --
  A (tile):   load the whole string as a tile and do a vectorized reduction (no scalar control)
  B (scalar): a sequential loop with a carried scalar state recurrence (automata-style)
Everything else is identical (language, data, harness, parallelism), so the A/B throughput cliff
is the cost the tile model imposes specifically on scalar data-dependent work -- the causal
control behind the abstraction-regret claim. (CUDA's thread model runs both at similar speed;
that cross-paradigm half is a separate kernel.) Writes paper/data/scalar_ablation_rtx4070.csv.

Usage:  python scripts/ablate_scalar_control.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import numpy as np

from gpufsm.bench.csvio import environment, guard_device, write_rows

FIELDS = ["n_strings", "tile_gbps", "scalar_gbps", "cliff", "gpu"]


def main() -> int:
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as e:  # pragma: no cover
        print(f"SKIP: triton/torch unavailable ({type(e).__name__}: {e})")
        return 0
    if not torch.cuda.is_available():
        print("SKIP: no CUDA device")
        return 0

    SLEN = 256

    @triton.jit
    def tile_kernel(inp, out, n, SLEN: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            offs = tl.arange(0, SLEN)
            x = tl.load(inp + pid * SLEN + offs)  # whole string as a tile
            tl.store(out + pid, tl.sum((x == 97).to(tl.int32)))  # vectorized reduction

    @triton.jit
    def scalar_kernel(inp, out, n, SLEN: tl.constexpr):
        pid = tl.program_id(0)
        if pid < n:
            state = tl.zeros((), tl.int32)
            for i in range(SLEN):
                b = tl.load(inp + pid * SLEN + i)  # scalar element
                state = (state * 31 + b) % 1000003  # carried data-dependent scalar recurrence
            tl.store(out + pid, state)

    def med_ms(kernel, inp, out, n, runs=9):
        grid = (n,)
        for _ in range(3):
            kernel[grid](inp, out, n, SLEN)
        torch.cuda.synchronize()
        ts = []
        for _ in range(runs):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            kernel[grid](inp, out, n, SLEN)
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        return statistics.median(ts)

    outp = Path("paper/data/scalar_ablation_rtx4070.csv")
    guard_device(outp)
    gpu = environment()["gpu"]

    rows: list[dict[str, object]] = []
    print("Scalar-control ablation in Triton (one program/string; tile vs scalar recurrence):")
    print(f"{'n_str':>7}{'tile_Gbps':>11}{'scalar_Gbps':>13}{'cliff(x)':>10}")
    rng = np.random.default_rng(0)
    for n in [1024, 4096, 16384]:
        data = torch.tensor(rng.integers(97, 102, size=n * SLEN, dtype=np.int32), device="cuda")
        out = torch.zeros(n, dtype=torch.int32, device="cuda")
        ta = med_ms(tile_kernel, data, out, n)
        tb = med_ms(scalar_kernel, data, out, n)
        bits = n * SLEN * 8
        ga, gb = bits / (ta * 1e-3) / 1e9, bits / (tb * 1e-3) / 1e9
        rows.append(
            {
                "n_strings": n,
                "tile_gbps": round(ga, 2),
                "scalar_gbps": round(gb, 2),
                "cliff": round(ga / gb, 1),
                "gpu": gpu,
            }
        )
        print(f"{n:7d}{ga:11.2f}{gb:13.2f}{ga / gb:9.1f}x")

    print(f"\nwrote {write_rows(outp, rows, FIELDS)} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
