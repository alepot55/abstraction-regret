#!/usr/bin/env python3
"""Generate the gpufsm paper figures — ONLY from versioned CSVs (reproducible).

Inputs (committed under paper/data/):
  - sweep_techniques.csv      throughput sweep (median + CI95) per backend/technique/size
  - costmodel_rtx4070.csv     cost-model calibration points

Outputs (paper/figures/*.pdf + *.png):
  - fig_throughput_vs_states  throughput vs num_states, log y (worklist vs full-scan; DSL gap)
  - fig_worklist_speedup      worklist speedup over CUDA full-scan multistream vs num_states
  - fig_memory_ablation       multistream vs _shared vs _async (within noise => compute-bound)
  - fig_abstraction_regret    Triton/CUDA/Warp throughput vs CUDA (regret = paradigm, not height)

Run:  python paper/figures.py    (needs the `paper` extra: matplotlib, pandas)

NOTE: supersedes the legacy generate_figures.py (MatMul/MLP/BitGen schema, out of scope).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
plt.rcParams.update(
    {
        "figure.figsize": (7, 4),
        "font.size": 11,
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.fontsize": 9,
        "savefig.bbox": "tight",
        "savefig.dpi": 200,
    }
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"

_LABEL = {
    "cuda/worklist": "CUDA worklist (work-efficient)",
    "cuda/multistream": "CUDA multistream (full-scan)",
    "cuda/multistream_shared": "CUDA multistream + shared CSR",
    "cuda/multistream_async": "CUDA multistream + async",
    "triton/multistream": "Triton multistream (full-scan)",
    "warp/multistream": "Warp multistream (full-scan)",
}


def _save(fig, name: str) -> None:
    """Write both formats, byte-reproducibly.

    matplotlib stamps a ``/CreationDate`` into every PDF, so re-running this script
    produced a diff in twelve files that differed only in a timestamp -- which makes
    "the figures rebuild from the committed CSVs" impossible to check with `git diff`.
    Omitting the date makes a rebuild a no-op when the data has not changed, and a real
    diff when it has.
    """
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", metadata={"CreationDate": None})
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print(f"wrote {name}.pdf / .png")


def fig_throughput_vs_states(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots()
    for (be, te), g in df.groupby(["backend", "technique"]):
        g = g.sort_values("num_states")
        key = f"{be}/{te}"
        ax.plot(g["num_states"], g["throughput_gbps"], marker="o", label=_LABEL.get(key, key))
    ax.set_xlabel("NFA states")
    ax.set_ylabel("Throughput (Gbps)")
    ax.set_yscale("log")
    ax.set_title("Throughput vs NFA size (batched multi-stream, RTX 4070)")
    ax.legend(loc="best")
    _save(fig, "fig_throughput_vs_states")


def fig_worklist_speedup(df: pd.DataFrame) -> None:
    base = df[(df.backend == "cuda") & (df.technique == "multistream")].set_index("num_states")
    wl = df[(df.backend == "cuda") & (df.technique == "worklist")].set_index("num_states")
    sizes = sorted(set(base.index) & set(wl.index))
    speedup = [base.loc[n, "median_ms"] / wl.loc[n, "median_ms"] for n in sizes]
    fig, ax = plt.subplots()
    ax.bar([str(n) for n in sizes], speedup, color="#2a7", width=0.6)
    for i, s in enumerate(speedup):
        ax.text(i, s, f"{s:.0f}x", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("NFA states")
    ax.set_ylabel("Speedup over full-scan (x)")
    ax.set_yscale("log")
    ax.set_title("Work-efficient worklist speedup over O(n^2) full-scan (CUDA)")
    _save(fig, "fig_worklist_speedup")


def fig_memory_ablation(df: pd.DataFrame) -> None:
    techs = ["multistream", "multistream_shared", "multistream_async"]
    d = df[(df.backend == "cuda") & (df.technique.isin(techs))]
    sizes = sorted(d.num_states.unique())
    fig, ax = plt.subplots()
    width = 0.25
    for j, te in enumerate(techs):
        ys = [
            d[(d.technique == te) & (d.num_states == n)]["throughput_gbps"].iloc[0] for n in sizes
        ]
        xs = [i + (j - 1) * width for i in range(len(sizes))]
        ax.bar(xs, ys, width=width, label=te.replace("multistream", "ms"))
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([str(n) for n in sizes])
    ax.set_xlabel("NFA states")
    ax.set_ylabel("Throughput (Gbps)")
    ax.set_yscale("log")
    ax.set_title("Memory-layout axes are within noise (compute-bound regime)")
    ax.legend(loc="best")
    _save(fig, "fig_memory_ablation")


def fig_abstraction_regret(df: pd.DataFrame) -> None:
    # Throughput ratio vs CUDA on the same full-scan multistream kernel, per DSL.
    d = df[df.technique == "multistream"]
    sizes = [32, 64]  # warp covers <=64
    backends = ["cuda", "triton", "warp"]
    fig, ax = plt.subplots()
    width = 0.25
    for j, be in enumerate(backends):
        ratios = []
        for n in sizes:
            cuda_tp = d[(d.backend == "cuda") & (d.num_states == n)]["throughput_gbps"].iloc[0]
            sub = d[(d.backend == be) & (d.num_states == n)]
            ratios.append(sub["throughput_gbps"].iloc[0] / cuda_tp if len(sub) else 0.0)
        xs = [i + (j - 1) * width for i in range(len(sizes))]
        ax.bar(xs, ratios, width=width, label=be)
    ax.axhline(1.0, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f"{n} states" for n in sizes])
    ax.set_ylabel("Throughput relative to CUDA")
    ax.set_title("Abstraction regret = execution paradigm: Triton (tile) << Warp (thread) ~ CUDA")
    ax.legend(loc="best")
    _save(fig, "fig_abstraction_regret")


def fig_costmodel_fit(cm: pd.DataFrame) -> None:
    """Predicted vs measured throughput: per-backend fit of time = a*traffic + b*n^2."""
    import numpy as np

    fig, ax = plt.subplots()
    for be, g in cm.groupby("backend"):
        traffic = g["traffic_bytes_per_sym"].to_numpy(float)
        n2 = (g["num_states"].to_numpy(float)) ** 2
        meas = g["throughput_gbps"].to_numpy(float)
        t_meas = 8e-9 / meas  # seconds/symbol
        coef, *_ = np.linalg.lstsq(np.stack([traffic, n2], axis=1), t_meas, rcond=None)
        a, b = max(coef[0], 1e-18), max(coef[1], 0.0)
        # Gbps from seconds-per-symbol is 8 bits / (t * 1e9) == 8e-9 / t. A second /1e9 here
        # put every predicted point nine decades under the y=x line it is plotted against.
        # gpufsm.costmodel.predict_throughput_gbps is the reference form of this conversion.
        pred = 8e-9 / (a * traffic + b * n2)
        ax.scatter(meas, pred, label=be, s=40)
    lim = [cm["throughput_gbps"].min() * 0.5, cm["throughput_gbps"].max() * 2]
    ax.plot(lim, lim, "k--", lw=0.8, label="y = x")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Measured throughput (Gbps)")
    ax.set_ylabel("Cost-model predicted (Gbps)")
    ax.set_title("Cost model: predicted vs measured (per-backend n^2 fit)")
    ax.legend(loc="best")
    _save(fig, "fig_costmodel_fit")


def fig_dfa_memory_bound(dfa: pd.DataFrame) -> None:
    """DFA (memory-bound face): CUDA/Warp peak at the L2 size then drop as the table
    crosses L2; Triton stays flat (scalar-gather-bound, never reaching the memory regime).
    Line plot over a fine table-size grid makes the L2 knee visible."""
    backends = ["cuda", "warp", "triton"]
    markers = {"cuda": "o", "warp": "s", "triton": "^"}
    fig, ax = plt.subplots()
    for be in backends:
        sub = dfa[dfa.backend == be].sort_values("num_states")
        mb = sub["table_kb"].to_numpy() / 1024.0  # table size in MB
        ax.plot(mb, sub["throughput_gbps"], marker=markers[be], label=be)
    # Mark where the thread-model backend actually peaks, read off the data. This used to
    # be a hardcoded "6 MB L2" line: a statement about the hardware that no committed CSV
    # records, on a device whose real L2 is larger. The observed peak is what was measured.
    cuda = dfa[dfa.backend == "cuda"].sort_values("num_states")
    if len(cuda):
        peak_mb = float(cuda.loc[cuda["throughput_gbps"].idxmax(), "table_kb"]) / 1024.0
        ax.axvline(peak_mb, color="0.5", ls="--", lw=1)
        ax.text(
            peak_mb,
            ax.get_ylim()[1] * 0.96,
            f" CUDA peak, {peak_mb:.0f} MB table",
            color="0.4",
            va="top",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("DFA transition-table size (MB, log scale)")
    ax.set_ylabel("Throughput (Gbps)")
    ax.set_title("DFA face: CUDA and Warp rise then fall with table size; Triton flat")
    ax.legend(loc="best")
    _save(fig, "fig_dfa_memory_bound")


def main() -> None:
    sweep = pd.read_csv(DATA / "sweep_techniques.csv")
    fig_throughput_vs_states(sweep)
    fig_worklist_speedup(sweep)
    fig_memory_ablation(sweep)
    fig_abstraction_regret(sweep)
    cm = pd.read_csv(DATA / "costmodel_rtx4070.csv")
    fig_costmodel_fit(cm)
    dfa = pd.read_csv(DATA / "dfa_regret_rtx4070.csv")
    fig_dfa_memory_bound(dfa)
    print(f"\nfigures written to {FIGS}")


if __name__ == "__main__":
    main()
