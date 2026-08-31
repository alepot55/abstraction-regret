# `paper/` — the published numbers and the code that draws them

This directory is the evidence half of the artifact: the committed measurement CSVs and the
single script that turns them into every figure in the paper.

```bash
python paper/figures.py     # rewrites paper/figures/*.{pdf,png} from paper/data/*.csv
```

It needs `matplotlib` and `pandas` (`pip install -e ".[paper]"`) and **no GPU**. Six figures
are generated, and those six are exactly the ones the paper includes; nothing here is a
hand-drawn image that cannot be rebuilt.

| Figure | Reads | Shows |
|---|---|---|
| `fig_throughput_vs_states` | `sweep_techniques.csv` | throughput against automaton size, per technique |
| `fig_worklist_speedup` | `sweep_techniques.csv` | the work-efficient worklist against the full-scan baseline |
| `fig_memory_ablation` | `sweep_techniques.csv` | the memory-layout axes, which turn out inert |
| `fig_abstraction_regret` | `sweep_techniques.csv` | the headline: regret per DSL against CUDA |
| `fig_costmodel_fit` | `costmodel_rtx4070.csv` | predicted against measured throughput |
| `fig_dfa_memory_bound` | `dfa_regret_rtx4070.csv` | the DFA L2 knee, and Triton's flat line through it |

Where each CSV came from -- which script, which GPU, which claim it backs, and which ones
were condensed by hand -- is in [`data/README.md`](data/README.md). `tests/test_paper_data.py`
pins every header, so a driver cannot change a schema and leave the committed data behind.

## The manuscript

The paper itself is not redistributed here. It is *The Two Faces of Abstraction Regret:
Control-Flow and Memory-Layout Limits of GPU DSLs on Irregular Automata*, IEEE HPEC 2026;
cite it via [`../CITATION.cff`](../CITATION.cff) and read it on IEEE Xplore. This repository
is the artifact the paper points to, not a mirror of the paper.
