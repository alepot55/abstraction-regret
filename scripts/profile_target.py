"""Minimal single-launch profiling target for Nsight Compute (ncu).

Usage:  python scripts/profile_target.py <backend> <technique> <num_states>
Builds one NFA + a small batch and issues exactly ONE run_batch (one kernel launch)
so ncu can profile that kernel cleanly. CUDA kernels have stable names
(worklist_multistream_kernel, bitpacked_multistream_kernel), so filter ncu with
--kernel-name-base to isolate them.
"""

from __future__ import annotations

import sys

import numpy as np

from gpufsm.api import run_batch
from gpufsm.bench import random_nfa


def main() -> None:
    # arg3 is either an int num_states (random NFA) or "anml:<key>" (real ANMLZoo automaton,
    # whose large CSR can exceed L2 -> tests whether the kernel becomes memory-bound).
    backend, technique, spec = sys.argv[1], sys.argv[2], sys.argv[3]
    n_strings = int(sys.argv[4]) if len(sys.argv) > 4 else 16384  # default saturates ~46 SMs
    slen = 256
    if spec.startswith("anml:"):
        from gpufsm.io.anml import load_anml
        from gpufsm.io.datasets import DATASETS, ensure

        nfa = load_anml(ensure(DATASETS[spec.split(":", 1)[1]], "data/anmlzoo"))
    else:
        nfa = random_nfa(int(spec))
    alpha = sorted({int(s) for s in nfa.sym_symbols if 0 <= int(s) <= 255}) or [ord("a")]
    rng = np.random.default_rng(0)
    flat = bytes(rng.choice(alpha, size=n_strings * slen).astype(np.uint8))
    batch = [flat[i * slen : (i + 1) * slen] for i in range(n_strings)]
    run_batch(nfa, batch, backend=backend, technique=technique)  # the single profiled launch


if __name__ == "__main__":
    main()
