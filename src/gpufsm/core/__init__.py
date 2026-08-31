"""Core data model: automata representations, the result format and the registry.

Everything here is dependency-free (numpy only) and backend-agnostic, so a backend
module can import it without dragging in torch, triton or the CUDA extension:

- :mod:`~gpufsm.core.nfa` / :mod:`~gpufsm.core.dfa` — the two automaton
  representations shared by every backend (CSR for the NFA, a dense table for the DFA).
- :mod:`~gpufsm.core.result` — the single result/timing format all backends return.
- :mod:`~gpufsm.core.registry` — the one extension point, ``@register(Kind, Backend, technique)``.
- :mod:`~gpufsm.core.bitmap` — the bit-packed NFA spec the GPU kernels mirror.
- :mod:`~gpufsm.core.packing` — host-side array packing shared by the GPU backends.

The correctness oracles live one level up in :mod:`gpufsm.reference`.
"""

from __future__ import annotations
