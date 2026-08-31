"""Bit-packed NFA simulation — the executable spec for the GPU bit-packed kernels.

The active state-set is a single packed bitmask (1 bit per state), not a byte/int
array (the legacy Triton default wasted 4 bytes per state — a 31x blow-up at 500
states). Transitions and epsilon-closure are expressed as bitwise OR over
per-state target masks. This module is the *reference for the memory-centric
thesis*: it computes the identical verdict as :func:`gpufsm.reference.simulate`,
using the compact representation the CUDA/Triton kernels must mirror.

Implementation note: a Python ``int`` is an arbitrary-width packed bitset, so the
bitwise ops here map 1:1 onto the word-level ``|`` / ``&`` operations a GPU kernel
performs on ``uint32`` words.
"""

from __future__ import annotations

from .nfa import ANY_SYMBOL, NFA


def _build_masks(nfa: NFA) -> tuple[list[dict[int, int]], list[int], list[int], int]:
    """Precompute per-state target masks (symbol-specific, any-symbol, epsilon)."""
    n = nfa.num_states
    sym_masks: list[dict[int, int]] = [dict() for _ in range(n)]
    any_masks: list[int] = [0] * n
    eps_masks: list[int] = [0] * n

    sp, st, ss = nfa.sym_row_ptr, nfa.sym_targets, nfa.sym_symbols
    for s in range(n):
        for k in range(int(sp[s]), int(sp[s + 1])):
            sym = int(ss[k])
            bit = 1 << int(st[k])
            if sym == ANY_SYMBOL:
                any_masks[s] |= bit
            else:
                sym_masks[s][sym] = sym_masks[s].get(sym, 0) | bit

    ep, et = nfa.eps_row_ptr, nfa.eps_targets
    for s in range(n):
        m = 0
        for k in range(int(ep[s]), int(ep[s + 1])):
            m |= 1 << int(et[k])
        eps_masks[s] = m

    accept_mask = 0
    for s in range(n):
        if nfa.accept[s]:
            accept_mask |= 1 << s
    return sym_masks, any_masks, eps_masks, accept_mask


def _iter_bits(mask: int):
    """Yield the indices of set bits in ``mask`` (the active states)."""
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _epsilon_closure(active: int, eps_masks: list[int]) -> int:
    frontier = active
    while frontier:
        new = 0
        for s in _iter_bits(frontier):
            new |= eps_masks[s]
        new &= ~active
        active |= new
        frontier = new
    return active


class BitmapProgram:
    """An NFA compiled to per-state bitmasks, ready to run over many inputs.

    Building the masks is O(states + transitions) and does not depend on the input, so it
    belongs with the executor rather than inside the timed loop. ``simulate_bitmap`` used to
    rebuild them on every call, which meant :func:`gpufsm.api.benchmark` timed the setup on
    every repeat: at 256 states that was more than the whole simulation. The GPU backends
    have always done this work once at executor construction; this makes the CPU spec match,
    so a comparison between them measures the same thing.
    """

    __slots__ = ("start_closure", "sym_masks", "any_masks", "eps_masks", "accept_mask")

    def __init__(self, nfa: NFA) -> None:
        self.sym_masks, self.any_masks, self.eps_masks, self.accept_mask = _build_masks(nfa)
        self.start_closure = _epsilon_closure(1 << nfa.start_state, self.eps_masks)

    def run(self, input_bytes: bytes) -> tuple[bool, int]:
        """``(accepted, match_len)`` under latch-first-match semantics."""
        accept_mask, eps_masks = self.accept_mask, self.eps_masks
        sym_masks, any_masks = self.sym_masks, self.any_masks

        active = self.start_closure
        if active & accept_mask:
            return True, 0

        for i, b in enumerate(input_bytes):
            nxt = 0
            for s in _iter_bits(active):
                nxt |= any_masks[s]
                sm = sym_masks[s]
                if b in sm:
                    nxt |= sm[b]
            if not nxt:
                break
            active = _epsilon_closure(nxt, eps_masks)
            if active & accept_mask:
                return True, i + 1

        return False, 0


def simulate_bitmap(nfa: NFA, input_bytes: bytes) -> tuple[bool, int]:
    """Bit-packed simulation with latch-first-match semantics.

    Returns ``(accepted, match_len)`` — identical to :func:`gpufsm.reference.simulate`.
    Compiles the automaton on every call; for repeated runs build a :class:`BitmapProgram`
    once instead.
    """
    return BitmapProgram(nfa).run(input_bytes)
