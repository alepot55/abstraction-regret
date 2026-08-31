"""Correctness gates: nothing reports a throughput before it agrees with the oracle.

This is the invariant the whole study rests on — a fast kernel that computes the wrong
verdict measures nothing. Every measurement script ran some version of this check
inline; having one implementation means the sample size and the failure message are
the same everywhere, and a script cannot quietly skip it.
"""

from __future__ import annotations

from ..api import run, run_batch
from ..core.registry import Automaton, Backend

DEFAULT_CHECK = 64


class OracleMismatch(AssertionError):
    """A backend disagreed with the CPU reference — the run must not report timings."""


def compare(
    automaton: Automaton,
    inputs: list[bytes],
    backend: Backend | str,
    technique: str | None = None,
    limit: int = DEFAULT_CHECK,
) -> list[tuple[int, tuple[bool, int], tuple[bool, int]]]:
    """Return ``(index, got, expected)`` for every input where the backend differs.

    An empty sample raises rather than returning "no mismatches". A gate that checked zero
    inputs and a gate that checked many and found nothing wrong are the same value here, and
    only one of them means the kernel is right.
    """
    if limit <= 0:
        raise ValueError(f"limit must be >= 1 to check anything, got {limit}")
    sample = inputs[:limit]
    if not sample:
        raise ValueError("no inputs to compare: the oracle gate would pass vacuously")
    got = run_batch(automaton, sample, backend=backend, technique=technique)
    mismatches = []
    for i, (data, res) in enumerate(zip(sample, got, strict=True)):
        ref = run(automaton, data, backend=Backend.CPU)
        if (res.accepted, res.match_len) != (ref.accepted, ref.match_len):
            mismatches.append((i, (res.accepted, res.match_len), (ref.accepted, ref.match_len)))
    return mismatches


def matches(
    automaton: Automaton,
    inputs: list[bytes],
    backend: Backend | str,
    technique: str | None = None,
    limit: int = DEFAULT_CHECK,
) -> bool:
    """True when the backend agrees with the CPU reference on the first ``limit`` inputs."""
    return not compare(automaton, inputs, backend, technique, limit)


def require(
    automaton: Automaton,
    inputs: list[bytes],
    backend: Backend | str,
    technique: str | None = None,
    limit: int = DEFAULT_CHECK,
) -> None:
    """Raise :class:`OracleMismatch` unless the backend agrees with the reference.

    Call this before timing anything. The message names the first disagreement so a
    failure is actionable without re-running under a debugger.
    """
    bad = compare(automaton, inputs, backend, technique, limit)
    if not bad:
        return
    i, got, expected = bad[0]
    name = backend.value if isinstance(backend, Backend) else backend
    raise OracleMismatch(
        f"{name}/{technique or 'default'} disagrees with the CPU reference on "
        f"{len(bad)}/{min(len(inputs), limit)} inputs; first at index {i}: "
        f"got {got}, expected {expected}"
    )
