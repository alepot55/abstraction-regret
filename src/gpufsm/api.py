"""The one public API: ``run`` (single execution) and ``benchmark`` (timed).

Both automaton kinds go through here. ``run(nfa, ...)`` simulates an NFA and
``run(dfa, ...)`` a DFA; the kind is derived from the object, so there is no second
entry point to keep in sync and no backend string parsed by hand.
"""

from __future__ import annotations

from . import backends as _backends  # noqa: F401  (triggers backend registration)
from .core.registry import Automaton, Backend, Kind, get_factory
from .core.result import BenchmarkStats, Result, batch_results


def run(
    automaton: Automaton,
    input_bytes: bytes,
    backend: Backend | str = Backend.CPU,
    technique: str | None = None,
) -> Result:
    """Run ``automaton`` over ``input_bytes`` on the chosen backend/technique."""
    backend = Backend(backend)
    technique, factory = get_factory(Kind.of(automaton), backend, technique)
    return factory(automaton, technique).run(input_bytes)


def run_batch(
    automaton: Automaton,
    inputs: list[bytes],
    backend: Backend | str = Backend.CPU,
    technique: str | None = None,
) -> list[Result]:
    """Run ``automaton`` over a batch of inputs (one :class:`Result` per input).

    Backends/techniques that expose a native ``run_batch`` (e.g. the multi-stream
    GPU kernels, one program/block per string) handle the whole batch in a single
    launch; everything else falls back to looping :meth:`run`, so every technique
    supports batching transparently.

    **The timing contract is the same on both paths**: ``results[0].kernel_ms`` is the time
    for the *whole batch* and the rest are zero, which is what every measurement driver
    reads. The fallback used to leave each result carrying its own per-string time, so
    ``results[0].kernel_ms`` silently meant "the first string" for looping techniques and
    "the batch" for batched ones -- the same expression, two different quantities, across
    the exact comparison this package exists to make.
    """
    backend = Backend(backend)
    technique, factory = get_factory(Kind.of(automaton), backend, technique)
    executor = factory(automaton, technique)
    batch = getattr(executor, "run_batch", None)
    if callable(batch):
        return batch(inputs)

    per_input = [executor.run(b) for b in inputs]
    if not per_input:
        return []
    return batch_results(
        [r.accepted for r in per_input],
        [r.match_len for r in per_input],
        kernel_ms=sum(r.kernel_ms for r in per_input),
        transfer_ms=sum(r.transfer_ms for r in per_input),
    )


def benchmark(
    automaton: Automaton,
    input_bytes: bytes,
    backend: Backend | str = Backend.CPU,
    technique: str | None = None,
    repeats: int = 10,
    warmup: int = 3,
) -> BenchmarkStats:
    """Time ``repeats`` runs (after ``warmup``) and aggregate mean/std/CI95."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    backend = Backend(backend)
    technique, factory = get_factory(Kind.of(automaton), backend, technique)
    executor = factory(automaton, technique)

    last: Result | None = None
    for _ in range(max(0, warmup)):
        last = executor.run(input_bytes)

    raw: list[float] = []
    for _ in range(repeats):
        last = executor.run(input_bytes)
        raw.append(last.kernel_ms)

    assert last is not None
    return BenchmarkStats(
        backend=backend.value,
        technique=technique,
        accepted=last.accepted,
        match_len=last.match_len,
        n=repeats,
        raw_ms=raw,
    )
