"""Backend/technique registry — the single extension point.

Adding a backend or a technique is one file plus one ``@register`` line. Each
registered factory builds an :class:`Executor` (anything with ``run(bytes) ->
Result``) for a given automaton. Backends declare an availability probe so the rest
of the system degrades gracefully when CUDA/Triton/Warp are absent.

Registrations are keyed by ``(Kind, Backend, technique)``. The :class:`Kind` axis is
what keeps the DFA workload — the memory-bound face of the study — inside this
registry instead of behind a parallel dispatch function: both faces are reached
through :func:`gpufsm.api.run`, and ``gpufsm list`` sees all of them.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Protocol, runtime_checkable

from .dfa import DFA
from .nfa import NFA
from .result import Result

Automaton = NFA | DFA


class Kind(str, Enum):
    """Which automaton model a technique simulates."""

    NFA = "nfa"  # control-flow-bound face: active-set traversal + epsilon closure
    DFA = "dfa"  # memory-bound face: one dense-table gather per input byte

    @classmethod
    def of(cls, automaton: Automaton) -> Kind:
        """The kind of a concrete automaton — how :mod:`gpufsm.api` dispatches."""
        if isinstance(automaton, NFA):
            return cls.NFA
        if isinstance(automaton, DFA):
            return cls.DFA
        raise TypeError(f"expected an NFA or a DFA, got {type(automaton).__name__}")


class Backend(str, Enum):
    CPU = "cpu"  # reference simulator — always available, the correctness oracle
    TRITON = "triton"
    CUDA = "cuda"
    WARP = "warp"  # NVIDIA Warp — Python thread-SIMT probe (abstraction spectrum)


@runtime_checkable
class Executor(Protocol):
    def run(self, input_bytes: bytes) -> Result: ...


ExecutorFactory = Callable[[Automaton, str], Executor]

_REGISTRY: dict[tuple[Kind, Backend, str], ExecutorFactory] = {}
_AVAILABILITY: dict[Backend, Callable[[], bool]] = {}
_DEFAULTS: dict[tuple[Kind, Backend], str] = {}


def register(
    kind: Kind, backend: Backend, technique: str, default: bool = False
) -> Callable[[ExecutorFactory], ExecutorFactory]:
    """Decorator: register a factory ``(automaton, technique) -> Executor``.

    ``default=True`` marks the technique ``run(automaton, backend=...)`` uses when the
    caller names none. Exactly one per (kind, backend). The default used to be whichever
    module happened to import first, so reordering a tuple of module names for readability
    silently changed what every default-argument call measured -- in software whose purpose
    is comparing techniques, that has to be declared rather than emergent.
    """

    def deco(factory: ExecutorFactory) -> ExecutorFactory:
        _REGISTRY[(kind, backend, technique)] = factory
        if default:
            previous = _DEFAULTS.get((kind, backend))
            if previous is not None and previous != technique:
                raise RuntimeError(
                    f"two defaults declared for {backend.value}/{kind.value}: "
                    f"{previous!r} and {technique!r}"
                )
            _DEFAULTS[(kind, backend)] = technique
        return factory

    return deco


def register_availability(backend: Backend, probe: Callable[[], bool]) -> None:
    _AVAILABILITY[backend] = probe


def is_available(backend: Backend) -> bool:
    probe = _AVAILABILITY.get(backend)
    try:
        return bool(probe()) if probe else any(b == backend for _, b, _ in _REGISTRY)
    except Exception:
        return False


def get_factory(kind: Kind, backend: Backend, technique: str | None) -> tuple[str, ExecutorFactory]:
    """Resolve a factory; if ``technique`` is None, use the backend's default."""
    techs = list_techniques(backend, kind)
    if not techs:
        # "nothing registered" has two very different causes, and saying which one saves a
        # user from debugging a typo that is really a missing CUDA build.
        reason = (
            "the backend is not available on this machine (import or build failed); "
            "run `gpufsm env` to see why"
            if not is_available(backend)
            else f"it registers no {kind.value} technique"
        )
        raise KeyError(
            f"backend {backend.value!r} has no {kind.value} technique: {reason}. "
            f"Available backends: {[b.value for b in available_backends()]}"
        )
    if technique is None:
        technique = _DEFAULTS.get((kind, backend))
        if technique is None:
            raise KeyError(
                f"backend {backend.value!r} declares no default {kind.value} technique; "
                f"name one explicitly (available: {techs}) or mark one with "
                f"register(..., default=True)"
            )
    if (kind, backend, technique) not in _REGISTRY:
        raise KeyError(
            f"{kind.value} technique {technique!r} not registered for backend "
            f"{backend.value!r}; available: {techs}"
        )
    return technique, _REGISTRY[(kind, backend, technique)]


def list_techniques(backend: Backend, kind: Kind = Kind.NFA) -> list[str]:
    return [t for (k, b, t) in _REGISTRY if b == backend and k == kind]


def list_kinds(backend: Backend) -> list[Kind]:
    """Which automaton kinds ``backend`` implements at least one technique for."""
    return [k for k in Kind if list_techniques(backend, k)]


def available_backends() -> list[Backend]:
    return [b for b in Backend if list_kinds(b) and is_available(b)]
