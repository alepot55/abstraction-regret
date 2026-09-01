"""The CUDA extension's three-way surface must agree, checked without a CUDA toolkit.

``native/include/api.hpp`` declares each host entry point, one ``.cu`` defines it, and
``bindings.cu`` exposes it to Python. The header says why that matters: the kernels are
compiled one translation unit per family without relocatable device code, so a declaration
that has drifted from its definition is an *undefined symbol at link time* -- not a warning,
and not something the Python side can diagnose.

CI does compile every translation unit with ``nvcc``, but compiling is not linking: it would
not catch a signature that differs between the header and the definition, nor an entry point
that exists but was never bound. These checks are pure text analysis, so they run in the CPU
suite on every commit, on a machine with no CUDA at all.

The parsing is deliberately narrow -- it matches the one declaration style the file uses. A
new entry point written differently makes `test_the_parse_found_the_expected_surface` fail
rather than silently reduce the coverage to nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

NATIVE = Path(__file__).resolve().parent.parent / "src" / "gpufsm" / "backends" / "cuda" / "native"
HEADER = NATIVE / "include" / "api.hpp"
BINDINGS = NATIVE / "bindings.cu"

_RETURN = r"(?:std::tuple<[^;{]+?>|void|int|float)"


def _normalize(params: str) -> str:
    """Collapse whitespace so formatting differences are not signature differences."""
    return re.sub(r"\s+", " ", params).strip()


def declarations() -> dict[str, str]:
    """Entry point -> parameter list, as declared in api.hpp."""
    pattern = rf"^({_RETURN})\s+(run_\w+)\s*\(([^;]*?)\);"
    return {
        m.group(2): _normalize(m.group(3))
        for m in re.finditer(pattern, HEADER.read_text(), re.S | re.M)
    }


def definitions() -> dict[str, tuple[str, str]]:
    """Entry point -> (file, parameter list), as defined across the .cu files."""
    pattern = rf"^({_RETURN})\s+(run_\w+)\s*\(([^{{]*?)\)\s*\{{"
    out: dict[str, tuple[str, str]] = {}
    for source in sorted(NATIVE.glob("*.cu")):
        for m in re.finditer(pattern, source.read_text(), re.S | re.M):
            out[m.group(2)] = (source.name, _normalize(m.group(3)))
    return out


def bound_names() -> set[str]:
    """The names bindings.cu exposes to Python."""
    return set(re.findall(r'"(run_\w+)"', BINDINGS.read_text()))


def test_the_parse_found_the_expected_surface() -> None:
    """Guard the guard: a regex that matched nothing would make every check below vacuous."""
    declared = declarations()
    assert len(declared) >= 10, f"only {len(declared)} declarations parsed from {HEADER.name}"
    for expected in ("run_dense", "run_multistream", "run_worklist_global", "run_dfa"):
        assert expected in declared, f"{expected} not parsed; the declaration style changed"


def test_every_declaration_has_a_definition() -> None:
    missing = sorted(set(declarations()) - set(definitions()))
    assert not missing, f"declared in api.hpp with no definition in any .cu: {missing}"


def test_every_definition_is_declared() -> None:
    """An entry point defined but not declared cannot be called from bindings.cu."""
    extra = sorted(set(definitions()) - set(declarations()))
    assert not extra, f"defined but absent from api.hpp: {extra}"


def test_signatures_match_exactly() -> None:
    """`uint64_t` and `unsigned long long` are different types on LP64 Linux.

    The header carries that warning; this is the check behind it. A mismatch links to an
    undefined symbol, which surfaces as an ImportError with no useful message.
    """
    declared, defined = declarations(), definitions()
    for name, params in sorted(declared.items()):
        if name not in defined:
            continue
        source, actual = defined[name]
        assert params == actual, (
            f"{name}: api.hpp and {source} disagree on the parameter list\n"
            f"  header: {params}\n  {source}: {actual}"
        )


def test_every_entry_point_is_exposed_to_python() -> None:
    """An unbound entry point is dead weight the Python side cannot reach."""
    unbound = sorted(set(declarations()) - bound_names())
    assert not unbound, f"declared but never bound in bindings.cu: {unbound}"


def test_nothing_is_bound_that_does_not_exist() -> None:
    """A typo in a binding name is a runtime AttributeError, far from its cause."""
    phantom = sorted(bound_names() - set(definitions()))
    assert not phantom, f"bound in bindings.cu with no definition: {phantom}"
