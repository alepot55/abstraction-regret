"""Command-line interface: ``gpufsm {env,list,verify,bench}``."""

from __future__ import annotations

import argparse
import platform
import sys

from .api import benchmark, run
from .backends import IMPORT_ERRORS
from .bench import sweep, write_csv
from .core.registry import (
    Backend,
    Kind,
    available_backends,
    list_kinds,
    list_techniques,
    unavailable_reason,
)
from .examples import EXAMPLES


def _techniques_line(backend: Backend, kind: Kind) -> str:
    return ", ".join(list_techniques(backend, kind)) or "-"


def _cmd_env(_: argparse.Namespace) -> int:
    print(f"python   : {platform.python_version()} ({sys.platform})")
    try:
        import numpy

        print(f"numpy    : {numpy.__version__}")
    except Exception:
        print("numpy    : (missing)")
    avail = available_backends()
    print(f"backends : {', '.join(b.value for b in avail) or '(none)'}")
    for b in avail:
        for k in list_kinds(b):
            print(f"  {b.value:7s} {k.value}: {_techniques_line(b, k)}")

    # Name the missing ones too, with the reason where there is one. "Which backends do I
    # have" is the question this command exists to answer, and a backend that is absent
    # because its import blew up looks exactly like one that was never installed.
    missing = [b for b in Backend if b not in avail]
    if missing:
        print(f"missing  : {', '.join(b.value for b in missing)}")
        for b in missing:
            why = IMPORT_ERRORS.get(b.value) or unavailable_reason(b)
            print(f"  {b.value:7s} {why}" if why else f"  {b.value:7s} not installed")
    return 0


def _cmd_list(_: argparse.Namespace) -> int:
    for b in Backend:
        status = "available" if b in available_backends() else "unavailable"
        kinds = list_kinds(b)
        if not kinds:
            print(f"{b.value:7s} [{status}]: -")
            continue
        for k in kinds:
            print(f"{b.value:7s} [{status}] {k.value}: {_techniques_line(b, k)}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Check every available backend and technique agrees with the CPU reference.

    Two properties this command must have and previously did not. It walks *every*
    registered NFA technique, not just each backend's default, because the default is
    whichever module imported first. And a kernel that raises is a failure to report, not a
    traceback to print: a verification command that dies on the first broken backend tells
    you less than one that finishes and lists what broke.

    ``scripts/oracle_gate.py`` remains the thorough check -- it covers DFAs, randomized
    inputs and required-backend enforcement. This one is the quick smoke test.
    """
    failures = 0
    backends = available_backends()
    for name, factory in EXAMPLES.items():
        nfa, inputs = factory()
        for data, expected in inputs:
            ref = run(nfa, data, backend=Backend.CPU)
            ok = ref.accepted == expected
            mark = "ok" if ok else "FAIL"
            if not ok:
                failures += 1
            print(f"[{mark}] cpu/{name:16s} {data!r:24s} -> {ref.accepted} (want {expected})")
            for b in backends:
                if b == Backend.CPU:
                    continue
                for tech in list_techniques(b, Kind.NFA):
                    label = f"{b.value}/{tech}"
                    try:
                        res = run(nfa, data, backend=b, technique=tech)
                    except Exception as exc:
                        failures += 1
                        print(f"     {label:28s} ERROR {type(exc).__name__}: {exc}")
                        continue
                    agree = res.matches(ref)
                    if not agree:
                        failures += 1
                    print(
                        f"     {label:28s} {'agrees' if agree else 'DIFFERS'} "
                        f"(accepted={res.accepted}, len={res.match_len})"
                    )
    print(f"\n{failures} failure(s).")
    return 1 if failures else 0


def _cmd_bench(args: argparse.Namespace) -> int:
    factory = EXAMPLES.get(args.example)
    if factory is None:
        print(f"unknown example {args.example!r}; choose from {list(EXAMPLES)}", file=sys.stderr)
        return 2
    nfa, _ = factory()
    data = (b"abcd" * (args.size // 4 + 1))[: args.size]
    backend = Backend(args.backend)
    stats = benchmark(nfa, data, backend=backend, repeats=args.repeats, warmup=args.warmup)
    print(
        f"{stats.backend}/{stats.technique}: "
        f"mean={stats.mean_ms:.4f} ms  std={stats.std_ms:.4f}  ci95=±{stats.ci95_ms:.4f}  "
        f"(n={stats.n}, accepted={stats.accepted}, len={stats.match_len})"
    )
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    factory = EXAMPLES.get(args.example)
    if factory is None:
        print(f"unknown example {args.example!r}; choose from {list(EXAMPLES)}", file=sys.stderr)
        return 2
    nfa, _ = factory()
    data = (b"abcd" * (args.size // 4 + 1))[: args.size]
    stats = sweep(nfa, data, repeats=args.repeats, warmup=args.warmup)
    for s in stats:
        print(
            f"{s.backend}/{s.technique:12s} mean={s.mean_ms:.4f} ms  "
            f"ci95=±{s.ci95_ms:.4f}  (accepted={s.accepted}, len={s.match_len})"
        )
    if args.out:
        path = write_csv(stats, args.out)
        print(f"\nwrote {len(stats)} rows -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gpufsm", description="GPU FSM/NFA processing toolkit.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("env", help="show environment and available backends").set_defaults(
        func=_cmd_env
    )
    sub.add_parser("list", help="list backends and techniques").set_defaults(func=_cmd_list)
    sub.add_parser("verify", help="check backends agree with the CPU reference").set_defaults(
        func=_cmd_verify
    )

    pb = sub.add_parser("bench", help="benchmark an example NFA")
    pb.add_argument("--backend", default="cpu", choices=[b.value for b in Backend])
    pb.add_argument("--example", default="ab_star_c_plus_d", choices=list(EXAMPLES))
    pb.add_argument("--size", type=int, default=4096, help="input length in bytes")
    pb.add_argument("--repeats", type=int, default=10)
    pb.add_argument("--warmup", type=int, default=3)
    pb.set_defaults(func=_cmd_bench)

    ps = sub.add_parser("sweep", help="benchmark all available backends/techniques -> CSV")
    ps.add_argument("--example", default="ab_star_c_plus_d", choices=list(EXAMPLES))
    ps.add_argument("--size", type=int, default=4096, help="input length in bytes")
    ps.add_argument("--repeats", type=int, default=10)
    ps.add_argument("--warmup", type=int, default=3)
    ps.add_argument("--out", default=None, help="CSV output path")
    ps.set_defaults(func=_cmd_sweep)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
