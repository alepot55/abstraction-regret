"""The oracle gate over EVERY registered kernel, in one command, with no silent skips.

`pytest -m gpu` is the closest thing the repo had to this, and it has two properties that
make it the wrong thing to hand an artifact evaluator:

* On a box where a backend failed to build, every test for it `skipif`s to a pass. The suite
  exits 0 having verified nothing, and nothing in the output says so.
* It does not reach every registered `(Kind, Backend, technique)` triple: the Warp backend
  has no pytest coverage at all, and neither does the Triton DFA path. This walks the
  registry, so the count is whatever is registered rather than a number in a comment.

This walks the registry itself, so it cannot drift from what is registered, and it is strict
by default: a backend that is expected and missing is a FAILURE, not a skip.

    python scripts/oracle_gate.py                        # everything that is available
    python scripts/oracle_gate.py --require cuda,triton  # and fail if those are absent
    python scripts/oracle_gate.py --inputs 256           # deepen the sample past the default 64

Exit code 0 only when every triple checked agreed with the CPU reference and every required
backend was present. The oracle is `gpufsm.reference`, latch-first-match, comparing the pair
(accepted, match_len) that is the whole correctness contract.
"""

from __future__ import annotations

import argparse

from gpufsm.bench.generators import random_batch, random_byte_batch, random_nfa
from gpufsm.bench.oracle import OracleMismatch, require
from gpufsm.core.dfa import random_dfa
from gpufsm.core.registry import Backend, Kind, is_available, list_techniques


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--require", default="", help="comma-separated backends that MUST be available")
    ap.add_argument(
        "--inputs", type=int, default=64, help="inputs compared per triple (default 64)"
    )
    ap.add_argument("--nfa-states", type=int, default=48)
    ap.add_argument("--dfa-states", type=int, default=1024)
    ap.add_argument("--length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1048)
    args = ap.parse_args()

    required = {b.strip().lower() for b in args.require.split(",") if b.strip()}

    print("== backend availability ==")
    missing = []
    for b in Backend:
        kinds = [k.value for k in Kind if list_techniques(b, k)]
        avail = is_available(b)
        mark = "available" if avail else "MISSING"
        print(f"  {b.value:8} {mark:10} kinds={kinds or '[]'}")
        if b.value in required and not avail:
            missing.append(b.value)
    if missing:
        # The whole point: a green run on a box where the backend never built is the
        # failure mode this script exists to make impossible.
        print(f"\nFAIL: required backend(s) not available: {', '.join(missing)}")
        print("Nothing below would have been verified for them; they would have skipped silently.")
        return 2

    nfa = random_nfa(args.nfa_states, seed=args.seed)
    nfa_batch, _ = random_batch(args.inputs, args.length, seed=1)
    dfa = random_dfa(args.dfa_states, accept_prob=0.02, seed=0)
    dfa_batch, _ = random_byte_batch(args.inputs, args.length, seed=2)

    print(f"\n== oracle gate: {args.inputs} inputs x {args.length} bytes per triple ==")
    ok = fail = skipped = 0
    for kind in Kind:
        automaton = nfa if kind is Kind.NFA else dfa
        batch = nfa_batch if kind is Kind.NFA else dfa_batch
        for backend in Backend:
            if backend is Backend.CPU:
                continue  # the reference cannot disagree with itself
            for technique in list_techniques(backend, kind):
                label = f"{kind.value}/{backend.value}/{technique}"
                if not is_available(backend):
                    print(f"  skip  {label:34} backend not available")
                    skipped += 1
                    continue
                try:
                    require(automaton, batch, backend, technique, limit=args.inputs)
                except OracleMismatch as exc:
                    print(f"  FAIL  {label:34} {exc}")
                    fail += 1
                except Exception as exc:  # a kernel that will not even run is not a pass
                    print(f"  ERROR {label:34} {type(exc).__name__}: {exc}")
                    fail += 1
                else:
                    print(f"  ok    {label:34}")
                    ok += 1

    print(f"\n{ok} ok, {fail} failed, {skipped} skipped (backend absent)")
    if ok == 0 and fail == 0:
        # Zero checks is the vacuous green this script exists to prevent: on a box with no
        # GPU backend nothing registers, nothing is enumerated, and a plain exit 0 would
        # read exactly like a clean run.
        print("FAIL: nothing was checked. No GPU backend registered a single technique.")
        return 3
    if skipped and not required:
        print(
            "NOTE: skipped triples were NOT verified. Pass --require to turn that into a failure."
        )
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
