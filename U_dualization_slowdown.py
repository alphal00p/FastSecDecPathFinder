#!/usr/bin/env python3
"""Minimal Symbolica dualization slowdown reproducer.

Run from this directory with:

    python3 U_dualization_slowdown.py

The default reproduces a slow dualization observed in FastSecDec's massless
three-loop triple-box DOT example.  It does not import FastSecDec, pySecDec, or
any project module.  It only uses Symbolica, a hard-coded U polynomial, and the
dual multi-index shape requested by one generated six-axis endpoint sector.

On the local machine where this was extracted, scalar evaluator construction is
sub-millisecond, while dualizing the same evaluator with the 5120-coefficient
shape takes minutes.  The script prints each stage so the expensive call is
obvious when shared as a standalone reproducer.
"""

from __future__ import annotations

import argparse
import copy
from itertools import product
import os
from pathlib import Path
import sys
import textwrap
import time


try:
    from symbolica import E, S
except ImportError as exc:  # pragma: no cover - helper path for manual runs.
    # This makes the requested "python3 U_dualization_slowdown.py" work from an
    # FSD_v2 checkout where dependencies were installed only in .venv.
    venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
        print(
            f"symbolica is not importable with {sys.executable}; "
            f"re-executing with {venv_python}",
            flush=True,
        )
        os.execv(str(venv_python), [str(venv_python), *sys.argv])
    print("Could not import symbolica. Install symbolica or run inside FSD_v2/.venv.", file=sys.stderr)
    raise SystemExit(1) from exc


U_TRIPLE_BOX = (
    "x0*x2*x5+x0*x2*x6+x0*x2*x9+x0*x2*x4+x0*x5*x9+x0*x5*x3+"
    "x0*x5*x8+x0*x6*x9+x0*x6*x3+x0*x6*x8+x0*x9*x3+x0*x9*x4+"
    "x0*x9*x8+x0*x3*x4+x0*x4*x8+x2*x5*x8+x2*x5*x1+x2*x5*x7+"
    "x2*x6*x8+x2*x6*x1+x2*x6*x7+x2*x9*x8+x2*x9*x1+x2*x9*x7+"
    "x2*x4*x8+x2*x4*x1+x2*x4*x7+x5*x9*x8+x5*x9*x1+x5*x9*x7+"
    "x5*x3*x8+x5*x3*x1+x5*x3*x7+x5*x8*x1+x5*x8*x7+x6*x9*x8+"
    "x6*x9*x1+x6*x9*x7+x6*x3*x8+x6*x3*x1+x6*x3*x7+x6*x8*x1+"
    "x6*x8*x7+x9*x3*x8+x9*x3*x1+x9*x3*x7+x9*x4*x8+x9*x4*x1+"
    "x9*x4*x7+x9*x8*x1+x9*x8*x7+x3*x4*x8+x3*x4*x1+x3*x4*x7+"
    "x4*x8*x1+x4*x8*x7"
)


def _timed(label: str, fn):
    """Run one callable and print a flush-safe timing line."""
    print(f"\n[START] {label}", flush=True)
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    print(f"[DONE ] {label}: {elapsed:.6f} s", flush=True)
    return result, elapsed


def _shape_from_maxima(maxima: list[int]) -> list[tuple[int, ...]]:
    """Build Symbolica's dualize shape from per-axis maximum derivative orders."""
    return [tuple(mi) for mi in product(*[range(maximum + 1) for maximum in maxima])]


def _shape_summary(shape: list[tuple[int, ...]]) -> str:
    """Return a compact summary of the chosen dual shape."""
    rank = len(shape[0]) if shape else 0
    maxima = [max(mi[i] for mi in shape) for i in range(rank)] if rank else []
    max_total = max((sum(mi) for mi in shape), default=0)
    return (
        f"rank={rank}, coefficient_count={len(shape)}, "
        f"axis_maxima={maxima}, max_total_degree={max_total}"
    )


def _shape_excerpt(shape: list[tuple[int, ...]], limit: int = 12) -> str:
    """Show the beginning and end of a large multi-index list."""
    if len(shape) <= limit:
        return repr(shape)
    half = max(limit // 2, 1)
    return f"{shape[:half]!r} ... {shape[-half:]!r}"


def _build_scalar_evaluator(expr, params, *, verbose: bool):
    """Build a scalar evaluator, using verbose=True when supported."""
    if verbose:
        try:
            return expr.evaluator(params, jit_compile=False, verbose=True)
        except TypeError as exc:
            print(f"verbose=True is not supported by this Symbolica build: {exc}", flush=True)
    return expr.evaluator(params, jit_compile=False)


def main() -> int:
    """Run the standalone slowdown reproducer."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            Reproduce a slow Symbolica Evaluator.dualize(...) call for the
            triple-box U polynomial.  The default shape is the problematic
            six-axis shape with maxima 3,3,3,3,3,4 and 5120 coefficients.
            """
        ),
    )
    parser.add_argument(
        "--expression",
        choices=["u", "constant"],
        default="u",
        help="Expression to dualize. Default is the triple-box U polynomial.",
    )
    parser.add_argument(
        "--maxima",
        nargs="+",
        type=int,
        default=[3, 3, 3, 3, 3, 4],
        help="Per-axis maximum derivative orders used to build the dual shape.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny shape [1, 1] for quick environment checks.",
    )
    parser.add_argument(
        "--no-verbose",
        action="store_true",
        help="Do not pass verbose=True to the scalar evaluator builder.",
    )
    parser.add_argument(
        "--jit",
        action="store_true",
        help="Use jit_compile=True for the scalar evaluator build.",
    )
    parser.add_argument(
        "--skip-dualize",
        action="store_true",
        help="Only time parsing/scalar evaluator construction.",
    )
    args = parser.parse_args()

    maxima = [1, 1] if args.quick else args.maxima
    if any(maximum < 0 for maximum in maxima):
        raise SystemExit("--maxima entries must be non-negative integers")
    shape = _shape_from_maxima(maxima)

    expression_text = U_TRIPLE_BOX if args.expression == "u" else "1"
    expression_label = "triple-box U polynomial" if args.expression == "u" else "constant 1"

    print("Symbolica U dualization slowdown reproducer")
    print(f"  python executable : {sys.executable}")
    print(f"  expression        : {expression_label}")
    print(f"  expression terms  : {expression_text.count('+') + 1}")
    print(f"  expression length : {len(expression_text)}")
    print(f"  dual shape        : {_shape_summary(shape)}")
    print(f"  dual shape sample : {_shape_excerpt(shape)}")
    print("  evaluator params  : x0..x9, s12, s23")
    print("  scalar build      : Symbolica expression evaluator")
    print("  dualize call      : evaluator.dualize(shape)")
    if not args.skip_dualize and not args.quick:
        print("  note              : default dualization can take several minutes")

    expr, _ = _timed("parse expression with symbolica.E(...)", lambda: E(expression_text))
    params = [S(name) for name in [*(f"x{i}" for i in range(10)), "s12", "s23"]]

    def scalar_builder():
        if args.jit:
            return expr.evaluator(params, jit_compile=True)
        return _build_scalar_evaluator(expr, params, verbose=not args.no_verbose)

    evaluator, _ = _timed("build scalar U evaluator", scalar_builder)
    copied, _ = _timed("copy scalar evaluator", lambda: copy.copy(evaluator))
    if args.skip_dualize:
        print("\nSkipping dualize by request.")
        return 0

    _timed("dualize copied evaluator", lambda: copied.dualize([list(mi) for mi in shape]))
    print("\nFinished.  If this last line took minutes while scalar build was fast,")
    print("the slowdown is isolated to Evaluator.dualize(...) for this shape.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
