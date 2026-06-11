#!/usr/bin/env python3
"""Minimal reproducer for Symbolica batch ``jit_compile`` evaluator issues."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np
from symbolica import E, S


def symbolica_version() -> str:
    """Return the installed Symbolica version if package metadata is present."""
    try:
        return version("symbolica")
    except PackageNotFoundError:
        return "unknown"


def evaluate(expr_text: str, names: list[str], rows: np.ndarray, jit_compile: bool) -> np.ndarray:
    """Evaluate one expression over a batch with or without Symbolica JIT."""
    expr = E(expr_text)
    evaluator = expr.evaluator([S(name) for name in names], jit_compile=jit_compile)
    return np.asarray(evaluator.evaluate(rows), dtype=float)[:, 0]


def show_case(
    title: str,
    expr_text: str,
    names: list[str],
    rows: np.ndarray,
    expected: np.ndarray,
) -> bool:
    """Print one reproducible case and report whether JIT-only failure occurs."""
    print(f"\n{title}")
    print(f"expr: {expr_text}")
    print(f"params: {names}")
    print(f"rows:\n{rows}")
    print(f"expected:        {expected}")

    observed_jit = evaluate(expr_text, names, rows, jit_compile=True)
    observed_nojit = evaluate(expr_text, names, rows, jit_compile=False)
    print(f"jit_compile=True:  {observed_jit}")
    print(f"jit_compile=False: {observed_nojit}")

    jit_ok = np.allclose(observed_jit, expected)
    nojit_ok = np.allclose(observed_nojit, expected)
    print(f"jit matches expected:    {jit_ok}")
    print(f"nojit matches expected:  {nojit_ok}")
    return (not jit_ok) and nojit_ok


def main() -> int:
    """Run all known reproducer cases."""
    print(f"Symbolica version: {symbolica_version()}")

    rows2 = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=float,
    )
    rows5 = np.asarray(
        [
            [0.61881555, 0.20415282, 0.17703163, -1.0, 0.0],
            [0.5, 0.38461538, 0.11538462, -1.0, 0.0],
        ],
        dtype=float,
    )

    reproduced = [
        show_case(
            "Case 1: expression depends only on first variable",
            "a",
            ["a", "b"],
            rows2,
            rows2[:, 0],
        ),
        show_case(
            "Case 2: affine expression depends only on first variable",
            "1-a",
            ["a", "b"],
            rows2,
            1.0 - rows2[:, 0],
        ),
        show_case(
            "Case 3: U polynomial batch evaluation",
            "x0+x1+x2",
            ["x0", "x1", "x2", "s", "m2"],
            rows5,
            rows5[:, 0] + rows5[:, 1] + rows5[:, 2],
        ),
    ]

    if any(reproduced):
        print("\nBUG REPRODUCED: jit_compile=True mis-evaluates at least one batch case.")
        return 1

    print("\nNo mismatch observed. The installed Symbolica version may already contain a fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
