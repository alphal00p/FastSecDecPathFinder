"""QMCPy rank-1 lattice helpers for FSD's independent QMC path."""

from __future__ import annotations

import numpy as np


def is_power_of_two(value: int) -> bool:
    """Return true when ``value`` is a positive power of two."""
    n = int(value)
    return n > 0 and (n & (n - 1)) == 0


def qmcpy_shifted_lattice_points(
    *,
    dimension: int,
    n_points: int,
    shift_count: int,
    seed: int,
    order: str,
) -> np.ndarray:
    """Generate QMCPy shifted rank-1 lattice points.

    QMCPy lattices are naturally base-two rules.  FSD validates this before
    calling the helper so that the requested sample count is the actual sample
    count and comparisons are not muddied by implicit resizing.
    """
    if not is_power_of_two(int(n_points)):
        raise ValueError("QMCPy QMC requires --samples-per-iter to be a power of two")
    try:
        from qmcpy import Lattice
    except ImportError as exc:  # pragma: no cover - requirements.txt includes qmcpy.
        raise RuntimeError(
            "--sampling-mode qmc requires the 'qmcpy' package; run "
            "'.venv/bin/python -m pip install qmcpy'"
        ) from exc

    lattice = Lattice(
        int(dimension),
        replications=int(shift_count),
        seed=int(seed),
        randomize="SHIFT",
        order=str(order).upper().replace("-", " "),
    )
    raw_points = np.asarray(lattice(int(n_points)), dtype=float)
    if raw_points.ndim == 2:
        raw_points = raw_points[np.newaxis, :, :]
    if raw_points.shape != (int(shift_count), int(n_points), int(dimension)):
        raise RuntimeError(
            "QMCPy returned an unexpected lattice shape "
            f"{raw_points.shape}, expected {(int(shift_count), int(n_points), int(dimension))}"
        )
    return raw_points
