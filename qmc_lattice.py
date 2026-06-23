"""QMCPy rank-1 lattice helpers for FSD's independent QMC path."""

from __future__ import annotations

import numpy as np


_CBCPT_DN1_100_VECTORS: dict[int, tuple[int, ...]] = {
    # Small subset of the public CBC/PT dn1 vectors used by pySecDec's QMC
    # defaults.  The backend below implements the rank-1 rule directly in
    # Python/NumPy; it does not call or link against pySecDec's QMC integrator.
    1021: (1, 374, 421, 220, 482, 449),
    1123: (1, 438, 413, 324, 169, 121),
    4261: (1, 1648, 1902, 1757, 1533, 2032),
    8311: (1, 3068, 1811, 1128, 1964, 516),
    17807: (1, 6801, 7999, 5312, 2438, 2316),
    34687: (1, 14564, 10745, 12209, 7027, 16829),
    67601: (1, 24821, 28748, 19803, 25712, 17700),
}


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


def cbcpt_dn1_shifted_lattice_points(
    *,
    dimension: int,
    n_points: int,
    shift_count: int,
    seed: int,
) -> np.ndarray:
    """Generate shifted rank-1 lattice points from bundled CBC/PT vectors.

    ``n_points`` is interpreted like pySecDec's ``minn``: the first bundled
    vector with size ``n >= n_points`` is used.  This intentionally produces
    prime-sized rules such as 4261 for a requested 4096 points.
    """
    dim = int(dimension)
    if dim < 1:
        raise ValueError("QMC dimension must be positive")
    if dim > 6:
        raise ValueError("cbcpt-dn1-100 backend currently bundles vectors only up to dimension 6")
    requested = int(n_points)
    vector_size = next((n for n in sorted(_CBCPT_DN1_100_VECTORS) if n >= requested), None)
    if vector_size is None:
        raise ValueError(
            f"cbcpt-dn1-100 backend has no bundled vector for requested n >= {requested}"
        )
    z = np.asarray(_CBCPT_DN1_100_VECTORS[vector_size][:dim], dtype=np.int64)
    offsets = np.arange(vector_size, dtype=np.int64)[:, np.newaxis]
    lattice = np.mod(offsets * z[np.newaxis, :], vector_size).astype(float) / float(vector_size)
    rng = np.random.default_rng(int(seed))
    shifts = rng.random((int(shift_count), dim), dtype=float)
    return np.mod(lattice[np.newaxis, :, :] + shifts[:, np.newaxis, :], 1.0)


def actual_lattice_point_count(*, backend: str, n_points: int) -> int:
    """Return the concrete point count used by a QMC backend.

    Some backends interpret ``n_points`` as a lower bound.  In particular the
    bundled CBC/PT vectors use the first available prime-sized rule with
    ``n >= n_points``.  Progress reporting and target-time scheduling must use
    this concrete size rather than the nominal request.
    """
    requested = int(n_points)
    if backend == "qmcpy":
        if not is_power_of_two(requested):
            raise ValueError("QMCPy QMC requires --samples-per-iter to be a power of two")
        return requested
    if backend == "cbcpt-dn1-100":
        vector_size = next((n for n in sorted(_CBCPT_DN1_100_VECTORS) if n >= requested), None)
        if vector_size is None:
            raise ValueError(
                f"cbcpt-dn1-100 backend has no bundled vector for requested n >= {requested}"
            )
        return int(vector_size)
    raise ValueError(f"unsupported QMC lattice backend {backend!r}")


def shifted_lattice_points(
    *,
    backend: str,
    dimension: int,
    n_points: int,
    shift_count: int,
    seed: int,
    order: str,
) -> np.ndarray:
    """Dispatch to the configured independent shifted-lattice backend."""
    if backend == "qmcpy":
        return qmcpy_shifted_lattice_points(
            dimension=dimension,
            n_points=n_points,
            shift_count=shift_count,
            seed=seed,
            order=order,
        )
    if backend == "cbcpt-dn1-100":
        return cbcpt_dn1_shifted_lattice_points(
            dimension=dimension,
            n_points=n_points,
            shift_count=shift_count,
            seed=seed,
        )
    raise ValueError(f"unsupported QMC lattice backend {backend!r}")
