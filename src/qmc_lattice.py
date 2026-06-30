"""Rank-1 lattice helpers for FSD's independent QMC path.

FSD evaluates QMC samples itself.  The ``pysecdec-default`` backend below only
mirrors pySecDec's published CBC/PT generating-vector tables and lower-bound
rule for choosing the lattice size; it does not call pySecDec's QMC
integrator.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

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

_PYSECDEC_VECTOR_FUNCTIONS = ("cbcpt_dn1_100",)


def _rank1_shifted_points(
    *,
    z: np.ndarray,
    vector_size: int,
    dimension: int,
    shift_count: int,
    seed: int,
) -> np.ndarray:
    """Generate shifted rank-1 points from a concrete integer vector."""
    offsets = np.arange(int(vector_size), dtype=np.int64)[:, np.newaxis]
    lattice = np.mod(offsets * z[np.newaxis, :], int(vector_size)).astype(float) / float(vector_size)
    shifts = _shift_vectors(int(seed), int(shift_count), int(dimension))
    return np.mod(lattice[np.newaxis, :, :] + shifts[:, np.newaxis, :], 1.0)


@lru_cache(maxsize=4096)
def _shift_vectors(seed: int, shift_count: int, dimension: int) -> np.ndarray:
    """Return deterministic random shifts cached inside one worker process."""
    rng = np.random.default_rng(int(seed))
    return rng.random((int(shift_count), int(dimension)), dtype=float)


def _pysecdec_qmc_header_path() -> Path:
    """Return the installed pySecDecContrib QMC header path."""
    try:
        import pySecDecContrib
    except ImportError as exc:  # pragma: no cover - requirements include pySecDec.
        raise RuntimeError(
            "the pysecdec-default QMC backend needs pySecDecContrib's qmc.hpp "
            "generating-vector table to be installed"
        ) from exc
    header = Path(pySecDecContrib.dirname) / "include" / "qmc.hpp"
    if not header.is_file():
        raise RuntimeError(f"could not find pySecDecContrib QMC header at {header}")
    return header


def _extract_cpp_function_body(text: str, function_name: str) -> str:
    """Extract the C++ body for a zero-argument vector-table function."""
    signature = f"inline std::map<U,std::vector<U>> {function_name}()"
    start = text.find(signature)
    if start < 0:
        raise RuntimeError(f"could not find {function_name} in pySecDecContrib qmc.hpp")
    brace = text.find("{", start)
    if brace < 0:
        raise RuntimeError(f"could not parse {function_name} body in pySecDecContrib qmc.hpp")
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : index]
    raise RuntimeError(f"unterminated {function_name} body in pySecDecContrib qmc.hpp")


@lru_cache(maxsize=1)
def _pysecdec_vector_tables() -> dict[str, dict[int, tuple[int, ...]]]:
    """Parse pySecDecContrib's bundled CBC/PT vector tables once."""
    text = _pysecdec_qmc_header_path().read_text(encoding="utf-8", errors="replace")
    assignment = re.compile(r"generatingvectors\[(\d+)\]\s*=\s*\{([^}]*)\}\s*;")
    tables: dict[str, dict[int, tuple[int, ...]]] = {}
    for function_name in _PYSECDEC_VECTOR_FUNCTIONS:
        body = _extract_cpp_function_body(text, function_name)
        table: dict[int, tuple[int, ...]] = {}
        for match in assignment.finditer(body):
            n = int(match.group(1))
            vector = tuple(
                int(piece.strip())
                for piece in match.group(2).split(",")
                if piece.strip()
            )
            table[n] = vector
        if not table:
            raise RuntimeError(f"no generating vectors parsed for {function_name}")
        tables[function_name] = table
    return tables


def _pysecdec_default_vector_table(dimension: int) -> dict[int, tuple[int, ...]]:
    """Return the combined vector table selected by pySecDec for a dimension."""
    dim = int(dimension)
    if dim < 1:
        raise ValueError("QMC dimension must be positive")
    if dim > 100:
        raise ValueError("pysecdec-default QMC backend mirrors pySecDec's cbcpt_dn1_100 table and supports dimensions up to 100")
    tables = _pysecdec_vector_tables()
    # The installed pySecDec Qmc constructor initializes its default
    # generating-vector map as cbcpt_dn1_100().  Keep this backend deliberately
    # narrow: it is for parity checks, not for selecting another QMC rule.
    return dict(tables["cbcpt_dn1_100"])


def pysecdec_default_vector_info(*, dimension: int, n_points: int) -> tuple[int, tuple[int, ...]]:
    """Return the concrete pySecDec-style lattice size and vector."""
    requested = int(n_points)
    table = _pysecdec_default_vector_table(int(dimension))
    vector_size = next((n for n in sorted(table) if n >= requested), None)
    if vector_size is None:
        raise ValueError(
            "pysecdec-default backend has no vector for requested "
            f"n >= {requested} in dimension {int(dimension)}"
        )
    vector = table[int(vector_size)]
    if len(vector) < int(dimension):
        raise RuntimeError(
            "parsed pySecDec vector is shorter than requested dimension: "
            f"{len(vector)} < {int(dimension)}"
        )
    return int(vector_size), tuple(int(value) for value in vector[: int(dimension)])


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
    return _rank1_shifted_points(
        z=z,
        vector_size=int(vector_size),
        dimension=dim,
        shift_count=int(shift_count),
        seed=int(seed),
    )


def _cbcpt_dn1_vector_info(*, dimension: int, n_points: int) -> tuple[int, tuple[int, ...]]:
    """Return the concrete vector used by the small bundled dn1 subset."""
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
    return int(vector_size), tuple(_CBCPT_DN1_100_VECTORS[int(vector_size)][:dim])


def pysecdec_default_shifted_lattice_points(
    *,
    dimension: int,
    n_points: int,
    shift_count: int,
    seed: int,
) -> np.ndarray:
    """Generate shifted rank-1 points with pySecDec's default vector table."""
    dim = int(dimension)
    vector_size, vector = pysecdec_default_vector_info(dimension=dim, n_points=int(n_points))
    return _rank1_shifted_points(
        z=np.asarray(vector, dtype=np.int64),
        vector_size=int(vector_size),
        dimension=dim,
        shift_count=int(shift_count),
        seed=int(seed),
    )


def supports_lattice_slices(backend: str) -> bool:
    """Return true when a backend can stream rank-1 slices by flat index."""
    return backend in {"cbcpt-dn1-100", "pysecdec-default"}


def shifted_lattice_point_slice(
    *,
    backend: str,
    dimension: int,
    n_points: int,
    shift_count: int,
    seed: int,
    order: str,
    start: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate a flat slice of shifted rank-1 points.

    The returned points are shift-major, matching ``shifted_lattice_points``:
    index ``k`` corresponds to ``shift = k // n`` and lattice point
    ``k % n``.  This lets workers stream pySecDec-style QMC work packages
    without the master process materializing the full ``m*n`` array.
    """
    if str(order) != "linear":
        raise ValueError("streamed CBC/PT QMC slices currently require --qmc-order linear")
    dim = int(dimension)
    if backend == "cbcpt-dn1-100":
        vector_size, vector = _cbcpt_dn1_vector_info(dimension=dim, n_points=int(n_points))
    elif backend == "pysecdec-default":
        vector_size, vector = pysecdec_default_vector_info(dimension=dim, n_points=int(n_points))
    else:
        raise ValueError(f"backend {backend!r} does not support streamed QMC slices")
    total = int(vector_size) * int(shift_count)
    offset = max(int(start), 0)
    length = max(min(int(count), total - offset), 0)
    if length <= 0:
        return np.empty((0, dim), dtype=float), np.empty(0, dtype=np.int64), int(vector_size)
    flat = np.arange(offset, offset + length, dtype=np.int64)
    shift_indices = flat // int(vector_size)
    lattice_indices = flat - shift_indices * int(vector_size)
    z = np.asarray(vector, dtype=np.int64)
    lattice = (
        np.mod(lattice_indices[:, np.newaxis] * z[np.newaxis, :], int(vector_size)).astype(float)
        / float(vector_size)
    )
    shifts = _shift_vectors(int(seed), int(shift_count), int(dim))
    points = np.mod(lattice + shifts[shift_indices, :], 1.0)
    return points, shift_indices.astype(np.int64, copy=False), int(vector_size)


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
    if backend == "pysecdec-default":
        raise ValueError("pysecdec-default actual lattice size requires a support dimension")
    raise ValueError(f"unsupported QMC lattice backend {backend!r}")


def actual_lattice_point_count_for_dimension(*, backend: str, n_points: int, dimension: int) -> int:
    """Return the concrete point count when a backend depends on dimension."""
    if backend == "pysecdec-default":
        vector_size, _vector = pysecdec_default_vector_info(
            dimension=int(dimension),
            n_points=int(n_points),
        )
        return int(vector_size)
    return actual_lattice_point_count(backend=backend, n_points=int(n_points))


def max_lattice_point_count(*, backend: str) -> int | None:
    """Return the largest concrete point count supported by a finite backend."""
    if backend == "qmcpy":
        return None
    if backend == "cbcpt-dn1-100":
        return max(_CBCPT_DN1_100_VECTORS)
    if backend == "pysecdec-default":
        # The largest available key depends on dimension, so the caller should
        # use ``max_lattice_point_count_for_dimension`` when it needs a hard cap.
        return None
    raise ValueError(f"unsupported QMC lattice backend {backend!r}")


def max_lattice_point_count_for_dimension(*, backend: str, dimension: int) -> int | None:
    """Return the largest supported point count, including dimension-aware backends."""
    if backend == "pysecdec-default":
        return max(_pysecdec_default_vector_table(int(dimension)))
    return max_lattice_point_count(backend=backend)


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
    if backend == "pysecdec-default":
        return pysecdec_default_shifted_lattice_points(
            dimension=dimension,
            n_points=n_points,
            shift_count=shift_count,
            seed=seed,
        )
    raise ValueError(f"unsupported QMC lattice backend {backend!r}")
