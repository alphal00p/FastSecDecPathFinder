"""Havana sampling driver and manual Laurent-coefficient accumulation.

Symbolica's Havana object is used as an adaptive sampler and grid owner.  The
code evaluates all Laurent coefficients itself, fills running statistics
manually, and trains Havana with a scalar finite-part importance function.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from colorama import Fore, Style
from symbolica import NumericalIntegrator

from definitions import HotPathTiming, IntegralRequest, IntegrationResult, SectorIntegrationResult, TargetDefinition
from formatting import (
    apply_global_convention,
    format_complex,
    format_complex_error,
    pull_value,
    selected_prefactor_values,
    summed_relative_error_percent,
)
from integrand import SectorProcessor, TopologyDefinition
from qmc_lattice import shifted_lattice_points
from sectors_generator import SectorDefinition
from utils import format_complex_uncertainty

try:
    import progressbar
except ImportError:  # pragma: no cover - requirements.txt includes progressbar2.
    progressbar = None


TARGET_PROGRESS_UNITS = 10_000


@dataclass
class RunningStats:
    """Streaming complex mean/error estimator for one Laurent coefficient."""

    count: int = 0
    sum_re: float = 0.0
    sum_im: float = 0.0
    sumsq_re: float = 0.0
    sumsq_im: float = 0.0
    mean_re_acc: float = 0.0
    mean_im_acc: float = 0.0
    m2_re: float = 0.0
    m2_im: float = 0.0

    def add(self, value: complex) -> None:
        """Add one weighted sample."""
        z = complex(value)
        self.count += 1
        self.sum_re += z.real
        self.sum_im += z.imag
        self.sumsq_re += z.real * z.real
        self.sumsq_im += z.imag * z.imag
        delta_re = z.real - self.mean_re_acc
        self.mean_re_acc += delta_re / self.count
        self.m2_re += delta_re * (z.real - self.mean_re_acc)
        delta_im = z.imag - self.mean_im_acc
        self.mean_im_acc += delta_im / self.count
        self.m2_im += delta_im * (z.imag - self.mean_im_acc)

    def add_many(self, values: np.ndarray) -> None:
        """Add a vector of weighted samples using NumPy reductions."""
        array = np.asarray(values, dtype=np.complex128)
        if array.size == 0:
            return
        count = int(array.size)
        sum_re = float(np.sum(array.real))
        sum_im = float(np.sum(array.imag))
        sumsq_re = float(np.sum(array.real * array.real))
        sumsq_im = float(np.sum(array.imag * array.imag))
        self.add_aggregate(count, sum_re, sum_im, sumsq_re, sumsq_im)

    def add_aggregate(
        self,
        count: int,
        sum_re: float,
        sum_im: float,
        sumsq_re: float,
        sumsq_im: float,
    ) -> None:
        """Add pre-reduced samples, allowing implicit zero contributions.

        Per-sector results are additive only if every sector accumulator sees
        every Monte Carlo sample, with zero contribution for samples assigned to
        another top-level discrete sector.  Passing the full batch count together
        with the nonzero sums implements exactly that without materializing
        dense zero arrays.
        """
        if count <= 0:
            return
        group_count = int(count)
        group_sum_re = float(sum_re)
        group_sum_im = float(sum_im)
        group_sumsq_re = float(sumsq_re)
        group_sumsq_im = float(sumsq_im)
        group_mean_re = group_sum_re / group_count
        group_mean_im = group_sum_im / group_count
        group_m2_re = max(group_sumsq_re - group_count * group_mean_re * group_mean_re, 0.0)
        group_m2_im = max(group_sumsq_im - group_count * group_mean_im * group_mean_im, 0.0)

        previous_count = self.count
        new_count = previous_count + group_count
        if previous_count == 0:
            self.mean_re_acc = group_mean_re
            self.mean_im_acc = group_mean_im
            self.m2_re = group_m2_re
            self.m2_im = group_m2_im
        else:
            delta_re = group_mean_re - self.mean_re_acc
            delta_im = group_mean_im - self.mean_im_acc
            self.mean_re_acc += delta_re * group_count / new_count
            self.mean_im_acc += delta_im * group_count / new_count
            self.m2_re += group_m2_re + delta_re * delta_re * previous_count * group_count / new_count
            self.m2_im += group_m2_im + delta_im * delta_im * previous_count * group_count / new_count

        self.count = new_count
        self.sum_re += float(sum_re)
        self.sum_im += float(sum_im)
        self.sumsq_re += float(sumsq_re)
        self.sumsq_im += float(sumsq_im)

    @property
    def mean(self) -> complex:
        """Return the current Monte Carlo mean."""
        if self.count == 0:
            return 0.0 + 0.0j
        return complex(self.sum_re / self.count, self.sum_im / self.count)

    @property
    def error(self) -> complex:
        """Return component-wise standard errors of the mean."""
        if self.count < 2:
            return complex(float("inf"), float("inf"))
        var_re = max(self.m2_re / (self.count - 1), 0.0)
        var_im = max(self.m2_im / (self.count - 1), 0.0)
        return complex(math.sqrt(var_re / self.count), math.sqrt(var_im / self.count))


@dataclass(frozen=True)
class EvaluationBatch:
    """Batch of Havana samples sent through the vectorized sector processor."""

    indices: np.ndarray
    sector_indices: np.ndarray
    coords: np.ndarray
    weights: np.ndarray
    sector_max_abs: np.ndarray
    max_weight_precision_xi: float


@dataclass(frozen=True)
class DemocraticBatch:
    """Uniform samples for one explicit sector in democratic mode."""

    sector_position: int
    sector_id: int
    coords: np.ndarray


@dataclass(frozen=True)
class DemocraticReducedBatch:
    """Pre-reduced democratic sector contribution returned by a worker."""

    sector_id: int
    count: int
    sums: np.ndarray
    sumsq_re: np.ndarray
    sumsq_im: np.ndarray
    max_abs: np.ndarray
    precision_counts: np.ndarray
    timing: HotPathTiming


@dataclass(frozen=True)
class QmcBatch:
    """One Korobov-transformed lattice chunk for one sector.

    ``shift_indices`` labels which randomized lattice shift produced each row.
    A QMC error estimate must be formed from full shift estimates rather than
    from the deterministic lattice points themselves.
    """

    sector_position: int
    sector_id: int
    coords: np.ndarray
    weights: np.ndarray
    shift_indices: np.ndarray
    shift_count: int
    coefficient_indices: tuple[int, ...]
    support_axes: tuple[int, ...]


@dataclass(frozen=True)
class QmcReducedBatch:
    """Per-shift sums for one QMC sector chunk returned by a worker."""

    sector_id: int
    coefficient_indices: tuple[int, ...]
    count: int
    counts_by_shift: np.ndarray
    sums_by_shift: np.ndarray
    max_abs: np.ndarray
    precision_counts: np.ndarray
    timing: HotPathTiming


_WORKER_SECTORS: list[SectorDefinition] | None = None
_WORKER_PROCESSOR: SectorProcessor | None = None
_PARENT_TOPOLOGY: TopologyDefinition | None = None
_PARENT_SECTORS: list[SectorDefinition] | None = None
_PARENT_REQUEST: IntegralRequest | None = None


def _make_sector_processor(topology: TopologyDefinition, request: IntegralRequest) -> SectorProcessor:
    """Construct a sector processor with runtime precision controls."""
    return SectorProcessor(
        topology,
        stability_threshold=request.stability_threshold,
        medium_precision_stability_threshold=request.medium_precision_stability_threshold,
        high_precision_stability_threshold=request.high_precision_stability_threshold,
        stability_precision=request.stability_precision,
        medium_precision_stability_precision=request.medium_precision_stability_precision,
        high_precision_stability_precision=request.high_precision_stability_precision,
        subtraction_backend=request.subtraction_backend,
    )


def _move_timing_precision_counts_to_high(timing: HotPathTiming, count: int) -> None:
    """Move already-counted rows to the high-precision tier.

    A large-weight guard first evaluates a sector row at its normal precision,
    then replaces selected rows with a forced high-precision value.  The lower
    precision attempt still contributes to timing, but precision sample counts
    should describe the final accepted precision tier.
    """
    remaining = max(int(count), 0)
    take = min(timing.ordinary_precision_samples, remaining)
    timing.ordinary_precision_samples -= take
    remaining -= take
    take = min(timing.stability_precision_samples, remaining)
    timing.stability_precision_samples -= take
    remaining -= take
    take = min(timing.medium_precision_samples, remaining)
    timing.medium_precision_samples -= take
    remaining -= take
    take = min(timing.high_precision_samples, remaining)
    timing.high_precision_samples -= take


def _terminate_executor_workers(executor: ProcessPoolExecutor) -> None:
    """Best-effort termination for running ProcessPoolExecutor workers.

    Python 3.12's public shutdown API can cancel queued futures, but it does
    not immediately stop workers already inside a heavy sector batch.  On a
    keyboard interrupt we prefer returning the accumulated partial result and
    terminating those workers rather than leaving orphan integrations alive.
    """
    processes = getattr(executor, "_processes", None)
    if not processes:
        return
    for process in list(processes.values()):
        if process is not None and process.is_alive():
            process.terminate()


def _init_worker_from_parent() -> None:
    """Attach fork-inherited prepared topology/sector state to a worker."""
    global _WORKER_SECTORS, _WORKER_PROCESSOR
    if _PARENT_TOPOLOGY is None or _PARENT_SECTORS is None or _PARENT_REQUEST is None:
        raise RuntimeError(
            "prepared FSD topology/sectors are unavailable in worker; "
            "DOT multiprocessing requires a fork-capable Python runtime"
        )
    _WORKER_SECTORS = _PARENT_SECTORS
    _WORKER_PROCESSOR = _make_sector_processor(_PARENT_TOPOLOGY, _PARENT_REQUEST)


def _evaluate_records_worker(
    batch: EvaluationBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HotPathTiming]:
    """Evaluate one batch in a worker process."""
    if _WORKER_PROCESSOR is None or _WORKER_SECTORS is None:
        raise RuntimeError("worker processor not initialized")
    return _evaluate_records(_WORKER_PROCESSOR, _WORKER_SECTORS, batch)


def _evaluate_democratic_batch_worker(batch: DemocraticBatch) -> DemocraticReducedBatch:
    """Evaluate one democratic sector batch in a worker process."""
    if _WORKER_PROCESSOR is None or _WORKER_SECTORS is None:
        raise RuntimeError("worker processor not initialized")
    return _evaluate_democratic_batch(_WORKER_PROCESSOR, _WORKER_SECTORS, batch)


def _evaluate_qmc_batch_worker(batch: QmcBatch) -> QmcReducedBatch:
    """Evaluate one QMC sector chunk in a worker process."""
    if _WORKER_PROCESSOR is None or _WORKER_SECTORS is None:
        raise RuntimeError("worker processor not initialized")
    return _evaluate_qmc_batch(_WORKER_PROCESSOR, _WORKER_SECTORS, batch)


def _evaluate_records(
    processor: SectorProcessor,
    sectors: list[SectorDefinition],
    batch: EvaluationBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HotPathTiming]:
    """Evaluate a batch and return weighted coefficients and training values."""
    hot_start = time.perf_counter()
    timing = HotPathTiming()
    if batch.indices.size == 0:
        return (
            np.empty(0, dtype=int),
            np.empty((0, processor.topology.coefficient_count), dtype=np.complex128),
            np.empty(0, dtype=float),
            np.zeros((len(sectors), 4), dtype=np.int64),
            timing,
        )

    indices = batch.indices
    sector_indices = batch.sector_indices
    coords = batch.coords
    weights = batch.weights
    weighted = np.zeros((indices.size, processor.topology.coefficient_count), dtype=np.complex128)
    training = np.zeros(indices.size, dtype=float)
    precision_counts = np.zeros((len(sectors), 4), dtype=np.int64)
    sector_max_abs = np.asarray(batch.sector_max_abs, dtype=float)
    max_weight_precision_xi = float(batch.max_weight_precision_xi)

    for sector_index in np.unique(sector_indices):
        sector_index = int(sector_index)
        sector = sectors[sector_index]
        mask = sector_indices == sector_index
        coeffs, train, sector_timing = processor.evaluate_batch(sector, coords[mask])
        sector_weights = weights[mask]
        weighted_coeffs = coeffs * sector_weights[:, np.newaxis]
        if (
            max_weight_precision_xi > 0.0
            and sector_max_abs.ndim == 2
            and sector_index < sector_max_abs.shape[0]
        ):
            current_max = sector_max_abs[sector_index]
            # Compare against the previously registered sector maximum.  Do not
            # fold the current batch maximum into the reference here: doing so
            # makes the largest row in every batch satisfy
            # row >= xi * max(current_batch), which turns the guard into a
            # near-deterministic high-precision replay for sparsely sampled
            # sectors.  The first finite row establishes the reference; later
            # rows that approach or exceed it are the ones worth replaying.
            active_orders = current_max > 0.0
            if np.any(active_orders):
                row_abs = np.abs(weighted_coeffs[:, active_orders])
                threshold = max_weight_precision_xi * current_max[active_orders]
                rescue_mask = np.any(row_abs >= threshold[np.newaxis, :], axis=1)
                if np.any(rescue_mask):
                    rescue_count = int(np.count_nonzero(rescue_mask))
                    _move_timing_precision_counts_to_high(sector_timing, rescue_count)
                    rescue_coeffs, rescue_train, rescue_timing = processor.evaluate_batch_at_precision(
                        sector,
                        coords[mask][rescue_mask],
                        processor.high_precision_stability_precision,
                    )
                    sector_timing.absorb(rescue_timing)
                    coeffs = np.array(coeffs, dtype=np.complex128, copy=True)
                    train = np.array(train, dtype=float, copy=True)
                    coeffs[rescue_mask] = rescue_coeffs
                    train[rescue_mask] = rescue_train
                    weighted_coeffs = coeffs * sector_weights[:, np.newaxis]
        timing.absorb(sector_timing)
        precision_counts[sector_index, :] += np.asarray(
            [
                sector_timing.ordinary_precision_samples,
                sector_timing.stability_precision_samples,
                sector_timing.medium_precision_samples,
                sector_timing.high_precision_samples,
            ],
            dtype=np.int64,
        )
        weighted[mask, :] = weighted_coeffs
        training[mask] = train

    hot_elapsed = time.perf_counter() - hot_start
    timing.add_python(hot_elapsed - timing.total_seconds)
    return indices, weighted, training, precision_counts, timing


def _evaluate_democratic_batch(
    processor: SectorProcessor,
    sectors: list[SectorDefinition],
    batch: DemocraticBatch,
) -> DemocraticReducedBatch:
    """Evaluate and reduce one uniform sector batch.

    Democratic mode estimates every sector integral separately.  The sample
    weight is therefore one: the aggregate integral is assembled as a sum of
    per-sector means rather than as a mean over a discrete Havana sector
    coordinate.
    """
    hot_start = time.perf_counter()
    timing = HotPathTiming()
    sector = sectors[int(batch.sector_position)]
    coords = np.asarray(batch.coords, dtype=float)
    if coords.size == 0:
        count = 0
        coeff_count = processor.topology.coefficient_count
        return DemocraticReducedBatch(
            sector_id=int(batch.sector_id),
            count=0,
            sums=np.zeros(coeff_count, dtype=np.complex128),
            sumsq_re=np.zeros(coeff_count, dtype=float),
            sumsq_im=np.zeros(coeff_count, dtype=float),
            max_abs=np.zeros(coeff_count, dtype=float),
            precision_counts=np.zeros(4, dtype=np.int64),
            timing=timing,
        )
    coeffs, _training, sector_timing = processor.evaluate_batch(sector, coords)
    timing.absorb(sector_timing)
    hot_elapsed = time.perf_counter() - hot_start
    timing.add_python(hot_elapsed - timing.total_seconds)
    return DemocraticReducedBatch(
        sector_id=int(batch.sector_id),
        count=int(coeffs.shape[0]),
        sums=np.sum(coeffs, axis=0),
        sumsq_re=np.sum(coeffs.real * coeffs.real, axis=0),
        sumsq_im=np.sum(coeffs.imag * coeffs.imag, axis=0),
        max_abs=np.max(np.abs(coeffs), axis=0),
        precision_counts=np.asarray(
            [
                sector_timing.ordinary_precision_samples,
                sector_timing.stability_precision_samples,
                sector_timing.medium_precision_samples,
                sector_timing.high_precision_samples,
            ],
            dtype=np.int64,
        ),
        timing=timing,
    )


def _evaluate_qmc_batch(
    processor: SectorProcessor,
    sectors: list[SectorDefinition],
    batch: QmcBatch,
) -> QmcReducedBatch:
    """Evaluate and reduce one Korobov-transformed QMC sector chunk."""
    hot_start = time.perf_counter()
    timing = HotPathTiming()
    sector = sectors[int(batch.sector_position)]
    coords = np.asarray(batch.coords, dtype=float)
    weights = np.asarray(batch.weights, dtype=float)
    shift_indices = np.asarray(batch.shift_indices, dtype=np.int64)
    coeff_count = processor.topology.coefficient_count
    shift_count = int(batch.shift_count)
    if coords.size == 0:
        return QmcReducedBatch(
            sector_id=int(batch.sector_id),
            coefficient_indices=tuple(int(index) for index in batch.coefficient_indices),
            count=0,
            counts_by_shift=np.zeros(shift_count, dtype=np.int64),
            sums_by_shift=np.zeros((shift_count, coeff_count), dtype=np.complex128),
            max_abs=np.zeros(coeff_count, dtype=float),
            precision_counts=np.zeros(4, dtype=np.int64),
            timing=timing,
        )

    coeffs, _training, sector_timing = processor.evaluate_batch(sector, coords)
    timing.absorb(sector_timing)
    weighted = np.zeros((coords.shape[0], coeff_count), dtype=np.complex128)
    selected = np.asarray(batch.coefficient_indices, dtype=np.int64)
    if selected.size:
        weighted[:, selected] = (
            np.asarray(coeffs, dtype=np.complex128)[:, selected]
            * weights[:, np.newaxis]
        )
    sums_by_shift = np.zeros((shift_count, coeff_count), dtype=np.complex128)
    counts_by_shift = np.bincount(shift_indices, minlength=shift_count).astype(np.int64)
    for coeff_index in selected:
        values = weighted[:, coeff_index]
        sums_by_shift[:, coeff_index] = (
            np.bincount(shift_indices, weights=values.real, minlength=shift_count)
            + 1j * np.bincount(shift_indices, weights=values.imag, minlength=shift_count)
        )
    hot_elapsed = time.perf_counter() - hot_start
    timing.add_python(hot_elapsed - timing.total_seconds)
    return QmcReducedBatch(
        sector_id=int(batch.sector_id),
        coefficient_indices=tuple(int(index) for index in batch.coefficient_indices),
        count=int(coords.shape[0]),
        counts_by_shift=counts_by_shift,
        sums_by_shift=sums_by_shift,
        max_abs=np.max(np.abs(weighted), axis=0),
        precision_counts=np.asarray(
            [
                sector_timing.ordinary_precision_samples,
                sector_timing.stability_precision_samples,
                sector_timing.medium_precision_samples,
                sector_timing.high_precision_samples,
            ],
            dtype=np.int64,
        ),
        timing=timing,
    )


def split_evaluation_batches(
    indices: np.ndarray,
    sector_indices: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    default_batches: int,
) -> list[EvaluationBatch]:
    """Split one Havana iteration into vectorized processor tasks."""
    n_samples = int(indices.size)
    if n_samples == 0:
        return []
    if batch_size <= 0:
        n_batches = max(int(default_batches), 1)
        step = max(math.ceil(n_samples / n_batches), 1)
    else:
        step = batch_size
    return [
        EvaluationBatch(
            indices=indices[start : start + step],
            sector_indices=sector_indices[start : start + step],
            coords=coords[start : start + step, :],
            weights=weights[start : start + step],
            sector_max_abs=np.zeros((0, 0), dtype=float),
            max_weight_precision_xi=0.0,
        )
        for start in range(0, n_samples, step)
    ]


def democratic_batches(
    request: IntegralRequest,
    active_sector_ids: list[int],
    active_sectors: list[SectorDefinition],
) -> list[DemocraticBatch]:
    """Create deterministic uniform batches covering every active sector.

    Batches are scheduled round-robin across sectors.  This matters for
    diagnostics: if early sector ids are expensive, a sector-by-sector schedule
    can spend the whole watchdog budget before later sectors receive any
    points, even though the final estimator would be democratic.
    """
    samples_per_sector = int(request.democratic_samples_per_sector)
    step = samples_per_sector if request.batch_size <= 0 else int(request.batch_size)
    step = max(step, 1)
    batches: list[DemocraticBatch] = []
    rngs = [
        np.random.default_rng(int(request.seed) + 1_000_003 * int(sector_id))
        for sector_id in active_sector_ids
    ]
    remaining_by_sector = [samples_per_sector for _ in active_sector_ids]
    while any(remaining > 0 for remaining in remaining_by_sector):
        for sector_position, (sector_id, sector) in enumerate(
            zip(active_sector_ids, active_sectors)
        ):
            remaining = remaining_by_sector[sector_position]
            if remaining <= 0:
                continue
            count = min(step, remaining)
            batches.append(
                DemocraticBatch(
                    sector_position=int(sector_position),
                    sector_id=int(sector_id),
                    coords=rngs[sector_position].random(
                        (count, sector.integration_dim),
                        dtype=float,
                    ),
                )
            )
            remaining_by_sector[sector_position] -= count
    return batches


def korobov_transform(points: np.ndarray, alpha: int) -> tuple[np.ndarray, np.ndarray]:
    """Apply the product Korobov periodization map to QMC points.

    For integer ``alpha`` the one-dimensional map is the polynomial

      phi(y) = C int_0^y u^alpha (1-u)^alpha du,

    with ``C = 1/B(alpha+1, alpha+1)``.  The returned weight is the product of
    ``phi'(y_j)`` over dimensions, which must multiply the sector integrand.
    """
    a = int(alpha)
    if a < 1:
        raise ValueError("Korobov alpha must be positive")
    y = np.asarray(points, dtype=float)
    y_clipped = np.clip(y, 0.0, 1.0)
    norm = float(math.factorial(2 * a + 1) / (math.factorial(a) * math.factorial(a)))
    phi = np.zeros_like(y_clipped)
    for k in range(a + 1):
        coefficient = norm * ((-1.0) ** k) * math.comb(a, k) / float(a + k + 1)
        phi += coefficient * np.power(y_clipped, a + k + 1)
    omega = norm * np.power(y_clipped, a) * np.power(1.0 - y_clipped, a)
    jacobian = np.prod(omega, axis=1)
    return phi, jacobian


def qmc_support_groups(
    topology: TopologyDefinition,
    sector: SectorDefinition,
    global_support_dims: tuple[int, ...] | None = None,
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Return coefficient groups and their effective QMC support axes.

    pySecDec does not always periodize every Laurent coefficient in the full
    sector dimension.  The deepest endpoint pole is a boundary term: all
    singular axes have been projected to zero, so only the nonsingular axes
    remain as integration variables.  Periodizing such a term in the full
    dimension leaves the integral invariant, but it multiplies by dummy
    Korobov weights and can inflate the random-shift variance by many orders.

    The group layout here preserves the ordinary FSD sector integrand while
    matching pySecDec's lower-support QMC transform for the deepest pole.  The
    higher Laurent coefficients keep the full support because they contain
    plus-distribution/logarithmic pieces depending on endpoint coordinates.
    """
    full_axes = tuple(range(int(sector.integration_dim)))
    full_dim = len(full_axes)
    endpoint_depth = len(sector.singular_axes)
    sector_min_order = -endpoint_depth if endpoint_depth else 0
    local_support: dict[int, tuple[int, ...]] = {}
    for index, order in enumerate(topology.laurent_orders):
        if int(order) >= sector_min_order:
            local_support[index] = full_axes
    if sector.singular_axes:
        deepest_order = -len(sector.singular_axes)
        if deepest_order in topology.laurent_orders:
            deepest_index = topology.laurent_orders.index(deepest_order)
            boundary_axes = tuple(
                axis for axis in full_axes if axis not in set(int(a) for a in sector.singular_axes)
            )
            if boundary_axes:
                local_support[deepest_index] = boundary_axes

    grouped: dict[tuple[int, ...], list[int]] = {}
    for index in range(topology.coefficient_count):
        if index not in local_support:
            continue
        axes = local_support[index]
        if global_support_dims is not None:
            global_dim = int(global_support_dims[index])
            if global_dim >= full_dim:
                axes = full_axes
            elif len(axes) != global_dim:
                # This mixed-support case has not appeared in the validation
                # examples.  Lifting to full support preserves correctness and
                # avoids silently guessing pySecDec's local coordinate lift.
                axes = full_axes
        grouped.setdefault(axes, []).append(index)
    return [(tuple(indices), axes) for axes, indices in grouped.items()]


def qmc_global_support_dims(
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
) -> tuple[int, ...]:
    """Return pySecDec-style aggregate support dimensions per Laurent column."""
    if not sectors:
        return tuple()
    full_dim = int(sectors[0].integration_dim)
    dims = [0 for _ in range(topology.coefficient_count)]
    for sector in sectors:
        groups = qmc_support_groups(topology, sector, None)
        for coeff_indices, axes in groups:
            for index in coeff_indices:
                dims[index] = max(dims[index], len(axes))
    return tuple(dim if dim > 0 else full_dim for dim in dims)


def qmc_batches_for_sector(
    request: IntegralRequest,
    sector_position: int,
    sector_id: int,
    sector: SectorDefinition,
    iteration: int,
    raw_points: np.ndarray | None = None,
    support_axes: tuple[int, ...] | None = None,
    coefficient_indices: tuple[int, ...] | None = None,
) -> list[QmcBatch]:
    """Generate Korobov-transformed shifted-lattice chunks for one sector."""
    n_points = int(request.samples_per_iter)
    shift_count = int(request.qmc_shifts)
    axes = tuple(range(int(sector.integration_dim))) if support_axes is None else tuple(int(axis) for axis in support_axes)
    coeffs = tuple(range(0)) if coefficient_indices is None else tuple(int(index) for index in coefficient_indices)
    if not coeffs:
        raise ValueError(f"{sector.name}: empty QMC coefficient group")
    support_dim = len(axes)
    if support_dim <= 0:
        raise ValueError(f"{sector.name}: zero-dimensional QMC support is not implemented")
    if raw_points is None:
        seed = int(request.seed) + 1_000_003 * int(sector_id) + 97_003 * int(iteration)
        raw_points = shifted_lattice_points(
            backend=str(request.qmc_lattice_backend),
            dimension=int(support_dim),
            n_points=n_points,
            shift_count=shift_count,
            seed=seed,
            order=str(request.qmc_order),
        )
    else:
        expected_prefix = (shift_count, int(support_dim))
        if raw_points.ndim != 3 or raw_points.shape[0] != expected_prefix[0] or raw_points.shape[2] != expected_prefix[1]:
            raise RuntimeError(
                "shared QMC lattice has unexpected shape "
                f"{tuple(raw_points.shape)}, expected shift/dimension {expected_prefix}"
            )

    actual_n_points = int(raw_points.shape[1])
    step = actual_n_points * shift_count if request.batch_size <= 0 else int(request.batch_size)
    step = max(step, 1)
    flat_points = raw_points.reshape(shift_count * actual_n_points, support_dim)
    support_coords, weights = korobov_transform(flat_points, int(request.qmc_korobov_alpha))
    coords = np.full(
        (shift_count * actual_n_points, int(sector.integration_dim)),
        0.5,
        dtype=float,
    )
    coords[:, list(axes)] = support_coords
    shift_indices = np.repeat(np.arange(shift_count, dtype=np.int64), actual_n_points)
    return [
        QmcBatch(
            sector_position=int(sector_position),
            sector_id=int(sector_id),
            coords=coords[start : start + step, :],
            weights=weights[start : start + step],
            shift_indices=shift_indices[start : start + step],
            shift_count=shift_count,
            coefficient_indices=coeffs,
            support_axes=axes,
        )
        for start in range(0, shift_count * actual_n_points, step)
    ]


def format_progress_scalar(value: float | None) -> str:
    """Compact scalar formatter for live progress fields."""
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-2 or abs(value) >= 1.0e3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def format_progress_percent(value: float | None) -> str:
    """Format a live percent with scientific notation below 0.001%."""
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    if value == 0.0:
        return "0.00"
    abs_value = abs(value)
    if abs_value < 1.0e-3:
        return f"{value:.2e}"
    if abs_value < 0.01:
        return f"{value:.5f}"
    if abs_value < 0.1:
        return f"{value:.4f}"
    if abs_value < 1.0:
        return f"{value:.3f}"
    if abs_value < 10.0:
        return f"{value:.2f}"
    if abs_value < 100.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def format_sample_count(value: int) -> str:
    """Format sample counts with 3-significant-digit K/M/B units."""
    count = int(value)
    if abs(count) < 1000:
        return str(count)
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(count) >= scale:
            scaled = count / scale
            if abs(scaled) >= 100.0:
                return f"{scaled:.0f}{suffix}"
            if abs(scaled) >= 10.0:
                return f"{scaled:.1f}{suffix}"
            return f"{scaled:.2f}{suffix}"
    return str(count)


def max_sample_budget(request: IntegralRequest) -> int | None:
    """Return the finite maximum sample budget, or None when unbounded."""
    if request.max_iter < 0:
        return None
    return request.max_iter * request.samples_per_iter


def format_max_iter(request: IntegralRequest) -> str:
    """Format the iteration budget for the progress bar."""
    return "∞" if request.max_iter < 0 else str(request.max_iter)


def format_sample_target(request: IntegralRequest) -> str:
    """Format the sample budget for the progress bar."""
    budget = max_sample_budget(request)
    return "∞" if budget is None else format_sample_count(budget)


def format_duration(seconds: float) -> str:
    """Format a non-negative duration as HH:MM:SS."""
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(seconds: float | None) -> str:
    """Format an ETA, preserving non-finite sentinel values."""
    if seconds is None:
        return "n/a"
    if math.isnan(seconds):
        return "nan"
    if math.isinf(seconds):
        return "inf"
    return format_duration(seconds)


def progress_color(text: str, color: str) -> str:
    """Apply ANSI color to a progress-bar label."""
    return f"{color}{text}{Style.RESET_ALL}"


def profile_text(timing: HotPathTiming) -> str:
    """Render the live profile tuple ``(python | evaluator | havana)``."""
    return (
        f"({100.0 * timing.python_overhead_fraction:.2f}% | "
        f"{100.0 * timing.evaluator_fraction:.2f}% | "
        f"{100.0 * timing.havana_fraction:.2f}%)"
    )


def avg_eval_us_per_sample_per_worker(timing: HotPathTiming, sample_count: int) -> float:
    """Normalize worker-summed evaluator time by accepted samples."""
    if sample_count <= 0:
        return 0.0
    # EvalT is already worker-summed: each worker reports its evaluator wall time
    # and the master absorbs all worker timings. Dividing by total samples gives
    # the average evaluator time for one sample on one worker.
    return timing.eval_seconds * 1.0e6 / sample_count


def make_progress_bar(request: IntegralRequest) -> Any | None:
    """Create the colored progress bar unless the output mode disables it."""
    if request.json or request.no_progress or progressbar is None:
        return None
    sample_budget = max_sample_budget(request)
    target_mode = request.target_rel_accuracy is not None
    unbounded_mode = sample_budget is None and not target_mode
    max_value = (
        progressbar.UnknownLength
        if unbounded_mode
        else TARGET_PROGRESS_UNITS
        if target_mode
        else sample_budget
    )
    live_widget = progressbar.FormatCustomText(
        (
            f"{progress_color('it', Fore.CYAN)}:%(iteration)s/%(max_iter)s "
            f"{progress_color('smpl', Fore.CYAN)}:%(samples)s/%(target_samples)s "
            f"{progress_color('err%%', Fore.GREEN)}:%(relerr)s "
            f"{progress_color('val', Fore.MAGENTA)}[{request.progress_value_order}]:%(value)s "
            f"{progress_color('pull', Fore.YELLOW)}: %(pull)s "
            f"{progress_color('t', Fore.BLUE)}:%(elapsed)s "
            f"{progress_color('eta', Fore.BLUE)}:%(eta)s "
            f"{progress_color('eval μs/smpl/wkr', Fore.MAGENTA)}:%(avg_us)s "
            f"{progress_color('prof py|eval|hav', Fore.CYAN)}:%(profile)s"
        ),
        {
            "iteration": "0",
            "max_iter": format_max_iter(request),
            "samples": "0",
            "target_samples": format_sample_target(request),
            "relerr": "n/a",
            "value": "n/a",
            "pull": "n/a",
            "elapsed": "00:00:00",
            "eta": "n/a",
            "avg_us": "n/a",
            "profile": "n/a",
        },
    )
    widgets = [
        progress_color(f"integrating {request.integral} ", Fore.CYAN),
        progressbar.AnimatedMarker() if unbounded_mode else progressbar.Percentage(),
        " ",
        "" if unbounded_mode else progressbar.Bar(),
        "" if unbounded_mode else " ",
        live_widget,
        " ",
    ]
    bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets)
    bar.fsd_live_widget = live_widget
    return bar


def live_progress_metrics(
    request: IntegralRequest,
    stats: list[RunningStats],
    target: TargetDefinition | None,
) -> tuple[float, float | None]:
    """Compute live summed relative error and max pull for display."""
    sector_coeffs = [stat.mean for stat in stats]
    sector_errors = [stat.error for stat in stats]
    raw_coeffs, raw_errors = apply_global_convention(sector_coeffs, sector_errors, request)
    display_coeffs, display_errors, _display_bench, _ = selected_prefactor_values(
        request, raw_coeffs, raw_errors, None
    )
    relerr = summed_relative_error_percent(display_coeffs, display_errors)
    if target is None:
        return relerr, None
    pulls = [
        pull_value(coeff - ref, err)
        for coeff, err, ref in zip(display_coeffs, display_errors, target.coefficients)
    ]
    numeric_pulls = [pull for pull in pulls if pull is not None]
    return relerr, max(numeric_pulls) if numeric_pulls else None


def _epsilon_order_from_label(label: str) -> int:
    """Parse labels like eps^0 or eps^-2."""
    text = str(label).strip()
    if text == "eps^0":
        return 0
    if text.startswith("eps^"):
        return int(text[4:])
    raise ValueError(f"unsupported epsilon order label {label!r}")


def live_progress_value(
    request: IntegralRequest,
    stats: list[RunningStats],
) -> str:
    """Return the selected Laurent coefficient with MC uncertainty."""
    sector_coeffs = [stat.mean for stat in stats]
    sector_errors = [stat.error for stat in stats]
    raw_coeffs, raw_errors = apply_global_convention(sector_coeffs, sector_errors, request)
    display_coeffs, display_errors, _display_bench, _ = selected_prefactor_values(
        request, raw_coeffs, raw_errors, None
    )
    try:
        selected_order = _epsilon_order_from_label(request.progress_value_order)
    except ValueError:
        selected_order = request.max_eps_order
    min_order = request.max_eps_order - len(display_coeffs) + 1
    index = selected_order - min_order
    if index < 0 or index >= len(display_coeffs):
        index = len(display_coeffs) - 1
    return format_complex_uncertainty(display_coeffs[index], display_errors[index])


def target_progress_fraction(relerr_percent: float, target_percent: float | None) -> float | None:
    """Estimate target completion fraction using error proportional to 1/sqrt(N)."""
    if target_percent is None:
        return None
    if target_percent <= 0.0:
        return None
    if relerr_percent == 0.0:
        return 1.0
    if not math.isfinite(relerr_percent):
        return 0.0
    return min((target_percent / relerr_percent) ** 2, 1.0)


def sample_budget_progress_fraction(request: IntegralRequest, samples_done: int) -> float | None:
    """Return the fraction of a finite sample budget already consumed."""
    budget = max_sample_budget(request)
    if budget is None or budget <= 0:
        return None
    return min(max(samples_done / budget, 0.0), 1.0)


def estimate_eta_seconds(
    request: IntegralRequest,
    samples_done: int,
    elapsed_seconds: float,
    relerr_percent: float,
) -> float | None:
    """Estimate ETA from target accuracy and finite sample budget candidates."""
    if samples_done <= 0 or elapsed_seconds <= 0.0:
        return None
    sample_rate = samples_done / elapsed_seconds
    if sample_rate <= 0.0:
        return None
    target = request.target_rel_accuracy
    budget = max_sample_budget(request)
    budget_eta = None
    if budget is not None:
        remaining_budget_samples = max(budget - samples_done, 0)
        budget_eta = remaining_budget_samples / sample_rate
    if target is None:
        return budget_eta
    if target <= 0.0:
        return budget_eta
    target_eta = None
    if relerr_percent == 0.0:
        target_eta = 0.0
    elif math.isfinite(relerr_percent):
        estimated_target_samples = samples_done * (relerr_percent / target) ** 2
        remaining = max(estimated_target_samples - samples_done, 0.0)
        target_eta = remaining / sample_rate

    candidates = [eta for eta in (target_eta, budget_eta) if eta is not None]
    return min(candidates) if candidates else None


def format_relerr_with_target(request: IntegralRequest, relerr_percent: float) -> str:
    """Format live relative error and append a target if configured."""
    relerr_text = format_progress_percent(relerr_percent)
    if request.target_rel_accuracy is None:
        return relerr_text
    target_text = format_progress_percent(request.target_rel_accuracy)
    return relerr_text + " " + maybe_blue_slash_target(target_text)


def maybe_blue_slash_target(target_text: str) -> str:
    """Color the target suffix shown next to ``err%``."""
    return progress_color(f"/ {target_text}", Fore.BLUE)


def update_progress_bar(
    bar: Any | None,
    request: IntegralRequest,
    stats: list[RunningStats],
    target: TargetDefinition | None,
    iteration: int,
    elapsed_seconds: float,
    avg_eval_us_per_sample_per_worker: float,
    timing: HotPathTiming,
) -> None:
    """Refresh progress widgets from the current accumulators."""
    if bar is None:
        return
    relerr, pull = live_progress_metrics(request, stats, target)
    eta = estimate_eta_seconds(request, stats[0].count, elapsed_seconds, relerr)
    live_widget = getattr(bar, "fsd_live_widget", None)
    if live_widget is not None:
        live_widget.update_mapping(
            iteration=str(iteration),
            samples=format_sample_count(stats[0].count),
            relerr=format_relerr_with_target(request, relerr),
            value=live_progress_value(request, stats),
            pull="N/A" if pull is None else f"{pull:.2f}σ",
            elapsed=format_duration(elapsed_seconds),
            eta=format_eta(eta),
            avg_us=format_progress_scalar(avg_eval_us_per_sample_per_worker),
            profile=profile_text(timing),
        )
    if request.target_rel_accuracy is None:
        if max_sample_budget(request) is None:
            progress_value = stats[0].count
        else:
            progress_value = min(stats[0].count, max_sample_budget(request) or stats[0].count)
    else:
        target_fraction = target_progress_fraction(relerr, request.target_rel_accuracy)
        sample_fraction = sample_budget_progress_fraction(request, stats[0].count)
        finite_fractions = [
            fraction
            for fraction in (target_fraction, sample_fraction)
            if fraction is not None and math.isfinite(fraction)
        ]
        fraction = max(finite_fractions) if finite_fractions else 0.0
        progress_value = int(TARGET_PROGRESS_UNITS * (fraction if fraction is not None else 0.0))
    bar.update(progress_value, force=True)


def update_progress_bar_timed(
    bar: Any | None,
    request: IntegralRequest,
    stats: list[RunningStats],
    target: TargetDefinition | None,
    iteration: int,
    elapsed_seconds: float,
    avg_eval_us_per_sample_per_worker: float,
    timing: HotPathTiming,
) -> None:
    """Refresh the progress bar and charge rendering work to PythonT."""
    if bar is None:
        return
    progress_start = time.perf_counter()
    update_progress_bar(
        bar,
        request,
        stats,
        target,
        iteration,
        elapsed_seconds,
        avg_eval_us_per_sample_per_worker,
        timing,
    )
    timing.add_python(time.perf_counter() - progress_start)


def target_accuracy_reached(
    request: IntegralRequest,
    stats: list[RunningStats],
    target: TargetDefinition | None,
    iteration: int,
) -> bool:
    """Return whether the requested relative target has been reached."""
    if request.target_rel_accuracy is None or iteration < request.min_iter:
        return False
    relerr, _ = live_progress_metrics(request, stats, target)
    return math.isfinite(relerr) and relerr <= request.target_rel_accuracy


def _stats_from_reduced_batches(
    sectors: list[SectorDefinition],
    active_sector_ids: list[int],
    topology: TopologyDefinition,
    sector_stats: list[list[RunningStats]],
    sector_hits: list[int],
    sector_precision_counts: list[dict[str, int]],
    sector_timing: list[HotPathTiming],
    sector_max_abs: list[np.ndarray],
    total_timing: HotPathTiming,
    start_time: float,
    interrupted: bool,
    diagnostics: dict[str, Any],
    aggregate_stats_override: list[RunningStats] | None = None,
) -> IntegrationResult:
    """Materialize a democratic integration result from per-sector statistics."""
    coeff_count = topology.coefficient_count
    aggregate_coeffs: list[complex] = []
    aggregate_errors: list[complex] = []
    if aggregate_stats_override is not None:
        for stat in aggregate_stats_override:
            aggregate_coeffs.append(stat.mean)
            aggregate_errors.append(stat.error)
    else:
        for coeff_index in range(coeff_count):
            coeff = sum(
                sector_stats[sector_id][coeff_index].mean
                for sector_id in active_sector_ids
            )
            err_re2 = sum(
                sector_stats[sector_id][coeff_index].error.real ** 2
                for sector_id in active_sector_ids
                if math.isfinite(sector_stats[sector_id][coeff_index].error.real)
            )
            err_im2 = sum(
                sector_stats[sector_id][coeff_index].error.imag ** 2
                for sector_id in active_sector_ids
                if math.isfinite(sector_stats[sector_id][coeff_index].error.imag)
            )
            aggregate_coeffs.append(coeff)
            aggregate_errors.append(complex(math.sqrt(err_re2), math.sqrt(err_im2)))

    total_samples = sum(sector_hits[sector_id] for sector_id in active_sector_ids)
    timing_per_sector: list[dict[str, Any]] = []
    for sector_id in active_sector_ids:
        hits = int(sector_hits[sector_id])
        timing = sector_timing[sector_id]
        avg_eval = avg_eval_us_per_sample_per_worker(timing, hits)
        timing_per_sector.append(
            {
                "sector_id": sector_id,
                "name": sectors[sector_id].name,
                "samples": hits,
                "avg_eval_us_per_sample": avg_eval,
                "eval_seconds": timing.eval_seconds,
                "python_seconds": timing.python_seconds,
                "profile": {
                    "python_fraction": timing.python_overhead_fraction,
                    "evaluator_fraction": timing.evaluator_fraction,
                    "havana_fraction": timing.havana_fraction,
                },
                "max_abs_weight": float(np.max(sector_max_abs[sector_id])) if hits else 0.0,
                "max_abs_by_order": [float(value) for value in sector_max_abs[sector_id]],
            }
        )
    nonzero_timings = [
        row for row in timing_per_sector if int(row["samples"]) > 0
    ]
    if nonzero_timings:
        min_row = min(nonzero_timings, key=lambda row: float(row["avg_eval_us_per_sample"]))
        max_row = max(nonzero_timings, key=lambda row: float(row["avg_eval_us_per_sample"]))
        max_weight_row = max(nonzero_timings, key=lambda row: float(row["max_abs_weight"]))
        diagnostics.update(
            {
                "min_avg_eval_us_per_sample_sector": min_row,
                "max_avg_eval_us_per_sample_sector": max_row,
                "max_abs_weight_sector": max_weight_row,
            }
        )
    diagnostics["sector_timing_summary"] = timing_per_sector
    sampling_mode = str(diagnostics.get("sampling_mode", "democratic"))
    return IntegrationResult(
        raw_sector_coeffs=aggregate_coeffs,
        raw_sector_errors=aggregate_errors,
        per_sector=[
            SectorIntegrationResult(
                sector_id=sector_index,
                sector_name=sectors[sector_index].name,
                samples=sector_hits[sector_index],
                raw_sector_coeffs=[stat.mean for stat in coeff_stats],
                raw_sector_errors=[stat.error for stat in coeff_stats],
                precision_counts=sector_precision_counts[sector_index].copy(),
                diagnostics={
                    "sampling_mode": sampling_mode,
                    "avg_eval_us_per_sample": avg_eval_us_per_sample_per_worker(
                        sector_timing[sector_index],
                        sector_hits[sector_index],
                    ),
                    "eval_seconds": sector_timing[sector_index].eval_seconds,
                    "python_seconds": sector_timing[sector_index].python_seconds,
                    "havana_seconds": sector_timing[sector_index].havana_seconds,
                    "max_abs_weight": float(np.max(sector_max_abs[sector_index]))
                    if sector_hits[sector_index]
                    else 0.0,
                    "max_abs_by_order": [
                        float(value) for value in sector_max_abs[sector_index]
                    ],
                },
            )
            for sector_index, coeff_stats in enumerate(sector_stats)
        ],
        samples=total_samples,
        elapsed_seconds=time.perf_counter() - start_time,
        avg_eval_us_per_sample_per_worker=avg_eval_us_per_sample_per_worker(
            total_timing, total_samples
        ),
        eval_seconds=total_timing.eval_seconds,
        python_seconds=total_timing.python_seconds,
        havana_seconds=total_timing.havana_seconds,
        python_overhead_fraction=total_timing.python_overhead_fraction,
        precision_counts=total_timing.precision_counts,
        interrupted=interrupted,
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class _LiveAggregateStat:
    """Duck-typed statistic used only for QMC progress display."""

    mean: complex
    error: complex
    count: int


def _qmc_live_stats(
    sector_stats: list[list[RunningStats]],
    active_sector_ids: list[int],
    coeff_count: int,
    raw_samples: int,
) -> list[_LiveAggregateStat]:
    """Build aggregate sector-sum stats with quadrature errors for QMC progress."""
    out: list[_LiveAggregateStat] = []
    for coeff_index in range(coeff_count):
        coeff = sum(
            sector_stats[sector_id][coeff_index].mean
            for sector_id in active_sector_ids
        )
        err_re2 = sum(
            sector_stats[sector_id][coeff_index].error.real ** 2
            for sector_id in active_sector_ids
            if math.isfinite(sector_stats[sector_id][coeff_index].error.real)
        )
        err_im2 = sum(
            sector_stats[sector_id][coeff_index].error.imag ** 2
            for sector_id in active_sector_ids
            if math.isfinite(sector_stats[sector_id][coeff_index].error.imag)
        )
        out.append(
            _LiveAggregateStat(
                mean=coeff,
                error=complex(math.sqrt(err_re2), math.sqrt(err_im2)),
                count=int(raw_samples),
            )
        )
    return out


def _qmc_live_stats_from_aggregate(
    aggregate_stats: list[RunningStats],
    raw_samples: int,
) -> list[_LiveAggregateStat]:
    """Build live QMC stats from already-correlated shift-sum estimates."""
    return [
        _LiveAggregateStat(
            mean=stat.mean,
            error=stat.error,
            count=int(raw_samples),
        )
        for stat in aggregate_stats
    ]


def integrate_qmc(
    request: IntegralRequest,
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    target: TargetDefinition | None,
) -> IntegrationResult:
    """Run randomized shifted-lattice QMC with Korobov periodization.

    ``samples_per_iter`` is interpreted as the number of lattice points per
    random shift and sector.  Each random shift produces one sector-integral
    estimate, and the QMC error is estimated from the distribution of these
    shift estimates.
    """
    if not sectors:
        raise ValueError("no sectors generated")
    active_sector_ids = list(request.sectors) if request.sectors is not None else list(range(len(sectors)))
    if not active_sector_ids:
        raise ValueError("no active sectors selected")
    active_sectors = [sectors[sector_id] for sector_id in active_sector_ids]
    continuous_dim = active_sectors[0].integration_dim
    if any(sector.integration_dim != continuous_dim for sector in active_sectors):
        raise ValueError("all sectors must have the same integration dimension for QMC mode")

    sector_stats = [
        [RunningStats() for _ in range(topology.coefficient_count)]
        for _sector in sectors
    ]
    sector_hits = [0 for _sector in sectors]
    sector_precision_counts = [
        {"ordinary": 0, "stability": 0, "medium_precision": 0, "high_precision": 0}
        for _sector in sectors
    ]
    sector_timing = [HotPathTiming() for _sector in sectors]
    sector_max_abs = [
        np.zeros(topology.coefficient_count, dtype=float)
        for _sector in sectors
    ]
    correlated_qmc = bool(request.qmc_correlate_sectors)
    global_support_dims = qmc_global_support_dims(topology, active_sectors)
    qmc_group_count = sum(
        len(qmc_support_groups(topology, sector, global_support_dims))
        for sector in active_sectors
    )
    aggregate_shift_stats = [
        RunningStats() for _coeff_index in range(topology.coefficient_count)
    ]
    total_timing = HotPathTiming()
    start_time = time.perf_counter()
    interrupted = False
    raw_samples_done = 0
    diagnostics: dict[str, Any] = {
        "sampling_mode": "qmc",
        "qmc_lattice_backend": "qmcpy",
        "qmc_software": "qmcpy.Lattice",
        "qmc_randomization": "SHIFT",
        "qmc_lattice_order": str(request.qmc_order),
        "qmc_lattice_points_per_shift": int(request.samples_per_iter),
        "qmc_shifts": int(request.qmc_shifts),
        "qmc_korobov_alpha": int(request.qmc_korobov_alpha),
        "qmc_correlate_sectors": correlated_qmc,
        "qmc_support_grouping": "deepest-endpoint-boundary",
        "qmc_global_support_dimensions": [int(value) for value in global_support_dims],
        "qmc_sector_group_count": int(qmc_group_count),
        "active_sector_count": len(active_sector_ids),
        "note": (
            "All sectors share the same randomized shifted lattice in each "
            "iteration; aggregate QMC errors are estimated from the "
            "shift-by-shift sector sum."
            if correlated_qmc
            else "Each sector is integrated by randomized shifted lattices. "
            "Errors are estimated from random-shift sector estimates and "
            "combined across sectors in quadrature."
        ),
    }

    progress_request = replace(
        request,
        samples_per_iter=int(request.samples_per_iter) * int(request.qmc_shifts) * int(qmc_group_count),
    )
    bar = make_progress_bar(progress_request)
    if bar is not None:
        bar.start()

    executor: ProcessPoolExecutor | None = None
    processor: SectorProcessor | None = None

    def absorb_reduced(
        reduced: QmcReducedBatch,
        sector_sums: np.ndarray,
        sector_counts: np.ndarray,
    ) -> None:
        nonlocal raw_samples_done
        aggregate_start = time.perf_counter()
        sector_id = int(reduced.sector_id)
        sector_sums += reduced.sums_by_shift
        sector_counts += reduced.counts_by_shift
        sector_precision_counts[sector_id]["ordinary"] += int(reduced.precision_counts[0])
        sector_precision_counts[sector_id]["stability"] += int(reduced.precision_counts[1])
        sector_precision_counts[sector_id]["medium_precision"] += int(reduced.precision_counts[2])
        sector_precision_counts[sector_id]["high_precision"] += int(reduced.precision_counts[3])
        sector_timing[sector_id].absorb(reduced.timing)
        total_timing.absorb(reduced.timing)
        sector_max_abs[sector_id] = np.maximum(sector_max_abs[sector_id], reduced.max_abs)
        raw_samples_done += int(reduced.count)
        total_timing.add_python(time.perf_counter() - aggregate_start)

    def finish_sector_shift_estimates(
        sector_id: int,
        sector_sums: np.ndarray,
        sector_counts: np.ndarray,
        coefficient_indices: tuple[int, ...],
    ) -> np.ndarray:
        aggregate_start = time.perf_counter()
        if np.any(sector_counts <= 0):
            raise RuntimeError(f"sector {sector_id} has incomplete QMC shift estimates")
        sector_hits[sector_id] += int(np.sum(sector_counts))
        means = sector_sums / sector_counts[:, np.newaxis]
        for shift_values in means:
            for coeff_index in coefficient_indices:
                stat = sector_stats[sector_id][coeff_index]
                stat.add(shift_values[coeff_index])
        total_timing.add_python(time.perf_counter() - aggregate_start)
        return means

    def update_qmc_progress(iteration: int) -> None:
        if correlated_qmc and aggregate_shift_stats[0].count > 0:
            live_stats = _qmc_live_stats_from_aggregate(
                aggregate_shift_stats,
                raw_samples_done,
            )
        else:
            live_stats = _qmc_live_stats(
                sector_stats,
                active_sector_ids,
                topology.coefficient_count,
                raw_samples_done,
            )
        elapsed = time.perf_counter() - start_time
        avg_eval_us = avg_eval_us_per_sample_per_worker(total_timing, raw_samples_done)
        update_progress_bar_timed(
            bar,
            progress_request,
            live_stats,  # type: ignore[arg-type]
            target,
            iteration,
            elapsed,
            avg_eval_us,
            total_timing,
        )

    try:
        if request.workers > 1:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("QMC multi-worker integration requires a fork-capable Python runtime")
            global _PARENT_TOPOLOGY, _PARENT_SECTORS, _PARENT_REQUEST
            _PARENT_TOPOLOGY = topology
            _PARENT_SECTORS = active_sectors
            _PARENT_REQUEST = request
            executor = ProcessPoolExecutor(
                max_workers=request.workers,
                mp_context=mp.get_context("fork"),
                initializer=_init_worker_from_parent,
            )
        else:
            processor = _make_sector_processor(topology, request)

        iteration = 0
        while request.max_iter < 0 or iteration < request.max_iter:
            iteration += 1
            shared_raw_points_by_group: dict[tuple[tuple[int, ...], int], np.ndarray] = {}
            iteration_shift_totals = np.zeros(
                (int(request.qmc_shifts), topology.coefficient_count),
                dtype=np.complex128,
            )
            for sector_position, (sector_id, sector) in enumerate(zip(active_sector_ids, active_sectors)):
                for coefficient_indices, support_axes in qmc_support_groups(
                    topology,
                    sector,
                    global_support_dims,
                ):
                    sector_sums = np.zeros(
                        (int(request.qmc_shifts), topology.coefficient_count),
                        dtype=np.complex128,
                    )
                    sector_counts = np.zeros(int(request.qmc_shifts), dtype=np.int64)
                    shared_raw_points: np.ndarray | None = None
                    if correlated_qmc:
                        group_key = (tuple(int(index) for index in coefficient_indices), len(support_axes))
                        shared_raw_points = shared_raw_points_by_group.get(group_key)
                        if shared_raw_points is None:
                            shared_seed = (
                                int(request.seed)
                                + 97_003 * int(iteration)
                                + 53_021 * len(shared_raw_points_by_group)
                            )
                            shared_raw_points = shifted_lattice_points(
                                backend=str(request.qmc_lattice_backend),
                                dimension=int(len(support_axes)),
                                n_points=int(request.samples_per_iter),
                                shift_count=int(request.qmc_shifts),
                                seed=shared_seed,
                                order=str(request.qmc_order),
                            )
                            shared_raw_points_by_group[group_key] = shared_raw_points
                    batches = qmc_batches_for_sector(
                        request,
                        sector_position,
                        sector_id,
                        sector,
                        iteration,
                        raw_points=shared_raw_points,
                        support_axes=support_axes,
                        coefficient_indices=coefficient_indices,
                    )
                    if executor is None:
                        assert processor is not None
                        for batch in batches:
                            absorb_reduced(
                                _evaluate_qmc_batch(processor, active_sectors, batch),
                                sector_sums,
                                sector_counts,
                            )
                    else:
                        pending = [
                            executor.submit(_evaluate_qmc_batch_worker, batch)
                            for batch in batches
                        ]
                        for future in pending:
                            absorb_reduced(future.result(), sector_sums, sector_counts)
                    sector_shift_means = finish_sector_shift_estimates(
                        sector_id,
                        sector_sums,
                        sector_counts,
                        coefficient_indices,
                    )
                    if correlated_qmc:
                        iteration_shift_totals += sector_shift_means
                    update_qmc_progress(iteration)
            if correlated_qmc:
                aggregate_start = time.perf_counter()
                for shift_values in iteration_shift_totals:
                    for coeff_index, stat in enumerate(aggregate_shift_stats):
                        stat.add(shift_values[coeff_index])
                total_timing.add_python(time.perf_counter() - aggregate_start)
                live_stats = _qmc_live_stats_from_aggregate(
                    aggregate_shift_stats,
                    raw_samples_done,
                )
            else:
                live_stats = _qmc_live_stats(
                    sector_stats,
                    active_sector_ids,
                    topology.coefficient_count,
                    raw_samples_done,
                )
            if target_accuracy_reached(
                progress_request,
                live_stats,  # type: ignore[arg-type]
                target,
                iteration,
            ):
                break
            if (
                request.target_rel_accuracy is None
                and request.max_iter >= 0
                and iteration >= request.min_iter
                and max(live_stats[topology.training_index].error.real, live_stats[topology.training_index].error.imag)
                < request.min_error
            ):
                break
        diagnostics["qmc_completed_iterations"] = int(iteration)
    except KeyboardInterrupt:
        interrupted = True
        if executor is not None:
            _terminate_executor_workers(executor)
        if not request.json:
            print(
                f"\n{Fore.YELLOW}Keyboard interrupt received; writing partial QMC result "
                f"with {raw_samples_done} raw lattice samples.{Style.RESET_ALL}"
            )
    finally:
        if bar is not None:
            bar.finish(dirty=True)
        if executor is not None:
            executor.shutdown(wait=not interrupted, cancel_futures=True)
        if request.workers > 1:
            _PARENT_TOPOLOGY = None
            _PARENT_SECTORS = None
            _PARENT_REQUEST = None

    return _stats_from_reduced_batches(
        sectors,
        active_sector_ids,
        topology,
        sector_stats,
        sector_hits,
        sector_precision_counts,
        sector_timing,
        sector_max_abs,
        total_timing,
        start_time,
        interrupted,
        diagnostics,
        aggregate_stats_override=aggregate_shift_stats
        if correlated_qmc and aggregate_shift_stats[0].count > 0
        else None,
    )


def integrate_democratic(
    request: IntegralRequest,
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    target: TargetDefinition | None,
) -> IntegrationResult:
    """Run explicit equal-statistics sampling over every active sector."""
    if not sectors:
        raise ValueError("no sectors generated")
    active_sector_ids = list(request.sectors) if request.sectors is not None else list(range(len(sectors)))
    if not active_sector_ids:
        raise ValueError("no active sectors selected")
    active_sectors = [sectors[sector_id] for sector_id in active_sector_ids]
    continuous_dim = active_sectors[0].integration_dim
    if any(sector.integration_dim != continuous_dim for sector in active_sectors):
        raise ValueError("all sectors must have the same integration dimension for democratic mode")

    sector_stats = [
        [RunningStats() for _ in range(topology.coefficient_count)]
        for _sector in sectors
    ]
    sector_hits = [0 for _sector in sectors]
    sector_precision_counts = [
        {"ordinary": 0, "stability": 0, "medium_precision": 0, "high_precision": 0}
        for _sector in sectors
    ]
    sector_timing = [HotPathTiming() for _sector in sectors]
    sector_max_abs = [
        np.zeros(topology.coefficient_count, dtype=float)
        for _sector in sectors
    ]
    total_timing = HotPathTiming()
    start_time = time.perf_counter()
    interrupted = False
    diagnostics: dict[str, Any] = {
        "sampling_mode": "democratic",
        "samples_per_sector": int(request.democratic_samples_per_sector),
        "active_sector_count": len(active_sector_ids),
        "requested_total_samples": len(active_sector_ids)
        * int(request.democratic_samples_per_sector),
        "note": (
            "Each active sector is sampled uniformly with equal statistics; "
            "the aggregate integral is the sum of per-sector means."
        ),
    }

    batches = democratic_batches(request, active_sector_ids, active_sectors)

    def absorb_reduced(reduced: DemocraticReducedBatch) -> None:
        aggregate_start = time.perf_counter()
        sector_id = int(reduced.sector_id)
        sector_hits[sector_id] += int(reduced.count)
        sector_precision_counts[sector_id]["ordinary"] += int(reduced.precision_counts[0])
        sector_precision_counts[sector_id]["stability"] += int(reduced.precision_counts[1])
        sector_precision_counts[sector_id]["medium_precision"] += int(reduced.precision_counts[2])
        sector_precision_counts[sector_id]["high_precision"] += int(reduced.precision_counts[3])
        sector_timing[sector_id].absorb(reduced.timing)
        total_timing.absorb(reduced.timing)
        sector_max_abs[sector_id] = np.maximum(sector_max_abs[sector_id], reduced.max_abs)
        for coeff_index, stat in enumerate(sector_stats[sector_id]):
            stat.add_aggregate(
                int(reduced.count),
                float(reduced.sums[coeff_index].real),
                float(reduced.sums[coeff_index].imag),
                float(reduced.sumsq_re[coeff_index]),
                float(reduced.sumsq_im[coeff_index]),
            )
        total_timing.add_python(time.perf_counter() - aggregate_start)

    executor: ProcessPoolExecutor | None = None
    processor: SectorProcessor | None = None
    try:
        if request.workers > 1:
            if "fork" not in mp.get_all_start_methods():
                raise RuntimeError("democratic multi-worker integration requires a fork-capable Python runtime")
            global _PARENT_TOPOLOGY, _PARENT_SECTORS, _PARENT_REQUEST
            _PARENT_TOPOLOGY = topology
            _PARENT_SECTORS = active_sectors
            _PARENT_REQUEST = request
            executor = ProcessPoolExecutor(
                max_workers=request.workers,
                mp_context=mp.get_context("fork"),
                initializer=_init_worker_from_parent,
            )
            pending: set[Future[DemocraticReducedBatch]] = set()
            batch_iter = iter(batches)
            pending_limit = max(int(request.workers), len(active_sector_ids), 1)
            diagnostics["pending_batch_limit"] = int(pending_limit)

            def submit_until_full() -> None:
                submit_start = time.perf_counter()
                try:
                    while len(pending) < pending_limit:
                        pending.add(executor.submit(_evaluate_democratic_batch_worker, next(batch_iter)))
                except StopIteration:
                    pass
                total_timing.add_python(time.perf_counter() - submit_start)

            submit_until_full()
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    absorb_reduced(future.result())
                submit_until_full()
        else:
            processor = _make_sector_processor(topology, request)
            for batch in batches:
                absorb_reduced(_evaluate_democratic_batch(processor, active_sectors, batch))
    except KeyboardInterrupt:
        interrupted = True
        if executor is not None:
            _terminate_executor_workers(executor)
    finally:
        if executor is not None:
            executor.shutdown(wait=not interrupted, cancel_futures=True)

    return _stats_from_reduced_batches(
        sectors,
        active_sector_ids,
        topology,
        sector_stats,
        sector_hits,
        sector_precision_counts,
        sector_timing,
        sector_max_abs,
        total_timing,
        start_time,
        interrupted,
        diagnostics,
    )


def integrate(
    request: IntegralRequest,
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    target: TargetDefinition | None,
) -> IntegrationResult:
    """Run the adaptive Monte Carlo integration and return raw coefficients."""
    if request.sampling_mode == "qmc":
        return integrate_qmc(request, topology, sectors, target)
    if request.sampling_mode == "democratic":
        return integrate_democratic(request, topology, sectors, target)
    if not sectors:
        raise ValueError("no sectors generated")
    active_sector_ids = list(request.sectors) if request.sectors is not None else list(range(len(sectors)))
    if not active_sector_ids:
        raise ValueError("no active sectors selected")
    active_sectors = [sectors[sector_id] for sector_id in active_sector_ids]
    active_sector_ids_array = np.asarray(active_sector_ids, dtype=int)
    continuous_dim = active_sectors[0].integration_dim
    if any(sector.integration_dim != continuous_dim for sector in active_sectors):
        raise ValueError("all sectors must have the same integration dimension for the current Havana driver")

    # One discrete dimension chooses the sector, and each sector owns an
    # adaptive continuous Havana grid of the same dimension.  Lower-support
    # subtraction terms are localized into this same dimension by the processor.
    grid = NumericalIntegrator.discrete(
        [NumericalIntegrator.continuous(continuous_dim, n_bins=request.bins) for _ in active_sectors]
    )
    rng = grid.rng(request.seed, 0)
    stats = [RunningStats() for _ in range(topology.coefficient_count)]
    per_sector_stats = [
        [RunningStats() for _ in range(topology.coefficient_count)]
        for _sector in sectors
    ]
    per_sector_hits = [0 for _sector in sectors]
    per_sector_precision_counts = [
        {"ordinary": 0, "stability": 0, "medium_precision": 0, "high_precision": 0}
        for _sector in sectors
    ]
    per_sector_max_abs = [
        np.zeros(topology.coefficient_count, dtype=float)
        for _sector in sectors
    ]

    processor: SectorProcessor | None = None
    executor: ProcessPoolExecutor | None = None
    if request.workers > 1:
        if "fork" not in mp.get_all_start_methods():
            if request.integral == "dot":
                raise RuntimeError(
                    "DOT multi-worker integration requires a fork-capable Python runtime so prepared "
                    "pySecDec-generated sectors are inherited without regenerating; use --workers 1"
                )
            raise RuntimeError("multi-worker integration requires a fork-capable Python runtime")
        global _PARENT_TOPOLOGY, _PARENT_SECTORS, _PARENT_REQUEST
        _PARENT_TOPOLOGY = topology
        _PARENT_SECTORS = active_sectors
        _PARENT_REQUEST = request
        executor = ProcessPoolExecutor(
            max_workers=request.workers,
            mp_context=mp.get_context("fork"),
            initializer=_init_worker_from_parent,
        )
    else:
        processor = _make_sector_processor(topology, request)

    bar = make_progress_bar(request)
    if bar is not None:
        bar.start()

    start_time = time.perf_counter()
    hot_timing = HotPathTiming()
    interrupted = False

    def build_result() -> IntegrationResult:
        """Materialize the current accumulators, including partial runs."""
        return IntegrationResult(
            raw_sector_coeffs=[stat.mean for stat in stats],
            raw_sector_errors=[stat.error for stat in stats],
            per_sector=[
                SectorIntegrationResult(
                    sector_id=sector_index,
                    sector_name=sectors[sector_index].name,
                    samples=per_sector_hits[sector_index],
                    raw_sector_coeffs=[stat.mean for stat in sector_stats],
                    raw_sector_errors=[stat.error for stat in sector_stats],
                    precision_counts=per_sector_precision_counts[sector_index].copy(),
                    diagnostics={
                        "max_abs_weight": float(np.max(per_sector_max_abs[sector_index]))
                        if per_sector_hits[sector_index]
                        else 0.0,
                        "max_abs_by_order": [
                            float(value) for value in per_sector_max_abs[sector_index]
                        ],
                    },
                )
                for sector_index, sector_stats in enumerate(per_sector_stats)
            ],
            samples=stats[0].count,
            elapsed_seconds=time.perf_counter() - start_time,
            avg_eval_us_per_sample_per_worker=avg_eval_us_per_sample_per_worker(
                hot_timing, stats[0].count
            ),
            eval_seconds=hot_timing.eval_seconds,
            python_seconds=hot_timing.python_seconds,
            havana_seconds=hot_timing.havana_seconds,
            python_overhead_fraction=hot_timing.python_overhead_fraction,
            precision_counts=hot_timing.precision_counts,
            interrupted=interrupted,
        )

    try:
        iteration = 0
        while request.max_iter < 0 or iteration < request.max_iter:
            iteration += 1
            havana_start = time.perf_counter()
            # Train a cloned grid during the iteration.  The live grid is only
            # merged/updated after the iteration completes, so mid-iteration
            # progress updates do not perturb the sampler state.
            training_grid = copy.copy(grid)
            samples = grid.sample(request.samples_per_iter, rng)
            hot_timing.add_havana(time.perf_counter() - havana_start)

            master_start = time.perf_counter()
            n_samples = len(samples)
            sample_indices = np.arange(n_samples, dtype=int)
            sector_indices = np.empty(n_samples, dtype=int)
            coords = np.empty((n_samples, continuous_dim), dtype=float)
            weights = np.empty(n_samples, dtype=float)
            for sample_index, sample in enumerate(samples):
                # Havana samples contain one discrete coordinate d[0] and one
                # continuous coordinate vector c.  The first weight is the full
                # MC weight used in the manual Laurent accumulators.
                sector_indices[sample_index] = int(sample.d[0])
                coords[sample_index, :] = sample.c
                weights[sample_index] = float(sample.weights[0])
            batches = split_evaluation_batches(
                sample_indices,
                sector_indices,
                coords,
                weights,
                request.batch_size,
                request.workers,
            )
            hot_timing.add_python(time.perf_counter() - master_start)

            def with_current_max_weight_guard(batch: EvaluationBatch) -> EvaluationBatch:
                """Attach the latest per-sector max-weight snapshot to a batch."""
                if request.max_weight_precision_xi <= 0.0:
                    return batch
                snapshot = np.asarray(
                    [per_sector_max_abs[sector_id] for sector_id in active_sector_ids],
                    dtype=float,
                )
                return replace(
                    batch,
                    sector_max_abs=snapshot,
                    max_weight_precision_xi=request.max_weight_precision_xi,
                )

            def register_batch(
                indices: np.ndarray,
                w_part: np.ndarray,
                t_part: np.ndarray,
                precision_part: np.ndarray,
            ) -> bool:
                # This is the point where a batch becomes part of the returned
                # result.  Target-accuracy termination is checked only after
                # these samples have been accumulated.
                aggregate_start = time.perf_counter()
                for coeff_index, stat in enumerate(stats):
                    stat.add_many(w_part[:, coeff_index])
                local_sector_part = sector_indices[indices]
                sector_part = active_sector_ids_array[local_sector_part]
                hit_counts = np.bincount(sector_part, minlength=len(sectors))
                for sector_index, hits in enumerate(hit_counts):
                    per_sector_hits[sector_index] += int(hits)
                for local_sector_index, sector_id in enumerate(active_sector_ids):
                    counts = precision_part[local_sector_index]
                    per_sector_precision_counts[sector_id]["ordinary"] += int(counts[0])
                    per_sector_precision_counts[sector_id]["stability"] += int(counts[1])
                    per_sector_precision_counts[sector_id]["medium_precision"] += int(counts[2])
                    per_sector_precision_counts[sector_id]["high_precision"] += int(counts[3])
                if w_part.size:
                    abs_part = np.abs(w_part)
                    for local_sector_index, sector_id in enumerate(active_sector_ids):
                        local_mask = local_sector_part == local_sector_index
                        if np.any(local_mask):
                            per_sector_max_abs[sector_id] = np.maximum(
                                per_sector_max_abs[sector_id],
                                np.max(abs_part[local_mask], axis=0),
                            )
                batch_count = int(indices.size)
                for coeff_index in range(topology.coefficient_count):
                    values = np.asarray(w_part[:, coeff_index], dtype=np.complex128)
                    re_sums = np.bincount(
                        sector_part,
                        weights=values.real,
                        minlength=len(sectors),
                    )
                    im_sums = np.bincount(
                        sector_part,
                        weights=values.imag,
                        minlength=len(sectors),
                    )
                    re_sumsq = np.bincount(
                        sector_part,
                        weights=values.real * values.real,
                        minlength=len(sectors),
                    )
                    im_sumsq = np.bincount(
                        sector_part,
                        weights=values.imag * values.imag,
                        minlength=len(sectors),
                    )
                    for sector_index in active_sector_ids:
                        sector_stats = per_sector_stats[sector_index]
                        sector_stats[coeff_index].add_aggregate(
                            batch_count,
                            re_sums[sector_index],
                            im_sums[sector_index],
                            re_sumsq[sector_index],
                            im_sumsq[sector_index],
                        )
                start_index = int(indices[0])
                stop_index = int(indices[-1]) + 1
                if stop_index - start_index == indices.size:
                    batch_samples = samples[start_index:stop_index]
                else:
                    batch_samples = [samples[int(index)] for index in indices]
                hot_timing.add_python(time.perf_counter() - aggregate_start)

                havana_batch_start = time.perf_counter()
                # The training observable is the norm of the last requested
                # Laurent coefficient.  It steers the adaptive grid, while all
                # coefficients themselves are accumulated in RunningStats above.
                training_grid.add_training_samples(batch_samples, t_part)
                hot_timing.add_havana(time.perf_counter() - havana_batch_start)

                elapsed_seconds = time.perf_counter() - start_time
                avg_eval_us = avg_eval_us_per_sample_per_worker(hot_timing, stats[0].count)
                update_progress_bar_timed(
                    bar,
                    request,
                    stats,
                    target,
                    iteration,
                    elapsed_seconds,
                    avg_eval_us,
                    hot_timing,
                )
                return target_accuracy_reached(request, stats, target, iteration)

            stop_requested = False
            if executor is None:
                assert processor is not None
                for batch in batches:
                    indices, w_part, t_part, precision_part, worker_timing = _evaluate_records(
                        processor,
                        active_sectors,
                        with_current_max_weight_guard(batch),
                    )
                    hot_timing.absorb(worker_timing)
                    if register_batch(indices, w_part, t_part, precision_part):
                        stop_requested = True
                        break
            else:
                pending: set[Future[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, HotPathTiming]]] = set()
                next_batch_index = 0

                def submit_until_full() -> None:
                    nonlocal next_batch_index
                    submit_start = time.perf_counter()
                    # Keep only a worker-sized window in flight.  This allows
                    # target-accuracy termination to stop inside an iteration
                    # without submitting the whole iteration up front.
                    while (
                        next_batch_index < len(batches)
                        and len(pending) < max(request.workers, 1)
                    ):
                        pending.add(
                            executor.submit(
                                _evaluate_records_worker,
                                with_current_max_weight_guard(batches[next_batch_index]),
                            )
                        )
                        next_batch_index += 1
                    hot_timing.add_python(time.perf_counter() - submit_start)

                submit_until_full()
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        if stop_requested:
                            future.cancel()
                            continue
                        result_start = time.perf_counter()
                        indices, w_part, t_part, precision_part, worker_timing = future.result()
                        hot_timing.add_python(time.perf_counter() - result_start)
                        hot_timing.absorb(worker_timing)
                        if register_batch(indices, w_part, t_part, precision_part):
                            stop_requested = True
                            # Futures that have not started can be cancelled.
                            # Running futures may finish in the background, but
                            # their samples are intentionally not accumulated.
                            for pending_future in pending:
                                pending_future.cancel()
                            pending.clear()
                            break
                    if stop_requested:
                        break
                    submit_until_full()

            if stop_requested:
                # A target-accuracy stop returns the already accumulated
                # partial iteration.  The cloned training grid is deliberately
                # not merged because the live grid will not be sampled again.
                break

            havana_update_start = time.perf_counter()
            # Full completed iterations update the live Havana grid from the
            # cloned training grid.  This preserves stable sampling during the
            # iteration while still adapting between iterations.
            grid.merge(training_grid)
            live_avg, live_err, live_chi = grid.update(1.5, 1.5)
            hot_timing.add_havana(time.perf_counter() - havana_update_start)

            training_err = stats[topology.training_index].error
            if request.show_stats:
                print(
                    f"iter {iteration:3d}: training raw sector-sum "
                    f"{format_complex(stats[topology.training_index].mean)} +- {format_complex_error(training_err)} "
                    f"(havana train {live_avg:.6g} +- {live_err:.3g}, chi={live_chi:.3g})"
                )

            elapsed_seconds = time.perf_counter() - start_time
            avg_eval_us = avg_eval_us_per_sample_per_worker(hot_timing, stats[0].count)
            update_progress_bar_timed(
                bar,
                request,
                stats,
                target,
                iteration,
                elapsed_seconds,
                avg_eval_us,
                hot_timing,
            )

            if (
                request.target_rel_accuracy is None
                and request.max_iter >= 0
                and iteration >= request.min_iter
                and max(training_err.real, training_err.imag) < request.min_error
            ):
                break
            if target_accuracy_reached(request, stats, target, iteration):
                break

        return build_result()
    except KeyboardInterrupt:
        interrupted = True
        if not request.json:
            try:
                sample_count = stats[0].count if stats else 0
                print(
                    f"\n{Fore.YELLOW}Keyboard interrupt received; writing partial result "
                    f"with {sample_count} accumulated samples.{Style.RESET_ALL}"
                )
            except KeyboardInterrupt:
                # A second Ctrl-C should not turn graceful interruption into a
                # traceback; continue returning the accumulated result.
                pass
        return build_result()
    finally:
        if bar is not None:
            bar.finish(dirty=True)
        if executor is not None:
            if interrupted:
                _terminate_executor_workers(executor)
            executor.shutdown(wait=not interrupted, cancel_futures=True)
        if request.workers > 1:
            _PARENT_TOPOLOGY = None
            _PARENT_SECTORS = None
            _PARENT_REQUEST = None
