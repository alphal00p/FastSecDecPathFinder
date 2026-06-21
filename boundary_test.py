"""Endpoint-boundary sector tests for the FSD CLI."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import itertools
import json
import math
import multiprocessing as mp
from pathlib import Path
import time
from typing import Any

from colorama import Fore, Style
import numpy as np
from prettytable import PrettyTable

from definitions import IntegralRequest

try:
    import progressbar
except ImportError:  # pragma: no cover - requirements.txt includes progressbar2.
    progressbar = None


_GROWTH_POWER_NUMERICAL_FLOOR = 1.0e-6


@dataclass(frozen=True)
class EndpointTestResult:
    """JSON-friendly endpoint test result for one sector."""

    sector_id: int
    name: str
    status: str
    probe_count: int
    elapsed_seconds: float
    avg_eval_us_per_probe: float
    integration_dim: int
    singular_axes: list[int]
    all_laurent_weights_finite: bool
    all_training_weights_finite: bool
    nonfinite_probe_count: int
    nonfinite_examples: list[dict[str, Any]]
    boundary_stability_ok: bool
    boundary_growth_power_tolerance: float | None
    boundary_stability_failed_pair_count: int
    max_boundary_growth_power: float | None
    worst_boundary_stability_pair: dict[str, Any]
    boundary_stability_failures: list[dict[str, Any]]
    max_abs_by_order: list[float | None]
    max_abs_laurent_weight: float | None
    max_abs_training_weight: float | None
    worst_probe: dict[str, Any]
    by_distance: list[dict[str, Any]]
    by_probe_kind: list[dict[str, Any]]
    profile: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector_id": self.sector_id,
            "name": self.name,
            "status": self.status,
            "probe_count": self.probe_count,
            "elapsed_seconds": self.elapsed_seconds,
            "avg_eval_us_per_probe": self.avg_eval_us_per_probe,
            "integration_dim": self.integration_dim,
            "singular_axes": self.singular_axes,
            "all_laurent_weights_finite": self.all_laurent_weights_finite,
            "all_training_weights_finite": self.all_training_weights_finite,
            "nonfinite_probe_count": self.nonfinite_probe_count,
            "nonfinite_examples": self.nonfinite_examples,
            "boundary_stability_ok": self.boundary_stability_ok,
            "boundary_growth_power_tolerance": self.boundary_growth_power_tolerance,
            "boundary_stability_failed_pair_count": self.boundary_stability_failed_pair_count,
            "max_boundary_growth_power": self.max_boundary_growth_power,
            "worst_boundary_stability_pair": self.worst_boundary_stability_pair,
            "boundary_stability_failures": self.boundary_stability_failures,
            "max_abs_by_order": self.max_abs_by_order,
            "max_abs_laurent_weight": self.max_abs_laurent_weight,
            "max_abs_training_weight": self.max_abs_training_weight,
            "worst_probe": self.worst_probe,
            "by_distance": self.by_distance,
            "by_probe_kind": self.by_probe_kind,
            "profile": self.profile,
        }


@dataclass(frozen=True)
class EndpointChunkResult:
    """Raw endpoint evaluations for one chunk of one sector."""

    chunk_index: int
    coeffs: np.ndarray
    training: np.ndarray
    elapsed_seconds: float
    profile: dict[str, Any]


_PARENT_TOPOLOGY: Any | None = None
_PARENT_SECTORS: list[Any] | None = None
_PARENT_REQUEST: IntegralRequest | None = None
_WORKER_PROCESSOR: Any | None = None
_WORKER_SECTORS: list[Any] | None = None


def _json_default(obj: Any) -> Any:
    """JSON serializer kept local to avoid importing Symbolica for test helpers."""
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _output_json(output: dict[str, Any]) -> str:
    return json.dumps(output, default=_json_default, indent=2)


def _make_processor(topology: Any, request: IntegralRequest) -> Any:
    """Create the real sector processor lazily so pure probe tests stay light."""
    from integrator import _make_sector_processor

    return _make_sector_processor(topology, request)


def _color(text: str, color: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def _base_point(dim: int) -> np.ndarray:
    pattern = np.asarray([0.31, 0.47, 0.63, 0.79], dtype=float)
    return np.asarray([pattern[axis % len(pattern)] for axis in range(dim)], dtype=float)


def _append_probe(
    rows: list[np.ndarray],
    labels: list[str],
    distances: list[float | None],
    kinds: list[str],
    seen: set[tuple[float, ...]],
    label: str,
    point: np.ndarray,
    distance: float | None,
    kind: str,
) -> None:
    clipped = np.clip(np.asarray(point, dtype=float), 1.0e-300, 1.0 - 1.0e-15)
    key = tuple(float(value) for value in clipped)
    if key in seen:
        return
    seen.add(key)
    rows.append(clipped)
    labels.append(label)
    distances.append(distance)
    kinds.append(kind)


def _corner_label(values: tuple[float, ...], low: float) -> str:
    return "".join("0" if value == low else "1" for value in values)


def endpoint_probe_points(
    sector: Any,
    distances: tuple[float, ...],
) -> tuple[np.ndarray, list[str], list[float | None], list[str]]:
    """Return deterministic endpoint probe rows for one sector.

    The core endpoint combination is every low/high hypercube corner for each
    requested distance.  Single-axis faces and grouped singular-axis faces are
    included as cheaper diagnostics for boundary subsets.
    """

    dim = int(sector.integration_dim)
    base = _base_point(dim)
    rows: list[np.ndarray] = []
    labels: list[str] = []
    row_distances: list[float | None] = []
    row_kinds: list[str] = []
    seen: set[tuple[float, ...]] = set()
    _append_probe(
        rows,
        labels,
        row_distances,
        row_kinds,
        seen,
        "interior",
        base,
        None,
        "interior",
    )
    singular_axes = [int(axis) for axis in sector.singular_axes]
    for distance in distances:
        delta = float(distance)
        for axis in range(dim):
            low = base.copy()
            low[axis] = delta
            _append_probe(
                rows,
                labels,
                row_distances,
                row_kinds,
                seen,
                f"axis_{axis}_low_{delta:.1e}",
                low,
                delta,
                "axis_face",
            )
            high = base.copy()
            high[axis] = 1.0 - delta
            _append_probe(
                rows,
                labels,
                row_distances,
                row_kinds,
                seen,
                f"axis_{axis}_high_{delta:.1e}",
                high,
                delta,
                "axis_face",
            )
        if singular_axes:
            low = base.copy()
            low[singular_axes] = delta
            _append_probe(
                rows,
                labels,
                row_distances,
                row_kinds,
                seen,
                f"singular_axes_low_{delta:.1e}",
                low,
                delta,
                "singular_axis_face",
            )
            high = base.copy()
            high[singular_axes] = 1.0 - delta
            _append_probe(
                rows,
                labels,
                row_distances,
                row_kinds,
                seen,
                f"singular_axes_high_{delta:.1e}",
                high,
                delta,
                "singular_axis_face",
            )
        for values in itertools.product((delta, 1.0 - delta), repeat=dim):
            _append_probe(
                rows,
                labels,
                row_distances,
                row_kinds,
                seen,
                f"corner_{_corner_label(values, delta)}_{delta:.1e}",
                np.asarray(values, dtype=float),
                delta,
                "hypercube_corner",
            )
    return np.vstack(rows), labels, row_distances, row_kinds


def _finite_max(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _float_or_none(value: float | np.floating[Any]) -> float | None:
    result = float(value)
    if not math.isfinite(result):
        return None
    return result


def _summarize_nonfinite(
    labels: list[str],
    rows: np.ndarray,
    coeffs: np.ndarray,
    training: np.ndarray,
    limit: int = 8,
) -> list[dict[str, Any]]:
    coeff_finite = np.isfinite(coeffs.real) & np.isfinite(coeffs.imag)
    training_finite = np.isfinite(training)
    bad_rows = np.where((~np.all(coeff_finite, axis=1)) | (~training_finite))[0]
    examples: list[dict[str, Any]] = []
    for row_index in bad_rows[:limit]:
        bad_coeff_indices = np.where(~coeff_finite[row_index])[0].astype(int).tolist()
        examples.append(
            {
                "probe": labels[int(row_index)],
                "coords": [float(value) for value in rows[int(row_index)]],
                "bad_coeff_indices": bad_coeff_indices,
                "bad_training_weight": bool(not training_finite[int(row_index)]),
            }
        )
    return examples


def _group_key(value: float | str | None) -> tuple[int, float | str]:
    if value is None:
        return (0, -1.0)
    if isinstance(value, str):
        return (1, value)
    return (1, float(value))


def _group_stats(
    groups: list[float | None] | list[str],
    coeff_abs: np.ndarray,
    coeff_finite: np.ndarray,
    training_finite: np.ndarray,
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for group in sorted(set(groups), key=_group_key):
        indices = [index for index, value in enumerate(groups) if value == group]
        if not indices:
            continue
        row_ok = np.all(coeff_finite[indices, :], axis=1) & training_finite[indices]
        grouped.append(
            {
                "group": "interior" if group is None else group,
                "probe_count": len(indices),
                "nonfinite_probe_count": int(len(indices) - np.count_nonzero(row_ok)),
                "max_abs_laurent_weight": _finite_max(coeff_abs[indices, :]),
            }
        )
    return grouped


def _probe_family(label: str) -> str | None:
    """Return the distance-independent probe family for stability comparisons."""

    if label == "interior":
        return None
    family, separator, suffix = str(label).rpartition("_")
    if not separator:
        return None
    try:
        float(suffix)
    except ValueError:
        return None
    return family


def _relative_vector_change(left: np.ndarray, right: np.ndarray) -> float | None:
    """Return max relative change between two Laurent coefficient vectors."""

    if not (
        np.all(np.isfinite(left.real))
        and np.all(np.isfinite(left.imag))
        and np.all(np.isfinite(right.real))
        and np.all(np.isfinite(right.imag))
    ):
        return None
    numerator = float(np.max(np.abs(right - left)))
    denominator = float(max(np.max(np.abs(left)), np.max(np.abs(right)), 1.0e-300))
    return numerator / denominator


def _relative_scalar_change(left: float, right: float) -> float | None:
    if not (math.isfinite(float(left)) and math.isfinite(float(right))):
        return None
    numerator = abs(float(right) - float(left))
    denominator = max(abs(float(left)), abs(float(right)), 1.0e-300)
    return numerator / denominator


def _growth_power(
    left_abs: float | None,
    right_abs: float | None,
    larger_distance: float,
    smaller_distance: float,
) -> float | None:
    """Return the effective p in |w| ~ distance**(-p), clipped at zero."""

    if left_abs is None or right_abs is None:
        return None
    if not (math.isfinite(left_abs) and math.isfinite(right_abs)):
        return None
    if right_abs <= left_abs:
        return 0.0
    distance_ratio = float(larger_distance) / float(smaller_distance)
    if not math.isfinite(distance_ratio) or distance_ratio <= 1.0:
        return None
    value_ratio = max(float(right_abs), 1.0e-300) / max(float(left_abs), 1.0e-300)
    return max(0.0, math.log(value_ratio) / math.log(distance_ratio))


def _endpoint_distance_count(row: np.ndarray, distance: float) -> int:
    """Count coordinates whose local endpoint distance is the probe distance."""

    tolerance = max(abs(float(distance)) * 1.0e-9, 1.0e-300)
    high_endpoint = 1.0 - float(distance)
    return sum(
        1
        for value in row
        if (
            math.isclose(
                float(value),
                float(distance),
                rel_tol=1.0e-9,
                abs_tol=tolerance,
            )
            or math.isclose(
                float(value),
                high_endpoint,
                rel_tol=1.0e-9,
                abs_tol=tolerance,
            )
        )
    )


def _boundary_stability_summary(
    rows: np.ndarray,
    labels: list[str],
    row_distances: list[float | None],
    coeffs: np.ndarray,
    training: np.ndarray,
    growth_power_tolerance: float | None,
    failure_limit: int = 8,
) -> tuple[bool, int, float | None, dict[str, Any], list[dict[str, Any]]]:
    """Compare matched endpoint probes at consecutive boundary distances."""

    if growth_power_tolerance is None:
        return True, 0, None, {}, []

    by_family: dict[str, dict[float, int]] = {}
    for index, (label, distance) in enumerate(zip(labels, row_distances, strict=True)):
        if distance is None:
            continue
        family = _probe_family(label)
        if family is None:
            continue
        by_family.setdefault(family, {})[float(distance)] = int(index)

    unique_distances = sorted({float(value) for value in row_distances if value is not None}, reverse=True)
    distance_pairs = list(zip(unique_distances, unique_distances[1:], strict=False))
    worst: dict[str, Any] = {}
    worst_growth_power: float | None = None
    failures: list[dict[str, Any]] = []
    failed_count = 0

    for larger_distance, smaller_distance in distance_pairs:
        for family in sorted(by_family):
            family_rows = by_family[family]
            left_index = family_rows.get(float(larger_distance))
            right_index = family_rows.get(float(smaller_distance))
            if left_index is None or right_index is None:
                continue
            laurent_relative_change = _relative_vector_change(
                coeffs[left_index],
                coeffs[right_index],
            )
            training_relative_change = _relative_scalar_change(
                float(training[left_index]),
                float(training[right_index]),
            )
            left_max_abs_laurent_weight = _finite_max(np.abs(coeffs[left_index]))
            right_max_abs_laurent_weight = _finite_max(np.abs(coeffs[right_index]))
            left_abs_training_weight = _float_or_none(abs(float(training[left_index])))
            right_abs_training_weight = _float_or_none(abs(float(training[right_index])))
            laurent_growth_power = _growth_power(
                left_max_abs_laurent_weight,
                right_max_abs_laurent_weight,
                larger_distance,
                smaller_distance,
            )
            training_growth_power = _growth_power(
                left_abs_training_weight,
                right_abs_training_weight,
                larger_distance,
                smaller_distance,
            )
            finite_changes = [
                value
                for value in (laurent_relative_change, training_relative_change)
                if value is not None
            ]
            relative_change = max(finite_changes) if len(finite_changes) == 2 else None
            finite_growth_powers = [
                value
                for value in (laurent_growth_power, training_growth_power)
                if value is not None
            ]
            growth_power = max(finite_growth_powers) if len(finite_growth_powers) == 2 else None
            endpoint_distance_coordinate_count = _endpoint_distance_count(
                rows[right_index],
                smaller_distance,
            )
            growth_power_threshold = (
                float(growth_power_tolerance) * endpoint_distance_coordinate_count
            )
            entry = {
                "probe_family": family,
                "distance_pair": [float(larger_distance), float(smaller_distance)],
                "probe_pair": [labels[left_index], labels[right_index]],
                "endpoint_distance_coordinate_count": int(endpoint_distance_coordinate_count),
                "low_endpoint_coordinate_count": int(endpoint_distance_coordinate_count),
                "zero_coordinate_count": int(endpoint_distance_coordinate_count),
                "growth_power_threshold": growth_power_threshold,
                "growth_power_numerical_floor": _GROWTH_POWER_NUMERICAL_FLOOR,
                "growth_power_failure_threshold": (
                    growth_power_threshold + _GROWTH_POWER_NUMERICAL_FLOOR
                ),
                "growth_power": growth_power,
                "laurent_growth_power": laurent_growth_power,
                "training_growth_power": training_growth_power,
                "relative_change": relative_change,
                "laurent_relative_change": laurent_relative_change,
                "training_relative_change": training_relative_change,
                "left_max_abs_laurent_weight": left_max_abs_laurent_weight,
                "right_max_abs_laurent_weight": right_max_abs_laurent_weight,
                "left_abs_training_weight": left_abs_training_weight,
                "right_abs_training_weight": right_abs_training_weight,
            }
            if growth_power is None:
                failed = True
            else:
                if worst_growth_power is None or growth_power > worst_growth_power:
                    worst_growth_power = float(growth_power)
                    worst = dict(entry)
                failed = (
                    growth_power
                    > growth_power_threshold + _GROWTH_POWER_NUMERICAL_FLOOR
                )
            if failed:
                failed_count += 1
                if len(failures) < failure_limit:
                    failures.append(entry)

    if worst_growth_power is None and failures:
        worst = dict(failures[0])
    return failed_count == 0, failed_count, worst_growth_power, worst, failures


def _profile_dict(timing: Any) -> dict[str, Any]:
    return {
        "eval_seconds": float(timing.eval_seconds),
        "python_seconds": float(timing.python_seconds),
        "havana_seconds": float(timing.havana_seconds),
        "total_profiled_seconds": float(timing.total_seconds),
        "evaluator_fraction": float(timing.evaluator_fraction),
        "python_fraction": float(timing.python_overhead_fraction),
        "havana_fraction": float(timing.havana_fraction),
        "precision_counts": timing.precision_counts,
    }


def _combine_profiles(profiles: list[dict[str, Any]]) -> dict[str, Any]:
    eval_seconds = sum(float(profile.get("eval_seconds", 0.0)) for profile in profiles)
    python_seconds = sum(float(profile.get("python_seconds", 0.0)) for profile in profiles)
    havana_seconds = sum(float(profile.get("havana_seconds", 0.0)) for profile in profiles)
    total = eval_seconds + python_seconds + havana_seconds
    counts = {
        "ordinary": 0,
        "stability": 0,
        "high_precision": 0,
    }
    for profile in profiles:
        profile_counts = profile.get("precision_counts", {})
        for key in counts:
            counts[key] += int(profile_counts.get(key, 0))
    return {
        "eval_seconds": eval_seconds,
        "python_seconds": python_seconds,
        "havana_seconds": havana_seconds,
        "total_profiled_seconds": total,
        "evaluator_fraction": eval_seconds / total if total > 0.0 else 0.0,
        "python_fraction": python_seconds / total if total > 0.0 else 0.0,
        "havana_fraction": havana_seconds / total if total > 0.0 else 0.0,
        "precision_counts": counts,
    }


def _result_from_arrays(
    sector_id: int,
    sector: Any,
    rows: np.ndarray,
    labels: list[str],
    row_distances: list[float | None],
    row_kinds: list[str],
    coeffs: np.ndarray,
    training: np.ndarray,
    elapsed: float,
    profile: dict[str, Any],
    boundary_growth_power_tolerance: float | None,
) -> EndpointTestResult:
    coeff_finite = np.isfinite(coeffs.real) & np.isfinite(coeffs.imag)
    training_finite = np.isfinite(training)
    finite = bool(np.all(coeff_finite) and np.all(training_finite))
    coeff_abs = np.abs(coeffs)
    max_abs_by_order = [
        _finite_max(coeff_abs[:, order])
        for order in range(coeff_abs.shape[1])
    ]
    row_max_abs = np.asarray(
        [
            _finite_max(coeff_abs[row_index, :]) or 0.0
            for row_index in range(coeff_abs.shape[0])
        ],
        dtype=float,
    )
    worst_index = int(np.argmax(row_max_abs)) if row_max_abs.size else 0
    (
        boundary_stability_ok,
        boundary_stability_failed_pair_count,
        max_boundary_growth_power,
        worst_boundary_stability_pair,
        boundary_stability_failures,
    ) = _boundary_stability_summary(
        rows,
        labels,
        row_distances,
        coeffs,
        training,
        boundary_growth_power_tolerance,
    )
    return EndpointTestResult(
        sector_id=int(sector_id),
        name=str(sector.name),
        status="ok" if finite and boundary_stability_ok else "failed",
        probe_count=int(rows.shape[0]),
        elapsed_seconds=elapsed,
        avg_eval_us_per_probe=elapsed * 1.0e6 / max(int(rows.shape[0]), 1),
        integration_dim=int(sector.integration_dim),
        singular_axes=[int(axis) for axis in sector.singular_axes],
        all_laurent_weights_finite=bool(np.all(coeff_finite)),
        all_training_weights_finite=bool(np.all(training_finite)),
        nonfinite_probe_count=int(
            np.count_nonzero((~np.all(coeff_finite, axis=1)) | (~training_finite))
        ),
        nonfinite_examples=_summarize_nonfinite(labels, rows, coeffs, training),
        boundary_stability_ok=boundary_stability_ok,
        boundary_growth_power_tolerance=boundary_growth_power_tolerance,
        boundary_stability_failed_pair_count=boundary_stability_failed_pair_count,
        max_boundary_growth_power=max_boundary_growth_power,
        worst_boundary_stability_pair=worst_boundary_stability_pair,
        boundary_stability_failures=boundary_stability_failures,
        max_abs_by_order=max_abs_by_order,
        max_abs_laurent_weight=_finite_max(coeff_abs),
        max_abs_training_weight=_finite_max(np.abs(training)),
        worst_probe={
            "probe": labels[worst_index] if labels else None,
            "max_abs_laurent_weight": (
                _float_or_none(row_max_abs[worst_index]) if labels else None
            ),
            "coords": [float(value) for value in rows[worst_index]] if labels else [],
        },
        by_distance=_group_stats(row_distances, coeff_abs, coeff_finite, training_finite),
        by_probe_kind=_group_stats(row_kinds, coeff_abs, coeff_finite, training_finite),
        profile=profile,
    )


def _evaluate_sector(
    processor: Any,
    sector_id: int,
    sector: Any,
    distances: tuple[float, ...],
    boundary_growth_power_tolerance: float | None,
) -> EndpointTestResult:
    rows, labels, row_distances, row_kinds = endpoint_probe_points(sector, distances)
    start = time.perf_counter()
    coeffs, training, timing = processor.evaluate_batch(sector, rows)
    elapsed = time.perf_counter() - start
    return _result_from_arrays(
        sector_id,
        sector,
        rows,
        labels,
        row_distances,
        row_kinds,
        coeffs,
        training,
        elapsed,
        _profile_dict(timing),
        boundary_growth_power_tolerance,
    )


def _init_worker() -> None:
    global _WORKER_PROCESSOR, _WORKER_SECTORS
    if _PARENT_TOPOLOGY is None or _PARENT_SECTORS is None or _PARENT_REQUEST is None:
        raise RuntimeError("endpoint test worker was not initialized from a parent topology")
    _WORKER_SECTORS = _PARENT_SECTORS
    _WORKER_PROCESSOR = _make_processor(_PARENT_TOPOLOGY, _PARENT_REQUEST)


def _worker_evaluate(task: tuple[int, tuple[float, ...], float | None]) -> EndpointTestResult:
    sector_id, distances, boundary_growth_power_tolerance = task
    if _WORKER_PROCESSOR is None or _WORKER_SECTORS is None:
        _init_worker()
    assert _WORKER_PROCESSOR is not None
    assert _WORKER_SECTORS is not None
    return _evaluate_sector(
        _WORKER_PROCESSOR,
        sector_id,
        _WORKER_SECTORS[sector_id],
        distances,
        boundary_growth_power_tolerance,
    )


def _worker_evaluate_points(task: tuple[int, int, np.ndarray]) -> EndpointChunkResult:
    sector_id, chunk_index, rows = task
    if _WORKER_PROCESSOR is None or _WORKER_SECTORS is None:
        _init_worker()
    assert _WORKER_PROCESSOR is not None
    assert _WORKER_SECTORS is not None
    start = time.perf_counter()
    coeffs, training, timing = _WORKER_PROCESSOR.evaluate_batch(
        _WORKER_SECTORS[sector_id],
        rows,
    )
    return EndpointChunkResult(
        chunk_index=chunk_index,
        coeffs=coeffs,
        training=training,
        elapsed_seconds=time.perf_counter() - start,
        profile=_profile_dict(timing),
    )


def _make_progress_bar(request: IntegralRequest, total: int) -> Any | None:
    if request.json or request.no_progress or progressbar is None:
        return None
    live_widget = progressbar.FormatCustomText(
        (
            f"{_color('pass', Fore.GREEN)}:%(passed)s "
            f"{_color('fail', Fore.RED)}:%(failed)s "
            f"{_color('current', Fore.CYAN)}:%(current)s "
            f"{_color('t', Fore.BLUE)}:%(elapsed)s "
            f"{_color('eta', Fore.BLUE)}:%(eta)s"
        ),
        {
            "passed": "0",
            "failed": "0",
            "current": "n/a",
            "elapsed": "00:00:00",
            "eta": "n/a",
        },
    )
    widgets = [
        _color("testing endpoints ", Fore.CYAN),
        progressbar.Percentage(),
        " ",
        progressbar.Bar(),
        " ",
        live_widget,
        " ",
    ]
    bar = progressbar.ProgressBar(max_value=max(total, 1), widgets=widgets)
    bar.fsd_live_widget = live_widget
    return bar


def _update_progress(
    bar: Any | None,
    done: int,
    total: int,
    passed: int,
    failed: int,
    current: str,
    start: float,
) -> None:
    if bar is None:
        return
    elapsed = max(time.perf_counter() - start, 0.0)
    eta = "n/a"
    if done > 0 and done < total:
        eta_seconds = elapsed * (total - done) / done
        eta = _format_duration(eta_seconds)
    bar.fsd_live_widget.update_mapping(
        passed=str(passed),
        failed=str(failed),
        current=current,
        elapsed=_format_duration(elapsed),
        eta=eta,
    )
    bar.update(done, force=True)


def _format_duration(seconds: float) -> str:
    seconds_i = int(max(seconds, 0.0))
    hours, rem = divmod(seconds_i, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}"


def _selected_sector_ids(request: IntegralRequest, sectors: list[Any]) -> list[int]:
    if request.sectors is None:
        return list(range(len(sectors)))
    return [int(sector_id) for sector_id in request.sectors]


def _run_serial(
    request: IntegralRequest,
    topology: Any,
    sectors: list[Any],
    selected_ids: list[int],
    progress: Any | None,
    start: float,
) -> list[EndpointTestResult]:
    processor = _make_processor(topology, request)
    results: list[EndpointTestResult] = []
    passed = 0
    failed = 0
    distances = tuple(float(value) for value in request.test_boundary_distances)
    boundary_growth_power_tolerance = request.test_boundary_growth_power_tolerance
    _update_progress(progress, 0, len(selected_ids), passed, failed, "starting", start)
    for done, sector_id in enumerate(selected_ids, start=1):
        result = _evaluate_sector(
            processor,
            sector_id,
            sectors[sector_id],
            distances,
            boundary_growth_power_tolerance,
        )
        results.append(result)
        if result.status == "ok":
            passed += 1
        else:
            failed += 1
        _update_progress(progress, done, len(selected_ids), passed, failed, result.name, start)
    return results


def _run_parallel(
    request: IntegralRequest,
    topology: Any,
    sectors: list[Any],
    selected_ids: list[int],
    progress: Any | None,
    start: float,
) -> list[EndpointTestResult]:
    global _PARENT_TOPOLOGY, _PARENT_SECTORS, _PARENT_REQUEST
    _PARENT_TOPOLOGY = topology
    _PARENT_SECTORS = sectors
    _PARENT_REQUEST = request
    distances = tuple(float(value) for value in request.test_boundary_distances)
    boundary_growth_power_tolerance = request.test_boundary_growth_power_tolerance
    tasks = [(sector_id, distances, boundary_growth_power_tolerance) for sector_id in selected_ids]
    results: list[EndpointTestResult] = []
    passed = 0
    failed = 0
    done = 0
    _update_progress(progress, done, len(tasks), passed, failed, "starting", start)
    context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=max(1, int(request.workers)),
        mp_context=context,
        initializer=_init_worker,
    ) as executor:
        future_to_sector = {
            executor.submit(_worker_evaluate, task): task[0]
            for task in tasks
        }
        for future in as_completed(future_to_sector):
            sector_id = future_to_sector[future]
            result = future.result()
            results.append(result)
            done += 1
            if result.status == "ok":
                passed += 1
            else:
                failed += 1
            _update_progress(
                progress,
                done,
                len(tasks),
                passed,
                failed,
                f"{result.name} ({sector_id})",
                start,
            )
    return results


def _chunk_bounds(length: int, chunk_count: int) -> list[tuple[int, int]]:
    count = min(max(int(chunk_count), 1), max(int(length), 1))
    indices = np.array_split(np.arange(length), count)
    bounds: list[tuple[int, int]] = []
    for chunk in indices:
        if chunk.size == 0:
            continue
        bounds.append((int(chunk[0]), int(chunk[-1]) + 1))
    return bounds


def _run_single_sector_point_parallel(
    request: IntegralRequest,
    topology: Any,
    sectors: list[Any],
    sector_id: int,
    start: float,
) -> list[EndpointTestResult]:
    global _PARENT_TOPOLOGY, _PARENT_SECTORS, _PARENT_REQUEST
    _PARENT_TOPOLOGY = topology
    _PARENT_SECTORS = sectors
    _PARENT_REQUEST = request
    distances = tuple(float(value) for value in request.test_boundary_distances)
    sector = sectors[sector_id]
    rows, labels, row_distances, row_kinds = endpoint_probe_points(sector, distances)
    worker_count = max(1, int(request.workers))
    bounds = _chunk_bounds(rows.shape[0], worker_count * 4)
    progress = _make_progress_bar(request, len(bounds))
    chunk_results: list[EndpointChunkResult] = []
    passed_chunks = 0
    failed_chunks = 0
    _update_progress(
        progress,
        0,
        len(bounds),
        passed_chunks,
        failed_chunks,
        f"{sector.name} chunks",
        start,
    )
    context = mp.get_context("fork")
    sector_start = time.perf_counter()
    try:
        with ProcessPoolExecutor(
            max_workers=max(1, int(request.workers)),
            mp_context=context,
            initializer=_init_worker,
        ) as executor:
            future_to_index = {
                executor.submit(
                    _worker_evaluate_points,
                    (sector_id, index, rows[start_index:end_index]),
                ): index
                for index, (start_index, end_index) in enumerate(bounds)
            }
            for done, future in enumerate(as_completed(future_to_index), start=1):
                chunk = future.result()
                chunk_results.append(chunk)
                chunk_ok = bool(
                    np.all(np.isfinite(chunk.coeffs.real))
                    and np.all(np.isfinite(chunk.coeffs.imag))
                    and np.all(np.isfinite(chunk.training))
                )
                if chunk_ok:
                    passed_chunks += 1
                else:
                    failed_chunks += 1
                _update_progress(
                    progress,
                    done,
                    len(bounds),
                    passed_chunks,
                    failed_chunks,
                    f"{sector.name} chunk {chunk.chunk_index + 1}/{len(bounds)}",
                    start,
                )
    finally:
        if progress is not None:
            progress.finish()
    chunk_results.sort(key=lambda item: item.chunk_index)
    coeffs = np.vstack([chunk.coeffs for chunk in chunk_results])
    training = np.concatenate([chunk.training for chunk in chunk_results])
    profile = _combine_profiles([chunk.profile for chunk in chunk_results])
    result = _result_from_arrays(
        sector_id,
        sector,
        rows,
        labels,
        row_distances,
        row_kinds,
        coeffs,
        training,
        time.perf_counter() - sector_start,
        profile,
        request.test_boundary_growth_power_tolerance,
    )
    return [result]


def _summary_table(results: list[EndpointTestResult]) -> PrettyTable:
    table = PrettyTable()
    table.field_names = [
        "sector",
        "status",
        "probes",
        "max |w|",
        "max p",
        "p fail",
        "nonfinite",
        "worst probe",
        "time [s]",
        "μs/probe",
        "prec O/S/H",
    ]
    for result in sorted(results, key=lambda item: item.sector_id):
        color = Fore.GREEN if result.status == "ok" else Fore.RED
        counts = result.profile.get("precision_counts", {})
        table.add_row(
            [
                f"{result.sector_id}:{result.name}",
                f"{color}{result.status}{Style.RESET_ALL}",
                result.probe_count,
                _format_scalar(result.max_abs_laurent_weight),
                _format_scalar(result.max_boundary_growth_power),
                result.boundary_stability_failed_pair_count,
                result.nonfinite_probe_count,
                result.worst_probe.get("probe"),
                f"{result.elapsed_seconds:.3g}",
                f"{result.avg_eval_us_per_probe:.3g}",
                (
                    f"{int(counts.get('ordinary', 0))}/"
                    f"{int(counts.get('stability', 0))}/"
                    f"{int(counts.get('high_precision', 0))}"
                ),
            ]
        )
    return table


def _format_scalar(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3g}"


def _write_report(path: str | None, report: dict[str, Any]) -> None:
    if not path:
        return
    report_path = Path(path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(report, default=_json_default, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_path.replace(report_path)


def run_endpoint_test_mode(
    request: IntegralRequest,
    topology: Any,
    sectors: list[Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run endpoint-corner finite and boundary-stability tests for selected sectors."""

    selected_ids = _selected_sector_ids(request, sectors)
    start = time.perf_counter()
    point_parallel = int(request.workers) > 1 and len(selected_ids) == 1
    if point_parallel:
        results = _run_single_sector_point_parallel(
            request,
            topology,
            sectors,
            selected_ids[0],
            start,
        )
    else:
        progress = _make_progress_bar(request, len(selected_ids))
        try:
            if int(request.workers) <= 1 or len(selected_ids) <= 1:
                results = _run_serial(request, topology, sectors, selected_ids, progress, start)
            else:
                results = _run_parallel(request, topology, sectors, selected_ids, progress, start)
        finally:
            if progress is not None:
                progress.finish()
    elapsed = time.perf_counter() - start
    passed = sum(1 for result in results if result.status == "ok")
    failed = len(results) - passed
    status = "ok" if failed == 0 else "failed"
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "sector_count": len(results),
        "passed": passed,
        "failed": failed,
        "total_seconds": elapsed,
        "workers": int(request.workers),
        "parallelism": "probe-points" if point_parallel else "sectors",
        "distances": [float(value) for value in request.test_boundary_distances],
        "boundary_growth_power_tolerance": request.test_boundary_growth_power_tolerance,
        "boundary_growth_power_threshold": (
            "tolerance * endpoint_distance_coordinate_count"
            if request.test_boundary_growth_power_tolerance is not None
            else None
        ),
        "boundary_growth_power_numerical_floor": _GROWTH_POWER_NUMERICAL_FLOOR,
        "selected_sector_ids": selected_ids,
        "summary": summary or {},
        "sectors": [result.to_dict() for result in sorted(results, key=lambda item: item.sector_id)],
    }
    _write_report(request.test_report_path, report)
    if request.json:
        print(_output_json(report))
    else:
        title_color = Fore.GREEN if status == "ok" else Fore.RED
        print(
            f"{title_color}endpoint test {status}:{Style.RESET_ALL} "
            f"{passed} passed, {failed} failed, {len(results)} sector(s), "
            f"{elapsed:.3g}s"
        )
        print(_summary_table(results))
        if request.test_report_path:
            print(f"report: {Fore.BLUE}{Path(request.test_report_path).expanduser().resolve()}{Style.RESET_ALL}")
    return report
