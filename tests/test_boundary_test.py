from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

import boundary_test
from boundary_test import _boundary_stability_summary
from boundary_test import EndpointTestResult
from boundary_test import endpoint_probe_points


@dataclass(frozen=True)
class DummySector:
    name: str = "dummy"
    integration_dim: int = 2
    singular_axes: tuple[int, ...] = (0,)


def test_endpoint_probe_points_include_all_axis_corners() -> None:
    rows, labels, row_distances, row_kinds = endpoint_probe_points(
        DummySector(),
        (1.0e-6, 1.0e-8),
    )

    assert rows.shape == (17, 2)
    assert len(labels) == rows.shape[0]
    assert len(set(labels)) == len(labels)
    assert labels[0] == "interior"
    assert row_distances[0] is None
    assert row_kinds[0] == "interior"

    expected = {
        "corner_00_1.0e-06",
        "corner_01_1.0e-06",
        "corner_10_1.0e-06",
        "corner_11_1.0e-06",
        "corner_00_1.0e-08",
        "corner_01_1.0e-08",
        "corner_10_1.0e-08",
        "corner_11_1.0e-08",
        "corner_0x_1.0e-06",
        "corner_x1_1.0e-08",
    }
    assert expected.issubset(set(labels))


def test_endpoint_probe_points_can_limit_simultaneous_approaches() -> None:
    sector = DummySector(integration_dim=3, singular_axes=(0, 1))
    rows, labels, row_distances, row_kinds = endpoint_probe_points(
        sector,
        (1.0e-6,),
        max_simultaneous_endpoint_approaches=1,
    )

    assert rows.shape == (7, 3)
    assert "corner_0xx_1.0e-06" in labels
    assert "corner_x1x_1.0e-06" in labels
    assert all(label == "interior" or label.count("x") == 2 for label in labels)


def test_boundary_growth_threshold_scales_with_endpoint_coordinate_count() -> None:
    labels = [
        "axis_0_low_1.0e-06",
        "axis_0_low_1.0e-08",
        "corner_00_1.0e-06",
        "corner_00_1.0e-08",
    ]
    rows = np.asarray(
        [
            [1.0e-6, 0.25],
            [1.0e-8, 0.25],
            [1.0e-6, 1.0e-6],
            [1.0e-8, 1.0e-8],
        ],
        dtype=float,
    )
    row_distances = [1.0e-6, 1.0e-8, 1.0e-6, 1.0e-8]
    coeffs = np.asarray([[1.0], [100.0], [1.0], [100.0]], dtype=np.complex128)
    training = np.asarray([1.0, 100.0, 1.0, 100.0], dtype=float)

    (
        ok,
        failed_count,
        worst_growth_power,
        worst_threshold_ratio,
        worst,
        failures,
    ) = _boundary_stability_summary(
        rows,
        labels,
        row_distances,
        coeffs,
        training,
        growth_power_tolerance=0.5,
    )

    assert not ok
    assert failed_count == 1
    assert worst_growth_power == 1.0
    assert worst_threshold_ratio > 1.0
    assert worst["probe_family"] == "axis_0_low"
    assert worst["endpoint_distance_coordinate_count"] == 1
    assert worst["growth_power_threshold"] == 0.5
    assert failures[0]["probe_family"] == "axis_0_low"


def test_boundary_growth_ignores_training_only_growth() -> None:
    labels = [
        "axis_0_low_1.0e-06",
        "axis_0_low_1.0e-08",
    ]
    rows = np.asarray(
        [
            [1.0e-6, 0.25],
            [1.0e-8, 0.25],
        ],
        dtype=float,
    )
    row_distances = [1.0e-6, 1.0e-8]
    coeffs = np.asarray([[1.0], [1.1]], dtype=np.complex128)
    training = np.asarray([1.0, 1.0e6], dtype=float)

    (
        ok,
        failed_count,
        worst_growth_power,
        worst_threshold_ratio,
        worst,
        failures,
    ) = _boundary_stability_summary(
        rows,
        labels,
        row_distances,
        coeffs,
        training,
        growth_power_tolerance=0.5,
    )

    assert ok
    assert failed_count == 0
    assert worst_growth_power < 0.5
    assert worst_threshold_ratio < 1.0
    assert np.isclose(worst["training_growth_power"], 3.0)
    assert failures == []


def test_serial_endpoint_scan_returns_partial_results_on_interrupt(
    monkeypatch,
) -> None:
    request = SimpleNamespace(
        test_boundary_distances=(1.0e-6,),
        test_boundary_growth_power_tolerance=0.5,
    )
    sectors = [DummySector(name="s0"), DummySector(name="s1")]

    def fake_make_processor(_topology, _request):
        return object()

    def fake_evaluate_sector(_processor, sector_id, sector, *_args):
        if sector_id == 1:
            raise KeyboardInterrupt
        return EndpointTestResult(
            sector_id=sector_id,
            name=sector.name,
            status="ok",
            probe_count=1,
            elapsed_seconds=0.01,
            avg_eval_us_per_probe=10.0,
            integration_dim=sector.integration_dim,
            singular_axes=list(sector.singular_axes),
            all_laurent_weights_finite=True,
            all_training_weights_finite=True,
            nonfinite_probe_count=0,
            nonfinite_examples=[],
            boundary_stability_ok=True,
            boundary_growth_power_tolerance=0.5,
            boundary_stability_failed_pair_count=0,
            max_boundary_growth_power=0.0,
            max_boundary_growth_threshold_ratio=0.0,
            worst_boundary_stability_pair={},
            boundary_stability_failures=[],
            max_abs_by_order=[1.0],
            max_abs_laurent_weight=1.0,
            max_abs_training_weight=1.0,
            worst_probe={"probe": "interior"},
            by_distance=[],
            by_probe_kind=[],
            profile={"precision_counts": {"ordinary": 1}},
        )

    monkeypatch.setattr(boundary_test, "_make_processor", fake_make_processor)
    monkeypatch.setattr(boundary_test, "_evaluate_sector", fake_evaluate_sector)

    result = boundary_test._run_serial(
        request,
        topology=None,
        sectors=sectors,
        selected_ids=[0, 1],
        progress=None,
        start=0.0,
    )

    assert result.interrupted
    assert [item.sector_id for item in result.results] == [0]


def test_endpoint_scan_retries_failed_sector_with_scaled_distances(
    monkeypatch,
) -> None:
    request = SimpleNamespace(
        json=True,
        no_progress=True,
        workers=1,
        test_boundary_distances=(1.0e-4, 1.0e-6),
        test_boundary_growth_power_tolerance=0.5,
        test_boundary_retry_scales=(1.0e-2,),
        test_report_path=None,
        sectors=None,
    )
    sectors = [DummySector(name="s0")]
    calls = []

    def make_result(status: str, distances: tuple[float, ...]) -> EndpointTestResult:
        return EndpointTestResult(
            sector_id=0,
            name="s0",
            status=status,
            probe_count=2,
            elapsed_seconds=0.01,
            avg_eval_us_per_probe=10.0,
            integration_dim=2,
            singular_axes=[0],
            all_laurent_weights_finite=True,
            all_training_weights_finite=True,
            nonfinite_probe_count=0,
            nonfinite_examples=[],
            boundary_stability_ok=status == "ok",
            boundary_growth_power_tolerance=0.5,
            boundary_stability_failed_pair_count=0 if status == "ok" else 1,
            max_boundary_growth_power=0.0 if status == "ok" else 1.0,
            max_boundary_growth_threshold_ratio=0.0 if status == "ok" else 2.0,
            worst_boundary_stability_pair={"distances": list(distances)},
            boundary_stability_failures=[],
            max_abs_by_order=[1.0],
            max_abs_laurent_weight=1.0,
            max_abs_training_weight=1.0,
            worst_probe={"probe": "interior"},
            by_distance=[],
            by_probe_kind=[],
            profile={"precision_counts": {"ordinary": 2}},
        )

    def fake_scan_once(scan_request, *_args):
        distances = tuple(float(value) for value in scan_request.test_boundary_distances)
        calls.append(distances)
        status = "failed" if len(calls) == 1 else "ok"
        return boundary_test.EndpointRunResult(
            results=[make_result(status, distances)],
            interrupted=False,
        )

    monkeypatch.setattr(boundary_test, "_run_endpoint_scan_once", fake_scan_once)
    monkeypatch.setattr(boundary_test, "_endpoint_test_total", lambda *_args, **_kwargs: 2)

    report = boundary_test.run_endpoint_test_mode(
        request,
        topology=None,
        sectors=sectors,
        summary={},
    )

    assert report["status"] == "ok"
    assert len(calls) == 2
    assert np.allclose(calls[0], (1.0e-4, 1.0e-6))
    assert np.allclose(calls[1], (1.0e-6, 1.0e-8))
    assert report["retry_attempts"][0]["scale"] == 1.0e-2
    assert report["retry_attempts"][0]["sector_statuses"] == {"0": "ok"}
