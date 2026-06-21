from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from boundary_test import _boundary_stability_summary
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
    }
    assert expected.issubset(set(labels))


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

    ok, failed_count, worst_growth_power, worst, failures = _boundary_stability_summary(
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
    assert worst["probe_family"] == "axis_0_low"
    assert worst["endpoint_distance_coordinate_count"] == 1
    assert worst["growth_power_threshold"] == 0.5
    assert failures[0]["probe_family"] == "axis_0_low"
