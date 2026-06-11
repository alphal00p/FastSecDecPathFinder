"""Pytest smoke coverage for the supported FSD_v2 integral modes."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FSD import compute_benchmark_quietly, validate_request
from definitions import IntegralRequest
from formatting import apply_global_convention, pull_value, selected_prefactor_values
from integrand import build_topology
from integrator import integrate
from sectors_generator import generate_sectors


def make_request(**overrides: Any) -> IntegralRequest:
    """Build a deterministic, low-statistics integration request for tests."""
    data = {
        "integral": "triangle",
        "mode": "massive",
        "s": None,
        "s12": None,
        "s23": None,
        "m": 1.0,
        "gamma_scheme": "oneloop",
        "prefactor_convention": "raw",
        "seed": 1,
        "max_iter": 1,
        "min_iter": 1,
        "samples_per_iter": 4096,
        "batch_size": 2048,
        "target_rel_accuracy": None,
        "min_error": 0.0,
        "bins": 32,
        "workers": 1,
        "jit_compile_evaluators": False,
        "show_stats": False,
        "no_progress": True,
        "quiet_summary": True,
        "json": True,
        "mu": None,
        "onshell_threshold": None,
    }
    data.update(overrides)
    return IntegralRequest(**data)


def assert_finite_complex(value: complex) -> None:
    """Assert that both complex components are finite."""
    z = complex(value)
    assert math.isfinite(z.real)
    assert math.isfinite(z.imag)


@pytest.mark.parametrize(
    ("integral_request", "expected_sector_count", "expected_singular_axis_counts"),
    [
        pytest.param(
            make_request(integral="triangle", mode="massive", s=1.0, m=1.0),
            2,
            [0, 0],
            id="triangle-massive",
        ),
        pytest.param(
            make_request(integral="triangle", mode="massless", s=-1.0, m=0.0),
            2,
            [2, 2],
            id="triangle-massless",
        ),
        pytest.param(
            make_request(integral="box", mode="massive", s12=0.5, s23=0.7, m=1.0),
            4,
            [0, 0, 0, 0],
            id="box-massive",
        ),
        pytest.param(
            make_request(integral="box", mode="massless", s12=-1.0, s23=-2.0, m=0.0),
            12,
            [1, 2, 2] * 4,
            id="box-massless",
        ),
    ],
)
def test_supported_integrals_match_oneloopbridge_smoke(
    integral_request: IntegralRequest,
    expected_sector_count: int,
    expected_singular_axis_counts: list[int],
) -> None:
    """Run all supported modes and compare coefficients with MC-aware pulls."""
    validate_request(integral_request)
    topology = build_topology(integral_request)
    sectors = generate_sectors(integral_request)

    assert len(sectors) == expected_sector_count
    assert [len(sector.singular_axes) for sector in sectors] == expected_singular_axis_counts

    benchmark = compute_benchmark_quietly(integral_request)
    result = integrate(integral_request, topology, sectors, benchmark)

    assert result.samples == integral_request.samples_per_iter
    assert result.eval_seconds >= 0.0
    assert result.python_seconds >= 0.0
    assert result.havana_seconds >= 0.0

    raw_coeffs, raw_errors = apply_global_convention(
        result.raw_sector_coeffs,
        result.raw_sector_errors,
        integral_request,
    )
    display_coeffs, display_errors, display_benchmark, _ = selected_prefactor_values(
        integral_request,
        raw_coeffs,
        raw_errors,
        benchmark,
    )

    for coeff, error, reference in zip(display_coeffs, display_errors, display_benchmark):
        assert_finite_complex(coeff)
        assert_finite_complex(error)
        assert_finite_complex(reference)
        pull = pull_value(coeff - reference, error)
        assert pull is not None
        assert pull <= 8.0


@pytest.mark.parametrize(
    "integral_request",
    [
        pytest.param(
            make_request(integral="triangle", mode="massless", s=1.0, m=0.0),
            id="triangle-massless-timelike",
        ),
        pytest.param(
            make_request(integral="box", mode="massless", s12=1.0, s23=2.0, m=0.0),
            id="box-massless-timelike",
        ),
    ],
)
def test_massless_timelike_kinematics_are_rejected(integral_request: IntegralRequest) -> None:
    """Massless timelike cases need contour deformation and are not supported."""
    with pytest.raises(ValueError, match="contour deformation|threshold regularization"):
        validate_request(integral_request)
