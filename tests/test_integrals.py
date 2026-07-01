"""Pytest smoke coverage for the supported FSD integral modes."""

from __future__ import annotations

import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_source_does_not_hard_code_special_constant_decimals() -> None:
    """Precision-sensitive source must get special constants from Symbolica."""
    checked_paths = [PROJECT_ROOT / "FSD.py", *sorted(SRC_ROOT.glob("*.py"))]
    forbidden_patterns = [
        r"\d+\.\d{12,}",
        r"math\.pi",
        r"mp\.euler",
        r"mp\.zeta",
        r"3\.14159",
        r"0\.57721",
    ]
    combined = re.compile("|".join(f"(?:{pattern})" for pattern in forbidden_patterns))
    offenders: list[str] = []
    for path in checked_paths:
        text = path.read_text()
        for match in combined.finditer(text):
            line_no = text[: match.start()].count("\n") + 1
            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{line_no}")
    assert offenders == []

from FSD import (
    _align_coefficients,
    _align_coefficients_by_order,
    _merge_pysecdec_package_generation_timings,
    build_request,
    compute_benchmark_quietly,
    configure_laurent_range,
    main,
    parse_args,
    resolve_target,
    validate_request,
)
from definitions import IntegralRequest, TargetDefinition
from definitions import HotPathTiming
from definitions import EpsilonExpansion, ParametricRepresentation
from generation_timing import GenerationTimings
from dot_parser import parse_dot_file
from dot_topology import GammaLoopDotTopologyBuilder, clear_dot_bundle_cache
import evaluator_utils as evaluator_utils_module
from formatting import (
    _ellipsis_table_text,
    apply_global_convention,
    combine_uncorrelated_errors,
    make_output,
    print_pysecdec_result_table,
    print_result_table,
    pull_value,
    selected_prefactor_values,
    summary_data,
)
import integrand as integrand_module
import subtraction_formula as subtraction_formula_module
from integrand import (
    EndpointProjectorFormulaDefinition,
    RegularTaylorFormulaDefinition,
    SectorProcessor,
    TopologyDefinition,
    _multi_set_cache_key,
    _decimal_complex,
    _expr_derivative_coefficient,
    _expr_series_mul,
    _expr_series_pow_real,
    _multi_indices,
    _regular_taylor_signature_axis_count,
    _regular_taylor_signature_volume,
    _series_log_allowed,
    _series_mul,
    _series_mul_allowed,
    _series_pow_real_and_log_allowed,
    _series_pow_real_allowed,
    build_regular_taylor_formula,
    build_subtraction_formula_legacy,
    build_topology,
)
from subtraction_formula import build_endpoint_projector_formula_symbolica
from subtraction_formula import endpoint_projector_formula_has_curated_cache
import integrator as integrator_module
from integrator import EvaluationBatch, _evaluate_records, integrate
from kinematics import load_kinematics
from kinematics import KinematicsDefinition
from numerator_reducer import parse_dot_product_numerator, reduce_dot_product_numerator
from prepared_bundle import load_prepared_bundle, save_prepared_bundle
from qmc_lattice import (
    actual_lattice_point_count_for_dimension,
    cbcpt_dn1_shifted_lattice_points,
    is_power_of_two,
    max_lattice_point_count,
    pysecdec_default_shifted_lattice_points,
    pysecdec_default_vector_info,
    qmcpy_shifted_lattice_points,
    shifted_lattice_point_slice,
)
from pysecdec_bridge import (
    _make_loop_integral,
    _prefactor_series,
    _polynomial_to_expr,
    _polynomial_to_symbolica_text,
    require_pysecdec,
)
from result_io import print_saved_results, target_from_result_file, write_result_json
from sectors_generator import SectorDefinition, generate_sectors
from symbolica import E
from symbolica import S


def _path_from_run_file(run_file: Path, key: str, fallback: Path) -> Path:
    """Resolve a path-valued option from an FSD run YAML file."""
    try:
        raw = yaml.safe_load(run_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return fallback
    if not isinstance(raw, dict) or key not in raw:
        return fallback
    value = raw[key]
    if value is None:
        return fallback
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (run_file.parent / path).resolve()


def make_request(**overrides: Any) -> IntegralRequest:
    """Build a deterministic, low-statistics integration request for tests."""
    data = {
        "run_file": None,
        "integral": "triangle",
        "dot_file": None,
        "kinematics_file": None,
        "graph_name": None,
        "sector_method": "iterative",
        "normaliz_executable": None,
        "dot_engine": "fsd",
        "numerator_reducer": "symbolica",
        "sectors": None,
        "pysecdec_workdir": ".pysecdec_build",
        "pysecdec_epsrel": 1.0e-2,
        "pysecdec_maxeval": 1000,
        "keep_pysecdec_workdir": False,
        "show_pysecdec_output": False,
        "progress_value_order": "eps^0",
        "max_eps_order": 0,
        "target_args": None,
        "refresh_target": False,
        "show_results": None,
        "sort_sector_results": "index",
        "result_path": str(Path.cwd() / "result.json"),
        "report_path": None,
        "log_level": "WARNING",
        "log_file": None,
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
        "sampling_mode": "havana",
        "democratic_samples_per_sector": 1000,
        "target_rel_accuracy": None,
        "min_error": 0.0,
        "bins": 32,
        "workers": 1,
        "jit_compile_evaluators": False,
        "evaluator_compile_mode": "eager",
        "real_evaluator": True,
        "jit_direct_translation": False,
        "dual_evaluator_mode": "pregenerate",
        "subtraction_backend": "formula",
        "ibp_reduce_to_log_endpoint": False,
        "ibp_power_goal": None,
        "direct_projector_cache_term_threshold": 54,
        "allow_fallback_for_missing_caches": True,
        "force_regular_taylor_formulas": False,
        "regular_taylor_signature_limit": 256,
        "regular_taylor_formula_volume_limit": 8192,
        "regular_taylor_formula_axis_limit": 6,
        "chain_rule_formula_signature_limit": 256,
        "chain_rule_formula_output_length_limit": 0,
        "stability_threshold": 1.0e-3,
        "medium_precision_stability_threshold": 1.0e-6,
        "high_precision_stability_threshold": 1.0e-8,
        "stability_precision": 32,
        "medium_precision_stability_precision": 100,
        "high_precision_stability_precision": 1000,
        "max_weight_precision_xi": 0.9,
        "show_stats": False,
        "no_progress": True,
        "quiet_summary": True,
        "json": True,
        "mu": None,
        "onshell_threshold": None,
        "qmc_initial_samples_per_iter": 1024,
        "qmc_initial_shifts": 16,
        "qmc_max_samples_per_iter": 4096,
        "qmc_lattice_backend": "qmcpy",
        "qmc_order": "linear",
        "qmc_correlate_sectors": True,
    }
    data.update(overrides)
    return IntegralRequest(**data)


def assert_finite_complex(value: complex) -> None:
    """Assert that both complex components are finite."""
    z = complex(value)
    assert math.isfinite(z.real)
    assert math.isfinite(z.imag)


def test_sector_stat_table_values_are_ellipsized_to_fixed_width() -> None:
    """Long sector-stat values should not widen the pre-integration table."""
    long_value = {"sector_value": "x" * 120, "other": "y" * 120}
    clipped = _ellipsis_table_text(long_value, width=50)

    assert len(clipped) == 50
    assert clipped.endswith("[...]")
    assert "x" * 60 not in clipped


def test_pysecdec_package_timings_merge_into_generation_summary() -> None:
    """Native pySecDec FORM/codegen and compile time are generation cost."""
    summary = {
        "generation_timings": {
            "details": [
                {
                    "name": "DOT parse",
                    "seconds": 0.25,
                    "detail": "",
                }
            ]
        }
    }
    pysecdec_timings = GenerationTimings()
    pysecdec_timings.add(
        "pySecDec package generation",
        10.0,
        "captured output: pysecdec_generation.log",
    )
    pysecdec_timings.add("pySecDec package compile", 2.0, "make pylink")
    pysecdec_timings.add("pySecDec integration", 5.0, "")

    _merge_pysecdec_package_generation_timings(summary, pysecdec_timings)

    details = summary["generation_timings"]["details"]
    names = [record["name"] for record in details]
    assert "DOT parse" in names
    assert "pySecDec package generation" in names
    assert "pySecDec package compile" in names
    assert "pySecDec integration" not in names
    headline = {
        record["name"]: record["seconds"]
        for record in summary["generation_timings"]["headline"]
    }
    assert headline["pySecDec package generation/compile"] == pytest.approx(12.0)


def test_pysecdec_result_table_avoids_raw_dict_dump(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Native pySecDec-only mode should render a table, not an internal dict."""
    output = {
        "prefactor_convention": "pysecdec",
        "summary": {
            "generation_timings": {
                "headline": [
                    {
                        "name": "pySecDec package generation/compile",
                        "seconds": 1.25,
                    }
                ],
                "total": 1.25,
            }
        },
        "pysecdec": {
            "orders": [-2, -1, 0],
            "coeffs": [1.0 + 0.0j, 2.0 + 0.0j, -3.0 + 0.0j],
            "errors": [0.1 + 0.0j, 0.2 + 0.0j, 0.3 + 0.0j],
            "timings": {
                "records": [
                    {
                        "name": "pySecDec package generation",
                        "seconds": 1.0,
                    },
                    {
                        "name": "pySecDec package compile",
                        "seconds": 0.25,
                    },
                    {
                        "name": "pySecDec integration",
                        "seconds": 0.5,
                    },
                ]
            },
        },
    }

    print_pysecdec_result_table(output)
    rendered = capsys.readouterr().out

    assert "pySecDec pysecdec" in rendered
    assert "eps^-2" in rendered
    assert "MC err" in rendered
    assert "pySecDec timing" in rendered
    assert "'summary':" not in rendered
    assert "{'pysecdec':" not in rendered


def test_progress_field_padding_is_fixed_width_and_ansi_safe() -> None:
    """Progress values should have stable visible widths, even with colors."""
    plain = integrator_module._fit_progress_field("46.2", 8)
    colored = integrator_module._fit_progress_field("\x1b[34m/ 0.03\x1b[0m", 10)
    long = integrator_module._fit_progress_field("123456789abcdef", 8)

    assert integrator_module._visible_progress_len(plain) == 8
    assert integrator_module._visible_progress_len(colored) == 10
    assert integrator_module._visible_progress_len(long) == 8
    assert long.endswith("…")


def test_qmc_progress_fields_are_compact_and_fixed_width() -> None:
    """QMC progress should expose step/local scheduler state compactly."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        sampling_mode="qmc",
        qmc_refine_sectors="adaptive",
    )
    widths = integrator_module._progress_widths(request)
    step = integrator_module._fit_progress_field(
        f"{integrator_module.format_sample_count(17_800_000)}/"
        f"{integrator_module.format_sample_count(84_800_000)}",
        widths["qmc_step"],
    )
    lattice = integrator_module._fit_progress_field(
        integrator_module.format_qmc_lattice(25_000, 32),
        widths["qmc_lattice"],
    )

    assert integrator_module._visible_progress_len(step) == widths["qmc_step"]
    assert integrator_module._visible_progress_len(lattice) == widths["qmc_lattice"]
    assert "2.04B" not in step


def test_worker_initializer_ignores_sigint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forked workers must not dump KeyboardInterrupt tracebacks on Ctrl+C."""
    calls: list[tuple[int, object]] = []
    monkeypatch.setattr(
        integrator_module.signal,
        "signal",
        lambda signum, handler: calls.append((signum, handler)),
    )

    integrator_module._ignore_worker_sigint()

    assert calls == [(signal.SIGINT, signal.SIG_IGN)]


def test_nonfinite_rows_are_retried_at_high_precision_before_training() -> None:
    """A NaN f64 row must be rescued before Havana receives training data."""

    class NaNThenFiniteProcessor(SectorProcessor):
        def __init__(self) -> None:
            self.high_precision_stability_precision = 100

        def _evaluate_batch_impl(self, _sector, rows, timing):
            if timing.precision_digits is None:
                return (
                    np.full((rows.shape[0], 2), complex(float("nan"), float("nan"))),
                    np.full(rows.shape[0], float("nan")),
                )
            return (
                np.tile(np.array([1.0 + 0.0j, 2.0 + 0.0j]), (rows.shape[0], 1)),
                np.full(rows.shape[0], 2.0),
            )

    processor = NaNThenFiniteProcessor()
    sector = SimpleNamespace(name="toy-sector", singular_axes=[0])
    coeffs, training, timing = processor._evaluate_precision_chunk(
        sector,
        np.array([[1.0e-6]], dtype=float),
        precision_digits=None,
        precision_tier="ordinary",
    )

    assert np.all(np.isfinite(coeffs))
    assert np.all(np.isfinite(training))
    assert timing.ordinary_precision_samples == 0
    assert timing.high_precision_samples == 1


def test_endpoint_precision_thresholds_select_three_rescue_tiers() -> None:
    """Endpoint distance masks must separate 32-, 100-, and 1000-digit tiers."""

    class CountingProcessor(SectorProcessor):
        def __init__(self) -> None:
            self.topology = SimpleNamespace(
                coefficient_count=1,
                laurent_max_order=0,
                endpoint_power=lambda _sector, _axis: SimpleNamespace(base=-1.0),
            )
            self.stability_threshold = 1.0e-3
            self.medium_precision_stability_threshold = 1.0e-6
            self.high_precision_stability_threshold = 1.0e-8
            self.stability_precision = 32
            self.medium_precision_stability_precision = 100
            self.high_precision_stability_precision = 1000

        def _evaluate_batch_impl(self, _sector, rows, _timing):
            return (
                np.ones((rows.shape[0], 1), dtype=np.complex128),
                np.ones(rows.shape[0], dtype=float),
            )

    processor = CountingProcessor()
    sector = SimpleNamespace(name="toy-sector", integration_dim=1, singular_axes=[0])
    _coeffs, _training, timing = processor.evaluate_batch(
        sector,
        np.array([[2.0e-3], [5.0e-4], [5.0e-7], [5.0e-9]], dtype=float),
    )

    assert timing.precision_counts == {
        "ordinary": 1,
        "stability": 1,
        "medium_precision": 1,
        "high_precision": 1,
    }


def test_large_weight_rows_are_retried_at_high_precision() -> None:
    """Rows close to a sector's max weight are recomputed before accumulation."""

    class LargeWeightProcessor:
        def __init__(self) -> None:
            self.topology = SimpleNamespace(coefficient_count=1)
            self.high_precision_stability_precision = 100
            self.precision_calls = 0

        def evaluate_batch(self, _sector, rows):
            timing = HotPathTiming()
            timing.add_precision_samples(ordinary=rows.shape[0])
            coeffs = np.array([[1.0 + 0.0j], [20.0 + 0.0j]], dtype=np.complex128)
            return coeffs[: rows.shape[0]], np.abs(coeffs[: rows.shape[0], 0]), timing

        def evaluate_batch_at_precision(self, _sector, rows, precision_digits):
            self.precision_calls += 1
            assert precision_digits == 100
            timing = HotPathTiming(precision_digits=precision_digits)
            timing.add_precision_samples(high=rows.shape[0])
            coeffs = np.full((rows.shape[0], 1), 3.0 + 0.0j, dtype=np.complex128)
            return coeffs, np.full(rows.shape[0], 3.0), timing

    processor = LargeWeightProcessor()
    sectors = [SimpleNamespace(name="S0", integration_dim=1)]
    batch = EvaluationBatch(
        indices=np.array([0, 1], dtype=int),
        sector_indices=np.array([0, 0], dtype=int),
        coords=np.array([[0.2], [0.4]], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
        sector_max_abs=np.array([[10.0]], dtype=float),
        max_weight_precision_xi=0.9,
    )

    _indices, weighted, training, precision_counts, timing = _evaluate_records(
        processor,
        sectors,
        batch,
    )

    assert processor.precision_calls == 1
    assert weighted[:, 0] == pytest.approx([1.0 + 0.0j, 3.0 + 0.0j])
    assert training == pytest.approx([1.0, 3.0])
    assert precision_counts[0].tolist() == [1, 0, 0, 1]
    assert timing.high_precision_samples == 1


def test_large_weight_guard_uses_previous_sector_maximum() -> None:
    """The current batch maximum must not become its own replay threshold."""

    class FirstBatchProcessor:
        def __init__(self) -> None:
            self.topology = SimpleNamespace(coefficient_count=1)
            self.high_precision_stability_precision = 100
            self.precision_calls = 0

        def evaluate_batch(self, _sector, rows):
            timing = HotPathTiming()
            timing.add_precision_samples(ordinary=rows.shape[0])
            coeffs = np.array([[1.0 + 0.0j], [20.0 + 0.0j]], dtype=np.complex128)
            return coeffs[: rows.shape[0]], np.abs(coeffs[: rows.shape[0], 0]), timing

        def evaluate_batch_at_precision(self, _sector, rows, precision_digits):
            self.precision_calls += 1
            timing = HotPathTiming(precision_digits=precision_digits)
            timing.add_precision_samples(high=rows.shape[0])
            coeffs = np.full((rows.shape[0], 1), 3.0 + 0.0j, dtype=np.complex128)
            return coeffs, np.full(rows.shape[0], 3.0), timing

    processor = FirstBatchProcessor()
    sectors = [SimpleNamespace(name="S0", integration_dim=1)]
    batch = EvaluationBatch(
        indices=np.array([0, 1], dtype=int),
        sector_indices=np.array([0, 0], dtype=int),
        coords=np.array([[0.2], [0.4]], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
        sector_max_abs=np.array([[0.0]], dtype=float),
        max_weight_precision_xi=0.9,
    )

    _indices, weighted, training, precision_counts, timing = _evaluate_records(
        processor,
        sectors,
        batch,
    )

    assert processor.precision_calls == 0
    assert weighted[:, 0] == pytest.approx([1.0 + 0.0j, 20.0 + 0.0j])
    assert training == pytest.approx([1.0, 20.0])
    assert precision_counts[0].tolist() == [2, 0, 0, 0]
    assert timing.high_precision_samples == 0


def prepare_generated_evaluators(
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    mode: str = "pregenerate",
    subtraction_backend: str = "formula",
) -> None:
    """Mirror the CLI generation phase for tests that bypass ``main``."""
    topology.prepare_dual_evaluators(sectors, mode)
    if subtraction_backend == "formula":
        topology.prepare_subtraction_formulas(sectors)
    elif subtraction_backend == "projector-formula":
        topology.prepare_endpoint_projector_formulas(sectors)
        topology.prepare_regular_taylor_formulas(sectors)
        topology.prepare_chain_rule_formulas(sectors)
    elif subtraction_backend != "recursive":
        raise ValueError(f"unsupported test subtraction backend {subtraction_backend!r}")


def test_two_stage_sector_backend_matches_projector_box(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two-evaluator sector path must reproduce the existing projector algebra."""
    monkeypatch.setenv("FSD_TWO_STAGE_SECTOR_CACHE_DIR", str(tmp_path / "two_stage_cache"))
    request = make_request(
        integral="box",
        mode="massless",
        s12=-1.0,
        s23=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
        ibp_reduce_to_log_endpoint=True,
        sector_evaluator_backend="two-stage-explicit",
    )
    two_stage_topology = build_topology(request)
    two_stage_sectors = generate_sectors(request)
    two_stage_topology.prepare_endpoint_projector_formulas(two_stage_sectors)
    two_stage_topology.prepare_two_stage_sector_formulas(two_stage_sectors)

    projector_request = replace(request, sector_evaluator_backend="projector")
    projector_topology = build_topology(projector_request)
    projector_sectors = generate_sectors(projector_request)
    prepare_generated_evaluators(
        projector_topology,
        projector_sectors,
        mode="pregenerate",
        subtraction_backend="projector-formula",
    )

    two_stage_processor = SectorProcessor(
        two_stage_topology,
        subtraction_backend="projector-formula",
    )
    projector_processor = SectorProcessor(
        projector_topology,
        subtraction_backend="projector-formula",
    )
    max_diff = 0.0
    checked = 0
    for two_stage_sector, projector_sector in zip(two_stage_sectors, projector_sectors):
        if not two_stage_sector.singular_axes:
            continue
        point = np.full((1, two_stage_sector.integration_dim), 0.41)
        two_stage_coeffs, _training, _timing = two_stage_processor.evaluate_batch(
            two_stage_sector,
            point,
        )
        projector_coeffs, _training, _timing = projector_processor.evaluate_batch(
            projector_sector,
            point,
        )
        max_diff = max(max_diff, float(np.max(np.abs(two_stage_coeffs - projector_coeffs))))
        checked += 1
    assert checked == 12
    assert max_diff < 1.0e-12


def test_explicit_sector_backend_matches_projector_triangle() -> None:
    """The single-evaluator explicit path must reproduce projector subtraction."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
        sector_evaluator_backend="explicit",
    )
    explicit_topology = build_topology(request)
    explicit_sectors = generate_sectors(request)
    explicit_topology.prepare_endpoint_projector_formulas(explicit_sectors)
    explicit_topology.prepare_explicit_sector_formulas(explicit_sectors)

    projector_request = replace(request, sector_evaluator_backend="projector")
    projector_topology = build_topology(projector_request)
    projector_sectors = generate_sectors(projector_request)
    prepare_generated_evaluators(
        projector_topology,
        projector_sectors,
        mode="pregenerate",
        subtraction_backend="projector-formula",
    )

    explicit_processor = SectorProcessor(
        explicit_topology,
        subtraction_backend="projector-formula",
    )
    projector_processor = SectorProcessor(
        projector_topology,
        subtraction_backend="projector-formula",
    )
    max_diff = 0.0
    checked = 0
    for explicit_sector, projector_sector in zip(explicit_sectors, projector_sectors):
        point = np.full((1, explicit_sector.integration_dim), 0.37)
        explicit_coeffs, _training, _timing = explicit_processor.evaluate_batch(
            explicit_sector,
            point,
        )
        projector_coeffs, _training, _timing = projector_processor.evaluate_batch(
            projector_sector,
            point,
        )
        max_diff = max(max_diff, float(np.max(np.abs(explicit_coeffs - projector_coeffs))))
        checked += 1

    assert checked == len(explicit_sectors)
    assert max_diff < 1.0e-12


def test_explicit_qmc_components_reconstruct_triangle_coefficients() -> None:
    """Support-resolved QMC components must sum back to explicit coefficients."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
        sector_evaluator_backend="explicit",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    topology.enable_qmc_component_outputs = True
    topology.prepare_endpoint_projector_formulas(sectors)
    topology.prepare_explicit_sector_formulas(sectors)
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")

    for sector in sectors:
        point = np.array([[0.37, 0.43]], dtype=float)
        coeffs, _training, _timing = processor.evaluate_batch(sector, point)
        component_values, component_layout = processor.explicit_qmc_component_batch(
            sector,
            point,
            HotPathTiming(),
        )
        reconstructed = np.zeros_like(coeffs)
        for component_index, (coefficient_index, _axes) in enumerate(component_layout):
            reconstructed[:, int(coefficient_index)] += component_values[:, component_index]
        assert np.max(np.abs(reconstructed - coeffs)) < 1.0e-12


def test_explicit_qmc_optimized_evaluator_matches_weighted_components() -> None:
    """Raw-QMC optimized evaluators fold in Korobov coordinates and weights."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
        sector_evaluator_backend="explicit",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    topology.enable_qmc_component_outputs = True
    topology.enable_qmc_optimized_evaluators = True
    topology.qmc_korobov_alpha = 3
    topology.prepare_endpoint_projector_formulas(sectors)
    topology.prepare_explicit_sector_formulas(sectors)
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")

    for sector in sectors:
        formula = topology.explicit_sector_formula_for(sector)
        assert formula is not None
        assert formula.qmc_optimized_evaluators
        support_axes = tuple(range(int(sector.integration_dim)))
        component_indices = tuple(range(len(formula.qmc_component_layout)))
        raw = np.array([[0.23, 0.41]], dtype=float)
        transformed, weight = integrator_module.korobov_transform(raw, 3)
        component_values, component_layout = processor.explicit_qmc_component_batch(
            sector,
            transformed,
            HotPathTiming(),
        )
        optimized_values, optimized_layout = processor.explicit_qmc_optimized_component_batch(
            sector,
            raw,
            support_axes,
            component_indices,
            3,
            HotPathTiming(),
        )
        assert optimized_layout == component_layout
        assert np.max(np.abs(optimized_values - component_values * weight[:, None])) < 1.0e-11


def test_explicit_sector_backend_matches_projector_box_with_numerator() -> None:
    """Explicit singular sectors include numerator epsilon-polynomial Taylor data."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box_rank2_numerator.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        mode="massless",
        subtraction_backend="projector-formula",
        sector_evaluator_backend="explicit",
        prefactor_convention="pysecdec",
    )
    explicit_topology = build_topology(request)
    explicit_sectors = generate_sectors(request)
    configure_laurent_range(request, explicit_topology, explicit_sectors)
    explicit_topology.prepare_endpoint_projector_formulas(explicit_sectors)
    explicit_topology.prepare_explicit_sector_formulas(explicit_sectors)

    projector_request = replace(request, sector_evaluator_backend="projector")
    projector_topology = build_topology(projector_request)
    projector_sectors = generate_sectors(projector_request)
    configure_laurent_range(projector_request, projector_topology, projector_sectors)
    prepare_generated_evaluators(
        projector_topology,
        projector_sectors,
        mode="pregenerate",
        subtraction_backend="projector-formula",
    )

    explicit_processor = SectorProcessor(
        explicit_topology,
        subtraction_backend="projector-formula",
        stability_threshold=0.0,
        high_precision_stability_threshold=0.0,
    )
    projector_processor = SectorProcessor(
        projector_topology,
        subtraction_backend="projector-formula",
        stability_threshold=0.0,
        high_precision_stability_threshold=0.0,
    )
    max_diff = 0.0
    for index, (explicit_sector, projector_sector) in enumerate(
        zip(explicit_sectors, projector_sectors)
    ):
        rng = np.random.default_rng(1701 + index)
        point = 0.2 + 0.6 * rng.random((2, explicit_sector.integration_dim))
        explicit_coeffs, _training, _timing = explicit_processor.evaluate_batch(
            explicit_sector,
            point,
        )
        projector_coeffs, _training, _timing = projector_processor.evaluate_batch(
            projector_sector,
            point,
        )
        max_diff = max(max_diff, float(np.max(np.abs(explicit_coeffs - projector_coeffs))))

    assert max_diff < 1.0e-12


def test_dot_explicit_bundle_cache_does_not_contaminate_projector_backend() -> None:
    """DOT bundle reuse must not make normal projector mode dispatch through explicit formulas."""
    clear_dot_bundle_cache()
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/double_box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/double_box_kinematics.yaml"),
        sector_method="iterative",
        subtraction_backend="projector-formula",
        sector_evaluator_backend="explicit",
        prefactor_convention="pysecdec",
    )
    explicit_topology = build_topology(request)
    explicit_sectors = generate_sectors(request)
    configure_laurent_range(request, explicit_topology, explicit_sectors)
    explicit_topology.prepare_endpoint_projector_formulas(explicit_sectors)
    explicit_topology.prepare_explicit_sector_formulas(explicit_sectors)
    assert len(explicit_topology._explicit_sector_formulas) == len(explicit_sectors)

    projector_request = replace(request, sector_evaluator_backend="projector")
    projector_topology = build_topology(projector_request)
    projector_sectors = generate_sectors(projector_request)
    configure_laurent_range(projector_request, projector_topology, projector_sectors)
    prepare_generated_evaluators(
        projector_topology,
        projector_sectors,
        mode="pregenerate",
        subtraction_backend="projector-formula",
    )

    assert len(projector_topology._explicit_sector_formulas) == 0
    assert projector_topology.explicit_sector_formula_for(projector_sectors[0]) is None


def test_dual_evaluator_cli_modes_are_mutually_exclusive_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI flags select exactly one dual evaluator generation mode."""
    monkeypatch.setattr(sys, "argv", ["FSD.py"])
    default_request = build_request(parse_args())
    assert default_request.command == "run"
    assert default_request.dual_evaluator_mode == "pregenerate"
    assert default_request.sector_method == "iterative"
    assert default_request.direct_projector_cache_term_threshold == 54
    assert default_request.subtraction_backend == "projector-formula"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "FSD.py",
            "generate",
            "--dot-file",
            "examples/graphs/triangle.dot",
            "--kinematics",
            "examples/graphs/triangle_kinematics.yaml",
            "--output",
            "prepared/triangle",
        ],
    )
    generated_request = build_request(parse_args())
    assert generated_request.command == "generate"
    assert generated_request.integral == "dot"
    assert generated_request.output == "prepared/triangle"

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--lazy-dual-evaluators-generation"])
    assert build_request(parse_args()).dual_evaluator_mode == "lazy"

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--pregenerate-single-overall-dual-evaluator"])
    assert build_request(parse_args()).dual_evaluator_mode == "single-overall"

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--symbolic-derivatives"])
    assert build_request(parse_args()).dual_evaluator_mode == "symbolic-derivatives"

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--sectors", "3", "7"])
    assert build_request(parse_args()).sectors == (3, 7)

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--regular-taylor-signature-limit", "512"])
    assert build_request(parse_args()).regular_taylor_signature_limit == 512

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--regular-taylor-formula-volume-limit", "81"])
    assert build_request(parse_args()).regular_taylor_formula_volume_limit == 81

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--regular-taylor-formula-axis-limit", "6"])
    assert build_request(parse_args()).regular_taylor_formula_axis_limit == 6

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--chain-rule-formula-signature-limit", "17"])
    assert build_request(parse_args()).chain_rule_formula_signature_limit == 17

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "FSD.py",
            "--target-integration-time",
            "12.5",
            "--target-abs-error",
            "1e-6",
            "--target-rel-error",
            "3e-4",
            "--target-rel-accuracy",
            "0.03",
        ],
    )
    target_request = build_request(parse_args())
    assert target_request.target_integration_time == pytest.approx(12.5)
    assert target_request.target_abs_error == pytest.approx(1.0e-6)
    assert target_request.target_rel_error == pytest.approx(3.0e-4)
    assert target_request.target_rel_accuracy == pytest.approx(0.03)

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--sampling-mode", "qmc"])
    qmc_request = build_request(parse_args())
    assert qmc_request.evaluator_compile_mode == "eager"
    assert qmc_request.jit_compile_evaluators is False
    assert qmc_request.real_evaluator is False

    monkeypatch.setattr(
        sys,
        "argv",
        ["FSD.py", "--sampling-mode", "qmc", "--jit-compile"],
    )
    qmc_jit_request = build_request(parse_args())
    assert qmc_jit_request.evaluator_compile_mode == "jit"
    assert qmc_jit_request.jit_compile_evaluators is True
    assert qmc_jit_request.real_evaluator is True

    monkeypatch.setattr(
        sys,
        "argv",
        ["FSD.py", "--sampling-mode", "qmc", "--real-evaluator"],
    )
    qmc_real_request = build_request(parse_args())
    assert qmc_real_request.evaluator_compile_mode == "eager"
    assert qmc_real_request.real_evaluator is True

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "FSD.py",
            "--subtraction-backend",
            "projector-formula",
            "--force-regular-taylor-formulas",
        ],
    )
    forced_request = build_request(parse_args())
    assert forced_request.force_regular_taylor_formulas is True
    assert forced_request.regular_taylor_signature_limit >= 1_000_000
    assert forced_request.regular_taylor_formula_volume_limit >= 1_000_000
    assert forced_request.regular_taylor_formula_axis_limit >= 32

    custom_result = PROJECT_ROOT / "custom-result.json"
    monkeypatch.setattr(sys, "argv", ["FSD.py", "--result-path", str(custom_result)])
    assert build_request(parse_args()).result_path == str(custom_result)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "FSD.py",
            "cache",
            "--cache-loop-counts",
            "1",
            "2",
            "--cache-verify-samples-per-sector",
            "3",
            "--cache-report-path",
            "docs/cache-test.json",
        ],
    )
    cache_request = build_request(parse_args())
    assert cache_request.command == "cache"
    assert cache_request.cache_loop_counts == (1, 2)
    assert cache_request.cache_verify_samples_per_sector == 3
    assert cache_request.cache_report_path == "docs/cache-test.json"
    validate_request(cache_request)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "FSD.py",
            "--pregenerate-dual-evaluators",
            "--lazy-dual-evaluators-generation",
        ],
    )
    with pytest.raises(SystemExit):
        parse_args()


def test_numerator_reducer_cli_switch_defaults_to_symbolica() -> None:
    """DOT momentum numerators use the FSD-owned reducer unless requested otherwise."""
    default_request = build_request(parse_args([]))
    assert default_request.numerator_reducer == "symbolica"

    pysecdec_request = build_request(parse_args(["--numerator-reducer", "pysecdec"]))
    assert pysecdec_request.numerator_reducer == "pysecdec"


def test_missing_formula_cache_fallback_is_explicit() -> None:
    """Normal CLI generation is cache-only unless the fallback flag is supplied."""
    default_request = build_request(parse_args([]))
    assert not default_request.allow_fallback_for_missing_caches

    fallback_request = build_request(parse_args(["--allow-fallback-for-missing-caches"]))
    assert fallback_request.allow_fallback_for_missing_caches

    cache_request = build_request(parse_args(["cache"]))
    assert cache_request.allow_fallback_for_missing_caches


def test_run_yaml_resolves_paths_and_cli_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run presets load before normal parsing, while explicit CLI flags win."""
    run_file = PROJECT_ROOT / "examples/runs/dot_double_box.yaml"
    args = parse_args(
        [
            "--run",
            str(run_file),
            "--samples-per-iter",
            "123",
            "--no-progress",
        ]
    )
    request = build_request(args)

    assert request.run_file == str(run_file.resolve())
    assert request.dot_file == str((PROJECT_ROOT / "examples/graphs/double_box.dot").resolve())
    assert request.kinematics_file == str(
        (PROJECT_ROOT / "examples/graphs/double_box_kinematics.yaml").resolve()
    )
    assert request.target_args == (
        str((PROJECT_ROOT / "examples/outputs/dot_double_box_pysecdec_target.json").resolve()),
    )
    assert request.samples_per_iter == 123
    assert request.no_progress is True
    assert request.dual_evaluator_mode == "symbolic-derivatives"
    assert request.ibp_reduce_to_log_endpoint is True


def test_run_yaml_ibp_can_be_disabled_from_cli() -> None:
    """The triple-box preset enables IBP, but CLI flags can turn it off."""
    run_file = PROJECT_ROOT / "examples/runs/dot_triple_box.yaml"
    request = build_request(
        parse_args(
            [
                "--run",
                str(run_file),
                "--no-ibp-reduce-to-log-endpoint",
                "--no-progress",
            ]
        )
    )

    assert request.ibp_reduce_to_log_endpoint is False
    assert request.ibp_power_goal is None


def test_direct_projector_cache_threshold_cli_option() -> None:
    """The curated direct-projector override threshold is user configurable."""
    request = build_request(
        parse_args(
            [
                "--direct-projector-cache-term-threshold",
                "64",
                "--no-progress",
            ]
        )
    )

    assert request.direct_projector_cache_term_threshold == 64


def test_explicit_backend_cli_shortcut() -> None:
    """The explicit backend is the default and has a compact CLI alias."""
    default_request = build_request(parse_args(["--no-progress"]))
    request = build_request(parse_args(["--explicit", "--no-progress"]))

    assert default_request.sector_evaluator_backend == "explicit"
    assert request.sector_evaluator_backend == "explicit"


def test_qmc_cli_default_uses_pysecdec_lattice() -> None:
    """The QMC auto-default should use the pySecDec-compatible CBC/PT table."""
    request = build_request(
        parse_args(
            [
                "--sampling-mode",
                "qmc",
                "--samples-per-iter",
                "123",
                "--s",
                "1.0",
                "--m",
                "1.0",
                "--no-progress",
            ]
        )
    )

    assert request.qmc_lattice_backend == "pysecdec-default"
    assert request.qmc_refine_sectors == "democratic"
    assert request.qmc_max_samples_per_iter == 4096
    assert request.qmc_optimized_evaluators is True
    assert request.restart is False
    validate_request(request)

    democratic = build_request(
        parse_args(
            [
                "--sampling-mode",
                "qmc",
                "--qmc-refine-sectors",
                "democratic",
                "--restart",
                "--s",
                "1.0",
                "--m",
                "1.0",
                "--no-progress",
            ]
        )
    )
    assert democratic.qmc_refine_sectors == "democratic"
    assert democratic.restart is True
    validate_request(democratic)

    optimized = build_request(
        parse_args(
            [
                "--sampling-mode",
                "qmc",
                "--qmc-optimized-evaluators",
                "--s",
                "1.0",
                "--m",
                "1.0",
                "--no-progress",
            ]
        )
    )
    assert optimized.qmc_optimized_evaluators is True
    validate_request(optimized)

    unoptimized = build_request(
        parse_args(
            [
                "--sampling-mode",
                "qmc",
                "--no-qmc-optimized-evaluators",
                "--s",
                "1.0",
                "--m",
                "1.0",
                "--no-progress",
            ]
        )
    )
    assert unoptimized.qmc_optimized_evaluators is False
    validate_request(unoptimized)


def test_projector_generation_cli_shortcut() -> None:
    """The old fast-generation black-box path has an explicit CLI alias."""
    request = build_request(parse_args(["--projector-generation", "--no-progress"]))

    assert request.sector_evaluator_backend == "projector"


def test_parametric_generation_legacy_alias() -> None:
    """The retired name remains accepted for old local run scripts."""
    request = build_request(parse_args(["--parametric-generation", "--no-progress"]))

    assert request.sector_evaluator_backend == "projector"


def test_ibp_power_goal_cli_and_aliases() -> None:
    """Numeric IBP goals and the legacy log-endpoint alias normalize consistently."""
    goal_request = build_request(
        parse_args(["--ibp-power-goal", "-3", "--no-progress"])
    )
    alias_request = build_request(parse_args(["--ibp-reduce-to-log-endpoint", "--no-progress"]))

    assert goal_request.ibp_power_goal == -3
    assert goal_request.ibp_reduce_to_log_endpoint is False
    assert alias_request.ibp_power_goal == -1
    assert alias_request.ibp_reduce_to_log_endpoint is True


def test_evaluator_backend_cli_defaults_and_validation() -> None:
    """Evaluator backend flags normalize to explicit request metadata."""
    default_request = build_request(parse_args(["--no-progress"]))
    eager_request = build_request(parse_args(["--eager-evaluator", "--no-progress"]))
    compile_request = build_request(parse_args(["--compile", "--no-progress"]))
    complex_request = build_request(parse_args(["--complex-evaluator", "--no-progress"]))
    invalid_request = build_request(
        parse_args(["--compile", "--complex-evaluator", "--no-progress"])
    )

    assert default_request.evaluator_compile_mode == "jit"
    assert default_request.jit_compile_evaluators is True
    assert default_request.real_evaluator is True
    assert default_request.jit_direct_translation is False
    assert eager_request.evaluator_compile_mode == "eager"
    assert eager_request.jit_compile_evaluators is False
    assert compile_request.evaluator_compile_mode == "compile"
    assert compile_request.real_evaluator is True
    assert complex_request.evaluator_compile_mode == "jit"
    assert complex_request.real_evaluator is False
    indirect_request = build_request(
        parse_args(["--no-jit-direct-translation", "--no-progress"])
    )
    explicit_direct_request = build_request(
        parse_args(["--jit-direct-translation", "--no-progress"])
    )
    assert indirect_request.jit_direct_translation is False
    assert explicit_direct_request.jit_direct_translation is True
    with pytest.raises(ValueError, match="--compile"):
        validate_request(invalid_request)


def test_jit_direct_translation_forwarded_to_symbolica(monkeypatch: pytest.MonkeyPatch) -> None:
    """FSD forwards direct/indirect JIT translation to Symbolica builders."""
    monkeypatch.delenv("FSD_SYMBOLICA_JIT_DIRECT_TRANSLATION", raising=False)
    single_calls: list[dict[str, Any]] = []

    class FakeExpr:
        def evaluator(self, params: list[Any], **kwargs: Any) -> SimpleNamespace:
            single_calls.append(dict(kwargs))
            return SimpleNamespace(evaluate=lambda rows: rows)

    evaluator_utils_module.build_evaluator(
        FakeExpr(),
        [],
        evaluator_compile_mode="jit",
        jit_direct_translation=False,
    )
    assert single_calls[0]["jit_compile"] is False
    assert single_calls[0]["jit_direct_translation"] is False
    assert single_calls[1]["jit_compile"] is True
    assert single_calls[1]["jit_direct_translation"] is False

    multi_calls: list[dict[str, Any]] = []

    class FakeExpression:
        @staticmethod
        def evaluator_multiple(
            exprs: list[Any],
            params: list[Any],
            **kwargs: Any,
        ) -> SimpleNamespace:
            multi_calls.append(dict(kwargs))
            return SimpleNamespace(evaluate=lambda rows: rows)

    monkeypatch.setattr(evaluator_utils_module, "Expression", FakeExpression)
    evaluator_utils_module.build_evaluator_multiple(
        [FakeExpr(), FakeExpr()],
        [],
        evaluator_compile_mode="jit",
        jit_direct_translation=True,
    )
    assert multi_calls[0]["jit_compile"] is False
    assert multi_calls[0]["jit_direct_translation"] is False
    assert multi_calls[1]["jit_compile"] is True
    assert multi_calls[1]["jit_direct_translation"] is True

    multi_calls.clear()
    monkeypatch.setenv("FSD_SYMBOLICA_JIT_DIRECT_TRANSLATION", "0")
    evaluator_utils_module.build_evaluator_multiple(
        [FakeExpr()],
        [],
        evaluator_compile_mode="jit",
        jit_direct_translation=True,
    )
    assert multi_calls[1]["jit_compile"] is True
    assert multi_calls[1]["jit_direct_translation"] is False


def test_all_example_run_presets_parse_and_resolve_paths() -> None:
    """Every shipped run preset remains wired to the reorganized examples tree."""
    run_files = sorted((PROJECT_ROOT / "examples/runs").glob("*.yaml"))

    assert {path.name for path in run_files} == {
        "builtin_box.yaml",
        "builtin_triangle.yaml",
        "dot_box.yaml",
        "dot_double_box.yaml",
        "dot_triangle.yaml",
        "dot_triple_box.yaml",
        "dot_triple_box_offshell.yaml",
        "dot_triple_box_offshell_rank2_numerator.yaml",
        "double_box_from_U_and_F.yaml",
        "four_loop_hard_from_U_and_F.yaml",
    }
    for run_file in run_files:
        request = build_request(parse_args(["--run", str(run_file), "--no-progress"]))
        assert request.run_file == str(run_file.resolve())
        assert request.result_path is not None
        assert Path(request.result_path).parent == (PROJECT_ROOT / "examples/outputs").resolve()
        if request.dot_file is not None:
            assert Path(request.dot_file).is_file()
            assert Path(request.kinematics_file or "").is_file()
        if request.integral == "uf":
            assert request.topology_source == "uf"
            assert isinstance(request.uf_topology, dict)
            assert request.uf_topology.get("variables")
        if request.target_args:
            for target in request.target_args:
                target_path = Path(target)
                if target_path.suffix == ".json":
                    assert target_path.parent == (PROJECT_ROOT / "examples/outputs").resolve()


def test_four_loop_hard_toml_cards_parse_and_extend_base_run() -> None:
    """The PSD2807 TOML cards inherit the long U/F definition from YAML."""
    fsd_card = PROJECT_ROOT / "examples/runs/four_loop_hard_psd2807_fsd_qmc.toml"
    pysecdec_card = PROJECT_ROOT / "examples/runs/four_loop_hard_psd2807_pysecdec_native.toml"
    all_sector_card = PROJECT_ROOT / "examples/runs/four_loop_hard_all_sectors_fsd_qmc.toml"

    fsd_request = build_request(parse_args(["--run", str(fsd_card), "--no-progress"]))
    assert fsd_request.run_file == str(fsd_card.resolve())
    assert fsd_request.integral == "uf"
    assert fsd_request.topology_source == "uf"
    assert fsd_request.uf_topology is not None
    assert fsd_request.sectors == (2807,)
    assert fsd_request.dot_engine == "fsd"
    assert fsd_request.sampling_mode == "qmc"
    assert fsd_request.qmc_optimized_evaluators is True
    assert fsd_request.qmc_lattice_backend == "pysecdec-default"
    assert fsd_request.evaluator_compile_mode == "jit"
    assert fsd_request.real_evaluator is True
    assert fsd_request.output is not None
    assert Path(fsd_request.output).parent == (PROJECT_ROOT / "output").resolve()
    assert Path(fsd_request.result_path).parent == (PROJECT_ROOT / "examples/outputs").resolve()

    pysecdec_request = build_request(parse_args(["--run", str(pysecdec_card), "--no-progress"]))
    assert pysecdec_request.integral == "uf"
    assert pysecdec_request.sectors == (2807,)
    assert pysecdec_request.dot_engine == "pysecdec"
    assert pysecdec_request.pysecdec_workdir is not None
    assert Path(pysecdec_request.pysecdec_workdir).parent == (PROJECT_ROOT / "output").resolve()
    assert pysecdec_request.keep_pysecdec_workdir is True
    assert pysecdec_request.pysecdec_maxeval == 1_000_000
    assert pysecdec_request.ibp_power_goal == -1

    all_sector_request = build_request(parse_args(["--run", str(all_sector_card), "--no-progress"]))
    assert all_sector_request.integral == "uf"
    assert all_sector_request.sectors is None
    assert all_sector_request.sampling_mode == "qmc"
    assert all_sector_request.qmc_optimized_evaluators is True
    assert all_sector_request.qmc_correlate_sectors is False
    assert all_sector_request.samples_per_iter == 2_000_000
    assert all_sector_request.qmc_shifts == 32
    assert all_sector_request.target_rel_error == 1.0e-3
    assert all_sector_request.report_path is not None
    assert Path(all_sector_request.report_path).parent == (PROJECT_ROOT / "examples/outputs").resolve()


def test_ibp_lowering_requires_projector_formula_backend() -> None:
    """IBP endpoint lowering is deliberately scoped to projector formulas."""
    request = make_request(
        ibp_reduce_to_log_endpoint=True,
        subtraction_backend="formula",
    )
    with pytest.raises(ValueError, match="projector-formula"):
        validate_request(request)


def test_force_regular_taylor_formulas_requires_projector_formula_backend() -> None:
    """The expensive cache-warming regular-Taylor mode is projector-only."""
    request = make_request(
        force_regular_taylor_formulas=True,
        subtraction_backend="formula",
    )
    with pytest.raises(ValueError, match="projector-formula"):
        validate_request(request)


def test_endpoint_projector_formula_cache_is_topology_independent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Endpoint projector cache signatures do not include topology names."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    signature = topology.endpoint_projector_signature(sectors[0])

    topology.prepare_endpoint_projector_formulas([sectors[0]])
    files = sorted(tmp_path.glob("endpoint_projector_*.json"))
    assert len(files) == 1
    payload = files[0].read_text(encoding="utf-8")
    assert "C0(" not in payload
    assert "triangle" not in payload
    assert "x0" not in payload

    second_topology = build_topology(request)
    second_topology.prepare_endpoint_projector_formulas([sectors[0]])
    assert sorted(tmp_path.glob("endpoint_projector_*.json")) == files
    assert signature in second_topology._endpoint_projector_formulas


def test_endpoint_projector_cache_loads_curated_assets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Formula assets can be shipped under the curated cache subdirectory."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    sector = sectors[0]
    signature = topology.endpoint_projector_signature(sector)

    topology.prepare_endpoint_projector_formulas([sector])
    files = sorted(tmp_path.glob("endpoint_projector_*.json"))
    assert len(files) == 1
    curated = tmp_path / "curated"
    curated.mkdir()
    curated_file = curated / files[0].name
    files[0].replace(curated_file)
    assert not sorted(tmp_path.glob("endpoint_projector_*.json"))

    def forbidden_build_outputs(_self: Any) -> list[Any]:
        raise AssertionError("endpoint projector generation bypassed curated cache")

    monkeypatch.setattr(
        "subtraction_formula._EndpointProjectorContext.build_outputs",
        forbidden_build_outputs,
    )
    second_topology = build_topology(request)
    second_topology.prepare_endpoint_projector_formulas([sector])
    assert signature in second_topology._endpoint_projector_formulas


def test_curated_direct_projector_overrides_large_ibp_compound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A shipped direct projector can replace a large IBP child-projector tree."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    base_topology = TopologyDefinition(
        family="toy",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("1"),
        u_power_base=0.0,
        f_power_base=0.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=0.0,
        expected_laurent_orders=["eps^-1", "eps^0"],
        convention_note="toy",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-2.0, 1.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="toy",
            convention_description="toy",
        ),
        ibp_reduce_to_log_endpoint=False,
    )
    sector = SectorDefinition(
        name="toy-sector",
        integration_dim=1,
        variable_names=["y"],
        map_exprs=[E("y")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[1],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1],
    )

    direct_signature = base_topology.endpoint_projector_signature(sector)
    base_topology.prepare_endpoint_projector_formulas([sector])
    generated = sorted(tmp_path.glob("endpoint_projector_*.json"))
    assert len(generated) == 1
    curated = tmp_path / "curated"
    curated.mkdir()
    generated[0].replace(curated / generated[0].name)
    endpoint_projector_formula_has_curated_cache.cache_clear()
    assert endpoint_projector_formula_has_curated_cache(direct_signature)

    hybrid_topology = replace(
        base_topology,
        ibp_reduce_to_log_endpoint=True,
        direct_projector_cache_term_threshold=1,
    )
    hybrid_signature = hybrid_topology.endpoint_projector_signature(sector)
    hybrid_topology.prepare_endpoint_projector_formulas([sector])
    formula = hybrid_topology.endpoint_projector_formula_for(sector)

    assert hybrid_signature == direct_signature
    assert formula.ibp_reduce_to_log_endpoint is False
    assert hybrid_topology.endpoint_projector_direct_cache_override_sectors == 1
    assert hybrid_topology.endpoint_projector_direct_cache_override_signatures == 1


def test_curated_direct_projector_threshold_zero_keeps_ibp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Threshold zero disables the direct cached-projector override."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("1"),
        u_power_base=0.0,
        f_power_base=0.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=0.0,
        expected_laurent_orders=["eps^-1", "eps^0"],
        convention_note="toy",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-2.0, 1.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="toy",
            convention_description="toy",
        ),
        ibp_reduce_to_log_endpoint=True,
        direct_projector_cache_term_threshold=0,
    )
    sector = SectorDefinition(
        name="toy-sector",
        integration_dim=1,
        variable_names=["y"],
        map_exprs=[E("y")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[1],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1],
    )

    assert topology.endpoint_projector_signature(sector)[2] == -1


def test_promote_subtraction_formula_asset_script(tmp_path: Path) -> None:
    """Validated generated formulas can be promoted into curated source assets."""
    cache_file = tmp_path / "endpoint_projector_test.json"
    cache_file.write_text(
        json.dumps(
            {
                "signature_payload": {
                    "schema_version": 1,
                    "kind": "endpoint-projector",
                    "signature": ["endpoint-projector", 2, False, 1, [[-1, 1.0]], [0], [-1, 0]],
                },
                "input_names": ["sf_y0", "sf_c_0_0_0_0_re", "sf_c_0_0_0_0_im"],
                "output_expressions": ["sf_c_0_0_0_0_re"],
                "laurent_orders": [-1, 0],
                "zero_subsets": [[]],
                "taylor_orders": [0],
                "coefficient_layout": [
                    {
                        "zero_subset": [],
                        "boundary_subset": [],
                        "multi_index": [0],
                        "regular_order": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    script = PROJECT_ROOT / "src" / "promote_subtraction_formula_asset.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--cache-dir",
            str(tmp_path),
            cache_file.name,
        ],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    curated_file = tmp_path / "curated" / cache_file.name
    assert curated_file.is_file()
    assert cache_file.read_text(encoding="utf-8") == curated_file.read_text(encoding="utf-8")


def test_sparse_integer_power_avoids_log_exp_path() -> None:
    """Integer U/F powers are expanded by binomial products in fallback mode."""
    n_rows = 2
    max_orders = [2]
    allowed = {(0,), (1,), (2,)}
    series = {
        (0,): np.asarray([2.0, 4.0], dtype=np.complex128),
        (1,): np.asarray([0.3, -0.8], dtype=np.complex128),
    }

    inverse_square = _series_pow_real_allowed(
        series,
        -2.0,
        max_orders,
        n_rows,
        allowed,
    )
    q = series[(1,)] / series[(0,)]
    np.testing.assert_allclose(inverse_square[(0,)], series[(0,)] ** -2)
    np.testing.assert_allclose(inverse_square[(1,)], series[(0,)] ** -2 * (-2.0 * q))
    np.testing.assert_allclose(inverse_square[(2,)], series[(0,)] ** -2 * (3.0 * q**2))

    square = _series_pow_real_allowed(series, 2.0, max_orders, n_rows, allowed)
    np.testing.assert_allclose(square[(0,)], series[(0,)] ** 2)
    np.testing.assert_allclose(square[(1,)], 2.0 * series[(0,)] * series[(1,)])
    np.testing.assert_allclose(square[(2,)], series[(1,)] ** 2)


def test_sparse_power_and_log_helper_matches_separate_paths() -> None:
    """Combined integer-power/log series reuse preserves the old algebra."""
    n_rows = 3
    max_orders = [2, 1]
    allowed = {
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (2, 1),
    }
    series = {
        (0, 0): np.asarray([2.0, 3.0, 5.0], dtype=np.complex128),
        (1, 0): np.asarray([0.2, -0.1, 0.4], dtype=np.complex128),
        (0, 1): np.asarray([0.05, 0.3, -0.2], dtype=np.complex128),
        (1, 1): np.asarray([0.01, -0.02, 0.03], dtype=np.complex128),
    }

    combined_power, combined_log = _series_pow_real_and_log_allowed(
        series,
        -3.0,
        max_orders,
        n_rows,
        allowed,
    )
    separate_power = _series_pow_real_allowed(series, -3.0, max_orders, n_rows, allowed)
    separate_log = _series_log_allowed(series, max_orders, n_rows, allowed)

    assert set(combined_power) == set(separate_power)
    assert set(combined_log) == set(separate_log)
    for key in separate_power:
        np.testing.assert_allclose(combined_power[key], separate_power[key], rtol=1.0e-12, atol=1.0e-12)
    for key in separate_log:
        np.testing.assert_allclose(combined_log[key], separate_log[key], rtol=1.0e-12, atol=1.0e-12)


def test_series_mul_allowed_fast_paths_match_rectangular_product() -> None:
    """Constant and single-term sparse products agree with the generic product."""
    n_rows = 2
    max_orders = [2, 2]
    allowed = {
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
    }
    constant = {
        (0, 0): np.asarray([2.0, -3.0], dtype=np.complex128),
    }
    single = {
        (1, 0): np.asarray([0.5, 0.25], dtype=np.complex128),
    }
    general = {
        (0, 0): np.asarray([1.0, 2.0], dtype=np.complex128),
        (0, 1): np.asarray([3.0, -1.0], dtype=np.complex128),
        (1, 0): np.asarray([0.25, 0.5], dtype=np.complex128),
    }

    for left, right in ((constant, general), (general, constant), (single, general), (general, single)):
        fast = _series_mul_allowed(left, right, allowed)
        dense = {
            key: value
            for key, value in _series_mul(left, right, max_orders).items()
            if key in allowed
        }
        assert set(fast) == set(dense)
        for key in dense:
            np.testing.assert_allclose(fast[key], dense[key], rtol=1.0e-12, atol=1.0e-12)


def test_regular_taylor_formula_cache_reuses_expression_strings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Warm regular-Taylor cache hits must not rebuild Symbolica expressions."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    sector = sectors[0]
    topology.prepare_endpoint_projector_formulas([sector])
    topology.prepare_regular_taylor_formulas([sector])
    files = sorted(tmp_path.glob("regular_taylor_*.json"))
    assert files
    payload = files[0].read_text(encoding="utf-8")
    assert "triangle" not in payload
    assert "C0(" not in payload

    def forbidden_build_outputs(_self: Any) -> list[Any]:
        raise AssertionError("regular Taylor expression generation bypassed cache")

    monkeypatch.setattr(
        "subtraction_formula._RegularTaylorContext.build_outputs",
        forbidden_build_outputs,
    )
    second_topology = build_topology(request)
    second_topology.prepare_endpoint_projector_formulas([sector])
    second_topology.prepare_regular_taylor_formulas([sector])
    assert sorted(tmp_path.glob("regular_taylor_*.json")) == files
    assert second_topology._regular_taylor_formulas


def test_curated_regular_taylor_assets_bypass_cold_build_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Curated high-axis assets are part of FSD and bypass cold-build guards."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
        dual_evaluator_mode="symbolic-derivatives",
    )
    topology = build_topology(request)
    topology.regular_taylor_dual_shape_limit = 0
    sectors = generate_sectors(request)
    sector = sectors[0]
    signature = (
        "regular-taylor",
        3,
        6,
        (((0, 0, 0, 0, 0, 0), 0),),
        float(topology.u_power_base),
        float(topology.f_power_base),
        float(topology.eps_log_u_coeff),
        float(topology.eps_log_f_coeff),
        tuple(topology.laurent_orders),
    )

    def fake_requests(_sector: SectorDefinition) -> list[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]]:
        return [(signature, (), (0, 0, 0, 0, 0, 0))]

    def fake_formula(
        _topology: TopologyDefinition,
        _sector: SectorDefinition,
        _signature: tuple[Any, ...],
    ) -> RegularTaylorFormulaDefinition:
        return RegularTaylorFormulaDefinition(
            signature=_signature,
            input_names=[],
            input_symbols=[],
            output_expressions=[],
            evaluators=[],
            output_layout=[],
            input_layout=[],
            max_orders=[0, 0, 0, 0, 0, 0],
            zero_positions=(),
        )

    monkeypatch.setattr(topology, "regular_taylor_requests_for_sector", fake_requests)
    monkeypatch.setattr(
        "integrand.regular_taylor_formula_has_curated_cache",
        lambda candidate: candidate == signature,
    )
    monkeypatch.setattr("integrand.build_regular_taylor_formula", fake_formula)

    topology.prepare_regular_taylor_formulas([sector])

    assert signature in topology._regular_taylor_formulas
    assert topology.regular_taylor_formulas_from_curated_cache == 1
    assert topology.regular_taylor_formulas_skipped == 0


def test_curated_regular_taylor_assets_bypass_signature_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cold-build count cap must not disable shipped curated formulas."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
        dual_evaluator_mode="symbolic-derivatives",
    )
    topology = build_topology(request)
    topology.regular_taylor_formula_signature_limit = 0
    topology.regular_taylor_formula_volume_limit = 64
    topology.regular_taylor_formula_axis_limit = 5
    sectors = generate_sectors(request)
    sector = sectors[0]
    cold_signature = (
        "regular-taylor",
        2,
        1,
        (0,),
        float(topology.u_power_base),
        float(topology.f_power_base),
        float(topology.eps_log_u_coeff),
        float(topology.eps_log_f_coeff),
        tuple(topology.laurent_orders),
    )
    curated_signature = (
        "regular-taylor",
        3,
        6,
        (((0, 0, 0, 0, 0, 0), 0),),
        float(topology.u_power_base),
        float(topology.f_power_base),
        float(topology.eps_log_u_coeff),
        float(topology.eps_log_f_coeff),
        tuple(topology.laurent_orders),
    )

    def fake_requests(_sector: SectorDefinition) -> list[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]]:
        return [
            (cold_signature, (0,), (1,)),
            (curated_signature, (), (0, 0, 0, 0, 0, 0)),
        ]

    def fake_formula(
        _topology: TopologyDefinition,
        _sector: SectorDefinition,
        _signature: tuple[Any, ...],
    ) -> RegularTaylorFormulaDefinition:
        assert _signature == curated_signature
        return RegularTaylorFormulaDefinition(
            signature=_signature,
            input_names=[],
            input_symbols=[],
            output_expressions=[],
            evaluators=[],
            output_layout=[],
            input_layout=[],
            max_orders=[0, 0, 0, 0, 0, 0],
            zero_positions=(),
        )

    monkeypatch.setattr(topology, "regular_taylor_requests_for_sector", fake_requests)
    monkeypatch.setattr(
        "integrand.regular_taylor_formula_has_curated_cache",
        lambda candidate: candidate == curated_signature,
    )
    monkeypatch.setattr("integrand.build_regular_taylor_formula", fake_formula)

    topology.prepare_regular_taylor_formulas([sector])

    assert set(topology._regular_taylor_formulas) == {curated_signature}
    assert topology.regular_taylor_formulas_from_curated_cache == 1
    assert topology.regular_taylor_formulas_skipped == 1


def test_regular_taylor_requests_skip_finite_sectors() -> None:
    """Finite sectors do not need endpoint-projector regular Taylor formulas."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        subtraction_backend="projector-formula",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)

    assert sectors[0].singular_axes == []
    assert topology.regular_taylor_requests_for_sector(sectors[0]) == []


def test_regular_taylor_low_signature_is_axis_permutation_invariant() -> None:
    """Large-sector regular formula cache keys ignore pure axis-order choices."""
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0", "x1"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("1"),
        u_power_base=0.0,
        f_power_base=0.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=0.0,
        expected_laurent_orders=["eps^-2", "eps^-1", "eps^0"],
        convention_note="toy",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0, 1.0),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-2.0, 1.0),
            parameter_weight_powers=(0.0, 0.0),
            prefactor_description="toy",
            convention_description="toy",
        ),
    )
    topology._regular_taylor_signature_version = 2
    sector_a = SectorDefinition(
        name="toy-a",
        integration_dim=2,
        variable_names=["u", "v"],
        map_exprs=[E("u"), E("v")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[1, 2],
        jacobian_monomial_powers=[0, 0],
        singular_axes=[0, 1],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1, 2],
    )
    sector_b = SectorDefinition(
        name="toy-b",
        integration_dim=2,
        variable_names=["v", "u"],
        map_exprs=[E("v"), E("u")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[2, 1],
        jacobian_monomial_powers=[0, 0],
        singular_axes=[0, 1],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[2, 1],
    )

    assert topology.regular_taylor_signature(sector_a, (), (1, 3)) == (
        topology.regular_taylor_signature(sector_b, (), (3, 1))
    )


def test_regular_taylor_volume_guard_skips_hard_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The all-sector guard can skip hard regular formulas individually."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
    )
    topology = build_topology(request)
    sectors = [sector for sector in generate_sectors(request) if sector.singular_axes][:2]
    configure_laurent_range(request, topology, sectors)
    topology.regular_taylor_formula_signature_limit = 256
    topology.regular_taylor_formula_volume_limit = 1
    signature = (
        "regular-taylor",
        2,
        1,
        (1,),
        0.0,
        0.0,
        0.0,
        0.0,
        tuple(topology.laurent_orders),
    )

    def fake_requests(sector: SectorDefinition) -> list[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]]:
        return [(signature, (0,), (1,))]

    def forbidden_build(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("guard should skip before regular formula generation")

    monkeypatch.setattr(topology, "regular_taylor_requests_for_sector", fake_requests)
    monkeypatch.setattr("integrand.build_regular_taylor_formula", forbidden_build)

    topology.prepare_regular_taylor_formulas(sectors)

    assert topology.regular_taylor_formulas_skipped == 1
    assert topology._regular_taylor_formulas == {}


def test_regular_taylor_axis_guard_skips_high_axis_formula(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Six-axis regular formulas are skipped by the default cold-build guard."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        subtraction_backend="projector-formula",
    )
    topology = build_topology(request)
    sectors = [sector for sector in generate_sectors(request) if sector.singular_axes][:1]
    configure_laurent_range(request, topology, sectors)
    topology.regular_taylor_formula_signature_limit = 256
    topology.regular_taylor_formula_volume_limit = 64
    topology.regular_taylor_formula_axis_limit = 5
    signature = (
        "regular-taylor",
        2,
        6,
        (0, 0, 0, 0, 0, 0),
        0.0,
        0.0,
        0.0,
        0.0,
        tuple(topology.laurent_orders),
    )

    def fake_requests(sector: SectorDefinition) -> list[tuple[tuple[Any, ...], tuple[int, ...], tuple[int, ...]]]:
        return [(signature, (), (0, 0, 0, 0, 0, 0))]

    def forbidden_build(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("axis guard should skip before regular formula generation")

    monkeypatch.setattr(topology, "regular_taylor_requests_for_sector", fake_requests)
    monkeypatch.setattr("integrand.build_regular_taylor_formula", forbidden_build)

    topology.prepare_regular_taylor_formulas(sectors)

    assert topology.regular_taylor_formulas_skipped == 1
    assert topology._regular_taylor_formulas == {}


def test_regular_taylor_signature_volume_supports_v1_and_v2_layouts() -> None:
    """The volume guard reads Taylor orders from the correct signature slot."""
    v1_signature = (
        "regular-taylor",
        1,
        3,
        (4, 5, 6),
        (),
        (),
        (),
        (),
        (),
        (),
        (),
        (0,),
        (2, 1, 0),
        (),
        ("eps^0",),
    )
    v2_signature = (
        "regular-taylor",
        2,
        3,
        (2, 1, 0),
        0.0,
        0.0,
        0.0,
        0.0,
        ("eps^0",),
    )
    v3_signature = (
        "regular-taylor",
        3,
        6,
        (
            ((0, 0, 0, 0, 0, 0), 0),
            ((0, 1, 0, 0, 0, 0), 0),
            ((0, 0, 0, 2, 0, 0), 1),
        ),
        0.0,
        0.0,
        0.0,
        0.0,
        ("eps^-6", "eps^-5"),
    )

    assert _regular_taylor_signature_volume(v1_signature) == 6
    assert _regular_taylor_signature_volume(v2_signature) == 6
    assert _regular_taylor_signature_volume(v3_signature) == 3
    assert _regular_taylor_signature_axis_count(v1_signature) == 3
    assert _regular_taylor_signature_axis_count(v2_signature) == 3
    assert _regular_taylor_signature_axis_count(v3_signature) == 6


def test_sparse_multi_set_cache_key_is_order_independent() -> None:
    """Sparse Taylor source-shape cache keys ignore set insertion order."""
    left = {(1, 0, 2), (0, 0, 0), (0, 2, 1)}
    right = {(0, 2, 1), (1, 0, 2), (0, 0, 0)}

    key_left = _multi_set_cache_key(left)
    key_right = _multi_set_cache_key(right)

    assert key_left == key_right
    assert key_left[0] == (0, 0, 0)
    assert key_left == tuple(sorted(left, key=lambda item: (sum(item), item)))


def test_summary_uses_compact_dual_shape_statistics() -> None:
    """Result summaries should not serialize every sector Taylor shape."""
    request = make_request(integral="triangle", mode="massless", s=-1.0, m=0.0)
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)

    data = summary_data(request, topology, sectors, benchmark_available=False)
    symanzik = data["symanzik"]

    assert "dual_shape_summary" in symanzik
    assert "dual_shapes" not in symanzik
    assert symanzik["dual_shape_summary"]["unique_shape_count"] >= 1
    assert "regular_taylor_formula_count" in symanzik
    assert "regular_taylor_formulas_from_curated_cache" in symanzik
    assert "regular_taylor_formulas_skipped" in symanzik
    assert "regular_taylor_formula_policy" in symanzik
    assert symanzik["regular_taylor_formula_policy"] == "not used by this subtraction backend"

    projector_request = replace(request, subtraction_backend="projector-formula")
    projector_data = summary_data(projector_request, topology, sectors, benchmark_available=False)
    assert (
        "curated endpoint projectors and regular Taylor formulas default-on"
        in projector_data["symanzik"]["regular_taylor_formula_policy"]
    )


def test_watchdog_wrapper_stops_on_stop_file(tmp_path: Path) -> None:
    """The local run wrapper can stop its own child without an external kill."""
    stop_file = tmp_path / "stop.order"
    sleep_executable = shutil.which("sleep") or "/bin/sleep"
    proc = subprocess.Popen(
        [
            sys.executable,
            str(PROJECT_ROOT / "src" / "run_with_memory_watch.py"),
            "--limit-gb",
            "35",
            "--timeout-seconds",
            "30",
            "--poll-seconds",
            "0.1",
            "--kill-grace-seconds",
            "0.1",
            "--stop-file",
            str(stop_file),
            "--",
            sleep_executable,
            "30",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    time.sleep(0.3)
    stop_file.write_text("", encoding="utf-8")
    output, _ = proc.communicate(timeout=10.0)

    assert proc.returncode == 130
    assert "stop file observed" in output
    assert not stop_file.exists()


def test_regular_taylor_v2_uses_dualized_evaluator(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The lowered regular formula can extract eps/tau coefficients by duals."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("1"),
        u_power_base=0.0,
        f_power_base=0.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=0.0,
        expected_laurent_orders=["eps^0", "eps^1"],
        convention_note="toy",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(0.0, 0.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="toy",
            convention_description="toy",
        ),
    )
    sector = SectorDefinition(
        name="toy-sector",
        integration_dim=1,
        variable_names=["y"],
        map_exprs=[E("y")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[0],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1],
    )
    signature = (
        "regular-taylor",
        2,
        1,
        (1,),
        0.0,
        0.0,
        0.0,
        0.0,
        tuple(topology.laurent_orders),
    )

    formula = build_regular_taylor_formula(topology, sector, signature)
    row_by_name = {
        "rg_monomial_pref": 5.0,
        "rg_monomial_log": 2.0,
        "rg_j_m0": 1.0,
        "rg_j_m1": 3.0,
        "rg_u_m0": 1.0,
        "rg_u_m1": 0.0,
        "rg_f_m0": 1.0,
        "rg_f_m1": 0.0,
    }
    row = np.asarray([[row_by_name[name] for name in formula.input_names]], dtype=np.complex128)
    values = formula.evaluate_complex_batch(row)

    assert formula.evaluator_dual_shape
    assert formula.output_layout == [((0,), 0), ((1,), 0), ((0,), 1), ((1,), 1)]
    np.testing.assert_allclose(values[0], np.asarray([5.0, 15.0, 10.0, 30.0]))

    cached = build_regular_taylor_formula(topology, sector, signature)
    assert cached.evaluator_dual_shape == formula.evaluator_dual_shape


def test_regular_taylor_v3_uses_sparse_requested_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Sparse v3 regular formulas close inputs but keep requested outputs sparse."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0", "x1"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("x0*x1"),
        u_power_base=0.0,
        f_power_base=0.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=0.0,
        expected_laurent_orders=["eps^0", "eps^1"],
        convention_note="toy",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0, 1.0),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(0.0, 0.0),
            parameter_weight_powers=(0.0, 0.0),
            prefactor_description="toy",
            convention_description="toy",
        ),
    )
    sector = SectorDefinition(
        name="toy-sector",
        integration_dim=2,
        variable_names=["y0", "y1"],
        map_exprs=[E("y0"), E("y1")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[0, 0],
        jacobian_monomial_powers=[0, 0],
        singular_axes=[0, 1],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1, 1],
    )
    signature = (
        "regular-taylor",
        3,
        2,
        (((1, 0), 1),),
        0.0,
        0.0,
        0.0,
        0.0,
        tuple(topology.laurent_orders),
    )

    formula = build_regular_taylor_formula(topology, sector, signature)
    row_by_name = {
        "rg_monomial_pref": 2.0,
        "rg_monomial_log": 3.0,
        "rg_j_m0_0": 1.0,
        "rg_j_m1_0": 4.0,
        "rg_u_m0_0": 1.0,
        "rg_u_m1_0": 0.0,
        "rg_f_m0_0": 1.0,
        "rg_f_m1_0": 0.0,
    }
    row = np.asarray([[row_by_name[name] for name in formula.input_names]], dtype=np.complex128)
    values = formula.evaluate_complex_batch(row)

    assert formula.evaluator_dual_shape
    assert formula.output_layout == [((1, 0), 1)]
    assert len(formula.evaluators) == 1
    np.testing.assert_allclose(values[0], np.asarray([24.0]))


def test_sparse_regular_taylor_fallback_matches_dense_path() -> None:
    """Skipped regular formulas still use sparse source shapes correctly."""
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0", "x1"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1 + x0 + 2*x1 + x0*x1"),
        f_expr=E("2 + x0 + x1 + x0*x1"),
        u_power_base=1.0,
        f_power_base=1.0,
        eps_log_u_coeff=1.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0", "eps^1"],
        convention_note="toy",
        dual_evaluator_mode="symbolic-derivatives",
    )
    sector = SectorDefinition(
        name="toy-sector",
        integration_dim=2,
        variable_names=["y0", "y1"],
        map_exprs=[E("y0"), E("y1")],
        regular_jacobian_expr=E("1 + y0*y1"),
        f_monomial_powers=[0, 0],
        jacobian_monomial_powers=[0, 0],
        singular_axes=[0, 1],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1, 1],
    )
    sector.prepare_evaluators(include_dual=False)
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")
    rows = np.asarray([[0.25, 0.375], [0.5, 0.125]], dtype=float)
    taylor_orders = [1, 1]
    output_pairs = (
        ((0, 0), 0),
        ((1, 0), 0),
        ((0, 1), 1),
        ((1, 1), 1),
    )

    dense = processor._g_taylor_eps_series_batch(
        sector,
        rows,
        {0, 1},
        taylor_orders,
        HotPathTiming(),
        max_orders_are_explicit=True,
    )
    sparse = processor._g_taylor_eps_series_batch(
        sector,
        rows,
        {0, 1},
        taylor_orders,
        HotPathTiming(),
        max_orders_are_explicit=True,
        output_pairs=output_pairs,
    )

    for multi_index, regular_order in output_pairs:
        np.testing.assert_allclose(
            sparse[regular_order][multi_index],
            dense[regular_order][multi_index],
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_ibp_shared_batch_g_cache_matches_direct_boundary_calls() -> None:
    """IBP boundary clustering must reproduce per-boundary regular Taylor calls."""
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0", "x1"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1 + x0 + 2*x1 + x0*x1"),
        f_expr=E("2 + x0 + x1 + x0*x1"),
        u_power_base=1.0,
        f_power_base=1.0,
        eps_log_u_coeff=1.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0", "eps^1"],
        convention_note="toy",
        dual_evaluator_mode="symbolic-derivatives",
    )
    sector = SectorDefinition(
        name="toy-sector",
        integration_dim=2,
        variable_names=["y0", "y1"],
        map_exprs=[E("y0"), E("y1")],
        regular_jacobian_expr=E("1 + y0*y1"),
        f_monomial_powers=[0, 0],
        jacobian_monomial_powers=[0, 0],
        singular_axes=[0, 1],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1, 1],
    )
    sector.prepare_evaluators(include_dual=False)
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")
    rows = np.asarray([[0.25, 0.375], [0.5, 0.125]], dtype=float)
    shared_max_orders = {
        ((), (0,)): (1, 0),
        ((1,), (0,)): (1, 0),
    }
    output_pairs = (
        ((0, 0), 0),
        ((1, 0), 0),
        ((0, 0), 1),
    )
    shared_output_pairs = {
        key: output_pairs
        for key in shared_max_orders
    }

    clustered = processor._precompute_ibp_shared_batch_g_cache(
        sector,
        rows,
        shared_max_orders,
        shared_output_pairs,
        HotPathTiming(),
    )

    for (boundary, zero), max_orders in shared_max_orders.items():
        direct = processor._g_taylor_eps_series_batch(
            sector,
            rows,
            set(zero),
            list(max_orders),
            HotPathTiming(),
            boundary_positions=set(boundary),
            max_orders_are_explicit=True,
            output_pairs=output_pairs,
        )
        cached = clustered[(boundary, zero, max_orders)]
        for multi_index, regular_order in output_pairs:
            np.testing.assert_allclose(
                cached[regular_order].get(multi_index, np.zeros(rows.shape[0], dtype=np.complex128)),
                direct[regular_order].get(multi_index, np.zeros(rows.shape[0], dtype=np.complex128)),
                rtol=1.0e-12,
                atol=1.0e-12,
            )


def test_direct_endpoint_g_cache_matches_direct_boundary_calls() -> None:
    """Direct endpoint projector boundary clustering preserves regular Taylor inputs."""
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0", "x1"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1 + x0 + 2*x1 + x0*x1"),
        f_expr=E("2 + x0 + x1 + x0*x1"),
        u_power_base=1.0,
        f_power_base=1.0,
        eps_log_u_coeff=1.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0", "eps^1"],
        convention_note="toy",
        dual_evaluator_mode="symbolic-derivatives",
    )
    sector = SectorDefinition(
        name="toy-sector",
        integration_dim=2,
        variable_names=["y0", "y1"],
        map_exprs=[E("y0"), E("y1")],
        regular_jacobian_expr=E("1 + y0*y1"),
        f_monomial_powers=[0, 0],
        jacobian_monomial_powers=[0, 0],
        singular_axes=[0, 1],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1, 1],
    )
    sector.prepare_evaluators(include_dual=False)
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")
    rows = np.asarray([[0.25, 0.375], [0.5, 0.125]], dtype=float)
    coefficient_layout = [
        ((), (0,), (0, 0), 0),
        ((), (0,), (1, 0), 0),
        ((), (0,), (0, 0), 1),
        ((1,), (0,), (0, 0), 0),
        ((1,), (0,), (1, 0), 0),
        ((1,), (0,), (0, 0), 1),
    ]
    formula = EndpointProjectorFormulaDefinition(
        signature=("toy-direct",),
        input_names=[],
        input_symbols=[],
        output_expressions=[],
        evaluators=[],
        laurent_orders=[0, 1],
        zero_subsets=[],
        taylor_orders=[1, 0],
        coefficient_layout=coefficient_layout,
    )

    clustered = processor._precompute_endpoint_projector_g_cache(
        sector,
        rows,
        formula,
        HotPathTiming(),
    )
    assert len(processor._endpoint_projector_plan_cache) == 1
    first_plan = processor._endpoint_projector_plan(sector, formula)
    assert processor._endpoint_projector_plan(sector, formula) is first_plan

    for boundary in ((), (1,)):
        output_pairs = (
            ((0, 0), 0),
            ((1, 0), 0),
            ((0, 0), 1),
        )
        direct = processor._g_taylor_eps_series_batch(
            sector,
            rows,
            {0},
            [1, 0],
            HotPathTiming(),
            boundary_positions=set(boundary),
            max_orders_are_explicit=True,
            output_pairs=output_pairs,
        )
        cached = clustered[(boundary, (0,))]
        for multi_index, regular_order in output_pairs:
            np.testing.assert_allclose(
                cached[regular_order].get(multi_index, np.zeros(rows.shape[0], dtype=np.complex128)),
                direct[regular_order].get(multi_index, np.zeros(rows.shape[0], dtype=np.complex128)),
                rtol=1.0e-12,
                atol=1.0e-12,
            )


def test_ibp_endpoint_projector_formula_builds_for_higher_endpoint_power(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A y^(-2+eps) endpoint produces an IBP-lowered projector formula."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("1"),
        u_power_base=0.0,
        f_power_base=0.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=0.0,
        expected_laurent_orders=["eps^-1", "eps^0"],
        convention_note="toy",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-2.0, 1.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="toy",
            convention_description="toy",
        ),
        ibp_reduce_to_log_endpoint=True,
    )
    sector = SectorDefinition(
        name="toy-sector",
        integration_dim=1,
        variable_names=["y"],
        map_exprs=[E("y")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[1],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[1],
    )
    signature = topology.endpoint_projector_signature(sector)
    formula = build_endpoint_projector_formula_symbolica(
        topology,
        sector,
        signature,
        EndpointProjectorFormulaDefinition,
        ibp_reduce_to_log_endpoint=True,
    )

    assert formula.ibp_reduce_to_log_endpoint is True
    assert any(key[0] == (0,) for key in formula.coefficient_layout)
    assert any(key[2] == (1,) for key in formula.coefficient_layout)
    assert list(tmp_path.glob("endpoint_projector_*.json"))


def test_expression_series_integer_inverse_cancels_endpoint_monomial() -> None:
    """Expression-side monomial division must not use float log/exp powers.

    The explicit IBP sector path builds residuals such as ``F/M_F`` as
    Symbolica Taylor series.  A previous implementation used ``exp(-log(M))``
    for the integer inverse of ``M``.  Near multi-axis corners this introduced
    tiny floating coefficient errors in huge inverse-monomial coefficients,
    which survived cancellation and looked like real endpoint power growth.
    """
    y_symbols = [S(f"toy_y{index}") for index in range(4)]
    monomial = y_symbols[0] * (y_symbols[1] ** 2) * y_symbols[2] * y_symbols[3]
    max_orders = [1, 1, 1, 1]
    monomial_series = {
        multi: _expr_derivative_coefficient(monomial, y_symbols, multi)
        for multi in _multi_indices(max_orders)
    }
    unity_series = _expr_series_mul(
        monomial_series,
        _expr_series_pow_real(monomial_series, -1.0, max_orders),
        max_orders,
    )
    probe_multi = (1, 1, 1, 1)
    evaluator = integrand_module.Expression.evaluator_multiple(
        [unity_series.get(probe_multi, E("0"))],
        y_symbols,
    )
    value = evaluator.evaluate_complex_with_prec(
        [_decimal_complex(1.0e-4, 1000) for _ in y_symbols],
        1000,
    )[0]

    assert value[0] == 0
    assert value[1] == 0


def test_ibp_shared_taylor_envelopes_cover_child_projectors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """IBP child projectors reuse one Taylor envelope per boundary/zero set."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0", "x1"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("1"),
        u_power_base=0.0,
        f_power_base=-3.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=1.0,
        expected_laurent_orders=["eps^-2", "eps^-1", "eps^0"],
        convention_note="toy",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0, 1.0),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-3.0, 1.0),
            parameter_weight_powers=(0.0, 0.0),
            prefactor_description="toy",
            convention_description="toy",
        ),
        ibp_reduce_to_log_endpoint=True,
    )
    sector = SectorDefinition(
        name="toy-sector-2d",
        integration_dim=2,
        variable_names=["y0", "y1"],
        map_exprs=[E("y0"), E("y1")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[1, 1],
        jacobian_monomial_powers=[0, 0],
        singular_axes=[0, 1],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[2, 2],
    )
    topology.prepare_endpoint_projector_formulas([sector])
    formula = topology.endpoint_projector_formula_for(sector)
    envelopes = SectorProcessor(topology)._ibp_shared_max_orders(sector, formula)

    exact_requests = []
    for term in formula.ibp_terms:
        child = formula.child_formulas[term.child_signature]
        active_positions = tuple(int(position) for position in term.active_positions)
        groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, ...]]] = {}
        for _child_boundary, child_zero, child_multi, _regular_order in child.coefficient_layout:
            original_zero = tuple(active_positions[position] for position in child_zero)
            original_multi = list(term.derivative_multi)
            for child_position, value in enumerate(child_multi):
                original_multi[active_positions[child_position]] += int(value)
            groups.setdefault(
                (tuple(term.boundary_positions), tuple(sorted(original_zero))),
                [],
            ).append(tuple(original_multi))
        for key, requests in groups.items():
            exact_requests.append((key, tuple(max(values) for values in zip(*requests))))

    assert len(envelopes) < len(exact_requests)
    for key, requested in exact_requests:
        envelope = envelopes[key]
        assert all(available >= needed for available, needed in zip(envelope, requested))


def test_ibp_child_projectors_are_batched_by_signature(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Batched IBP child evaluation matches term-by-term evaluation."""
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(tmp_path))
    topology = TopologyDefinition(
        family="toy",
        x_names=["x0", "x1"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("1"),
        u_power_base=0.0,
        f_power_base=-3.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=1.0,
        expected_laurent_orders=["eps^-2", "eps^-1", "eps^0"],
        convention_note="toy",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0, 1.0),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-3.0, 1.0),
            parameter_weight_powers=(0.0, 0.0),
            prefactor_description="toy",
            convention_description="toy",
        ),
        ibp_reduce_to_log_endpoint=True,
    )
    sector = SectorDefinition(
        name="toy-sector-2d",
        integration_dim=2,
        variable_names=["y0", "y1"],
        map_exprs=[E("y0"), E("y1")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[1, 1],
        jacobian_monomial_powers=[0, 0],
        singular_axes=[0, 1],
        subtraction_type="toy",
        description="toy",
        endpoint_taylor_orders=[2, 2],
    )
    topology.prepare_endpoint_projector_formulas([sector])
    formula = topology.endpoint_projector_formula_for(sector)
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")
    rows = np.asarray([[0.25, 0.375], [0.5, 0.125]], dtype=float)

    shared_max_orders = processor._ibp_shared_max_orders(sector, formula)
    shared_output_pairs = processor._ibp_shared_output_pairs(sector, formula)
    shared_g_cache: dict[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
        list[dict[tuple[int, ...], np.ndarray]],
    ] = {}
    reference = np.zeros((rows.shape[0], topology.coefficient_count), dtype=np.complex128)
    with np.errstate(divide="ignore", invalid="ignore"):
        for term in formula.ibp_terms:
            child = formula.child_formulas[term.child_signature]
            child_inputs = processor._ibp_child_input_matrix(
                sector,
                rows,
                child,
                term,
                HotPathTiming(),
                shared_g_cache=shared_g_cache,
                shared_max_orders=shared_max_orders,
                shared_output_pairs=shared_output_pairs,
            )
            child_values = child.evaluate_complex_batch(child_inputs, HotPathTiming())
            reference += processor._convolve_regular_prefactor_array(
                child_values,
                term.prefactor_coeffs,
            )

    calls: list[tuple[tuple[Any, ...], int]] = []
    for signature, child in formula.child_formulas.items():
        original = child.evaluate_complex_batch

        def wrapped(
            input_rows: np.ndarray,
            timing: HotPathTiming | None = None,
            *,
            original: Any = original,
            signature: tuple[Any, ...] = signature,
        ) -> np.ndarray:
            calls.append((signature, int(input_rows.shape[0])))
            return original(input_rows, timing)

        child.evaluate_complex_batch = wrapped  # type: ignore[method-assign]

    with np.errstate(divide="ignore", invalid="ignore"):
        batched, _training = processor._ibp_endpoint_projector_subtraction_batch(
            sector,
            rows,
            formula,
            HotPathTiming(),
        )

    np.testing.assert_allclose(batched, reference, rtol=1.0e-12, atol=1.0e-12)
    assert len(calls) == len(formula.child_formulas)
    assert sum(row_count for _signature, row_count in calls) == len(formula.ibp_terms) * rows.shape[0]


def test_symbolic_derivative_pair_reuses_map_context() -> None:
    """Paired symbolic derivative Taylor evaluation matches separate U/F calls."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/double_box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/double_box_kinematics.yaml"),
        mode="massless",
        m=0.0,
        sector_method="iterative",
        dual_evaluator_mode="symbolic-derivatives",
        prefactor_convention="pysecdec",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    sector = next(sector for sector in sectors if len(sector.singular_axes) >= 3)
    topology.prepare_dual_evaluators([sector], request.dual_evaluator_mode)
    rows = np.full((2, sector.integration_dim), 0.31, dtype=float)
    rows[1, :] = np.linspace(0.23, 0.77, sector.integration_dim)

    paired_u, paired_f = topology._symbolic_derivative_taylor_pair_batch(sector, rows)
    separate_u = topology.u_taylor_batch(sector, rows)
    separate_f = topology.f_taylor_batch(sector, rows)

    assert np.allclose(paired_u, separate_u, rtol=1.0e-12, atol=1.0e-12)
    assert np.allclose(paired_f, separate_f, rtol=1.0e-12, atol=1.0e-12)


def test_symbolic_derivative_context_uses_structural_map_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Runtime map Taylor context should not rescan every jet with np.any."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/double_box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/double_box_kinematics.yaml"),
        mode="massless",
        m=0.0,
        sector_method="iterative",
        dual_evaluator_mode="symbolic-derivatives",
        prefactor_convention="pysecdec",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    sector = next(sector for sector in sectors if len(sector.singular_axes) >= 3)
    output_shape = sector.dual_shape[: min(len(sector.dual_shape), 16)]
    topology._chain_rule_h_layout(sector, output_shape)

    def forbidden_any(*_args: Any, **_kwargs: Any) -> bool:
        raise AssertionError("runtime context should use the cached structural layout")

    monkeypatch.setattr("integrand.np.any", forbidden_any)
    rows = np.asarray([np.linspace(0.19, 0.71, sector.integration_dim)], dtype=float)
    context = topology._symbolic_derivative_taylor_context_batch(
        sector,
        rows,
        output_shape=output_shape,
    )

    assert context["h_series"]


def test_symbolic_derivative_values_use_multi_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ordinary derivative batches should not loop over single-output evaluators."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        dual_evaluator_mode="symbolic-derivatives",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    topology.prepare_dual_evaluators(sectors, request.dual_evaluator_mode)
    derivative_indices = topology._symbolic_derivative_indices("f", 2)
    x_values = np.asarray([[0.5, 0.2, 0.3], [0.5, 0.4, 0.1]], dtype=float)

    def forbidden_single_evaluate(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("single-output derivative evaluator was used")

    monkeypatch.setattr(topology, "_timed_evaluate", forbidden_single_evaluate)
    values = topology._derivative_values_batch("f", x_values, derivative_indices, None)

    assert set(values) == set(derivative_indices)
    assert topology._f_derivative_multi_evaluators
    assert all(value.shape == (2,) for value in values.values())


def test_chain_rule_formula_matches_python_symbolic_composition() -> None:
    """Prepared chain-rule formulas reproduce the retained Python composer."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/double_box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/double_box_kinematics.yaml"),
        mode="massless",
        m=0.0,
        sector_method="iterative",
        dual_evaluator_mode="symbolic-derivatives",
        prefactor_convention="pysecdec",
    )
    try:
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")
    configure_laurent_range(request, topology, sectors)
    sector = next(sector for sector in sectors if len(sector.singular_axes) >= 3)
    rows = np.asarray(
        [
            np.linspace(0.19, 0.71, sector.integration_dim),
            np.linspace(0.31, 0.83, sector.integration_dim),
        ],
        dtype=float,
    )
    output_shape = sector.dual_shape[: min(len(sector.dual_shape), 16)]
    context = topology._symbolic_derivative_taylor_context_batch(
        sector,
        rows,
        output_shape=output_shape,
    )

    formula_values = topology._compose_symbolic_derivative_taylor_batch(
        sector,
        context,
        "f",
        output_shape=output_shape,
        use_chain_formula=True,
    )
    python_values = topology._compose_symbolic_derivative_taylor_batch(
        sector,
        context,
        "f",
        output_shape=output_shape,
        use_chain_formula=False,
    )

    assert topology._chain_rule_formulas
    np.testing.assert_allclose(formula_values, python_values, rtol=1.0e-12, atol=1.0e-12)


def test_chain_rule_expression_sidecar_resume_skips_expression_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A native expression sidecar should restart evaluator generation directly."""
    cache_dir = tmp_path / "formula_cache"
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("FSD_CHAIN_RULE_MONITOR", "true")
    monkeypatch.setenv("FSD_CHAIN_RULE_EXPRESSION_SIDECAR_REQUIRED", "true")
    monkeypatch.setenv("FSD_CHAIN_RULE_EXPRESSION_PROGRESS_EVERY", "1")
    monkeypatch.setenv("FSD_CHAIN_RULE_EXPRESSION_COMPRESSION_LEVEL", "0")
    monkeypatch.setenv("FSD_SYMBOLICA_EVALUATOR_CORES", "1")
    monkeypatch.setenv("FSD_SYMBOLICA_EVALUATOR_VERBOSE", "false")
    monkeypatch.setattr(integrand_module, "formula_cache_read_roots", lambda: [cache_dir])

    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        dual_evaluator_mode="symbolic-derivatives",
    )
    topology = build_topology(request)

    class DummySector:
        name = "sidecar_resume_sector"

        def structurally_active_map_indices(self) -> tuple[int, ...]:
            return (0,)

    sector = DummySector()
    output_shape = [(0,), (1,)]
    signature = topology._chain_rule_formula_signature(sector, "f", output_shape)
    cache_json_path = integrand_module._chain_rule_formula_cache_path(signature)

    generated = topology._build_chain_rule_formula(sector, "f", output_shape, signature)
    assert generated.cache_expression_manifest_file is not None
    assert len(generated.cache_expression_files) == len(output_shape)
    assert (cache_dir / generated.cache_expression_manifest_file).is_file()
    assert all((cache_dir / name).is_file() for name in generated.cache_expression_files)
    assert not cache_json_path.exists()

    rows = np.asarray([[0.25, 2.0, 5.0], [0.5, 3.0, -7.0]], dtype=np.complex128)
    generated_values = generated.evaluate_complex_batch(rows)
    capsys.readouterr()

    def forbidden_expression_generation(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("expression generation was entered despite sidecar cache")

    monkeypatch.setattr(
        integrand_module,
        "_expr_series_mul_allowed",
        forbidden_expression_generation,
    )
    resumed = topology.chain_rule_formula_for(sector, "f", output_shape)
    captured = capsys.readouterr()

    assert resumed.cache_expression_manifest_file == generated.cache_expression_manifest_file
    assert resumed.cache_expression_files == generated.cache_expression_files
    assert cache_json_path.exists()
    assert "expressions_loaded_sidecar" in captured.err
    assert "evaluator_start source=expression-sidecar" in captured.err
    assert "expressions_done" not in captured.err
    np.testing.assert_allclose(
        resumed.evaluate_complex_batch(rows),
        generated_values,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_regular_taylor_expression_checkpoint_rebuilds_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regular Taylor expression checkpoints should skip expression generation."""
    cache_dir = tmp_path / "formula_cache"
    monkeypatch.setenv("FSD_SUBTRACTION_FORMULA_CACHE_DIR", str(cache_dir))
    monkeypatch.setenv("FSD_REGULAR_TAYLOR_EXPRESSION_COMPRESSION_LEVEL", "0")
    monkeypatch.setattr(subtraction_formula_module, "formula_cache_read_roots", lambda: [cache_dir])

    signature = ("regular-taylor", 3, 6, "checkpoint-test")
    input_names = ["x"]
    output_layout = [((0,), 0), ((1,), 1)]
    input_layout = [("u", (0,)), ("u", (1,))]
    max_orders = [1]
    zero_positions: tuple[int, ...] = ()
    x = S("x")
    outputs = [x + E("1"), x * x]

    manifest_name, sidecar_names = (
        subtraction_formula_module._write_regular_expression_checkpoint_to_cache(
            signature,
            input_names,
            output_layout,
            input_layout,
            max_orders,
            zero_positions,
            outputs,
        )
    )
    cache_json_path = subtraction_formula_module._regular_taylor_cache_path(signature)
    assert cache_json_path.is_file()
    assert manifest_name is not None
    assert (cache_dir / manifest_name).is_file()
    assert all((cache_dir / name).is_file() for name in sidecar_names)

    calls: list[int] = []

    def fake_build_evaluator_multiple(
        expressions: list[Any],
        input_symbols: list[Any],
        *,
        jit_compile: bool,
        jit_direct_translation: bool = False,
        monitor: Any = None,
    ) -> tuple[list[Any], str]:
        calls.append(len(expressions))
        assert [str(symbol) for symbol in input_symbols] == input_names
        assert jit_compile is False
        assert jit_direct_translation is False
        return ["rebuilt-evaluator"], "multiple"

    class DummyTopology:
        jit_compile_evaluators = False
        jit_direct_translation = False

    class DummyFormula:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    monkeypatch.setattr(
        subtraction_formula_module,
        "_build_evaluator_multiple",
        fake_build_evaluator_multiple,
    )
    loaded = subtraction_formula_module._load_regular_taylor_formula_from_cache(
        DummyTopology(),
        signature,
        DummyFormula,
    )

    assert loaded is not None
    assert calls == [len(outputs)]
    assert loaded.evaluators == ["rebuilt-evaluator"]
    assert loaded.evaluator_mode == "multiple"
    assert len(loaded.output_expressions) == len(outputs)
    assert loaded.output_layout == output_layout
    assert loaded.input_layout == input_layout


def test_chain_rule_signature_ignores_sector_name() -> None:
    """Equivalent map-jet layouts share one chain-rule formula signature."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        dual_evaluator_mode="symbolic-derivatives",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    sector = sectors[0]
    clone = replace(sector, name=f"{sector.name}_renamed")
    clone.prepare_evaluators(include_dual=False)
    topology.prepare_dual_evaluators([sector, clone], request.dual_evaluator_mode)
    output_shape = sector.dual_shape[: min(len(sector.dual_shape), 8)]

    original_signature = topology._chain_rule_formula_signature(sector, "f", output_shape)
    renamed_signature = topology._chain_rule_formula_signature(clone, "f", output_shape)

    assert original_signature == renamed_signature


def test_chain_rule_formula_guard_skips_large_pregeneration_request() -> None:
    """The chain-rule guard avoids cold-building too many composition formulas."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        dual_evaluator_mode="symbolic-derivatives",
        subtraction_backend="projector-formula",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    topology.prepare_dual_evaluators(sectors, request.dual_evaluator_mode)
    topology.prepare_endpoint_projector_formulas(sectors)
    topology.prepare_regular_taylor_formulas(sectors)

    topology.chain_rule_formula_signature_limit = 0
    topology.prepare_chain_rule_formulas(sectors)

    assert topology.chain_rule_formulas_skipped > 0
    assert topology._chain_rule_formulas == {}


def test_single_overall_dual_evaluator_matches_per_sector_shape() -> None:
    """Envelope dual evaluators are remapped back to sector-native columns."""
    base_request = make_request(integral="triangle", mode="massless", s=-1.0, m=0.0)
    envelope_request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        dual_evaluator_mode="single-overall",
    )

    base_topology = build_topology(base_request)
    base_sectors = generate_sectors(base_request)
    base_topology.prepare_dual_evaluators(base_sectors, base_request.dual_evaluator_mode)

    envelope_topology = build_topology(envelope_request)
    envelope_sectors = generate_sectors(envelope_request)
    envelope_topology.prepare_dual_evaluators(envelope_sectors, envelope_request.dual_evaluator_mode)

    y_values = np.asarray([[0.0, 0.25], [0.0, 0.5]], dtype=float)
    base = base_topology.f_taylor_batch(base_sectors[0], y_values)
    envelope = envelope_topology.f_taylor_batch(envelope_sectors[0], y_values)

    assert base.shape == envelope.shape
    assert np.allclose(base, envelope)
    assert envelope_topology._overall_dual_shapes


def test_single_overall_dual_evaluator_handles_mixed_axis_counts() -> None:
    """Envelope mode pads one-axis sectors into the two-axis box envelope."""
    base_request = make_request(integral="box", mode="massless", s12=-1.0, s23=-2.0, m=0.0)
    envelope_request = make_request(
        integral="box",
        mode="massless",
        s12=-1.0,
        s23=-2.0,
        m=0.0,
        dual_evaluator_mode="single-overall",
    )

    base_topology = build_topology(base_request)
    base_sectors = generate_sectors(base_request)
    base_topology.prepare_dual_evaluators(base_sectors, base_request.dual_evaluator_mode)

    envelope_topology = build_topology(envelope_request)
    envelope_sectors = generate_sectors(envelope_request)
    envelope_topology.prepare_dual_evaluators(envelope_sectors, envelope_request.dual_evaluator_mode)

    one_axis_index = next(i for i, sector in enumerate(base_sectors) if len(sector.singular_axes) == 1)
    y_values = np.asarray([[0.0, 0.25, 0.5], [0.0, 0.5, 0.25]], dtype=float)
    base = base_topology.f_taylor_batch(base_sectors[one_axis_index], y_values)
    envelope = envelope_topology.f_taylor_batch(envelope_sectors[one_axis_index], y_values)

    assert base.shape == envelope.shape
    assert np.allclose(base, envelope)


def test_symbolic_derivative_taylor_matches_dualized_topology_evaluator() -> None:
    """Symbolic U/F derivatives compose to the same sector Taylor coefficients."""
    cases = [
        (
            make_request(integral="triangle", mode="massless", s=-1.0, m=0.0),
            np.asarray([[0.0, 0.25], [0.0, 0.5], [0.125, 0.0]], dtype=float),
            0,
        ),
        (
            make_request(integral="box", mode="massless", s12=-1.0, s23=-2.0, m=0.0),
            np.asarray([[0.0, 0.25, 0.5], [0.0, 0.5, 0.25]], dtype=float),
            None,
        ),
    ]
    for base_request, y_values, explicit_sector_index in cases:
        symbolic_request = make_request(
            **{
                **base_request.__dict__,
                "dual_evaluator_mode": "symbolic-derivatives",
            }
        )
        base_topology = build_topology(base_request)
        base_sectors = generate_sectors(base_request)
        base_topology.prepare_dual_evaluators(base_sectors, base_request.dual_evaluator_mode)

        symbolic_topology = build_topology(symbolic_request)
        symbolic_sectors = generate_sectors(symbolic_request)
        symbolic_topology.prepare_dual_evaluators(
            symbolic_sectors,
            symbolic_request.dual_evaluator_mode,
        )

        sector_index = (
            explicit_sector_index
            if explicit_sector_index is not None
            else next(i for i, sector in enumerate(base_sectors) if sector.dual_shape)
        )
        base_sector = base_sectors[sector_index]
        symbolic_sector = symbolic_sectors[sector_index]

        assert np.allclose(
            base_topology.f_taylor_batch(base_sector, y_values),
            symbolic_topology.f_taylor_batch(symbolic_sector, y_values),
        )
        assert np.allclose(
            base_topology.u_taylor_batch(base_sector, y_values),
            symbolic_topology.u_taylor_batch(symbolic_sector, y_values),
        )


def test_dot_generation_timing_has_requested_headline_buckets() -> None:
    """DOT generation timings expose the three requested headline buckets."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/triangle.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/triangle_kinematics.yaml"),
        mode="massless",
        m=0.0,
        prefactor_convention="sector",
    )
    try:
        validate_request(request)
        from dot_topology import get_dot_bundle

        timing_data = get_dot_bundle(request).timings.to_summary_dict()
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")

    headline_names = {record["name"] for record in timing_data["headline"]}
    assert {
        "Generation U and F polynomial",
        "Generating sectors",
        "Generating Symbolica evaluators",
    } <= headline_names


def test_direct_uf_double_box_matches_dot_construction() -> None:
    """The direct U/F double-box preset reproduces DOT topology/sector metadata."""
    clear_dot_bundle_cache()
    from uf_topology import clear_uf_bundle_cache

    clear_uf_bundle_cache()
    uf_args = parse_args(
        [
            "--run",
            str(PROJECT_ROOT / "examples/runs/double_box_from_U_and_F.yaml"),
            "--samples-per-iter",
            "16",
            "--batch-size",
            "16",
            "--max-iter",
            "1",
            "--no-progress",
            "--quiet-summary",
        ]
    )
    dot_args = parse_args(
        [
            "--run",
            str(PROJECT_ROOT / "examples/runs/dot_double_box.yaml"),
            "--samples-per-iter",
            "16",
            "--batch-size",
            "16",
            "--max-iter",
            "1",
            "--no-progress",
            "--quiet-summary",
        ]
    )
    uf_request = build_request(uf_args)
    dot_request = build_request(dot_args)
    try:
        validate_request(uf_request)
        validate_request(dot_request)
        uf_topology = build_topology(uf_request)
        dot_topology = build_topology(dot_request)
        uf_sectors = generate_sectors(uf_request)
        dot_sectors = generate_sectors(dot_request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")

    assert uf_request.integral == "uf"
    assert uf_topology.laurent_min_order == dot_topology.laurent_min_order == -4
    assert uf_topology.laurent_max_order == dot_topology.laurent_max_order
    assert uf_topology.global_prefactor_min_order == dot_topology.global_prefactor_min_order
    assert np.allclose(
        uf_topology.global_prefactor_coeffs,
        dot_topology.global_prefactor_coeffs,
    )
    assert len(uf_sectors) == len(dot_sectors)
    assert Counter(len(sector.singular_axes) for sector in uf_sectors) == Counter(
        len(sector.singular_axes) for sector in dot_sectors
    )

    x_sample = [0.05, 0.1, 0.2, 0.07, 0.11, 0.13, 0.17]
    assert abs(uf_topology.u_value(x_sample) - dot_topology.u_value(x_sample)) < 1.0e-12
    assert abs(uf_topology.f_value(x_sample) - dot_topology.f_value(x_sample)) < 1.0e-12
    y_sample = [0.23 for _ in range(uf_sectors[0].integration_dim)]
    assert np.allclose(uf_sectors[0].map_eval(y_sample), dot_sectors[0].map_eval(y_sample))
    assert uf_sectors[0].u_monomial_powers == dot_sectors[0].u_monomial_powers
    assert uf_sectors[0].f_monomial_powers == dot_sectors[0].f_monomial_powers
    assert uf_sectors[0].measure_monomial_powers == dot_sectors[0].measure_monomial_powers


def test_direct_uf_prepared_bundle_round_trips(tmp_path: Path) -> None:
    """Direct U/F topology metadata and evaluators survive prepared-bundle IO."""
    from uf_topology import clear_uf_bundle_cache

    clear_uf_bundle_cache()
    output_dir = tmp_path / "prepared_uf_double_box"
    request = build_request(
        parse_args(
            [
                "generate",
                "--run",
                str(PROJECT_ROOT / "examples/runs/double_box_from_U_and_F.yaml"),
                "--output",
                str(output_dir),
                "--samples-per-iter",
                "16",
                "--batch-size",
                "16",
                "--max-iter",
                "1",
                "--no-progress",
                "--quiet-summary",
            ]
        )
    )
    try:
        validate_request(request)
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")

    manifest = save_prepared_bundle(
        output_dir,
        request,
        topology,
        sectors,
        generation_timings={},
    )
    loaded_topology, loaded_sectors, loaded_manifest = load_prepared_bundle(output_dir)

    assert manifest["source_files"]["topology_source"] == "uf"
    assert loaded_manifest["source_files"]["topology_source"] == "uf"
    assert loaded_topology.family == topology.family
    assert loaded_topology.parametric_representation.propagator_powers == (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert len(loaded_sectors) == len(sectors)
    assert loaded_sectors[0].measure_monomial_powers == sectors[0].measure_monomial_powers


def test_dot_nonunit_power_reaches_sector_measure_metadata(tmp_path: Path) -> None:
    """DOT propagator powers are represented as sector measure monomials."""
    clear_dot_bundle_cache()
    dot_text = (PROJECT_ROOT / "examples/graphs/triangle.dot").read_text(encoding="utf-8")
    dot_file = tmp_path / "triangle_power.dot"
    dot_file.write_text(
        dot_text.replace('name="e0", mass="mt"', 'name="e0", mass="mt", power=2'),
        encoding="utf-8",
    )
    request = make_request(
        integral="dot",
        dot_file=str(dot_file),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/triangle_kinematics.yaml"),
        mode="massless",
        m=0.0,
        prefactor_convention="sector",
        subtraction_backend="projector-formula",
    )
    try:
        validate_request(request)
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")

    assert topology.parametric_representation.propagator_powers[0] == 2.0
    assert topology.parametric_representation.parameter_weight_powers[0] == 1.0
    assert any(any(abs(power) > 0.0 for power in sector.measure_monomial_powers) for sector in sectors)


def test_dot_fsd_integration_does_not_reenter_pysecdec_after_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared DOT topology/sectors are sufficient for FSD integration."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/triangle.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/triangle_kinematics.yaml"),
        mode="massless",
        m=0.0,
        prefactor_convention="sector",
        samples_per_iter=64,
        batch_size=32,
        workers=2,
    )
    try:
        validate_request(request)
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")
    prepare_generated_evaluators(topology, sectors, request.dual_evaluator_mode)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("pySecDec generation was re-entered during integration")

    monkeypatch.setattr("pysecdec_bridge.require_pysecdec", forbidden)
    monkeypatch.setattr("dot_topology.build_dot_bundle", forbidden)

    try:
        result = integrate(request, topology, sectors, None)
    except PermissionError as exc:
        pytest.skip(f"multiprocessing semaphores unavailable in this sandbox: {exc}")

    assert result.samples == request.samples_per_iter


def _prepared_triangle_bundle(tmp_path: Path) -> tuple[IntegralRequest, Path]:
    """Generate a tiny DOT triangle prepared bundle for serialization tests."""
    output_dir = tmp_path / "prepared_triangle"
    request = make_request(
        command="generate",
        output=str(output_dir),
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/triangle.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/triangle_kinematics.yaml"),
        mode="massless",
        m=0.0,
        prefactor_convention="sector",
        subtraction_backend="projector-formula",
        samples_per_iter=64,
        batch_size=32,
        workers=1,
    )
    try:
        validate_request(request)
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(
        topology,
        sectors,
        request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )
    save_prepared_bundle(
        output_dir,
        request,
        topology,
        sectors,
        generation_timings={"headline": [], "total": 0.0},
    )
    return request, output_dir


def test_prepared_bundle_round_trips_symbolica_evaluators(tmp_path: Path) -> None:
    """A prepared bundle reloads evaluator bytes instead of rebuilding expressions."""
    _request, output_dir = _prepared_triangle_bundle(tmp_path)

    topology, sectors, manifest = load_prepared_bundle(output_dir, lru_size=2)

    assert manifest["artifact_counts"]["evaluator_files"] > 0
    assert len(sectors) == 2
    values = topology.u_values(np.asarray([[0.2, 0.3, 0.5]], dtype=float))
    assert np.all(np.isfinite(values))


def test_prepared_bundle_missing_evaluator_fails_strictly(tmp_path: Path) -> None:
    """Integrate-mode bundle loading fails when a serialized evaluator is absent."""
    _request, output_dir = _prepared_triangle_bundle(tmp_path)
    first_evaluator = next((output_dir / "evaluators").glob("*.bin*"))
    first_evaluator.unlink()

    with pytest.raises(RuntimeError, match="missing evaluator artifact"):
        load_prepared_bundle(output_dir)


def test_prepared_bundle_integrates_without_generation_reentry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Disk-prepared DOT bundles can be integrated after disabling generation hooks."""
    _request, output_dir = _prepared_triangle_bundle(tmp_path)
    topology, sectors, _manifest = load_prepared_bundle(output_dir)
    request = make_request(
        command="integrate",
        output=str(output_dir),
        integral="dot",
        dot_file=None,
        kinematics_file=None,
        mode="massless",
        m=0.0,
        prefactor_convention="sector",
        subtraction_backend="projector-formula",
        dual_evaluator_mode=topology.dual_evaluator_mode,
        ibp_reduce_to_log_endpoint=topology.ibp_reduce_to_log_endpoint,
        samples_per_iter=64,
        batch_size=32,
        workers=1,
        target_args=("0", "0", "0", "0", "0", "0"),
        result_path=str(output_dir / "result.json"),
        dot_global_prefactor_coeffs=tuple(topology.global_prefactor_coeffs or []),
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("prepared integrate mode re-entered generation")

    monkeypatch.setattr("pysecdec_bridge.require_pysecdec", forbidden)
    monkeypatch.setattr("dot_topology.build_dot_bundle", forbidden)

    result = integrate(request, topology, sectors, None)

    assert result.samples == request.samples_per_iter


def test_prepared_bundle_can_display_lower_laurent_subrange(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A bundle prepared through eps^0 can persist a leading-order result view."""
    _request, output_dir = _prepared_triangle_bundle(tmp_path)
    result_path = output_dir / "subrange.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "FSD.py",
            "integrate",
            "--output",
            str(output_dir),
            "--prefactor-convention",
            "sector",
            "--max-eps-order",
            "-1",
            "--samples-per-iter",
            "64",
            "--batch-size",
            "32",
            "--workers",
            "1",
            "--target",
            "0",
            "0",
            "0",
            "0",
            "--result-path",
            str(result_path),
            "--json",
            "--no-progress",
            "--quiet-summary",
        ],
    )

    assert main() == 0
    data = json.loads(result_path.read_text(encoding="utf-8"))

    assert data["laurent_labels"] == ["eps^-2", "eps^-1"]
    assert len(data["aggregate_results"]["display"]["coefficients"]) == 2
    assert len(data["sector_results"][0]["display"]["coefficients"]) == 2


def test_democratic_sampling_hits_every_sector_equally() -> None:
    """Democratic mode gives each active sector the same explicit sample count."""
    request = make_request(
        integral="triangle",
        mode="massless",
        m=0.0,
        s=-1.0,
        subtraction_backend="projector-formula",
        sampling_mode="democratic",
        democratic_samples_per_sector=3,
        batch_size=3,
        samples_per_iter=9,
        workers=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(
        topology,
        sectors,
        request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )

    result = integrate(request, topology, sectors, None)

    assert result.samples == 3 * len(sectors)
    assert result.diagnostics["sampling_mode"] == "democratic"
    assert result.diagnostics["samples_per_sector"] == 3
    for sector_result in result.per_sector:
        assert sector_result.samples == 3
        assert sector_result.diagnostics["sampling_mode"] == "democratic"
        assert "max_abs_weight" in sector_result.diagnostics


def test_democratic_batches_are_round_robin() -> None:
    """Small democratic chunks should cover every sector before the next round."""
    request = make_request(
        integral="triangle",
        mode="massless",
        m=0.0,
        s=-1.0,
        sampling_mode="democratic",
        democratic_samples_per_sector=3,
        batch_size=1,
    )
    sectors = generate_sectors(request)

    batches = integrator_module.democratic_batches(
        request,
        list(range(len(sectors))),
        sectors,
    )

    first_round = [batch.sector_id for batch in batches[: len(sectors)]]
    second_round = [batch.sector_id for batch in batches[len(sectors) : 2 * len(sectors)]]
    assert first_round == list(range(len(sectors)))
    assert second_round == list(range(len(sectors)))
    assert [batch.coords.shape[0] for batch in batches] == [1] * (3 * len(sectors))


def test_korobov_transform_alpha_three_is_normalized() -> None:
    """The QMC Korobov map fixes endpoints and has unit average Jacobian."""
    points = np.linspace(0.0, 1.0, 1001, dtype=float)[:, np.newaxis]
    coords, weights = integrator_module.korobov_transform(points, 3)

    assert coords[0, 0] == pytest.approx(0.0)
    assert coords[-1, 0] == pytest.approx(1.0)
    assert np.all(coords[:, 0] >= -1.0e-15)
    assert np.all(coords[:, 0] <= 1.0 + 1.0e-15)
    assert np.trapezoid(weights, points[:, 0]) == pytest.approx(1.0, rel=1.0e-6)


def test_qmcpy_qmc_lattice_requires_power_of_two_samples() -> None:
    """The independent QMCPy backend should not silently resize lattices."""
    assert is_power_of_two(1024)
    assert not is_power_of_two(1123)

    with pytest.raises(ValueError, match="power of two"):
        qmcpy_shifted_lattice_points(
            dimension=2,
            n_points=1123,
            shift_count=4,
            seed=11,
            order="linear",
        )


def test_qmcpy_shifted_rank1_lattice_shape_and_range() -> None:
    """The QMCPy rank-1 lattice helper returns shift-major unit-cube points."""
    points = qmcpy_shifted_lattice_points(
        dimension=2,
        n_points=8,
        shift_count=4,
        seed=11,
        order="linear",
    )

    assert points.shape == (4, 8, 2)
    assert np.all(points >= 0.0)
    assert np.all(points < 1.0)
    # The same seed must reproduce the same random shifts and lattice points.
    assert np.allclose(
        points,
        qmcpy_shifted_lattice_points(
            dimension=2,
            n_points=8,
            shift_count=4,
            seed=11,
            order="linear",
        ),
    )


def test_cbcpt_shifted_rank1_lattice_uses_prime_rule_size() -> None:
    """The bundled CBC/PT backend treats n_points as a lower bound."""
    points = cbcpt_dn1_shifted_lattice_points(
        dimension=3,
        n_points=4096,
        shift_count=4,
        seed=11,
    )

    assert points.shape == (4, 4261, 3)
    assert np.all(points >= 0.0)
    assert np.all(points < 1.0)


def test_pysecdec_default_lattice_supports_high_dimension() -> None:
    """The pySecDec-compatible table should cover PSD-style 8/9D sectors."""
    vector_size, vector = pysecdec_default_vector_info(dimension=9, n_points=4096)
    points = pysecdec_default_shifted_lattice_points(
        dimension=9,
        n_points=4096,
        shift_count=3,
        seed=11,
    )

    assert vector_size >= 4096
    assert len(vector) == 9
    assert points.shape == (3, vector_size, 9)
    assert np.all(points >= 0.0)
    assert np.all(points < 1.0)
    assert actual_lattice_point_count_for_dimension(
        backend="pysecdec-default",
        n_points=4096,
        dimension=9,
    ) == vector_size
    with pytest.raises(ValueError, match="dimensions up to 100"):
        pysecdec_default_vector_info(dimension=101, n_points=4096)


def test_pysecdec_default_lattice_slice_matches_full_points() -> None:
    """Worker-side streamed lattice chunks must reproduce full-array points."""
    full = pysecdec_default_shifted_lattice_points(
        dimension=5,
        n_points=4096,
        shift_count=3,
        seed=17,
    )
    flat = full.reshape(full.shape[0] * full.shape[1], full.shape[2])
    points, shifts, vector_size = shifted_lattice_point_slice(
        backend="pysecdec-default",
        dimension=5,
        n_points=4096,
        shift_count=3,
        seed=17,
        order="linear",
        start=123,
        count=1000,
    )

    assert vector_size == full.shape[1]
    assert np.allclose(points, flat[123:1123])
    assert np.array_equal(shifts, np.arange(123, 1123, dtype=np.int64) // vector_size)


def test_qmc_support_groups_skip_zero_poles_and_promote_global_support() -> None:
    """QMC support grouping should mirror pySecDec coefficient containers."""
    topology = SimpleNamespace(coefficient_count=3, laurent_orders=[-2, -1, 0])
    two_axis = SectorDefinition(
        name="two_axis",
        integration_dim=3,
        variable_names=["x1", "x2", "x3"],
        map_exprs=[E("x1"), E("x2"), E("x3")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[1, 1, 0],
        jacobian_monomial_powers=[0, 0, 0],
        singular_axes=[0, 1],
        subtraction_type="test",
        description="test",
    )
    one_axis = SectorDefinition(
        name="one_axis",
        integration_dim=3,
        variable_names=["x1", "x2", "x3"],
        map_exprs=[E("x1"), E("x2"), E("x3")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[0, 1, 0],
        jacobian_monomial_powers=[0, 0, 0],
        singular_axes=[1],
        subtraction_type="test",
        description="test",
    )

    assert integrator_module.qmc_support_groups(topology, two_axis) == [
        ((0,), (2,)),
        ((1, 2), (0, 1, 2)),
    ]
    assert integrator_module.qmc_support_groups(topology, one_axis) == [
        ((1,), (0, 2)),
        ((2,), (0, 1, 2)),
    ]

    global_dims = integrator_module.qmc_global_support_dims(topology, [two_axis, one_axis])
    assert global_dims == (1, 3, 3)
    assert integrator_module.qmc_support_groups(topology, one_axis, global_dims) == [
        ((1, 2), (0, 1, 2)),
    ]


def test_qmc_adaptive_scheduler_focuses_on_dominant_sector() -> None:
    """After the pilot iteration, adaptive QMC should refine error-dominant sectors."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        sampling_mode="qmc",
        qmc_refine_sectors="adaptive",
    )
    sector_stats = [
        [integrator_module.RunningStats() for _ in range(2)]
        for _sector in range(4)
    ]
    for sector_id in range(4):
        for coeff_index in range(2):
            if sector_id == 2:
                sector_stats[sector_id][coeff_index].add(0.0)
                sector_stats[sector_id][coeff_index].add(10.0)
            else:
                sector_stats[sector_id][coeff_index].add(1.0)
                sector_stats[sector_id][coeff_index].add(1.001)

    selected = integrator_module._qmc_select_iteration_sector_ids(
        request,
        2,
        [0, 1, 2, 3],
        sector_stats,
        2,
    )

    assert selected == [2]


def test_qmc_leader_percentages_are_colored_blue() -> None:
    """Only the percentages in the QMC leader text should carry blue ANSI."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        sampling_mode="qmc",
        qmc_refine_sectors="adaptive",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    stats = [
        [integrator_module.RunningStats() for _ in range(topology.coefficient_count)]
        for _sector in sectors
    ]
    for sector_id, scale in enumerate((10.0, 5.0)):
        for coeff_index in range(topology.coefficient_count):
            stats[sector_id][coeff_index].add(0.0)
            stats[sector_id][coeff_index].add(scale)

    text, _records = integrator_module._qmc_leader_text(
        sectors,
        stats,
        list(range(len(sectors))),
        topology.coefficient_count,
    )

    assert "\x1b[34m" in text
    assert "%" in text


def test_qmc_adaptive_ramp_uses_explicit_small_pilot() -> None:
    """Large production QMC settings should still begin with the pilot lattice."""
    request = make_request(
        sampling_mode="qmc",
        qmc_refine_sectors="adaptive",
        samples_per_iter=100_000,
        qmc_shifts=64,
        qmc_initial_samples_per_iter=1024,
        qmc_initial_shifts=16,
    )

    first = integrator_module._qmc_iteration_request(request, 1)
    second = integrator_module._qmc_iteration_request(request, 2)

    assert first.samples_per_iter == 1024
    assert first.qmc_shifts == 16
    assert second.samples_per_iter == 2048
    assert second.qmc_shifts == 32


def test_qmc_adaptive_ramp_respects_default_lattice_cap() -> None:
    """Adaptive QMC should stop ramping at the configured max lattice size."""
    request = make_request(
        sampling_mode="qmc",
        qmc_refine_sectors="adaptive",
        samples_per_iter=100_000,
        qmc_initial_samples_per_iter=1024,
        qmc_max_samples_per_iter=4096,
    )

    assert integrator_module._qmc_iteration_request(request, 4).samples_per_iter == 4096

    uncapped = replace(request, qmc_max_samples_per_iter=0)
    assert integrator_module._qmc_iteration_request(uncapped, 4).samples_per_iter == 8192


def test_qmc_lattice_cap_applies_to_democratic_mode() -> None:
    """The safety cap should also limit fixed democratic QMC schedules."""
    request = make_request(
        sampling_mode="qmc",
        qmc_refine_sectors="democratic",
        samples_per_iter=100_000,
        qmc_max_samples_per_iter=4096,
    )

    capped = integrator_module._qmc_iteration_request(request, 1)
    assert capped.samples_per_iter == 4096

    uncapped = integrator_module._qmc_iteration_request(
        replace(request, qmc_max_samples_per_iter=0),
        1,
    )
    assert uncapped.samples_per_iter == 100_000


def test_qmc_cbc_progress_uses_actual_lattice_size() -> None:
    """CBC/PT progress targets should use the selected prime rule size."""
    request = make_request(
        sampling_mode="qmc",
        qmc_lattice_backend="cbcpt-dn1-100",
        samples_per_iter=2048,
    )

    assert integrator_module.qmc_actual_points_per_shift(request, 3) == 4261
    assert integrator_module.qmc_actual_points_per_shift(request, 0) == 1


def test_qmc_sampling_hits_every_sector_with_shift_estimates() -> None:
    """QMC mode estimates errors from random shifts and records raw samples."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        subtraction_backend="projector-formula",
        sampling_mode="qmc",
        qmc_shifts=4,
        qmc_korobov_alpha=3,
        qmc_lattice_backend="qmcpy",
        samples_per_iter=32,
        max_iter=1,
        min_iter=1,
        batch_size=32,
        workers=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(
        topology,
        sectors,
        request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )

    result = integrate(request, topology, sectors, None)

    assert result.samples == 4 * 32 * len(sectors)
    assert result.diagnostics["sampling_mode"] == "qmc"
    assert result.diagnostics["qmc_software"] == "QMCPy Lattice"
    assert result.diagnostics["qmc_lattice_backend"] == "qmcpy"
    assert result.diagnostics["qmc_korobov_alpha"] == 3
    assert result.havana_seconds > 0.0
    for value in result.raw_sector_coeffs:
        assert_finite_complex(value)
    for sector_result in result.per_sector:
        assert sector_result.samples == 4 * 32
        assert sector_result.diagnostics["sampling_mode"] == "qmc"


def test_qmc_iteration_callback_receives_completed_aggregate() -> None:
    """QMC writes can be triggered once a reliable iteration has completed."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        subtraction_backend="projector-formula",
        sampling_mode="qmc",
        qmc_refine_sectors="adaptive",
        qmc_shifts=4,
        qmc_initial_shifts=4,
        qmc_lattice_backend="qmcpy",
        samples_per_iter=16,
        qmc_initial_samples_per_iter=16,
        qmc_max_samples_per_iter=16,
        max_iter=2,
        min_iter=1,
        batch_size=16,
        workers=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(
        topology,
        sectors,
        request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )
    snapshots: list[int] = []

    result = integrate(
        request,
        topology,
        sectors,
        None,
        iteration_callback=lambda partial: snapshots.append(partial.samples),
    )

    assert snapshots
    assert snapshots[-1] == result.samples
    assert all(sample_count > 0 for sample_count in snapshots)


def test_qmc_adaptive_shift_ramp_uses_iteration_shift_shape() -> None:
    """Adaptive QMC may use fewer shifts in early iterations than requested."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        subtraction_backend="projector-formula",
        sampling_mode="qmc",
        qmc_refine_sectors="adaptive",
        qmc_shifts=64,
        qmc_korobov_alpha=3,
        qmc_lattice_backend="qmcpy",
        samples_per_iter=16,
        max_iter=1,
        min_iter=1,
        batch_size=16,
        workers=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(
        topology,
        sectors,
        request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )

    result = integrate(request, topology, sectors, None)

    assert result.diagnostics["qmc_last_iteration"]["shifts"] == 16
    assert result.samples == 16 * 16 * len(sectors)
    for sector_result in result.per_sector:
        assert sector_result.diagnostics["qmc_shift_estimate_counts_by_order"][-1] == 16


def test_target_time_tuning_uses_steady_state_warmup_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Target-time tuning should not be pinned to conservative cold warm-up wall time."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        sampling_mode="qmc",
        qmc_lattice_backend="cbcpt-dn1-100",
        qmc_shifts=2,
        qmc_refine_sectors="adaptive",
        target_integration_time=10.0,
        samples_per_iter=1,
        max_iter=5,
        min_iter=1,
        workers=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)

    def fake_warmup(_request, _topology, _active_sectors):
        return {
            "warmup_records": 100,
            "warmup_seconds": 10.0,
            "warmup_setup_seconds": 0.5,
            "records_per_second": 10.0,
            "records_per_second_for_tuning": 25.0,
            "steady_state_warmup_seconds_for_tuning": 4.0,
            "workers": 1,
            "avg_eval_us_per_sample_per_worker": 1.0,
            "profile": {
                "python_fraction": 0.0,
                "evaluator_fraction": 1.0,
                "havana_fraction": 0.0,
            },
        }

    monkeypatch.setattr(integrator_module, "_measure_record_throughput", fake_warmup)
    tuned_request, diagnostics = integrator_module.autotune_request_for_target_time(
        request,
        topology,
        sectors,
    )

    assert diagnostics is not None
    qmc_group_count = diagnostics["qmc_group_count"]
    expected_target_records = 250
    effective_group_iterations = diagnostics["qmc_effective_group_iterations"]
    expected_samples = max(
        expected_target_records
        // max(int(request.qmc_shifts) * int(qmc_group_count) * int(effective_group_iterations), 1),
        1,
    )
    assert diagnostics["records_per_second"] == pytest.approx(10.0)
    assert diagnostics["tuning_records_per_second"] == pytest.approx(25.0)
    assert diagnostics["estimated_target_records"] == expected_target_records
    assert diagnostics["qmc_refine_sectors"] == "adaptive"
    assert tuned_request.samples_per_iter == expected_samples


def test_target_time_tuning_raises_qmc_lattice_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Target-time QMC tuning must not be silently clipped by the safety cap."""
    request = make_request(
        integral="box",
        mode="massless",
        s12=-1.0,
        s23=-1.0,
        sampling_mode="qmc",
        qmc_refine_sectors="democratic",
        qmc_lattice_backend="cbcpt-dn1-100",
        qmc_shifts=64,
        qmc_max_samples_per_iter=4096,
        samples_per_iter=50_000,
        max_iter=3,
        min_iter=1,
        workers=1,
        target_integration_time=30.0,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)

    def fake_warmup(_request, _topology, _active_sectors):
        return {
            "warmup_records": 100,
            "warmup_seconds": 0.1,
            "warmup_setup_seconds": 0.0,
            "records_per_second": 1.0e6,
            "records_per_second_for_tuning": 1.0e6,
            "steady_state_warmup_seconds_for_tuning": 0.1,
            "discounted_warmup_seconds_for_tuning": 0.025,
            "workers": 1,
            "avg_eval_us_per_sample_per_worker": 0.1,
            "profile": {
                "python_fraction": 0.0,
                "evaluator_fraction": 1.0,
                "havana_fraction": 0.0,
            },
        }

    monkeypatch.setattr(integrator_module, "_measure_record_throughput", fake_warmup)
    tuned_request, diagnostics = integrator_module.autotune_request_for_target_time(
        request,
        topology,
        sectors,
    )

    assert diagnostics is not None
    assert tuned_request.samples_per_iter > request.qmc_max_samples_per_iter
    assert tuned_request.qmc_max_samples_per_iter == tuned_request.samples_per_iter
    assert tuned_request.qmc_initial_samples_per_iter == tuned_request.samples_per_iter
    assert diagnostics["tuned_qmc_max_samples_per_iter"] == tuned_request.samples_per_iter


def test_target_time_tuning_respects_cbc_lattice_table_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Target-time QMC tuning should not request unavailable CBC/PT vectors."""
    request = make_request(
        integral="box",
        mode="massless",
        s12=-1.0,
        s23=-1.0,
        sampling_mode="qmc",
        qmc_refine_sectors="democratic",
        qmc_lattice_backend="cbcpt-dn1-100",
        qmc_shifts=64,
        qmc_max_samples_per_iter=4096,
        samples_per_iter=50_000,
        max_iter=3,
        min_iter=1,
        workers=1,
        target_integration_time=30.0,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)

    def fake_warmup(_request, _topology, _active_sectors):
        return {
            "warmup_records": 100,
            "warmup_seconds": 0.1,
            "warmup_setup_seconds": 0.0,
            "records_per_second": 1.0e9,
            "records_per_second_for_tuning": 1.0e9,
            "steady_state_warmup_seconds_for_tuning": 0.1,
            "discounted_warmup_seconds_for_tuning": 0.025,
            "workers": 1,
            "avg_eval_us_per_sample_per_worker": 0.1,
            "profile": {
                "python_fraction": 0.0,
                "evaluator_fraction": 1.0,
                "havana_fraction": 0.0,
            },
        }

    monkeypatch.setattr(integrator_module, "_measure_record_throughput", fake_warmup)
    tuned_request, diagnostics = integrator_module.autotune_request_for_target_time(
        request,
        topology,
        sectors,
    )

    assert diagnostics is not None
    assert tuned_request.samples_per_iter == max_lattice_point_count(backend="cbcpt-dn1-100")
    assert tuned_request.qmc_max_samples_per_iter == tuned_request.samples_per_iter
    assert diagnostics["qmc_lattice_backend_limit_applied"] is True


def test_dot_triangle_qmc_auto_defaults_match_target(tmp_path: Path) -> None:
    """Automatic QMC evaluator defaults should reproduce the pySecDec target."""
    result_path = tmp_path / "triangle_qmc.json"
    subprocess.run(
        [
            sys.executable,
            "FSD.py",
            "--run",
            "examples/runs/dot_triangle.yaml",
            "--sampling-mode",
            "qmc",
            "--qmc-shifts",
            "4",
            "--samples-per-iter",
            "64",
            "--max-iter",
            "1",
            "--batch-size",
            "64",
            "--workers",
            "1",
            "--prefactor-convention",
            "pysecdec",
            "--target",
            "examples/outputs/dot_triangle_pysecdec_target.json",
            "--no-progress",
            "--quiet-summary",
            "--json",
            "--result-path",
            str(result_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    data = json.loads(result_path.read_text())
    assert data["request"]["evaluator_compile_mode"] == "eager"
    assert data["request"]["real_evaluator"] is False
    pulls = [float(value) for value in data["aggregate_results"]["pull"]]
    assert max(abs(value) for value in pulls) < 3.0


def test_qmc_qmcpy_backend_records_lattice_metadata() -> None:
    """QMC diagnostics should expose the independent QMCPy lattice metadata."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        subtraction_backend="projector-formula",
        sampling_mode="qmc",
        qmc_shifts=2,
        qmc_korobov_alpha=3,
        samples_per_iter=1024,
        max_iter=1,
        min_iter=1,
        batch_size=2048,
        workers=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(
        topology,
        sectors,
        request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )

    result = integrate(request, topology, sectors, None)

    assert result.diagnostics["qmc_software"] == "QMCPy Lattice"
    assert result.diagnostics["qmc_lattice_backend"] == "qmcpy"
    assert result.diagnostics["qmc_lattice_order"] == "linear"
    assert result.diagnostics["qmc_lattice_points_per_shift"] == 1024
    assert result.samples == 2 * 1024 * len(sectors)


def test_qmc_cbcpt_backend_records_lattice_metadata() -> None:
    """QMC diagnostics should expose the bundled CBC table used for pySecDec checks."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        subtraction_backend="projector-formula",
        sampling_mode="qmc",
        qmc_shifts=2,
        qmc_korobov_alpha=3,
        qmc_lattice_backend="cbcpt-dn1-100",
        samples_per_iter=32,
        max_iter=1,
        min_iter=1,
        batch_size=2048,
        workers=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(
        topology,
        sectors,
        request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )

    result = integrate(request, topology, sectors, None)

    assert result.diagnostics["qmc_software"] == "bundled CBC/PT dn1 subset"
    assert result.diagnostics["qmc_lattice_backend"] == "cbcpt-dn1-100"
    assert result.diagnostics["qmc_lattice_order"] == "linear"
    assert result.diagnostics["qmc_lattice_points_per_shift"] == 32
    assert result.samples == 2 * 1021 * len(sectors)


def test_result_table_marks_missing_dot_reference_as_na(capsys: pytest.CaptureFixture[str]) -> None:
    """DOT/FSD-only output must not compute a pull against the zero placeholder."""
    request = make_request(integral="dot", prefactor_convention="sector")
    summary = {
        "validation": {
            "expected_laurent_orders": ["eps^0"],
            "benchmark_available": False,
        },
        "symanzik": {"dual_evaluator_build_seconds": 0.0},
    }
    output = make_output(
        request=request,
        raw_coeffs=[1.0 + 0.0j],
        raw_errors=[0.1 + 0.0j],
        target=None,
        samples=10,
        elapsed_seconds=0.0,
        avg_eval_us_per_sample_per_worker=0.0,
        eval_seconds=0.0,
        python_seconds=0.0,
        havana_seconds=0.0,
        python_overhead_fraction=0.0,
        summary=summary,
    )

    print_result_table(output)
    rendered = capsys.readouterr().out

    assert "N/A" in rendered
    assert "10.00σ" not in rendered
    assert "IntegratorT" in rendered
    assert "HavanaT" not in rendered


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

    try:
        benchmark = compute_benchmark_quietly(integral_request)
    except RuntimeError as exc:
        pytest.skip(f"OneLOopBridge unavailable: {exc}")
    prepare_generated_evaluators(topology, sectors, integral_request.dual_evaluator_mode)
    result = integrate(integral_request, topology, sectors, None)

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


def test_dot_box_oneloop_target_matches_pysecdec_convention() -> None:
    """The DOT box preset should use a high-precision OneLOopBridge target."""
    try:
        request = build_request(
            parse_args(
                [
                    "--run",
                    "examples/runs/dot_box.yaml",
                    "--no-progress",
                    "--quiet-summary",
                ]
            )
        )
        topology = build_topology(request)
        sectors = generate_sectors(request)
        configure_laurent_range(request, topology, sectors)
        target = resolve_target(request, topology, summary_data(request, topology, sectors, False))
    except RuntimeError as exc:
        pytest.skip(f"OneLOopBridge unavailable: {exc}")

    assert target is not None
    assert target.source == "oneloop"
    assert target.convention == "pysecdec"
    assert target.errors == [0.0 + 0.0j for _ in target.coefficients]
    assert target.coefficients[0].real == pytest.approx(4.0, abs=1.0e-14)
    assert target.coefficients[1].real == pytest.approx(-2.30886265960613, abs=1.0e-13)
    assert target.coefficients[2].real == pytest.approx(-12.4931166871698, abs=1.0e-12)


def test_dot_double_box_havana_preset_uses_safe_evaluator_mode() -> None:
    """The double-box Havana preset must avoid the known real-JIT mismatch."""
    request = build_request(
        parse_args(
            [
                "--run",
                "examples/runs/dot_double_box.yaml",
                "--sampling-mode",
                "havana",
                "--quiet-summary",
                "--no-progress",
            ]
        )
    )

    assert request.evaluator_compile_mode == "eager"
    assert request.real_evaluator is False


def test_pull_combines_target_uncertainty() -> None:
    """A noisy target file should not create an artificial many-sigma pull."""
    diff = 1.25e-11 + 0.0j
    fsd_error = 8.3e-13 + 0.0j
    target_error = 3.1e-10 + 0.0j
    fsd_only_pull = pull_value(diff, fsd_error)
    combined_pull = pull_value(diff, combine_uncorrelated_errors(fsd_error, target_error))

    assert fsd_only_pull is not None and fsd_only_pull > 10.0
    assert combined_pull is not None and combined_pull < 0.1


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


def test_dual_f_evaluator_is_cloned_without_mutating_scalar_evaluator() -> None:
    """Dualizing cached F evaluators must not mutate ordinary scalar F calls."""
    integral_request = make_request(integral="triangle", mode="massless", s=-1.0, m=0.0)
    topology = build_topology(integral_request)
    sector = generate_sectors(integral_request)[0]
    x_values = sector.map_eval_batch(np.asarray([[0.5, 0.25]], dtype=float))

    scalar_before = topology.f_values(x_values)
    taylor = topology.f_taylor_batch(sector, np.asarray([[0.0, 0.25]], dtype=float))
    scalar_after = topology.f_values(x_values)

    assert taylor.shape == (1, len(sector.dual_shape))
    assert topology.f_dual_evaluator(sector.dual_shape) is topology.f_dual_evaluator(sector.dual_shape)
    assert np.allclose(scalar_before, scalar_after)


def test_endpoint_powers_are_assembled_from_parametric_metadata() -> None:
    """Endpoint powers come from Jacobian, U/F, and topology exponents."""
    integral_request = make_request(integral="triangle", mode="massless", s=-1.0, m=0.0)
    topology = build_topology(integral_request)
    sector = generate_sectors(integral_request)[0]

    t_power = topology.endpoint_power(sector, 0)
    z_power = topology.endpoint_power(sector, 1)

    assert t_power.base == pytest.approx(-1.0)
    assert t_power.eps_coeff == pytest.approx(-2.0)
    assert z_power.base == pytest.approx(-1.0)
    assert z_power.eps_coeff == pytest.approx(-1.0)


def test_dot_file_request_reaches_topology_placeholder(tmp_path: Path) -> None:
    """DOT input requires an explicit kinematics YAML file."""
    dot_file = tmp_path / "toy.dot"
    dot_file.write_text("digraph toy { a -> b; }\n", encoding="utf-8")
    integral_request = make_request(integral="dot", dot_file=str(dot_file), mode="massless", m=0.0)

    with pytest.raises(ValueError, match="--kinematics"):
        validate_request(integral_request)


def test_dot_file_printout_has_topology_and_sector_schema(tmp_path: Path) -> None:
    """DOT mode can print the generic topology/sector schema."""
    dot_file = tmp_path / "toy.dot"
    dot_file.write_text("digraph toy { a -> b; }\n", encoding="utf-8")
    kin_file = tmp_path / "kinematics.yaml"
    kin_file.write_text("values: {}\nreplacements: {}\n", encoding="utf-8")
    integral_request = make_request(
        integral="dot",
        dot_file=str(dot_file),
        kinematics_file=str(kin_file),
        mode="massless",
        m=0.0,
    )

    printout = GammaLoopDotTopologyBuilder.from_request(integral_request).printout_placeholder()
    text = str(printout)
    data = printout.to_dict()

    assert "DOT topology printout" in text
    assert "U polynomial" in text
    assert "sector schema" in text
    assert any(row[0] == "U monomial" for row in data["sector_schema"])
    assert any(row[0] == "endpoint powers" for row in data["sector_schema"])


def test_dot_file_request_reaches_sector_placeholder(tmp_path: Path) -> None:
    """DOT-backed sector generation fails clearly when pySecDec is unavailable."""
    dot_file = tmp_path / "toy.dot"
    dot_file.write_text("digraph toy { a -> b; }\n", encoding="utf-8")
    kin_file = tmp_path / "kinematics.yaml"
    kin_file.write_text("values: {}\nreplacements: {}\n", encoding="utf-8")
    integral_request = make_request(
        integral="dot",
        dot_file=str(dot_file),
        kinematics_file=str(kin_file),
        mode="massless",
        m=0.0,
    )

    validate_request(integral_request)
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        generate_sectors(integral_request)


def test_dot_file_request_validates_file_path(tmp_path: Path) -> None:
    """DOT mode rejects missing files and non-DOT suffixes before parsing."""
    missing = make_request(integral="dot", dot_file=str(tmp_path / "missing.dot"))
    with pytest.raises(ValueError, match="does not exist"):
        validate_request(missing)

    not_dot = tmp_path / "toy.txt"
    not_dot.write_text("digraph toy { a -> b; }\n", encoding="utf-8")
    wrong_suffix = make_request(integral="dot", dot_file=str(not_dot))
    with pytest.raises(ValueError, match=".dot suffix"):
        validate_request(wrong_suffix)


def test_example_dot_parser_preserves_external_direction_and_masses() -> None:
    """The example DOT parser finds invisible half-edges and mass attributes."""
    parsed = parse_dot_file(PROJECT_ROOT / "examples/graphs/triangle.dot")

    assert parsed.graph_name == "triangle"
    assert parsed.loop_count == 1
    assert [line.mass for line in parsed.internal_lines] == ["mt", "mt", "mt"]
    assert [line.momentum for line in parsed.external_lines] == ["p0", "-p1", "-p2"]
    assert parsed.pysecdec_internal_lines()[0][0] == "mt"


def test_dot_parser_accepts_positive_integer_propagator_powers(tmp_path: Path) -> None:
    """DOT edge ``power=``/``pow=`` attributes become propagator powers."""
    dot_file = tmp_path / "powers.dot"
    dot_file.write_text(
        """
        digraph powers {
          v1; v2; v3;
          v1 -> v2 [id=0, name="e0", mass="0", power=2];
          v2 -> v3 [id=1, name="e1", mass="0", pow=3];
        }
        """,
        encoding="utf-8",
    )

    parsed = parse_dot_file(dot_file)

    assert [line.power for line in parsed.internal_lines] == [2, 3]


@pytest.mark.parametrize("power", ["0", "-1", "1.5", "eps"])
def test_dot_parser_rejects_unsupported_propagator_powers(tmp_path: Path, power: str) -> None:
    """Only positive integer DOT propagator powers are supported."""
    dot_file = tmp_path / "bad_power.dot"
    dot_file.write_text(
        f"""
        digraph bad_power {{
          v1; v2;
          v1 -> v2 [id=0, name="e0", mass="0", power="{power}"];
        }}
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="power|non-integer"):
        parse_dot_file(dot_file)


def test_dot_powerlist_is_forwarded_to_pysecdec_graph_constructor(tmp_path: Path) -> None:
    """The DOT bridge passes parsed propagator powers as pySecDec powerlist."""
    dot_file = tmp_path / "powers.dot"
    dot_file.write_text(
        """
        digraph powers {
          v1; v2; v3;
          v1 -> v2 [id=0, name="e0", mass="0", power=2];
          v2 -> v3 [id=1, name="e1", mass="0", power=1];
        }
        """,
        encoding="utf-8",
    )
    parsed = parse_dot_file(dot_file)
    captured: dict[str, Any] = {}

    def fake_loop_integral_from_graph(*_args: Any, **kwargs: Any) -> object:
        captured["powerlist"] = kwargs.get("powerlist")
        return object()

    modules = {
        "LoopIntegralFromGraph": fake_loop_integral_from_graph,
        "LoopIntegralFromPropagators": lambda *_args, **_kwargs: object(),
    }
    kin = KinematicsDefinition(
        path=tmp_path / "kinematics.yaml",
        values={},
        replacements=[],
        replacement_expressions=[],
    )

    _make_loop_integral(parsed, kin, make_request(integral="dot"), modules)

    assert captured["powerlist"] == [2, 1]


def test_dot_parser_reads_graph_level_momentum_numerator() -> None:
    """Graph-level ``num`` attributes are preserved for pySecDec numerator mode."""
    triangle = parse_dot_file(PROJECT_ROOT / "examples/graphs/triangle_numerator.dot")
    box = parse_dot_file(PROJECT_ROOT / "examples/graphs/box_numerator.dot")

    assert triangle.numerator == "2*k(mu)*p1(mu)"
    assert triangle.graph_attributes["loop_momenta"] == "k"
    assert triangle.graph_attr_list("lorentz_indices") == ["mu"]
    assert triangle.graph_attr_list("propagators", separator=";") == [
        "k**2",
        "(k+p0)**2",
        "(k+p0-p1)**2",
    ]
    assert box.numerator == "2*k(mu)*p3(mu)"
    assert box.graph_attr_list("external_momenta") == ["p1", "p2", "p3"]


def _eval_symbolica_scalar(expr: Any, names: list[str], row: list[float]) -> complex:
    """Evaluate one Symbolica expression in tests."""
    evaluator = expr.evaluator([S(name) for name in names])
    value = evaluator.evaluate([row])[0][0]
    return complex(float(value), 0.0)


def _assert_custom_numerator_matches_pysecdec(
    *,
    li: Any,
    numerator: str,
    loop_momenta: list[str],
    external_momenta: list[str],
    kinematics: KinematicsDefinition,
    x_sample: list[float],
    eps_samples: list[float],
) -> None:
    """Compare the FSD reducer with pySecDec's preliminary numerator."""
    reduced = reduce_dot_product_numerator(
        numerator=numerator,
        loop_momenta=loop_momenta,
        external_momenta=external_momenta,
        li=li,
        kinematics=kinematics,
    )
    x_names = [str(symbol) for symbol in list(li.integration_variables)]
    u_value = _eval_symbolica_scalar(_polynomial_to_expr(li.U), x_names, x_sample)
    f_value = _eval_symbolica_scalar(_polynomial_to_expr(li.F), x_names, x_sample)
    pysecdec_expr = E(_polynomial_to_symbolica_text(li.numerator))
    pysecdec_names = [*x_names, "U", "F", "eps"]
    coefficient_values = [
        _eval_symbolica_scalar(expr, x_names, x_sample)
        for expr in reduced.eps_coefficients
    ]
    for eps_value in eps_samples:
        fsd_value = sum(
            coefficient * (eps_value**order)
            for order, coefficient in enumerate(coefficient_values)
        )
        pysecdec_value = _eval_symbolica_scalar(
            pysecdec_expr,
            pysecdec_names,
            [*x_sample, u_value.real, f_value.real, eps_value],
        )
        assert fsd_value.real == pytest.approx(pysecdec_value.real, rel=1.0e-10, abs=1.0e-10)
        assert fsd_value.imag == pytest.approx(pysecdec_value.imag, abs=1.0e-14)


@pytest.mark.parametrize(
    ("graph_name", "kinematics_name", "x_sample"),
    [
        pytest.param("triangle_numerator", "triangle_kinematics", [0.19, 0.31, 0.50], id="one-loop-rank-1-triangle"),
        pytest.param("box_rank2_numerator", "box_kinematics", [0.11, 0.23, 0.29, 0.37], id="one-loop-rank-2-box"),
    ],
)
def test_symbolica_numerator_reducer_matches_pysecdec_dot_examples(
    graph_name: str,
    kinematics_name: str,
    x_sample: list[float],
) -> None:
    """The FSD reducer reproduces pySecDec preliminary numerator polynomials."""
    modules = require_pysecdec()
    graph = parse_dot_file(PROJECT_ROOT / f"examples/graphs/{graph_name}.dot")
    kinematics = load_kinematics(PROJECT_ROOT / f"examples/graphs/{kinematics_name}.yaml")
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / f"examples/graphs/{graph_name}.dot"),
        kinematics_file=str(PROJECT_ROOT / f"examples/graphs/{kinematics_name}.yaml"),
    )
    li = _make_loop_integral(graph, kinematics, request, modules)

    _assert_custom_numerator_matches_pysecdec(
        li=li,
        numerator=graph.numerator or "1",
        loop_momenta=graph.graph_attr_list("loop_momenta"),
        external_momenta=graph.graph_attr_list("external_momenta"),
        kinematics=kinematics,
        x_sample=x_sample,
        eps_samples=[0.0, 0.17, -0.23],
    )


def test_symbolica_numerator_reducer_matches_pysecdec_two_loop_dot_products() -> None:
    """The custom reducer handles multiple loop momenta and external dot products."""
    modules = require_pysecdec()
    kinematics = KinematicsDefinition(
        path=PROJECT_ROOT / "inline-two-loop-kinematics.yaml",
        values={},
        replacements=[("p1*p1", -1.0)],
        replacement_expressions=[("p1*p1", "-1")],
    )
    li = modules["LoopIntegralFromPropagators"](
        ["k1**2", "k2**2", "(k1+k2+p1)**2"],
        loop_momenta=["k1", "k2"],
        external_momenta=["p1"],
        Lorentz_indices=["mu", "nu"],
        numerator="k1(mu)*k2(mu)+2*k1(nu)*p1(nu)",
        replacement_rules=kinematics.pysecdec_replacement_rules(),
        Feynman_parameters="x",
        regulators=["eps"],
        dimensionality="4-2*eps",
    )

    _assert_custom_numerator_matches_pysecdec(
        li=li,
        numerator="k1(mu)*k2(mu)+2*k1(nu)*p1(nu)",
        loop_momenta=["k1", "k2"],
        external_momenta=["p1"],
        kinematics=kinematics,
        x_sample=[0.17, 0.29, 0.54],
        eps_samples=[0.0, 0.11, -0.19],
    )


def test_symbolica_numerator_reducer_matches_pysecdec_odd_rank() -> None:
    """Odd tensor rank contractions are reduced by the same generic pairing code."""
    modules = require_pysecdec()
    kinematics = KinematicsDefinition(
        path=PROJECT_ROOT / "inline-rank-three-kinematics.yaml",
        values={},
        replacements=[("p1*p1", 0.0), ("p2*p2", 0.0), ("p1*p2", -0.5)],
        replacement_expressions=[("p1*p1", "0"), ("p2*p2", "0"), ("p1*p2", "-1/2")],
    )
    numerator = "k(mu)*p1(mu)*k(nu)*p1(nu)*k(rho)*p2(rho)"
    li = modules["LoopIntegralFromPropagators"](
        ["k**2", "(k+p1)**2", "(k+p1+p2)**2"],
        loop_momenta=["k"],
        external_momenta=["p1", "p2"],
        Lorentz_indices=["mu", "nu", "rho"],
        numerator=numerator,
        replacement_rules=kinematics.pysecdec_replacement_rules(),
        Feynman_parameters="x",
        regulators=["eps"],
        dimensionality="4-2*eps",
    )

    _assert_custom_numerator_matches_pysecdec(
        li=li,
        numerator=numerator,
        loop_momenta=["k"],
        external_momenta=["p1", "p2"],
        kinematics=kinematics,
        x_sample=[0.21, 0.34, 0.45],
        eps_samples=[0.0, 0.13, -0.07],
    )


def test_symbolica_numerator_parser_rejects_open_lorentz_indices() -> None:
    """Every Lorentz index must be paired in every expanded term."""
    with pytest.raises(ValueError, match="exactly twice"):
        parse_dot_product_numerator(
            "k(mu)*p1(nu)",
            loop_momenta=["k"],
            external_momenta=["p1"],
            values={},
        )


def test_numerator_epsilon_polynomial_stability_path_matches_ordinary_path() -> None:
    """Forced high-precision rescue keeps all numerator epsilon coefficients."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box_rank2_numerator.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        prefactor_convention="sector",
        samples_per_iter=128,
        batch_size=64,
    )
    try:
        validate_request(request)
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")
    prepare_generated_evaluators(
        topology,
        sectors,
        mode=request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )
    sector = next(candidate for candidate in sectors if candidate.singular_axes)
    point = np.full((1, sector.integration_dim), 0.37)
    ordinary = SectorProcessor(
        topology,
        stability_threshold=0.0,
        subtraction_backend=request.subtraction_backend,
    )
    precise = SectorProcessor(
        topology,
        stability_threshold=1.0,
        high_precision_stability_threshold=1.0,
        high_precision_stability_precision=80,
        subtraction_backend=request.subtraction_backend,
    )

    ordinary_coeffs, _ordinary_training, _ordinary_timing = ordinary.evaluate_batch(sector, point)
    precise_coeffs, _precise_training, precise_timing = precise.evaluate_batch(sector, point)

    assert precise_timing.high_precision_samples == 1
    assert np.max(np.abs(ordinary_coeffs - precise_coeffs)) < 1.0e-9


def test_pysecdec_numerator_reducer_backend_matches_symbolica_backend() -> None:
    """The optional pySecDec reducer feeds the same eps-polynomial sector data."""
    common = dict(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box_rank2_numerator.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        prefactor_convention="sector",
        samples_per_iter=128,
        batch_size=64,
    )
    symbolica_request = make_request(**common, numerator_reducer="symbolica")
    pysecdec_request = make_request(**common, numerator_reducer="pysecdec")
    try:
        validate_request(symbolica_request)
        symbolica_topology = build_topology(symbolica_request)
        symbolica_sectors = generate_sectors(symbolica_request)
        validate_request(pysecdec_request)
        pysecdec_topology = build_topology(pysecdec_request)
        pysecdec_sectors = generate_sectors(pysecdec_request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")

    assert len(symbolica_sectors) == len(pysecdec_sectors)
    point = np.full((1, symbolica_sectors[0].integration_dim), 0.37)
    for symbolica_sector, pysecdec_sector in zip(symbolica_sectors[:4], pysecdec_sectors[:4]):
        symbolica_values = symbolica_sector.numerator_eps_eval_batch(
            point,
            symbolica_topology.coefficient_count,
        )
        pysecdec_values = pysecdec_sector.numerator_eps_eval_batch(
            point,
            pysecdec_topology.coefficient_count,
        )
        assert len(symbolica_values) == len(pysecdec_values)
        for left, right in zip(symbolica_values, pysecdec_values):
            assert np.max(np.abs(left - right)) < 1.0e-12


def test_symbolica_global_prefactor_series_handles_gamma_pole() -> None:
    """DOT global prefactors can now be Laurent-expanded with Symbolica."""
    min_order, coeffs = _prefactor_series("gamma(eps)", 2)

    assert min_order == -1
    assert coeffs[0] == pytest.approx(1.0)
    assert coeffs[1] == pytest.approx(-0.5772156649015329)
    assert coeffs[2] == pytest.approx(0.9890559953279725)


def test_symbolica_global_prefactor_series_handles_shifted_gamma() -> None:
    """The pySecDec box prefactor ``gamma(2+eps)`` needs correct constants."""
    min_order, coeffs = _prefactor_series("gamma(2+eps)", 2)

    euler_gamma = 0.5772156649015329
    expected_eps_2 = 0.5 * euler_gamma * euler_gamma - euler_gamma + math.pi**2 / 12.0
    assert min_order == 0
    assert coeffs[0] == pytest.approx(1.0)
    assert coeffs[1] == pytest.approx(1.0 - euler_gamma)
    assert coeffs[2] == pytest.approx(expected_eps_2)


def test_symbolica_global_prefactor_series_handles_signed_scaled_gamma() -> None:
    """Signed pySecDec Gamma factors must stay on the analytic path."""
    min_order, coeffs = _prefactor_series("-gamma(3+2*eps)", 4)

    expected = [
        -2.0,
        -3.6911373403938686,
        -4.9858599838053861,
        -4.5999533513648978,
        -3.681199052065825,
    ]
    assert min_order == 0
    assert coeffs == pytest.approx(expected)


def test_laurent_global_prefactor_convolution_extends_displayed_pole_range() -> None:
    """A singular DOT prefactor shifts raw sector orders into displayed orders."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box_high_rank_numerator.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        prefactor_convention="pysecdec",
        max_eps_order=0,
        dot_global_prefactor_min_order=-1,
        dot_global_prefactor_coeffs=(1.0 + 0.0j, 10.0 + 0.0j, 100.0 + 0.0j, 1000.0 + 0.0j),
        dot_sector_laurent_min_order=-2,
        dot_sector_laurent_max_order=1,
    )

    coeffs, errors = apply_global_convention(
        [2.0 + 0.0j, 3.0 + 0.0j, 5.0 + 0.0j, 7.0 + 0.0j],
        [0.2 + 0.0j, 0.3 + 0.0j, 0.5 + 0.0j, 0.7 + 0.0j],
        request,
    )

    assert coeffs == pytest.approx([
        2.0 + 0.0j,
        23.0 + 0.0j,
        235.0 + 0.0j,
        2357.0 + 0.0j,
    ])
    assert errors == pytest.approx([
        0.2 + 0.0j,
        2.3 + 0.0j,
        23.5 + 0.0j,
        235.7 + 0.0j,
    ])


def test_dot_gamma_prefactor_extends_raw_sector_epsilon_depth() -> None:
    """DOT generation requests extra raw sector coefficients for Gamma poles."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box_high_rank_numerator.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        mode="massless",
        m=0.0,
        prefactor_convention="pysecdec",
        max_eps_order=0,
        sector_method="iterative",
        numerator_reducer="symbolica",
    )

    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)

    assert topology.global_prefactor_min_order == -1
    assert topology.laurent_min_order == -2
    assert topology.laurent_max_order == 1
    assert topology.expected_laurent_orders == ["eps^-2", "eps^-1", "eps^0", "eps^1"]


def test_output_labels_follow_displayed_prefactor_convention() -> None:
    """Output labels describe the prefactor-convolved coefficients."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box_high_rank_numerator.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        prefactor_convention="pysecdec",
        max_eps_order=0,
        dot_global_prefactor_min_order=-1,
        dot_sector_laurent_min_order=-2,
    )
    output = make_output(
        request=request,
        raw_coeffs=[0.0 + 0.0j for _ in range(4)],
        raw_errors=[0.0 + 0.0j for _ in range(4)],
        target=None,
        samples=0,
        elapsed_seconds=0.0,
        avg_eval_us_per_sample_per_worker=0.0,
        eval_seconds=0.0,
        python_seconds=0.0,
        havana_seconds=0.0,
        python_overhead_fraction=0.0,
        summary={"validation": {"expected_laurent_orders": ["eps^-2", "eps^-1", "eps^0", "eps^1"]}},
    )

    assert output["laurent_labels"] == ["eps^-3", "eps^-2", "eps^-1", "eps^0"]


def test_kinematics_yaml_uses_symbolica_expression_evaluation() -> None:
    """YAML values and replacements are evaluated without SymPy/SciPy."""
    kin = load_kinematics(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml")

    assert kin.values["s12"] == pytest.approx(-1.0)
    assert kin.values["mt"] == pytest.approx(0.0)
    replacements = dict(kin.replacements)
    assert replacements["p1*p2"] == pytest.approx(-0.5)
    assert replacements["p1*p3"] == pytest.approx(1.0)


def test_qmc_compare_helper_resolves_kinematics_from_run_yaml() -> None:
    """The comparison helper should not default box runs to triangle kinematics."""
    resolved = _path_from_run_file(
        PROJECT_ROOT / "examples/runs/dot_box.yaml",
        "kinematics",
        PROJECT_ROOT / "examples/graphs/triangle_kinematics.yaml",
    )

    assert resolved == (PROJECT_ROOT / "examples/graphs/box_kinematics.yaml").resolve()


def test_dot_triangle_pysecdec_generation_matches_expected_endpoint_metadata() -> None:
    """When pySecDec is installed, DOT triangle generation recovers endpoint sectors."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/triangle.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/triangle_kinematics.yaml"),
        mode="massless",
        m=0.0,
        prefactor_convention="pysecdec",
    )
    try:
        validate_request(request)
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")

    assert topology.u_value([1.0, 2.0, 3.0]) == pytest.approx(6.0)
    assert len(sectors) == 2
    assert sorted(len(sector.singular_axes) for sector in sectors) == [1, 2]
    assert sorted(tuple(sector.f_monomial_powers) for sector in sectors) == [
        (0, 1),
        (1, 1),
    ]


def test_dot_box_pysecdec_generation_matches_expected_polynomial() -> None:
    """DOT box generation reproduces the massless box polynomial and sectors."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        mode="massless",
        m=0.0,
        prefactor_convention="pysecdec",
    )
    try:
        validate_request(request)
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")

    sample = [0.1, 0.2, 0.3, 0.4]
    assert topology.u_value(sample) == pytest.approx(1.0)
    assert topology.f_value(sample) == pytest.approx(0.11)
    assert len(sectors) == 3
    assert sorted(set(len(sector.singular_axes) for sector in sectors)) == [1, 2]


@pytest.mark.parametrize(
    ("name", "expected_loop_count", "expected_sector_count", "expected_dimension"),
    [
        pytest.param("kite_2loop", 2, 4, 4, id="kite-2-loop"),
        pytest.param("self_energy_3loop", 3, 74, 6, id="self-energy-3-loop"),
        pytest.param("three_point_2loop", 2, 6, 4, id="three-point-2-loop"),
        pytest.param("three_point_3loop", 3, 74, 6, id="three-point-3-loop"),
        pytest.param("three_point_2loop_6line", 2, 6, 5, id="three-point-2-loop-6-line"),
        pytest.param("three_point_3loop_8line", 3, 117, 7, id="three-point-3-loop-8-line"),
    ],
)
def test_dot_multiloop_two_and_three_point_examples_generate_finite_sector_sets(
    name: str,
    expected_loop_count: int,
    expected_sector_count: int,
    expected_dimension: int,
) -> None:
    """The smaller multi-loop DOT examples generate finite FSD sectors."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / f"examples/graphs/{name}.dot"),
        kinematics_file=str(PROJECT_ROOT / f"examples/graphs/{name}_kinematics.yaml"),
        mode="massive",
        m=1.0,
        prefactor_convention="pysecdec",
    )
    try:
        validate_request(request)
        topology = build_topology(request)
        sectors = generate_sectors(request)
    except RuntimeError as exc:
        pytest.skip(f"pySecDec unavailable: {exc}")

    assert topology.parametric_representation.loop_count == expected_loop_count
    assert len(sectors) == expected_sector_count
    assert {sector.integration_dim for sector in sectors} == {expected_dimension}
    assert all(len(sector.singular_axes) == 0 for sector in sectors)
    assert topology.expected_laurent_orders == [
        f"eps^{order}" for order in range(-2 * expected_loop_count, 1)
    ]

    parsed = parse_dot_file(PROJECT_ROOT / f"examples/graphs/{name}.dot")
    expected_external_count = 3 if name.startswith("three_point") else 2
    assert len(parsed.external_lines) == expected_external_count


@pytest.mark.skipif(
    os.environ.get("FSD_RUN_PYSECDEC_COMPARE") != "1",
    reason="set FSD_RUN_PYSECDEC_COMPARE=1 to run slow generated-pySecDec comparisons",
)
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("kite_2loop", id="kite-2-loop"),
        pytest.param("self_energy_3loop", id="self-energy-3-loop"),
    ],
)
def test_optional_multiloop_fsd_low_stat_compare_to_pysecdec(name: str) -> None:
    """Optional slow comparison of simple multi-loop FSD and pySecDec outputs."""
    from dot_topology import get_dot_bundle
    from pysecdec_bridge import run_pysecdec_package

    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / f"examples/graphs/{name}.dot"),
        kinematics_file=str(PROJECT_ROOT / f"examples/graphs/{name}_kinematics.yaml"),
        mode="massive",
        m=1.0,
        prefactor_convention="pysecdec",
        samples_per_iter=512,
        batch_size=256,
        pysecdec_maxeval=512,
        pysecdec_epsrel=5.0e-1,
        pysecdec_workdir=f".pysecdec_build_{name}_pytest",
    )
    validate_request(request)
    topology = build_topology(request)
    sectors = generate_sectors(request)
    prepare_generated_evaluators(topology, sectors, request.dual_evaluator_mode)

    pysecdec = run_pysecdec_package(get_dot_bundle(request), request)
    result = integrate(request, topology, sectors, None)
    raw_coeffs, raw_errors = apply_global_convention(
        result.raw_sector_coeffs,
        result.raw_sector_errors,
        request,
    )

    assert pysecdec.coeffs
    assert abs(raw_coeffs[-1] - pysecdec.coeffs[-1]) <= 8.0 * (
        abs(raw_errors[-1]) + abs(pysecdec.errors[-1]) + 1.0e-12
    )


def test_regular_monomial_factors_are_kept_in_finite_sectors() -> None:
    """Finite sectors still include positive endpoint monomial powers in g_s."""
    topology = TopologyDefinition(
        family="toy-regular-monomial",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("1"),
        u_power_base=0.0,
        f_power_base=0.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=0.0,
        expected_laurent_orders=["eps^0"],
        convention_note="toy finite sector",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(0.0, 0.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="none",
            convention_description="toy",
        ),
    )
    sector = SectorDefinition(
        name="toy-positive-jacobian-power",
        integration_dim=1,
        variable_names=["y0"],
        map_exprs=[E("y0")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[0],
        jacobian_monomial_powers=[1],
        singular_axes=[],
        subtraction_type="finite",
        description="finite sector with a positive extracted Jacobian power",
    )

    coeffs, training = SectorProcessor(topology).evaluate(sector, [0.3])

    assert coeffs[0].real == pytest.approx(0.3)
    assert coeffs[0].imag == pytest.approx(0.0)
    assert training == pytest.approx(0.3)


@pytest.mark.parametrize("axis_count", [1, 2, 3, 4])
def test_recursive_log_subtraction_for_constant_residual(axis_count: int) -> None:
    """Recursive subtraction returns the expected pure pole for prod y^(-1-eps)."""
    variables = [f"y{i}" for i in range(axis_count)]
    f_expr = "*".join(f"x{i}" for i in range(axis_count))
    topology = TopologyDefinition(
        family=f"toy-{axis_count}",
        x_names=[f"x{i}" for i in range(axis_count)],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E(f_expr),
        u_power_base=0.0,
        f_power_base=1.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0"],
        convention_note="toy recursive subtraction test",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=tuple(1.0 for _ in range(axis_count)),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-1.0, -1.0),
            parameter_weight_powers=tuple(0.0 for _ in range(axis_count)),
            prefactor_description="none",
            convention_description="toy",
        ),
    )
    topology.set_laurent_range(-axis_count, 0)
    sector = SectorDefinition(
        name=f"toy-{axis_count}",
        integration_dim=axis_count,
        variable_names=variables,
        map_exprs=[E(name) for name in variables],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[1 for _ in range(axis_count)],
        jacobian_monomial_powers=[0 for _ in range(axis_count)],
        singular_axes=list(range(axis_count)),
        subtraction_type="recursive endpoint subtraction",
        description="toy recursive subtraction sector",
    )
    prepare_generated_evaluators(topology, [sector])
    coeffs, _training = SectorProcessor(topology).evaluate(sector, [0.37 for _ in range(axis_count)])

    assert coeffs[0].real == pytest.approx((-1.0) ** axis_count)
    assert coeffs[0].imag == pytest.approx(0.0)
    for coeff in coeffs[1:]:
        assert abs(coeff) < 1.0e-12


def _multiply_series(
    left: dict[int, complex],
    right: dict[int, complex],
    min_order: int,
    max_order: int,
) -> dict[int, complex]:
    """Multiply Laurent series and retain the requested order window."""
    out: dict[int, complex] = {}
    for left_order, left_value in left.items():
        for right_order, right_value in right.items():
            order = left_order + right_order
            if min_order <= order <= max_order:
                out[order] = out.get(order, 0.0 + 0.0j) + left_value * right_value
    return out


def _denominator_series(
    beta: int,
    monomial_order: int,
    eps_coeff: float,
    min_order: int,
    max_order: int,
) -> dict[int, complex]:
    """Expand int_0^1 dy y^(beta+n+c eps) as a Laurent series."""
    offset = beta + monomial_order + 1
    if offset == 0:
        return {-1: 1.0 / eps_coeff}
    return {
        order: ((-eps_coeff / offset) ** order) / offset
        for order in range(max_order - min_order + 1)
    }


def test_three_axis_taylor_subtraction_matches_exact_polynomial_integral() -> None:
    """Three-axis sectors with y^-2 endpoints use the correct Taylor data."""
    topology = TopologyDefinition(
        family="toy-three-axis-first-taylor",
        x_names=["x0", "x1", "x2"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("x0^2*x1*x2^2"),
        u_power_base=0.0,
        f_power_base=1.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0"],
        convention_note="toy three-axis subtraction test",
        parametric_representation=ParametricRepresentation(
            loop_count=2,
            propagator_powers=(1.0, 1.0, 1.0),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-1.0, -1.0),
            parameter_weight_powers=(0.0, 0.0, 0.0),
            prefactor_description="none",
            convention_description="toy",
        ),
    )
    topology.set_laurent_range(-3, 0)
    sector = SectorDefinition(
        name="toy-three-axis",
        integration_dim=3,
        variable_names=["y0", "y1", "y2"],
        map_exprs=[E("y0"), E("y1"), E("y2")],
        regular_jacobian_expr=E("(1+2*y0)*(3+5*y2)"),
        f_monomial_powers=[2, 1, 2],
        jacobian_monomial_powers=[0, 0, 0],
        singular_axes=[0, 1, 2],
        subtraction_type="recursive endpoint subtraction",
        description="toy sector with two first-order Taylor subtractions",
        endpoint_taylor_orders=[1, 0, 1],
    )
    prepare_generated_evaluators(topology, [sector])

    coeffs, _training = SectorProcessor(topology).evaluate(sector, [0.23, 0.41, 0.67])

    min_order = -3
    max_order = 0
    work_max_order = max_order - min_order + 1
    exact: dict[int, complex] = {order: 0.0 + 0.0j for order in range(min_order, max_order + 1)}
    polynomial_terms = {
        (0, 0, 0): 3.0,
        (1, 0, 0): 6.0,
        (0, 0, 1): 5.0,
        (1, 0, 1): 10.0,
    }
    endpoint_data = [(-2, -2.0), (-1, -1.0), (-2, -2.0)]
    for powers, coefficient in polynomial_terms.items():
        series = {0: coefficient + 0.0j}
        for beta, eps_coeff, monomial_order in zip(
            [item[0] for item in endpoint_data],
            [item[1] for item in endpoint_data],
            powers,
        ):
            series = _multiply_series(
                series,
                _denominator_series(beta, monomial_order, eps_coeff, min_order, work_max_order),
                min_order,
                work_max_order,
            )
        for order, value in series.items():
            if min_order <= order <= max_order:
                exact[order] += value

    assert coeffs == pytest.approx([exact[order] for order in range(min_order, max_order + 1)])


def test_taylor_subtraction_differentiates_regular_polynomial_without_monomial() -> None:
    """Higher endpoints must Taylor-expand regular U/F factors too."""
    topology = TopologyDefinition(
        family="toy-regular-u-derivative",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1+x0"),
        f_expr=E("x0^2"),
        u_power_base=1.0,
        f_power_base=1.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0"],
        convention_note="toy regular U derivative subtraction test",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(1.0, 0.0),
            f_exponent=EpsilonExpansion(-1.0, -1.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="none",
            convention_description="toy",
        ),
    )
    topology.set_laurent_range(-1, 0)
    sector = SectorDefinition(
        name="toy-regular-u-derivative",
        integration_dim=1,
        variable_names=["y0"],
        map_exprs=[E("y0")],
        regular_jacobian_expr=E("1"),
        f_monomial_powers=[2],
        u_monomial_powers=[0],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="recursive endpoint subtraction",
        description="toy sector with y^-2 endpoint and regular U variation",
        endpoint_taylor_orders=[1],
    )
    prepare_generated_evaluators(topology, [sector])

    coeffs, _training = SectorProcessor(topology).evaluate(sector, [0.31])

    assert coeffs == pytest.approx([-0.5 + 0.0j, -1.0 + 0.0j])


def test_complex_prec_dualized_evaluators_support_constant_and_nonconstant() -> None:
    """Symbolica complex multiprecision works for dualized coefficient evaluators."""
    x = S("x")
    constant = E("1").evaluator([x])
    constant.dualize([[0], [1], [2]])
    polynomial = E("x^2").evaluator([x])
    polynomial.dualize([[0], [1], [2]])

    row = [(1.0e-10, 0.0), (1.0, 0.0), (0.0, 0.0)]
    constant_values = constant.evaluate_complex_with_prec(row, 32)
    polynomial_values = polynomial.evaluate_complex_with_prec(row, 32)

    assert complex(float(constant_values[0][0]), float(constant_values[0][1])) == pytest.approx(1.0 + 0.0j)
    assert complex(float(constant_values[1][0]), float(constant_values[1][1])) == pytest.approx(0.0 + 0.0j)
    assert complex(float(polynomial_values[0][0]), float(polynomial_values[0][1])) == pytest.approx(1.0e-20 + 0.0j)
    assert complex(float(polynomial_values[1][0]), float(polynomial_values[1][1])) == pytest.approx(2.0e-10 + 0.0j)
    assert complex(float(polynomial_values[2][0]), float(polynomial_values[2][1])) == pytest.approx(1.0 + 0.0j)


def test_generated_formula_precision_stabilizes_yminus2_cancellation() -> None:
    """The pregenerated formula cures y^-2 Taylor-remainder cancellation near zero."""
    topology = TopologyDefinition(
        family="toy-yminus2-formula-stability",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("x0^2"),
        u_power_base=0.0,
        f_power_base=1.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0"],
        convention_note="toy generated formula cancellation stability test",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-1.0, -1.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="none",
            convention_description="toy",
        ),
    )
    topology.set_laurent_range(-1, 0)
    sector = SectorDefinition(
        name="toy-yminus2-formula-stability",
        integration_dim=1,
        variable_names=["y0"],
        map_exprs=[E("y0")],
        regular_jacobian_expr=E("1+2*y0+3*y0^2"),
        f_monomial_powers=[2],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="recursive endpoint subtraction",
        description="toy sector with a known quadratic Taylor tail",
        endpoint_taylor_orders=[1],
    )
    prepare_generated_evaluators(topology, [sector])

    processor = SectorProcessor(
        topology,
        stability_threshold=1.0e-8,
        high_precision_stability_threshold=1.0e-8,
        high_precision_stability_precision=1000,
    )
    coeffs, _training, timing = processor.evaluate_batch(sector, np.asarray([[1.0e-10]], dtype=float))

    assert timing.precision_counts["high_precision"] == 1
    assert coeffs[0, 0] == pytest.approx(-1.0 + 0.0j)
    assert coeffs[0, 1] == pytest.approx(2.0 + 0.0j, abs=1.0e-8)


def test_endpoint_projector_precision_path_avoids_double_input_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Projector rescue rows assemble U/F/J Taylor data in multiprecision."""
    topology = TopologyDefinition(
        family="toy-yminus2-projector-stability",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("x0^2"),
        u_power_base=0.0,
        f_power_base=1.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0"],
        convention_note="toy endpoint-projector precision test",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-1.0, -1.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="none",
            convention_description="toy",
        ),
    )
    topology.set_laurent_range(-1, 0)
    sector = SectorDefinition(
        name="toy-yminus2-projector-stability",
        integration_dim=1,
        variable_names=["y0"],
        map_exprs=[E("y0")],
        regular_jacobian_expr=E("1+2*y0+3*y0^2"),
        f_monomial_powers=[2],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="endpoint projector subtraction",
        description="toy sector with a known quadratic Taylor tail",
        endpoint_taylor_orders=[1],
    )
    prepare_generated_evaluators(topology, [sector], subtraction_backend="projector-formula")

    def forbidden_matrix(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("projector precision path used the double input matrix")

    monkeypatch.setattr(SectorProcessor, "_endpoint_projector_input_matrix", forbidden_matrix)
    processor = SectorProcessor(
        topology,
        stability_threshold=1.0e-8,
        high_precision_stability_threshold=1.0e-8,
        high_precision_stability_precision=100,
        subtraction_backend="projector-formula",
    )
    coeffs, _training, timing = processor.evaluate_batch(sector, np.asarray([[1.0e-10]], dtype=float))

    assert timing.precision_counts["high_precision"] == 1
    assert coeffs[0, 0] == pytest.approx(-1.0 + 0.0j)
    assert coeffs[0, 1] == pytest.approx(2.0 + 0.0j, abs=1.0e-8)


def test_generated_formula_uses_integer_coordinate_powers() -> None:
    """Endpoint powers in generated formulas must never be encoded as floats."""
    topology = TopologyDefinition(
        family="toy-integer-formula-powers",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1"),
        f_expr=E("x0^2"),
        u_power_base=0.0,
        f_power_base=1.0,
        eps_log_u_coeff=0.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0"],
        convention_note="toy generated formula integer-power test",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(0.0, 0.0),
            f_exponent=EpsilonExpansion(-1.0, -1.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="none",
            convention_description="toy",
        ),
    )
    topology.set_laurent_range(-1, 0)
    sector = SectorDefinition(
        name="toy-integer-formula-powers",
        integration_dim=1,
        variable_names=["y0"],
        map_exprs=[E("y0")],
        regular_jacobian_expr=E("1+2*y0+3*y0^2"),
        f_monomial_powers=[2],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="recursive endpoint subtraction",
        description="toy sector checking generated coordinate powers",
        endpoint_taylor_orders=[1],
    )
    prepare_generated_evaluators(topology, [sector])

    formula_text = "\n".join(str(expr) for expr in topology.subtraction_formula_for(sector).output_expressions)

    assert re.search(r"sf_y\d+\^-?\d+\.\d", formula_text) is None
    assert "sf_y0^2" in formula_text


def test_symbolica_formula_generator_matches_legacy_builder_on_toy_sector() -> None:
    """The Symbolica-rule formula agrees with the legacy Python-built formula."""
    topology = TopologyDefinition(
        family="toy-formula-compare",
        x_names=["x0"],
        parameter_names=[],
        parameter_values=[],
        u_expr=E("1+x0"),
        f_expr=E("x0^2"),
        u_power_base=1.0,
        f_power_base=1.0,
        eps_log_u_coeff=1.0,
        eps_log_f_coeff=-1.0,
        expected_laurent_orders=["eps^0"],
        convention_note="toy formula comparison",
        parametric_representation=ParametricRepresentation(
            loop_count=1,
            propagator_powers=(1.0,),
            dimension=EpsilonExpansion(4.0, -2.0),
            gamma_argument=EpsilonExpansion(0.0, 0.0),
            u_exponent=EpsilonExpansion(1.0, 1.0),
            f_exponent=EpsilonExpansion(-1.0, -1.0),
            parameter_weight_powers=(0.0,),
            prefactor_description="none",
            convention_description="toy",
        ),
    )
    topology.set_laurent_range(-1, 0)
    sector = SectorDefinition(
        name="toy-formula-compare",
        integration_dim=1,
        variable_names=["y0"],
        map_exprs=[E("y0")],
        regular_jacobian_expr=E("1+2*y0+3*y0^2"),
        f_monomial_powers=[2],
        u_monomial_powers=[0],
        jacobian_monomial_powers=[0],
        singular_axes=[0],
        subtraction_type="formula comparison",
        description="toy sector comparing old and new formula builders",
        endpoint_taylor_orders=[1],
    )
    prepare_generated_evaluators(topology, [sector])
    signature = topology.subtraction_formula_signature(sector)
    new_formula = topology.subtraction_formula_for(sector)
    legacy_formula = build_subtraction_formula_legacy(topology, sector, signature)
    processor = SectorProcessor(topology)
    rows = np.asarray([[0.19], [0.53]], dtype=float)
    timing = HotPathTiming()
    inputs = processor._subtraction_formula_input_matrix(sector, rows, new_formula, timing)

    new_values = new_formula.evaluate_complex_batch(inputs)
    legacy_values = legacy_formula.evaluate_complex_batch(inputs)

    assert np.allclose(new_values, legacy_values, rtol=1.0e-11, atol=1.0e-11)


def test_symbolica_formula_generator_matches_legacy_builder_on_dot_box_sector() -> None:
    """The new formula generator matches the legacy builder on pySecDec sector data."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        mode="massless",
        m=0.0,
        sector_method="iterative",
        dual_evaluator_mode="symbolic-derivatives",
        prefactor_convention="pysecdec",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(topology, sectors, request.dual_evaluator_mode)
    sector = next(sector for sector in sectors if sector.singular_axes)
    signature = topology.subtraction_formula_signature(sector)
    new_formula = topology.subtraction_formula_for(sector)
    legacy_formula = build_subtraction_formula_legacy(topology, sector, signature)
    processor = SectorProcessor(topology)
    rows = np.full((2, sector.integration_dim), 0.37, dtype=float)
    rows[1, :] = np.linspace(0.21, 0.79, sector.integration_dim)
    inputs = processor._subtraction_formula_input_matrix(sector, rows, new_formula, HotPathTiming())

    new_values = new_formula.evaluate_complex_batch(inputs)
    legacy_values = legacy_formula.evaluate_complex_batch(inputs)

    assert np.allclose(new_values, legacy_values, rtol=1.0e-10, atol=1.0e-10)


def test_endpoint_projector_backend_matches_recursive_for_triangle_and_box() -> None:
    """The lower-signature projector reproduces recursive subtraction."""
    cases = [
        make_request(integral="triangle", mode="massless", s=-1.0, m=0.0),
        make_request(integral="box", mode="massless", s12=-1.0, s23=-1.0, m=0.0),
    ]
    for request in cases:
        topology = build_topology(request)
        sectors = generate_sectors(request)
        configure_laurent_range(request, topology, sectors)
        prepare_generated_evaluators(
            topology,
            sectors,
            request.dual_evaluator_mode,
            subtraction_backend="projector-formula",
        )
        recursive = SectorProcessor(
            topology,
            subtraction_backend="recursive",
            stability_threshold=0.0,
        )
        projector = SectorProcessor(
            topology,
            subtraction_backend="projector-formula",
            stability_threshold=0.0,
        )
        for sector in sectors:
            if not sector.singular_axes:
                continue
            rows = np.full((3, sector.integration_dim), 0.37, dtype=float)
            rows[1, :] = np.linspace(0.21, 0.79, sector.integration_dim)
            rows[2, :] = np.linspace(0.79, 0.21, sector.integration_dim)
            recursive_values = recursive.evaluate_batch(sector, rows)[0]
            projector_values = projector.evaluate_batch(sector, rows)[0]
            assert np.allclose(projector_values, recursive_values, rtol=1.0e-11, atol=1.0e-11)


def test_endpoint_projector_backend_matches_recursive_for_dot_double_box_sector() -> None:
    """A multi-axis DOT sector works with the endpoint-only projector cache."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/double_box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/double_box_kinematics.yaml"),
        mode="massless",
        m=0.0,
        sector_method="iterative",
        dual_evaluator_mode="symbolic-derivatives",
        prefactor_convention="pysecdec",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    sector = next(sector for sector in sectors if len(sector.singular_axes) >= 4)
    prepare_generated_evaluators(
        topology,
        [sector],
        request.dual_evaluator_mode,
        subtraction_backend="projector-formula",
    )
    rows = np.full((2, sector.integration_dim), 0.37, dtype=float)
    rows[1, :] = np.linspace(0.21, 0.79, sector.integration_dim)
    recursive_values = SectorProcessor(
        topology,
        subtraction_backend="recursive",
        stability_threshold=0.0,
    ).evaluate_batch(sector, rows)[0]
    projector_values = SectorProcessor(
        topology,
        subtraction_backend="projector-formula",
        stability_threshold=0.0,
    ).evaluate_batch(sector, rows)[0]

    assert len(topology._endpoint_projector_formulas) == 1
    assert topology._regular_taylor_formulas
    assert topology._regular_taylor_dual_signatures
    assert np.allclose(projector_values, recursive_values, rtol=1.0e-10, atol=1.0e-10)


def test_endpoint_projector_signature_is_lower_than_full_dot_box_signature() -> None:
    """Endpoint projector signatures intentionally ignore sector-specific U/F/J data."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/graphs/box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/graphs/box_kinematics.yaml"),
        mode="massless",
        m=0.0,
        sector_method="iterative",
        dual_evaluator_mode="symbolic-derivatives",
        prefactor_convention="pysecdec",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    full_signatures = {
        topology.subtraction_formula_signature(sector)
        for sector in sectors
        if sector.singular_axes
    }
    endpoint_signatures = {
        topology.endpoint_projector_signature(sector)
        for sector in sectors
        if sector.singular_axes
    }

    assert len(endpoint_signatures) < len(full_signatures)


def test_integration_does_not_generate_subtraction_formulas_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared formula evaluators are sufficient once integration starts."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        samples_per_iter=64,
        batch_size=32,
        workers=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(topology, sectors, request.dual_evaluator_mode)

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("subtraction formula generation happened during integration")

    monkeypatch.setattr("integrand.build_subtraction_formula", forbidden)
    result = integrate(request, topology, sectors, None)

    assert result.samples == request.samples_per_iter


def test_projector_formula_backend_does_not_generate_formulas_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared endpoint projectors are sufficient once integration starts."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        samples_per_iter=64,
        batch_size=32,
        workers=1,
        subtraction_backend="projector-formula",
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(
        topology,
        sectors,
        request.dual_evaluator_mode,
        subtraction_backend=request.subtraction_backend,
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("endpoint projector generation happened during integration")

    monkeypatch.setattr("integrand.build_endpoint_projector_formula", forbidden)
    monkeypatch.setattr("integrand.build_regular_taylor_formula", forbidden)
    result = integrate(request, topology, sectors, None)

    assert result.samples == request.samples_per_iter


def test_keyboard_interrupt_returns_partial_integration_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Ctrl+C after some batches returns the samples accumulated so far."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        samples_per_iter=128,
        batch_size=64,
        max_iter=1,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(topology, sectors, request.dual_evaluator_mode)
    original = integrator_module._evaluate_records
    calls = {"count": 0}

    def interrupt_after_first_batch(*args: Any, **kwargs: Any) -> Any:
        calls["count"] += 1
        if calls["count"] == 2:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(integrator_module, "_evaluate_records", interrupt_after_first_batch)

    result = integrate(request, topology, sectors, None)

    assert result.interrupted is True
    assert result.samples == request.batch_size


def test_numeric_target_parsing_zero_fills_and_rejects_odd_pairs() -> None:
    """Numeric targets are deepest-pole ordered and zero-filled to the range."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        target_args=("1.0", "0.0", "2.0", "0.5"),
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)

    target = resolve_target(request, topology, {})

    assert target is not None
    assert target.source == "numeric"
    assert target.coefficients == [1.0 + 0.0j, 2.0 + 0.5j, 0.0 + 0.0j]

    odd = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        target_args=("1.0",),
    )
    odd_topology = build_topology(odd)
    with pytest.raises(ValueError, match="re/im pairs"):
        resolve_target(odd, odd_topology, {})


def test_target_alignment_keeps_deepest_pole_first() -> None:
    """Targets are deepest-pole ordered even when max epsilon order truncates."""
    full_target = [complex(order, 0.0) for order in (-4, -3, -2, -1, 0)]

    assert _align_coefficients(full_target, 3) == [
        -4.0 + 0.0j,
        -3.0 + 0.0j,
        -2.0 + 0.0j,
    ]
    assert _align_coefficients(full_target[:2], 4) == [
        -4.0 + 0.0j,
        -3.0 + 0.0j,
        0.0 + 0.0j,
        0.0 + 0.0j,
    ]


def test_max_eps_order_truncates_builtin_range_and_training_index() -> None:
    """The requested highest epsilon order controls coefficient count."""
    request = make_request(
        integral="triangle",
        mode="massless",
        s=-1.0,
        m=0.0,
        max_eps_order=-1,
        samples_per_iter=64,
        batch_size=32,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(topology, sectors, request.dual_evaluator_mode)

    assert topology.expected_laurent_orders == ["eps^-2", "eps^-1"]
    assert topology.training_index == 1

    result = integrate(request, topology, sectors, None)

    assert len(result.raw_sector_coeffs) == 2
    assert result.samples == request.samples_per_iter


def test_per_sector_results_are_additive_contributions() -> None:
    """Per-sector stored means sum back to the aggregate coefficient vector."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        samples_per_iter=128,
        batch_size=64,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(topology, sectors, request.dual_evaluator_mode)

    result = integrate(request, topology, sectors, None)
    summed = [
        sum(sector.raw_sector_coeffs[index] for sector in result.per_sector)
        for index in range(len(result.raw_sector_coeffs))
    ]

    assert summed == pytest.approx(result.raw_sector_coeffs)
    assert sum(sector.samples for sector in result.per_sector) == result.samples


def test_sector_selection_uses_canonical_sector_ids() -> None:
    """A filtered run samples only requested canonical sector ids."""
    request = make_request(
        integral="triangle",
        mode="massive",
        s=1.0,
        m=1.0,
        samples_per_iter=64,
        batch_size=32,
        sectors=(1,),
        max_weight_precision_xi=0.0,
    )
    topology = build_topology(request)
    sectors = generate_sectors(request)
    configure_laurent_range(request, topology, sectors)
    prepare_generated_evaluators(topology, sectors, request.dual_evaluator_mode)

    result = integrate(request, topology, sectors, None)

    assert result.per_sector[0].samples == 0
    assert result.per_sector[1].samples == result.samples
    assert result.per_sector[1].precision_counts["ordinary"] == result.samples
    assert result.raw_sector_coeffs == pytest.approx(result.per_sector[1].raw_sector_coeffs)


def test_result_json_roundtrip_target_and_viewer(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stored results can be loaded as targets and viewed with sector sorting."""
    request = make_request(
        integral="dot",
        prefactor_convention="sector",
        result_path=str(tmp_path / "result.json"),
    )
    summary = {
        "validation": {
            "expected_laurent_orders": ["eps^-1", "eps^0"],
            "benchmark_available": True,
        },
        "symanzik": {
            "dual_evaluator_build_seconds": 0.0,
            "endpoint_projector_formula_count": 2,
            "regular_taylor_formula_count": 3,
            "regular_taylor_formulas_from_curated_cache": 1,
            "regular_taylor_formulas_skipped": 4,
        },
    }
    output = make_output(
        request=request,
        raw_coeffs=[1.0 + 0.0j, 2.0 + 0.0j],
        raw_errors=[0.1 + 0.0j, 0.2 + 0.0j],
        target=None,
        samples=10,
        elapsed_seconds=0.0,
        avg_eval_us_per_sample_per_worker=0.0,
        eval_seconds=0.0,
        python_seconds=0.0,
        havana_seconds=0.0,
        python_overhead_fraction=0.0,
        summary=summary,
        sector_results=[
            {
                "sector_id": 1,
                "name": "B",
                "samples": 4,
                "display": {
                    "coefficients": [0.25 + 0.0j, 0.5 + 0.0j],
                    "errors": [0.04 + 0.0j, 0.05 + 0.0j],
                },
                "sort_keys": {"abs_central": 0.5, "abs_error": 0.05},
            },
            {
                "sector_id": 0,
                "name": "A",
                "samples": 6,
                "display": {
                    "coefficients": [0.75 + 0.0j, 1.5 + 0.0j],
                    "errors": [0.01 + 0.0j, 0.02 + 0.0j],
                },
                "sort_keys": {"abs_central": 1.5, "abs_error": 0.02},
            },
        ],
    )
    path = write_result_json(output, request.result_path)

    target = target_from_result_file(path, "sector")
    assert target.source == "file"
    assert target.coefficients == [1.0 + 0.0j, 2.0 + 0.0j]

    with pytest.raises(ValueError, match="does not match"):
        target_from_result_file(path, "pysecdec")

    print_saved_results(path, sort_mode="abs-error")
    rendered = capsys.readouterr().out
    plain = re.sub(r"\x1b\[[0-9;]*m", "", rendered)
    assert "FSD result file" in rendered
    assert "coefficients" in rendered
    assert "curated regular Taylor assets" in rendered
    assert "regular Taylor formulas skipped" in rendered
    assert plain.index(" B    |") < plain.index(" A    |")
    assert "-1:" in plain
    assert "eps^-1:" not in plain


def test_result_file_target_prefers_stored_pysecdec_target(tmp_path: Path) -> None:
    """A comparison file reuses stored pySecDec coefficients as the target."""
    request = make_request(
        integral="dot",
        prefactor_convention="pysecdec",
        result_path=str(tmp_path / "result.json"),
    )
    summary = {
        "validation": {
            "expected_laurent_orders": ["eps^-1", "eps^0"],
            "benchmark_available": True,
        },
        "symanzik": {"dual_evaluator_build_seconds": 0.0},
    }
    output = make_output(
        request=request,
        raw_coeffs=[1.0 + 0.0j, 2.0 + 0.0j],
        raw_errors=[0.1 + 0.0j, 0.2 + 0.0j],
        target=TargetDefinition(
            source="pysecdec",
            convention="pysecdec",
            coefficients=[10.0 + 0.0j, 20.0 + 0.0j],
            errors=[0.3 + 0.0j, 0.4 + 0.0j],
            metadata={},
        ),
        samples=10,
        elapsed_seconds=0.0,
        avg_eval_us_per_sample_per_worker=0.0,
        eval_seconds=0.0,
        python_seconds=0.0,
        havana_seconds=0.0,
        python_overhead_fraction=0.0,
        summary=summary,
    )
    path = write_result_json(output, request.result_path)

    target = target_from_result_file(path, "pysecdec")

    assert target.source == "file:pysecdec"
    assert target.coefficients == [10.0 + 0.0j, 20.0 + 0.0j]
    assert target.errors == [0.3 + 0.0j, 0.4 + 0.0j]


def test_result_file_target_prefers_stored_numeric_target(tmp_path: Path) -> None:
    """Stored explicit numeric targets remain targets on a later run."""
    request = make_request(
        integral="dot",
        prefactor_convention="pysecdec",
        result_path=str(tmp_path / "result.json"),
    )
    summary = {
        "validation": {
            "expected_laurent_orders": ["eps^-1", "eps^0"],
            "benchmark_available": True,
        },
        "symanzik": {"dual_evaluator_build_seconds": 0.0},
    }
    output = make_output(
        request=request,
        raw_coeffs=[1.0 + 0.0j, 2.0 + 0.0j],
        raw_errors=[0.1 + 0.0j, 0.2 + 0.0j],
        target=TargetDefinition(
            source="numeric",
            convention="pysecdec",
            coefficients=[3.0 + 0.0j, 4.0 + 0.0j],
            errors=[0.0 + 0.0j, 0.0 + 0.0j],
            metadata={},
        ),
        samples=10,
        elapsed_seconds=0.0,
        avg_eval_us_per_sample_per_worker=0.0,
        eval_seconds=0.0,
        python_seconds=0.0,
        havana_seconds=0.0,
        python_overhead_fraction=0.0,
        summary=summary,
    )
    path = write_result_json(output, request.result_path)

    target = target_from_result_file(path, "pysecdec")

    assert target.source == "file:numeric"
    assert target.coefficients == [3.0 + 0.0j, 4.0 + 0.0j]


def test_result_file_target_reads_pysecdec_only_output(tmp_path: Path) -> None:
    """A pySecDec-only result file can be reused as a target."""
    path = tmp_path / "result.json"
    write_result_json(
        {
            "schema_version": 1,
            "prefactor_convention": "pysecdec",
            "pysecdec": {
                "coeffs": [1.5 + 0.0j, 2.5 + 0.0j],
                "errors": [0.05 + 0.0j, 0.06 + 0.0j],
                "orders": [-1, 0],
            },
        },
        path,
    )

    target = target_from_result_file(path, "pysecdec")

    assert target.source == "file:pysecdec"
    assert target.coefficients == [1.5 + 0.0j, 2.5 + 0.0j]
    assert target.errors == [0.05 + 0.0j, 0.06 + 0.0j]
    assert target.metadata["orders"] == [-1, 0]


def test_pysecdec_targets_align_by_laurent_order_when_poles_cancel() -> None:
    """Reduced pySecDec Laurent ranges must not shift comparison rows."""
    aligned = _align_coefficients_by_order(
        [1.5 + 0.0j, 2.5 + 0.0j],
        [-1, 0],
        [-2, -1, 0],
    )

    assert aligned == [0.0 + 0.0j, 1.5 + 0.0j, 2.5 + 0.0j]


def test_show_results_bypasses_generation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """--show-results reads a file and exits before validation/integration setup."""
    result_path = tmp_path / "result.json"
    result_path.write_text('{"schema_version": 1}', encoding="utf-8")
    calls: list[str] = []

    def fake_print(path: str, sort_mode: str) -> None:
        calls.append(f"{path}:{sort_mode}")

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("show-results should not validate or generate")

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--show-results", str(result_path)])
    monkeypatch.setattr("FSD.print_saved_results", fake_print)
    monkeypatch.setattr("FSD.validate_request", forbidden)

    assert main() == 0
    assert calls == [f"{result_path}:index"]
