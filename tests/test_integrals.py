"""Pytest smoke coverage for the supported FSD integral modes."""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from FSD import (
    _align_coefficients,
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
from dot_parser import parse_dot_file
from dot_topology import GammaLoopDotTopologyBuilder
from formatting import (
    apply_global_convention,
    make_output,
    print_result_table,
    pull_value,
    selected_prefactor_values,
)
from integrand import (
    SectorProcessor,
    TopologyDefinition,
    build_subtraction_formula_legacy,
    build_topology,
)
import integrator as integrator_module
from integrator import integrate
from kinematics import load_kinematics
from result_io import print_saved_results, target_from_result_file, write_result_json
from sectors_generator import SectorDefinition, generate_sectors
from symbolica import E
from symbolica import S


def make_request(**overrides: Any) -> IntegralRequest:
    """Build a deterministic, low-statistics integration request for tests."""
    data = {
        "integral": "triangle",
        "dot_file": None,
        "kinematics_file": None,
        "graph_name": None,
        "sector_method": "iterative",
        "normaliz_executable": None,
        "dot_engine": "fsd",
        "sectors": None,
        "pysecdec_workdir": ".pysecdec_build",
        "pysecdec_epsrel": 1.0e-2,
        "pysecdec_maxeval": 1000,
        "keep_pysecdec_workdir": False,
        "progress_value_order": "eps^0",
        "max_eps_order": 0,
        "target_args": None,
        "show_results": None,
        "sort_sector_results": "index",
        "result_path": str(Path.cwd() / "result.json"),
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
        "target_rel_accuracy": None,
        "min_error": 0.0,
        "bins": 32,
        "workers": 1,
        "jit_compile_evaluators": False,
        "dual_evaluator_mode": "pregenerate",
        "subtraction_backend": "formula",
        "stability_threshold": 1.0e-8,
        "high_precision_stability_threshold": 1.0e-12,
        "stability_precision": 32,
        "high_precision_stability_precision": 1000,
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
    elif subtraction_backend != "recursive":
        raise ValueError(f"unsupported test subtraction backend {subtraction_backend!r}")


def test_dual_evaluator_cli_modes_are_mutually_exclusive_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """CLI flags select exactly one dual evaluator generation mode."""
    monkeypatch.setattr(sys, "argv", ["FSD.py"])
    default_request = build_request(parse_args())
    assert default_request.dual_evaluator_mode == "pregenerate"
    assert default_request.sector_method == "iterative"

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--lazy-dual-evaluators-generation"])
    assert build_request(parse_args()).dual_evaluator_mode == "lazy"

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--pregenerate-single-overall-dual-evaluator"])
    assert build_request(parse_args()).dual_evaluator_mode == "single-overall"

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--symbolic-derivatives"])
    assert build_request(parse_args()).dual_evaluator_mode == "symbolic-derivatives"

    monkeypatch.setattr(sys, "argv", ["FSD.py", "--sectors", "3", "7"])
    assert build_request(parse_args()).sectors == (3, 7)

    custom_result = PROJECT_ROOT / "custom-result.json"
    monkeypatch.setattr(sys, "argv", ["FSD.py", "--result-path", str(custom_result)])
    assert build_request(parse_args()).result_path == str(custom_result)

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
        dot_file=str(PROJECT_ROOT / "examples/dot/triangle.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/dot/triangle_kinematics.yaml"),
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


def test_dot_fsd_integration_does_not_reenter_pysecdec_after_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prepared DOT topology/sectors are sufficient for FSD integration."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/dot/triangle.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/dot/triangle_kinematics.yaml"),
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

    result = integrate(request, topology, sectors, None)

    assert result.samples == request.samples_per_iter


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
    parsed = parse_dot_file(PROJECT_ROOT / "examples/dot/triangle.dot")

    assert parsed.graph_name == "triangle"
    assert parsed.loop_count == 1
    assert [line.mass for line in parsed.internal_lines] == ["mt", "mt", "mt"]
    assert [line.momentum for line in parsed.external_lines] == ["p0", "-p1", "-p2"]
    assert parsed.pysecdec_internal_lines()[0][0] == "mt"


def test_kinematics_yaml_uses_symbolica_expression_evaluation() -> None:
    """YAML values and replacements are evaluated without SymPy/SciPy."""
    kin = load_kinematics(PROJECT_ROOT / "examples/dot/box_kinematics.yaml")

    assert kin.values["s12"] == pytest.approx(-1.0)
    assert kin.values["mt"] == pytest.approx(0.0)
    replacements = dict(kin.replacements)
    assert replacements["p1*p2"] == pytest.approx(-0.5)
    assert replacements["p1*p3"] == pytest.approx(1.0)


def test_dot_triangle_pysecdec_generation_matches_expected_endpoint_metadata() -> None:
    """When pySecDec is installed, DOT triangle generation recovers endpoint sectors."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/dot/triangle.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/dot/triangle_kinematics.yaml"),
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
    assert len(sectors) == 3
    assert sorted(len(sector.singular_axes) for sector in sectors) == [1, 1, 2]
    assert sorted(tuple(sector.f_monomial_powers) for sector in sectors) == [
        (0, 1),
        (1, 0),
        (1, 1),
    ]


def test_dot_box_pysecdec_generation_matches_expected_polynomial() -> None:
    """DOT box generation reproduces the massless box polynomial and sectors."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/dot/box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/dot/box_kinematics.yaml"),
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
    assert len(sectors) == 12
    assert sorted(set(len(sector.singular_axes) for sector in sectors)) == [1, 2]


@pytest.mark.parametrize(
    ("name", "expected_loop_count", "expected_sector_count", "expected_dimension"),
    [
        pytest.param("kite_2loop", 2, 16, 4, id="kite-2-loop"),
        pytest.param("self_energy_3loop", 3, 117, 6, id="self-energy-3-loop"),
        pytest.param("three_point_2loop", 2, 16, 4, id="three-point-2-loop"),
        pytest.param("three_point_3loop", 3, 117, 6, id="three-point-3-loop"),
        pytest.param("three_point_2loop_6line", 2, 22, 5, id="three-point-2-loop-6-line"),
        pytest.param("three_point_3loop_8line", 3, 162, 7, id="three-point-3-loop-8-line"),
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
        dot_file=str(PROJECT_ROOT / f"examples/dot/{name}.dot"),
        kinematics_file=str(PROJECT_ROOT / f"examples/dot/{name}_kinematics.yaml"),
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

    parsed = parse_dot_file(PROJECT_ROOT / f"examples/dot/{name}.dot")
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
        dot_file=str(PROJECT_ROOT / f"examples/dot/{name}.dot"),
        kinematics_file=str(PROJECT_ROOT / f"examples/dot/{name}_kinematics.yaml"),
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
        dot_file=str(PROJECT_ROOT / "examples/dot/box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/dot/box_kinematics.yaml"),
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
        dot_file=str(PROJECT_ROOT / "examples/dot/double_box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/dot/double_box_kinematics.yaml"),
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
    assert np.allclose(projector_values, recursive_values, rtol=1.0e-10, atol=1.0e-10)


def test_endpoint_projector_signature_is_lower_than_full_dot_box_signature() -> None:
    """Endpoint projector signatures intentionally ignore sector-specific U/F/J data."""
    request = make_request(
        integral="dot",
        dot_file=str(PROJECT_ROOT / "examples/dot/box.dot"),
        kinematics_file=str(PROJECT_ROOT / "examples/dot/box_kinematics.yaml"),
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
        "symanzik": {"dual_evaluator_build_seconds": 0.0},
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
            },
        },
        path,
    )

    target = target_from_result_file(path, "pysecdec")

    assert target.source == "file:pysecdec"
    assert target.coefficients == [1.5 + 0.0j, 2.5 + 0.0j]
    assert target.errors == [0.05 + 0.0j, 0.06 + 0.0j]


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
