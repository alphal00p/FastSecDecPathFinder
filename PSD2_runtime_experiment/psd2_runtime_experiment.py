#!/usr/bin/env python3
"""PSD2 runtime experiment for the three-loop triple-box sector.

This script compares two deliberately different implementations of one fixed
sector, PSD2:

* ``fsd-style``: load the prepared FSD bundle and evaluate sector PSD2 through
  the normal black-box SectorProcessor path.
* ``fused``: explicitly substitute the PSD2 sector map into U and F, build the
  IBP-lowered endpoint subtraction as Symbolica expressions, and compile one
  fused evaluator per Laurent coefficient.

The fused path intentionally violates the FSD black-box design boundary.  Its
purpose is to estimate the runtime ceiling of a pySecDec-style generated
integrand for this sector.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from definitions import EpsilonExpansion, HotPathTiming, ParametricRepresentation
from integrand import (
    CHAIN_RULE_MAX_DERIVATIVE_DEGREE_1_TO_3_LOOPS,
    EndpointProjectorFormulaDefinition,
    SectorProcessor,
    TopologyDefinition,
    _ancestor_closed_multi_set,
    _dense_total_degree_multi_indices,
    _expr_series_add,
    _expr_series_coefficient,
    _expr_series_constant,
    _expr_series_log,
    _expr_series_mul,
    _expr_series_mul_allowed,
    _expr_series_pow_real,
    _expr_series_scale,
    _merge_multi_shapes,
    _multi_indices,
    _series_add,
    _series_coefficient,
    _series_constant,
    _series_filter_allowed,
    _series_mul_allowed,
    _series_pow_real_and_log_allowed,
    _series_scale,
    _zero_multi,
    build_endpoint_projector_formula_symbolica,
)
from prepared_bundle import load_prepared_bundle
from sectors_generator import SectorDefinition
from symbolica import E, Evaluator, Expression, Replacement, S


@dataclass
class FusedBuildResult:
    """Artifacts produced by the explicit PSD2 fused construction."""

    expressions: list[Any]
    evaluators: list[Any]
    generation_seconds: float
    evaluator_build_seconds: float
    expression_build_seconds: float
    expression_bytes: int
    evaluator_bytes: int
    artifact_dir: Path
    coefficient_count: int
    laurent_orders: list[int]
    evaluator_laurent_orders: list[int]


@dataclass
class SourceFusedBuildResult:
    """Assembler evaluator fed by black-box regular Taylor coefficients."""

    evaluator: Any
    output_expressions: list[Any]
    input_names: list[str]
    coefficient_keys: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]]
    laurent_orders: list[int]
    expression_build_seconds: float
    evaluator_build_seconds: float
    expression_bytes: int
    evaluator_bytes: int


@dataclass
class SourceCoefficientBuildResult:
    """One-shot evaluator for all PSD2 regular Taylor source coefficients."""

    evaluator: Any
    coefficient_keys: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]]
    expression_build_seconds: float
    evaluator_build_seconds: float
    expression_bytes: int
    evaluator_bytes: int


@dataclass
class DerivativeFusedBuildResult:
    """Two-call generic split: U/F derivatives first, universal-style assembler second."""

    source_evaluator: Any
    assembler_evaluator: Any
    derivative_slots: list[tuple[tuple[int, ...], tuple[int, ...], str, tuple[int, ...]]]
    groups: list[tuple[tuple[int, ...], tuple[int, ...]]]
    laurent_orders: list[int]
    source_expression_build_seconds: float
    source_evaluator_build_seconds: float
    assembler_expression_build_seconds: float
    assembler_evaluator_build_seconds: float
    source_expression_bytes: int
    source_evaluator_bytes: int
    assembler_expression_bytes: int
    assembler_evaluator_bytes: int


@dataclass
class DualEnvelopeSourceBuildResult:
    """Black-box dual-source context feeding the source-fused assembler."""

    topology: TopologyDefinition
    sector: SectorDefinition
    processor: SectorProcessor
    source_fused: SourceFusedBuildResult
    envelope_shape: list[tuple[int, ...]]
    grouped_pairs: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[tuple[tuple[int, ...], int], ...],
    ]
    scalar_evaluator_build_seconds: float
    envelope_shape_build_seconds: float
    dual_evaluator_build_seconds: float
    sector_evaluator_build_seconds: float


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the standalone experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=EXPERIMENT_DIR / "inputs" / "psd2_sector.json",
        help="PSD2 JSON expression input.",
    )
    parser.add_argument(
        "--prepared-bundle",
        type=Path,
        default=PROJECT_ROOT / "examples" / "outputs" / "prepared_triple_box_dual_stream_probe",
        help="Prepared FSD bundle used for the reference FSD-style timing.",
    )
    parser.add_argument("--sector-id", type=int, default=2)
    parser.add_argument("--points", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--fsd-max-eps-order",
        type=int,
        help=(
            "Restrict the prepared-bundle FSD reference to eps^(-6)..eps^N "
            "before evaluating the sector.  Use -2 to compare only the first "
            "five PSD2 coefficients."
        ),
    )
    parser.add_argument(
        "--sample",
        nargs=9,
        type=float,
        help="Fixed PSD2 sector point. When supplied it is reused for every repeat.",
    )
    parser.add_argument("--skip-fsd", action="store_true")
    parser.add_argument("--skip-fused", action="store_true")
    parser.add_argument(
        "--run-source-fused",
        action="store_true",
        help=(
            "Run the intermediate black-box experiment: acquire regular Taylor "
            "coefficients as in FSD, then assemble all IBP child projectors with "
            "one Symbolica evaluator."
        ),
    )
    parser.add_argument(
        "--source-fused-max-eps-order",
        type=int,
        default=-2,
        help="Maximum epsilon order for the source-coefficient fused assembler.",
    )
    parser.add_argument(
        "--run-two-stage-fused",
        action="store_true",
        help=(
            "Run the two-evaluator PSD2 experiment: one evaluator computes all "
            "regular source coefficients and one evaluator assembles the final "
            "Laurent integrand."
        ),
    )
    parser.add_argument(
        "--two-stage-max-eps-order",
        type=int,
        default=-2,
        help="Maximum epsilon order for the two-stage fused PSD2 experiment.",
    )
    parser.add_argument(
        "--run-derivative-fused",
        action="store_true",
        help=(
            "Run the generic split experiment: one evaluator computes stacked "
            "x-space U/F derivatives, another evaluator performs chain/source/"
            "projector assembly into Laurent coefficients."
        ),
    )
    parser.add_argument(
        "--run-dual-envelope-source",
        action="store_true",
        help=(
            "Run the FSD-generalizable two-call experiment: one envelope "
            "dualized U/F black-box pass acquires all regular Taylor sources, "
            "then one Symbolica assembler emits the Laurent coefficients."
        ),
    )
    parser.add_argument(
        "--dual-envelope-max-eps-order",
        type=int,
        default=-2,
        help="Maximum epsilon order for the dual-envelope source experiment.",
    )
    parser.add_argument(
        "--derivative-fused-max-eps-order",
        type=int,
        default=-2,
        help="Maximum epsilon order for the derivative-fused PSD2 experiment.",
    )
    parser.add_argument(
        "--derivative-source",
        choices=["symbolic", "dual"],
        default="symbolic",
        help="Derivative evaluator construction used by --run-derivative-fused.",
    )
    parser.add_argument(
        "--derivative-fused-max-groups",
        type=int,
        help="Diagnostic limit on endpoint groups included in the derivative-fused assembler.",
    )
    parser.add_argument(
        "--derivative-fused-regular-method",
        choices=["series", "symbolic-diff"],
        default="symbolic-diff",
        help=(
            "How evaluator B extracts regular Taylor coefficients from U/F "
            "residual series in the derivative-fused PSD2 experiment."
        ),
    )
    parser.add_argument(
        "--fused-max-terms",
        type=int,
        default=54,
        help="Maximum number of IBP terms to fuse. PSD2 has 54; smaller values are diagnostic only.",
    )
    parser.add_argument(
        "--fused-max-build-seconds",
        type=float,
        default=900.0,
        help="Soft wall-time guard checked between fused IBP terms.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=EXPERIMENT_DIR / "artifacts",
        help="Directory where fused expression/evaluator artifacts are written.",
    )
    parser.add_argument(
        "--load-fused-expressions",
        action="store_true",
        help="Load previously saved fused expressions from --artifact-dir instead of rebuilding them.",
    )
    parser.add_argument(
        "--load-evaluators-from",
        "--load_evaluators_from",
        type=Path,
        help=(
            "Load PSD2 source/assembler evaluator artifacts from this directory. "
            "This is intended for repeated precision/runtime probes."
        ),
    )
    parser.add_argument(
        "--save-evaluators-to",
        "--save_evaluators_to",
        type=Path,
        help=(
            "Save PSD2 source/assembler evaluator artifacts to this directory "
            "after they are built."
        ),
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=EXPERIMENT_DIR / "results.json",
        help="Machine-readable result summary.",
    )
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Build/evaluate fused expressions but do not write expression/evaluator artifacts.",
    )
    parser.add_argument(
        "--jit-compile",
        action="store_true",
        help="Pass jit_compile=True to fused Symbolica evaluator construction.",
    )
    parser.add_argument(
        "--skip-fused-evaluator-build",
        action="store_true",
        help="Build and save fused expressions but do not lower them to Symbolica evaluators.",
    )
    parser.add_argument(
        "--fused-evaluator-orders",
        nargs="*",
        type=int,
        help="Only build evaluators for these epsilon orders, for example -6 -5 0.",
    )
    parser.add_argument("--evaluator-iterations", type=int, default=1)
    parser.add_argument("--evaluator-cpe-iterations", type=int)
    parser.add_argument("--evaluator-n-cores", type=int, default=4)
    parser.add_argument("--evaluator-verbose", action="store_true")
    parser.add_argument("--evaluator-direct-translation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--evaluator-jit-direct-translation", action="store_true")
    parser.add_argument("--evaluator-max-horner-vars", type=int, default=500)
    parser.add_argument("--evaluator-max-common-pair-cache-entries", type=int, default=1_000_000)
    parser.add_argument("--evaluator-max-common-pair-distance", type=int, default=100)
    return parser.parse_args()


def load_psd2_input(path: Path) -> dict[str, Any]:
    """Load the hard-coded PSD2 expression input."""
    return json.loads(path.read_text(encoding="utf-8"))


def complex_json(value: complex) -> dict[str, float]:
    """Return a JSON-friendly complex number."""
    z = complex(value)
    return {"re": float(z.real), "im": float(z.imag)}


def expression_power(expr: Any, power: int) -> Any:
    """Raise a Symbolica expression to an integer power without float exponents."""
    p = int(power)
    if p == 0:
        return E("1")
    if p > 0:
        return expr ** p
    return E("1") / (expr ** abs(p))


def monomial_expr(variable_names: list[str], powers: list[int | float]) -> Any:
    """Build a monomial expression from integer powers."""
    out = E("1")
    for name, power in zip(variable_names, powers):
        rounded = round(float(power))
        if abs(float(power) - rounded) > 1.0e-12:
            raise ValueError(f"PSD2 experiment only supports integer monomial powers, got {power!r}")
        out = out * expression_power(S(name), int(rounded))
    return out


def replace_many(expr: Any, replacements: list[tuple[str, Any]]) -> Any:
    """Apply simultaneous Symbolica replacements."""
    return expr.replace_multiple([Replacement(S(name), rhs) for name, rhs in replacements])


def as_expression(expr: Any) -> Any:
    """Return a Symbolica expression object from an expression or stored string."""
    if hasattr(expr, "replace_multiple"):
        return expr
    return E(str(expr))


def instantiate_topology_and_sector(
    data: dict[str, Any],
    *,
    skip_evaluator_build: bool = True,
) -> tuple[TopologyDefinition, SectorDefinition]:
    """Construct lightweight FSD objects from the PSD2 JSON."""
    top = data["topology"]
    sec = data["sector"]
    propagator_powers = tuple(1.0 for _ in top["x_names"])
    parametric = ParametricRepresentation(
        loop_count=3,
        propagator_powers=propagator_powers,
        dimension=EpsilonExpansion(4.0, -2.0),
        gamma_argument=EpsilonExpansion(4.0, 3.0),
        u_exponent=EpsilonExpansion(float(top["u_power_base"]), float(top["eps_log_u_coeff"])),
        f_exponent=EpsilonExpansion(-float(top["f_power_base"]), float(top["eps_log_f_coeff"])),
        parameter_weight_powers=tuple(0.0 for _ in top["x_names"]),
        prefactor_description="PSD2 runtime experiment",
        convention_description="sector convention",
    )
    topology = TopologyDefinition(
        family=str(top["family"]),
        x_names=[str(name) for name in top["x_names"]],
        parameter_names=[str(name) for name in top["parameter_names"]],
        parameter_values=[float(value) for value in top["parameter_values"]],
        u_expr=E(str(top["u_expr"])),
        f_expr=E(str(top["f_expr"])),
        u_power_base=float(top["u_power_base"]),
        f_power_base=float(top["f_power_base"]),
        eps_log_u_coeff=float(top["eps_log_u_coeff"]),
        eps_log_f_coeff=float(top["eps_log_f_coeff"]),
        expected_laurent_orders=[f"eps^{int(order)}" for order in top["laurent_orders"]],
        convention_note="PSD2 runtime experiment sector convention",
        jit_compile_evaluators=False,
        dual_evaluator_mode="pregenerate",
        ibp_reduce_to_log_endpoint=True,
        skip_evaluator_build=bool(skip_evaluator_build),
        parametric_representation=parametric,
    )
    topology._regular_taylor_signature_version = 3
    topology.direct_projector_cache_term_threshold = 0
    sector = SectorDefinition(
        name=str(sec["name"]),
        integration_dim=int(sec["integration_dim"]),
        variable_names=[str(name) for name in sec["variable_names"]],
        map_exprs=[E(str(expr)) for expr in sec["map_exprs"]],
        regular_jacobian_expr=E(str(sec["regular_jacobian_expr"])),
        f_monomial_powers=[int(value) for value in sec["f_monomial_powers"]],
        u_monomial_powers=[int(value) for value in sec["u_monomial_powers"]],
        jacobian_monomial_powers=[int(value) for value in sec["jacobian_monomial_powers"]],
        measure_monomial_powers=[float(value) for value in sec["measure_monomial_powers"]],
        numerator_monomial_powers=[float(value) for value in sec["numerator_monomial_powers"]],
        endpoint_taylor_orders=[int(value) for value in sec["endpoint_taylor_orders"]],
        singular_axes=[int(value) for value in sec["singular_axes"]],
        subtraction_type=str(sec["subtraction_type"]),
        description=str(sec.get("description", "hard-coded PSD2")),
    )
    return topology, sector


def build_sector_regular_expression(
    data: dict[str, Any],
    topology: TopologyDefinition,
    sector: SectorDefinition,
) -> tuple[list[Any], dict[str, Any]]:
    """Build explicit PSD2 regular g_s coefficient expressions."""
    top = data["topology"]
    x_names = [str(name) for name in top["x_names"]]
    y_names = sector.variable_names
    map_exprs = [E(str(expr)) for expr in data["sector"]["map_exprs"]]
    replacements = list(zip(x_names, map_exprs))
    u_sub = replace_many(E(str(top["u_expr"])), replacements).expand()
    f_sub = replace_many(E(str(top["f_expr"])), replacements).expand()
    u_monomial = monomial_expr(y_names, sector.u_monomial_powers)
    f_monomial = monomial_expr(y_names, sector.f_monomial_powers)
    u_residual = (u_sub / u_monomial).expand()
    f_residual = (f_sub / f_monomial).expand()
    jacobian = E(str(data["sector"]["regular_jacobian_expr"]))

    monomial_pref = E("1")
    monomial_log = E("0")
    singular = set(sector.singular_axes)
    for axis, name in enumerate(y_names):
        if axis in singular:
            continue
        endpoint_power = topology.endpoint_power(sector, axis)
        base = round(endpoint_power.base)
        if abs(endpoint_power.base - base) > 1.0e-12:
            raise ValueError(f"non-integer regular monomial base in PSD2: {endpoint_power.base}")
        monomial_pref = monomial_pref * expression_power(S(name), int(base))
        if abs(endpoint_power.eps_coeff) > 1.0e-15:
            monomial_log = monomial_log + E(repr(float(endpoint_power.eps_coeff))) * S(name).log()

    prefactor = (
        monomial_pref
        * jacobian
        * expression_power(u_residual, int(topology.u_power_base))
        * expression_power(f_residual, -int(topology.f_power_base))
    )
    eps_log = (
        monomial_log
        + E(repr(float(topology.eps_log_u_coeff))) * u_residual.log()
        + E(repr(float(topology.eps_log_f_coeff))) * f_residual.log()
    )

    g_by_regular_order: list[Any] = []
    log_power = E("1")
    factorial = 1
    for regular_order in range(topology.coefficient_count):
        if regular_order > 0:
            log_power = log_power * eps_log
            factorial *= regular_order
        g_by_regular_order.append(prefactor * log_power / E(str(factorial)))
    return g_by_regular_order, {
        "u_substituted": str(u_sub),
        "f_substituted": str(f_sub),
        "u_residual": str(u_residual),
        "f_residual": str(f_residual),
        "monomial_prefactor": str(monomial_pref),
        "monomial_epsilon_log": str(monomial_log),
    }


def derivative_coefficient(
    g_by_regular_order: list[Any],
    sector: SectorDefinition,
    boundary_positions: tuple[int, ...],
    zero_positions: tuple[int, ...],
    multi_index: tuple[int, ...],
    regular_order: int,
    cache: dict[tuple[Any, ...], Any],
) -> Any:
    """Return one Taylor coefficient of explicit g_s at a PSD2 endpoint."""
    key = (
        tuple(boundary_positions),
        tuple(zero_positions),
        tuple(int(value) for value in multi_index),
        int(regular_order),
    )
    cached = cache.get(key)
    if cached is not None:
        return cached

    expr = g_by_regular_order[int(regular_order)]
    # Endpoint coordinates that are not differentiated can be specialized
    # before taking derivatives.  This is algebraically equivalent and avoids
    # carrying obviously dead variables through the hardest PSD2 derivatives.
    replacements_before: list[tuple[str, Any]] = []
    replacements_after: list[tuple[str, Any]] = []
    endpoint_values: dict[int, Any] = {}
    for position in boundary_positions:
        endpoint_values[int(position)] = E("1")
    for position in zero_positions:
        endpoint_values[int(position)] = E("0")
    for position, value in endpoint_values.items():
        axis = sector.singular_axes[int(position)]
        entry = (sector.variable_names[axis], value)
        if int(multi_index[int(position)]) == 0:
            replacements_before.append(entry)
        else:
            replacements_after.append(entry)
    if replacements_before:
        expr = replace_many(expr, replacements_before)

    denominator = 1
    for position, count in enumerate(multi_index):
        axis = sector.singular_axes[int(position)]
        symbol = S(sector.variable_names[axis])
        for _ in range(int(count)):
            expr = expr.derivative(symbol)
        denominator *= math.factorial(int(count))
    if denominator != 1:
        expr = expr / E(str(denominator))

    if replacements_after:
        expr = replace_many(expr, replacements_after)
    cache[key] = expr
    return expr


def build_fused_subtracted_evaluator(
    data: dict[str, Any],
    artifact_dir: Path,
    *,
    write_artifacts: bool,
    jit_compile: bool,
    max_terms: int,
    max_build_seconds: float,
    build_evaluators: bool,
    evaluator_orders: set[int] | None,
    evaluator_options: dict[str, Any],
) -> FusedBuildResult:
    """Build one explicit fully subtracted Symbolica evaluator per Laurent order."""
    total_start = time.perf_counter()
    topology, sector = instantiate_topology_and_sector(data)
    g_by_regular_order, diagnostics = build_sector_regular_expression(data, topology, sector)
    topology.prepare_endpoint_projector_formulas([sector])
    formula = topology.endpoint_projector_formula_for(sector)
    if not formula.ibp_reduce_to_log_endpoint:
        raise RuntimeError("PSD2 fused experiment expects an IBP-lowered projector formula")

    expression_start = time.perf_counter()
    outputs = [E("0") for _ in topology.laurent_orders]
    derivative_cache: dict[tuple[Any, ...], Any] = {}
    terms = formula.ibp_terms[: max(int(max_terms), 0)]
    if len(terms) != len(formula.ibp_terms):
        print(
            f"warning: building only {len(terms)}/{len(formula.ibp_terms)} IBP terms; "
            "this fused result is diagnostic only",
            flush=True,
        )
    for term_index, term in enumerate(terms):
        if time.perf_counter() - total_start > max_build_seconds:
            raise TimeoutError(
                f"fused expression build exceeded soft guard after {term_index}/"
                f"{len(formula.ibp_terms)} IBP terms"
            )
        child = formula.child_formulas[term.child_signature]
        if not child.output_expressions:
            raise RuntimeError(f"child formula {term.child_signature!r} has no output expressions")

        replacements: list[tuple[str, Any]] = []
        active_positions = tuple(int(position) for position in term.active_positions)
        for local_position, original_position in enumerate(active_positions):
            axis = sector.singular_axes[int(original_position)]
            replacements.append((child.input_names[local_position], S(sector.variable_names[axis])))

        offset = len(active_positions)
        for column, (_child_boundary, child_zero, child_multi, regular_order) in enumerate(
            child.coefficient_layout
        ):
            original_zero = tuple(active_positions[int(position)] for position in child_zero)
            original_multi = list(int(value) for value in term.derivative_multi)
            for child_position, value in enumerate(child_multi):
                original_multi[active_positions[int(child_position)]] += int(value)
            coeff_expr = derivative_coefficient(
                g_by_regular_order,
                sector,
                tuple(int(position) for position in term.boundary_positions),
                tuple(sorted(int(position) for position in original_zero)),
                tuple(original_multi),
                int(regular_order),
                derivative_cache,
            )
            replacements.append((child.input_names[offset + column], coeff_expr))

        for child_index, child_expr in enumerate(child.output_expressions):
            substituted = replace_many(child_expr, replacements)
            for pref_index, prefactor in enumerate(term.prefactor_coeffs):
                output_index = child_index + pref_index
                if output_index >= len(outputs):
                    break
                pref = complex(prefactor)
                if abs(pref.imag) > 0.0:
                    raise ValueError("PSD2 fused experiment expected real IBP prefactors")
                if abs(pref.real) <= 0.0:
                    continue
                outputs[output_index] = outputs[output_index] + E(repr(float(pref.real))) * substituted
        print(
            f"  fused term {term_index + 1:02d}/{len(terms)}: "
            f"derivative-cache={len(derivative_cache)} elapsed={time.perf_counter() - total_start:.1f}s",
            flush=True,
        )

    expression_seconds = time.perf_counter() - expression_start
    expression_bytes = 0
    evaluator_bytes = 0
    if write_artifacts:
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        (artifact_dir / "expressions").mkdir(parents=True, exist_ok=True)
        (artifact_dir / "evaluators").mkdir(parents=True, exist_ok=True)
        (artifact_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
        (artifact_dir / "diagnostics" / "regular_expressions.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for index, expr in enumerate(outputs):
            expr_path = artifact_dir / "expressions" / f"eps_order_{topology.laurent_orders[index]}.txt.gz"
            expr_path.write_bytes(gzip.compress(expr.format_plain().encode("utf-8"), compresslevel=9))
            expr.save(str(artifact_dir / "expressions" / f"eps_order_{topology.laurent_orders[index]}.expr.bin.gz"))
        expression_bytes = sum(path.stat().st_size for path in (artifact_dir / "expressions").glob("*.gz"))

    if not build_evaluators:
        return FusedBuildResult(
            expressions=outputs,
            evaluators=[],
            generation_seconds=time.perf_counter() - total_start,
            evaluator_build_seconds=0.0,
            expression_build_seconds=expression_seconds,
            expression_bytes=int(expression_bytes or sum(len(expr.format_plain().encode("utf-8")) for expr in outputs)),
            evaluator_bytes=0,
            artifact_dir=artifact_dir,
            coefficient_count=len(outputs),
            laurent_orders=list(topology.laurent_orders),
            evaluator_laurent_orders=[],
        )

    params = [S(name) for name in sector.variable_names]
    evaluator_start = time.perf_counter()
    evaluators: list[Any | None] = []
    evaluator_artifact_paths: list[Path] = []
    for index, expr in enumerate(outputs):
        order = int(topology.laurent_orders[index])
        if evaluator_orders is not None and order not in evaluator_orders:
            evaluators.append(None)
            continue
        start = time.perf_counter()
        print(f"  lowering fused evaluator eps^{order}...", flush=True)
        evaluator = expr.evaluator(
            params,
            jit_compile=jit_compile,
            **evaluator_options,
        )
        print(
            f"  lowered fused evaluator eps^{order} in {time.perf_counter() - start:.3f}s",
            flush=True,
        )
        evaluators.append(evaluator)
        if write_artifacts:
            evaluator_path = artifact_dir / "evaluators" / f"eps_order_{order}.bin.gz"
            evaluator_path.write_bytes(gzip.compress(evaluator.save(), compresslevel=9))
            evaluator_artifact_paths.append(evaluator_path)
    evaluator_seconds = time.perf_counter() - evaluator_start
    if write_artifacts:
        evaluator_bytes = sum(path.stat().st_size for path in evaluator_artifact_paths)
    else:
        expression_bytes = sum(len(expr.format_plain().encode("utf-8")) for expr in outputs)

    return FusedBuildResult(
        expressions=outputs,
        evaluators=[evaluator for evaluator in evaluators if evaluator is not None],
        generation_seconds=time.perf_counter() - total_start,
        evaluator_build_seconds=evaluator_seconds,
        expression_build_seconds=expression_seconds,
        expression_bytes=int(expression_bytes),
        evaluator_bytes=int(evaluator_bytes),
        artifact_dir=artifact_dir,
        coefficient_count=len(outputs),
        laurent_orders=list(topology.laurent_orders),
        evaluator_laurent_orders=[
            int(order)
            for order, evaluator in zip(topology.laurent_orders, evaluators)
            if evaluator is not None
        ],
    )


def build_fused_evaluators_from_artifacts(
    artifact_dir: Path,
    laurent_orders: list[int],
    variable_names: list[str],
    *,
    write_artifacts: bool,
    jit_compile: bool,
    evaluator_orders: set[int] | None,
    evaluator_options: dict[str, Any],
) -> FusedBuildResult:
    """Load saved fused expressions and lower selected evaluators."""
    start_all = time.perf_counter()
    expressions: list[Any | None] = []
    expression_bytes = 0
    for order in laurent_orders:
        if evaluator_orders is not None and int(order) not in evaluator_orders:
            expressions.append(None)
            continue
        binary_path = artifact_dir / "expressions" / f"eps_order_{int(order)}.expr.bin.gz"
        text_path = artifact_dir / "expressions" / f"eps_order_{int(order)}.txt.gz"
        if binary_path.is_file():
            expressions.append(Expression.load(str(binary_path)))
            expression_bytes += binary_path.stat().st_size
        else:
            if not text_path.is_file():
                raise FileNotFoundError(f"missing fused expression artifact: {text_path}")
            raw = gzip.decompress(text_path.read_bytes())
            expression_bytes += text_path.stat().st_size
            expressions.append(E(raw.decode("utf-8")))

    params = [S(name) for name in variable_names]
    evaluators: list[Any | None] = []
    evaluator_artifact_paths: list[Path] = []
    evaluator_start = time.perf_counter()
    for order, expr in zip(laurent_orders, expressions):
        if expr is None:
            evaluators.append(None)
            continue
        print(f"  lowering saved fused evaluator eps^{int(order)}...", flush=True)
        start = time.perf_counter()
        evaluator = expr.evaluator(
            params,
            jit_compile=jit_compile,
            **evaluator_options,
        )
        elapsed = time.perf_counter() - start
        print(f"  lowered saved fused evaluator eps^{int(order)} in {elapsed:.3f}s", flush=True)
        evaluators.append(evaluator)
        if write_artifacts:
            (artifact_dir / "evaluators").mkdir(parents=True, exist_ok=True)
            path = artifact_dir / "evaluators" / f"eps_order_{int(order)}.bin.gz"
            path.write_bytes(gzip.compress(evaluator.save(), compresslevel=9))
            evaluator_artifact_paths.append(path)
    evaluator_seconds = time.perf_counter() - evaluator_start
    evaluator_bytes = (
        sum(path.stat().st_size for path in evaluator_artifact_paths)
        if write_artifacts
        else 0
    )
    return FusedBuildResult(
        expressions=[expr for expr in expressions if expr is not None],
        evaluators=[evaluator for evaluator in evaluators if evaluator is not None],
        generation_seconds=time.perf_counter() - start_all,
        evaluator_build_seconds=evaluator_seconds,
        expression_build_seconds=0.0,
        expression_bytes=int(expression_bytes),
        evaluator_bytes=int(evaluator_bytes),
        artifact_dir=artifact_dir,
        coefficient_count=len(expressions),
        laurent_orders=list(laurent_orders),
        evaluator_laurent_orders=[
            int(order)
            for order, evaluator in zip(laurent_orders, evaluators)
            if evaluator is not None
        ],
    )


def evaluate_fused(build: FusedBuildResult, coords: np.ndarray) -> tuple[np.ndarray, float, list[int]]:
    """Evaluate fused coefficient evaluators for a batch."""
    if not build.evaluators:
        raise RuntimeError("no fused evaluators were built")
    start = time.perf_counter()
    columns = [
        np.asarray(evaluator.evaluate_complex(coords), dtype=np.complex128)[:, 0]
        for evaluator in build.evaluators
    ]
    return np.stack(columns, axis=1), time.perf_counter() - start, list(build.evaluator_laurent_orders)


def build_source_fused_assembler(
    prepared_bundle: Path,
    sector_id: int,
    *,
    max_eps_order: int,
    evaluator_options: dict[str, Any],
    jit_compile: bool,
) -> tuple[SourceFusedBuildResult, TopologyDefinition, SectorDefinition, SectorProcessor]:
    """Build one Symbolica evaluator for the PSD2 endpoint-projector assembly.

    This is the middle-ground experiment requested in the discussion.  U and F
    remain black boxes: runtime still acquires regular Taylor coefficients via
    the prepared FSD coefficient machinery.  The new evaluator receives those
    coefficients as inputs and performs all IBP child-projector algebra and
    Laurent assembly in Symbolica instead of Python.
    """
    topology, sectors, _manifest = load_prepared_bundle(prepared_bundle, lru_size=0)
    if int(max_eps_order) < topology.laurent_min_order:
        raise ValueError(
            f"--source-fused-max-eps-order {max_eps_order} is below prepared minimum "
            f"eps^{topology.laurent_min_order}"
        )
    if int(max_eps_order) > topology.laurent_max_order:
        raise ValueError(
            f"--source-fused-max-eps-order {max_eps_order} is above prepared maximum "
            f"eps^{topology.laurent_max_order}"
        )
    topology.set_laurent_range(topology.laurent_min_order, int(max_eps_order))
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")
    sector = sectors[int(sector_id)]
    formula = topology.endpoint_projector_formula_for(sector)
    if not formula.ibp_reduce_to_log_endpoint:
        raise RuntimeError("source-fused PSD2 experiment expects an IBP-lowered projector")
    child_formulas = {
        signature: build_endpoint_projector_formula_symbolica(
            topology,
            None,
            signature,
            EndpointProjectorFormulaDefinition,
            ibp_reduce_to_log_endpoint=False,
        )
        for signature in formula.child_formulas
    }

    expression_start = time.perf_counter()
    y_input_names = [f"y{index}" for index in range(sector.integration_dim)]
    coefficient_names: dict[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int], str] = {}
    coefficient_keys: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]] = []

    def coefficient_symbol(
        key: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int],
    ) -> Any:
        name = coefficient_names.get(key)
        if name is None:
            name = f"c{len(coefficient_keys)}"
            coefficient_names[key] = name
            coefficient_keys.append(key)
        return S(name)

    active_order_to_child_index: dict[tuple[Any, ...], list[int]] = {}
    for child_signature, child in child_formulas.items():
        index_by_order = {int(order): index for index, order in enumerate(child.laurent_orders)}
        active_order_to_child_index[child_signature] = [
            index_by_order.get(int(order), -1) for order in topology.laurent_orders
        ]

    outputs = [E("0") for _ in topology.laurent_orders]
    y_symbols = [S(name) for name in y_input_names]
    for term in formula.ibp_terms:
        child = child_formulas[term.child_signature]
        active_positions = tuple(int(position) for position in term.active_positions)
        replacements: list[tuple[str, Any]] = []
        for local_position, original_position in enumerate(active_positions):
            axis = sector.singular_axes[int(original_position)]
            replacements.append((child.input_names[local_position], y_symbols[int(axis)]))

        offset = len(active_positions)
        for column, (_child_boundary, child_zero, child_multi, regular_order) in enumerate(
            child.coefficient_layout
        ):
            if int(regular_order) >= topology.coefficient_count:
                replacements.append((child.input_names[offset + column], E("0")))
                continue
            original_zero = tuple(active_positions[int(position)] for position in child_zero)
            original_multi = list(int(value) for value in term.derivative_multi)
            for child_position, value in enumerate(child_multi):
                original_multi[active_positions[int(child_position)]] += int(value)
            key = (
                tuple(int(position) for position in term.boundary_positions),
                tuple(sorted(int(position) for position in original_zero)),
                tuple(int(value) for value in original_multi),
                int(regular_order),
            )
            replacements.append((child.input_names[offset + column], coefficient_symbol(key)))

        active_child_exprs = [
            E("0") if index < 0 else as_expression(child.output_expressions[index])
            for index in active_order_to_child_index[term.child_signature]
        ]
        for value_index, child_expr in enumerate(active_child_exprs):
            substituted = replace_many(child_expr, replacements)
            for pref_index, prefactor in enumerate(term.prefactor_coeffs):
                out_index = value_index + pref_index
                if out_index >= len(outputs):
                    break
                pref = complex(prefactor)
                if abs(pref.imag) > 1.0e-15:
                    raise ValueError("source-fused PSD2 experiment expected real IBP prefactors")
                if abs(pref.real) <= 0.0:
                    continue
                outputs[out_index] = outputs[out_index] + E(repr(float(pref.real))) * substituted

    expression_seconds = time.perf_counter() - expression_start
    input_names = [*y_input_names, *[coefficient_names[key] for key in coefficient_keys]]
    input_symbols = [S(name) for name in input_names]
    evaluator_start = time.perf_counter()
    evaluator = Expression.evaluator_multiple(
        outputs,
        input_symbols,
        jit_compile=jit_compile,
        **evaluator_options,
    )
    evaluator_seconds = time.perf_counter() - evaluator_start
    evaluator_bytes = len(evaluator.save())
    expression_bytes = sum(len(expr.format_plain().encode("utf-8")) for expr in outputs)
    return (
        SourceFusedBuildResult(
            evaluator=evaluator,
            output_expressions=outputs,
            input_names=input_names,
            coefficient_keys=coefficient_keys,
            laurent_orders=list(topology.laurent_orders),
            expression_build_seconds=float(expression_seconds),
            evaluator_build_seconds=float(evaluator_seconds),
            expression_bytes=int(expression_bytes),
            evaluator_bytes=int(evaluator_bytes),
        ),
        topology,
        sector,
        processor,
    )


def source_fused_input_matrix(
    build: SourceFusedBuildResult,
    processor: SectorProcessor,
    sector: SectorDefinition,
    rows: np.ndarray,
    timing: HotPathTiming,
) -> np.ndarray:
    """Assemble source-fused evaluator inputs from regular Taylor coefficients."""
    sample_rows = np.asarray(rows, dtype=float)
    n_rows = int(sample_rows.shape[0])
    input_matrix = np.zeros((n_rows, len(build.input_names)), dtype=np.complex128)
    input_matrix[:, : sector.integration_dim] = sample_rows.astype(np.complex128)

    formula = processor.topology.endpoint_projector_formula_for(sector)
    shared_max_orders = processor._ibp_shared_max_orders(sector, formula)
    shared_output_pairs = processor._ibp_shared_output_pairs(sector, formula)
    shared_cache = processor._precompute_ibp_shared_batch_g_cache(
        sector,
        sample_rows,
        shared_max_orders,
        shared_output_pairs,
        timing,
    )

    grouped_pairs: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        set[tuple[tuple[int, ...], int]],
    ] = {}
    for boundary, zero, multi_index, regular_order in build.coefficient_keys:
        grouped_pairs.setdefault((tuple(boundary), tuple(zero)), set()).add(
            (tuple(multi_index), int(regular_order))
        )

    g_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], list[dict[tuple[int, ...], np.ndarray]]] = {}
    for (boundary, zero), pairs in grouped_pairs.items():
        max_orders = tuple(
            int(value)
            for value in shared_max_orders.get(
                (boundary, zero),
                tuple(
                    max(int(multi[position]) for multi, _order in pairs)
                    for position in range(len(sector.singular_axes))
                ),
            )
        )
        cached = shared_cache.get((boundary, zero, max_orders))
        if cached is not None:
            g_cache[(boundary, zero)] = cached
            continue
        output_pairs = tuple(
            shared_output_pairs.get(
                (boundary, zero),
                tuple(sorted(pairs, key=lambda item: (item[1], sum(item[0]), item[0]))),
            )
        )
        g_cache[(boundary, zero)] = processor._g_taylor_eps_series_batch(
            sector,
            sample_rows,
            set(zero),
            list(max_orders),
            timing,
            boundary_positions=set(boundary),
            max_orders_are_explicit=True,
            output_pairs=output_pairs,
        )

    offset = sector.integration_dim
    for boundary, zero, multi_index, regular_order in build.coefficient_keys:
        series = g_cache[(tuple(boundary), tuple(zero))][int(regular_order)]
        input_matrix[:, offset] = _series_coefficient(series, tuple(multi_index), n_rows)
        offset += 1
    return input_matrix


def evaluate_source_fused(
    build: SourceFusedBuildResult,
    processor: SectorProcessor,
    sector: SectorDefinition,
    coords: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], HotPathTiming]:
    """Evaluate the source-coefficient fused PSD2 implementation."""
    timing = HotPathTiming()
    total_start = time.perf_counter()
    source_start = time.perf_counter()
    input_matrix = source_fused_input_matrix(build, processor, sector, coords, timing)
    source_wall = time.perf_counter() - source_start
    eval_before = timing.eval_seconds
    assembler_start = time.perf_counter()
    values = np.asarray(build.evaluator.evaluate_complex(input_matrix), dtype=np.complex128)
    assembler_wall = time.perf_counter() - assembler_start
    timing.add_eval(assembler_wall)
    wall = time.perf_counter() - total_start
    stats = {
        "wall_seconds": float(wall),
        "source_wall_seconds": float(source_wall),
        "source_eval_seconds": float(eval_before),
        "source_python_seconds": float(max(source_wall - eval_before, 0.0)),
        "assembler_eval_seconds": float(assembler_wall),
        "python_seconds": float(max(wall - timing.eval_seconds, 0.0)),
        "eval_seconds": float(timing.eval_seconds),
    }
    return values, stats, timing


def _group_source_fused_keys(
    coefficient_keys: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]],
) -> dict[
    tuple[tuple[int, ...], tuple[int, ...]],
    tuple[tuple[tuple[int, ...], int], ...],
]:
    """Group source-fused coefficient requests by endpoint projector."""
    grouped: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        set[tuple[tuple[int, ...], int]],
    ] = {}
    for boundary, zero, multi_index, regular_order in coefficient_keys:
        grouped.setdefault((tuple(boundary), tuple(zero)), set()).add(
            (tuple(multi_index), int(regular_order))
        )
    return {
        key: tuple(sorted(pairs, key=lambda item: (item[1], sum(item[0]), item[0])))
        for key, pairs in grouped.items()
    }


def build_dual_envelope_source_context(
    data: dict[str, Any],
    source_fused: SourceFusedBuildResult,
    *,
    max_eps_order: int,
) -> DualEnvelopeSourceBuildResult:
    """Prepare a black-box dual-envelope source path for PSD2.

    This is the generalizable split: U/F are never opened or substituted into
    the sector map.  The only topology-specific evaluator work is cloning and
    dualizing the scalar U/F black-box evaluators for one envelope Taylor shape.
    """
    scalar_start = time.perf_counter()
    topology, sector = instantiate_topology_and_sector(data, skip_evaluator_build=False)
    topology.set_laurent_range(topology.laurent_min_order, int(max_eps_order))
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")
    scalar_seconds = time.perf_counter() - scalar_start

    sector_start = time.perf_counter()
    sector.prepare_evaluators(include_dual=False)
    sector_seconds = time.perf_counter() - sector_start

    shape_start = time.perf_counter()
    grouped_pairs = _group_source_fused_keys(source_fused.coefficient_keys)
    envelope_shapes: list[list[tuple[int, ...]]] = []
    for (_boundary, zero), pairs in grouped_pairs.items():
        residual_multis = _ancestor_closed_multi_set(
            {tuple(multi) for multi, _regular_order in pairs},
            len(sector.singular_axes),
        )
        u_shape = topology.sparse_regular_source_shape_for_monomial_powers(
            sector,
            zero,
            residual_multis,
            sector.u_monomial_powers,
        )
        f_shape = topology.sparse_regular_source_shape_for_monomial_powers(
            sector,
            zero,
            residual_multis,
            sector.f_monomial_powers,
        )
        envelope_shapes.extend([u_shape, f_shape])
    envelope_shape = _merge_multi_shapes(*envelope_shapes)
    shape_seconds = time.perf_counter() - shape_start

    dual_start = time.perf_counter()
    # Build all hot-path evaluators explicitly here.  In FSD generation this is
    # the part that should be serialized into a prepared bundle.
    sector.prepare_dual_evaluators_for_shape(envelope_shape)
    topology.u_dual_evaluator(envelope_shape)
    topology.f_dual_evaluator(envelope_shape)
    dual_seconds = time.perf_counter() - dual_start

    return DualEnvelopeSourceBuildResult(
        topology=topology,
        sector=sector,
        processor=processor,
        source_fused=source_fused,
        envelope_shape=envelope_shape,
        grouped_pairs=grouped_pairs,
        scalar_evaluator_build_seconds=float(scalar_seconds),
        envelope_shape_build_seconds=float(shape_seconds),
        dual_evaluator_build_seconds=float(dual_seconds),
        sector_evaluator_build_seconds=float(sector_seconds),
    )


def dual_envelope_source_input_matrix(
    context: DualEnvelopeSourceBuildResult,
    rows: np.ndarray,
    timing: HotPathTiming,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build assembler inputs using one envelope dualized U/F source pass."""
    source_fused = context.source_fused
    topology = context.topology
    sector = context.sector
    processor = context.processor
    sample_rows = np.asarray(rows, dtype=float)
    n_rows = int(sample_rows.shape[0])
    input_matrix = np.zeros((n_rows, len(source_fused.input_names)), dtype=np.complex128)
    input_matrix[:, : sector.integration_dim] = sample_rows.astype(np.complex128)
    source_index = {multi: index for index, multi in enumerate(context.envelope_shape)}

    stack_start = time.perf_counter()
    stacked_rows: list[np.ndarray] = []
    slices: dict[tuple[tuple[int, ...], tuple[int, ...]], tuple[int, int]] = {}
    offset = 0
    for boundary, zero in context.grouped_pairs:
        endpoint_rows = sample_rows.copy()
        for position in boundary:
            endpoint_rows[:, sector.singular_axes[int(position)]] = 1.0
        for position in zero:
            endpoint_rows[:, sector.singular_axes[int(position)]] = 0.0
        stacked_rows.append(endpoint_rows)
        slices[(boundary, zero)] = (offset, offset + n_rows)
        offset += n_rows
    stacked = np.vstack(stacked_rows) if stacked_rows else sample_rows[:0, :]
    stack_seconds = time.perf_counter() - stack_start

    eval_start = time.perf_counter()
    u_taylor = topology._taylor_batch(
        sector,
        stacked,
        topology.u_dual_evaluator(context.envelope_shape),
        evaluator_shape=context.envelope_shape,
        timing=timing,
    )
    f_taylor = topology._taylor_batch(
        sector,
        stacked,
        topology.f_dual_evaluator(context.envelope_shape),
        evaluator_shape=context.envelope_shape,
        timing=timing,
    )
    dual_eval_seconds = time.perf_counter() - eval_start

    assembly_start = time.perf_counter()
    g_cache: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[dict[tuple[int, ...], np.ndarray]],
    ] = {}
    for (boundary, zero), pairs in context.grouped_pairs.items():
        row_start, row_stop = slices[(boundary, zero)]
        endpoint_rows = stacked[row_start:row_stop, :]
        residual_multis = _ancestor_closed_multi_set(
            {tuple(multi) for multi, _regular_order in pairs},
            len(sector.singular_axes),
        )
        max_orders = [
            max((int(multi[position]) for multi in residual_multis), default=0)
            for position in range(len(sector.singular_axes))
        ]
        u_series = processor._residual_taylor_series_from_values(
            sector=sector,
            endpoint_rows=endpoint_rows,
            monomial_powers=sector.u_monomial_powers,
            taylor=u_taylor[row_start:row_stop, :],
            zero_positions=set(zero),
            max_orders=max_orders,
            taylor_index=source_index,
            residual_multis=residual_multis,
        )
        f_series = processor._residual_taylor_series_from_values(
            sector=sector,
            endpoint_rows=endpoint_rows,
            monomial_powers=sector.f_monomial_powers,
            taylor=f_taylor[row_start:row_stop, :],
            zero_positions=set(zero),
            max_orders=max_orders,
            taylor_index=source_index,
            residual_multis=residual_multis,
        )
        allowed_multis = set(residual_multis)
        u_power_series, u_log_series = _series_pow_real_and_log_allowed(
            u_series,
            topology.u_power_base,
            max_orders,
            n_rows,
            allowed_multis,
        )
        f_power_series, f_log_series = _series_pow_real_and_log_allowed(
            f_series,
            -topology.f_power_base,
            max_orders,
            n_rows,
            allowed_multis,
        )
        jacobian_series = _series_filter_allowed(
            _series_constant(1.0 + 0.0j, max_orders, n_rows),
            allowed_multis,
        )
        pref_series = _series_mul_allowed(
            jacobian_series,
            _series_mul_allowed(u_power_series, f_power_series, allowed_multis),
            allowed_multis,
        )
        monomial_pref, monomial_log = processor._regular_monomial_base_log_batch(
            sector,
            endpoint_rows,
        )
        pref_series = _series_mul_allowed(
            _series_constant(monomial_pref, max_orders, n_rows),
            pref_series,
            allowed_multis,
        )
        log_series = _series_filter_allowed(
            _series_add(
                _series_constant(monomial_log, max_orders, n_rows),
                _series_add(
                    _series_scale(u_log_series, topology.eps_log_u_coeff),
                    _series_scale(f_log_series, topology.eps_log_f_coeff),
                ),
            ),
            allowed_multis,
        )
        requested_multis_by_order: dict[int, set[tuple[int, ...]]] = {}
        for multi_index, regular_order in pairs:
            requested_multis_by_order.setdefault(int(regular_order), set()).add(tuple(multi_index))
        out: list[dict[tuple[int, ...], np.ndarray]] = []
        log_power = _series_constant(1.0 + 0.0j, max_orders, n_rows)
        factorial = 1.0
        for regular_order in range(topology.coefficient_count):
            if regular_order > 0:
                factorial *= float(regular_order)
                log_power = _series_mul_allowed(log_power, log_series, allowed_multis)
            final_allowed = requested_multis_by_order.get(regular_order, set())
            if not final_allowed:
                out.append({})
                continue
            out.append(
                _series_scale(
                    _series_mul_allowed(pref_series, log_power, final_allowed),
                    1.0 / factorial,
                )
            )
        g_cache[(boundary, zero)] = out

    column = sector.integration_dim
    for boundary, zero, multi_index, regular_order in source_fused.coefficient_keys:
        series = g_cache[(tuple(boundary), tuple(zero))][int(regular_order)]
        input_matrix[:, column] = _series_coefficient(series, tuple(multi_index), n_rows)
        column += 1
    assembly_seconds = time.perf_counter() - assembly_start
    return input_matrix, {
        "stack_seconds": float(stack_seconds),
        "dual_source_eval_seconds": float(dual_eval_seconds),
        "source_assembly_seconds": float(assembly_seconds),
        "stacked_rows": int(stacked.shape[0]),
    }


def evaluate_dual_envelope_source(
    context: DualEnvelopeSourceBuildResult,
    coords: np.ndarray,
) -> tuple[np.ndarray, dict[str, float], HotPathTiming]:
    """Evaluate the dual-envelope source plus Symbolica assembler path."""
    timing = HotPathTiming()
    total_start = time.perf_counter()
    source_start = time.perf_counter()
    input_matrix, source_stats = dual_envelope_source_input_matrix(context, coords, timing)
    source_wall = time.perf_counter() - source_start
    eval_before_assembler = timing.eval_seconds
    assembler_start = time.perf_counter()
    values = np.asarray(context.source_fused.evaluator.evaluate_complex(input_matrix), dtype=np.complex128)
    assembler_seconds = time.perf_counter() - assembler_start
    timing.add_eval(assembler_seconds)
    wall = time.perf_counter() - total_start
    eval_total = timing.eval_seconds
    stats = {
        "wall_seconds": float(wall),
        "source_wall_seconds": float(source_wall),
        "source_eval_seconds": float(eval_before_assembler),
        "source_python_seconds": float(max(source_wall - eval_before_assembler, 0.0)),
        "assembler_eval_seconds": float(assembler_seconds),
        "eval_seconds": float(eval_total),
        "python_seconds": float(max(wall - eval_total, 0.0)),
        **source_stats,
    }
    return values, stats, timing


def build_source_coefficient_evaluator(
    data: dict[str, Any],
    coefficient_keys: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]],
    *,
    max_eps_order: int,
    evaluator_options: dict[str, Any],
    jit_compile: bool,
) -> SourceCoefficientBuildResult:
    """Build evaluator A for the two-stage PSD2 experiment.

    The outputs are the regular Taylor coefficients consumed by the assembler
    evaluator.  These coefficients are obtained from the explicit PSD2 U/F
    expressions only in this standalone experiment; in FSD proper the same
    role would be played by a generated black-box U/F derivative evaluator.
    """
    topology, sector = instantiate_topology_and_sector(data)
    topology.set_laurent_range(topology.laurent_min_order, int(max_eps_order))
    g_by_regular_order, _diagnostics = build_sector_regular_expression(data, topology, sector)
    expression_start = time.perf_counter()
    derivative_cache: dict[tuple[Any, ...], Any] = {}
    outputs: list[Any] = []
    for index, (boundary, zero, multi_index, regular_order) in enumerate(coefficient_keys):
        outputs.append(
            derivative_coefficient(
                g_by_regular_order,
                sector,
                tuple(boundary),
                tuple(zero),
                tuple(multi_index),
                int(regular_order),
                derivative_cache,
            )
        )
        if (index + 1) % 250 == 0 or index + 1 == len(coefficient_keys):
            print(
                f"  source coefficient expression {index + 1}/{len(coefficient_keys)} "
                f"cache={len(derivative_cache)} elapsed={time.perf_counter() - expression_start:.1f}s",
                flush=True,
            )
    expression_seconds = time.perf_counter() - expression_start
    params = [S(name) for name in sector.variable_names]
    evaluator_start = time.perf_counter()
    evaluator = Expression.evaluator_multiple(
        outputs,
        params,
        jit_compile=jit_compile,
        **evaluator_options,
    )
    evaluator_seconds = time.perf_counter() - evaluator_start
    return SourceCoefficientBuildResult(
        evaluator=evaluator,
        coefficient_keys=list(coefficient_keys),
        expression_build_seconds=float(expression_seconds),
        evaluator_build_seconds=float(evaluator_seconds),
        expression_bytes=int(sum(len(expr.format_plain().encode("utf-8")) for expr in outputs)),
        evaluator_bytes=int(len(evaluator.save())),
    )


def evaluate_two_stage_fused(
    source_build: SourceCoefficientBuildResult,
    assembler_build: SourceFusedBuildResult,
    sector: SectorDefinition,
    coords: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate PSD2 with exactly two Symbolica evaluator calls per batch."""
    rows = np.asarray(coords, dtype=float)
    total_start = time.perf_counter()
    source_start = time.perf_counter()
    source_values = np.asarray(source_build.evaluator.evaluate_complex(rows), dtype=np.complex128)
    source_seconds = time.perf_counter() - source_start
    input_matrix = np.zeros((rows.shape[0], len(assembler_build.input_names)), dtype=np.complex128)
    input_matrix[:, : sector.integration_dim] = rows.astype(np.complex128)
    input_matrix[:, sector.integration_dim :] = source_values
    assembler_start = time.perf_counter()
    coeffs = np.asarray(assembler_build.evaluator.evaluate_complex(input_matrix), dtype=np.complex128)
    assembler_seconds = time.perf_counter() - assembler_start
    wall = time.perf_counter() - total_start
    return coeffs, {
        "wall_seconds": float(wall),
        "source_eval_seconds": float(source_seconds),
        "assembler_eval_seconds": float(assembler_seconds),
        "eval_seconds": float(source_seconds + assembler_seconds),
        "python_seconds": float(max(wall - source_seconds - assembler_seconds, 0.0)),
    }


def _coefficient_key_to_json(
    key: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int],
) -> list[Any]:
    """Serialize one endpoint regular-coefficient key."""
    boundary, zero, multi_index, regular_order = key
    return [
        [int(value) for value in boundary],
        [int(value) for value in zero],
        [int(value) for value in multi_index],
        int(regular_order),
    ]


def _coefficient_key_from_json(
    value: list[Any],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]:
    """Inverse of ``_coefficient_key_to_json``."""
    boundary, zero, multi_index, regular_order = value
    return (
        tuple(int(item) for item in boundary),
        tuple(int(item) for item in zero),
        tuple(int(item) for item in multi_index),
        int(regular_order),
    )


def _evaluator_cache_paths(root: Path, prefix: str, max_eps_order: int) -> tuple[Path, Path]:
    """Return ``(metadata,json, evaluator.bin.gz)`` paths for a PSD2 cache item."""
    directory = Path(root).expanduser()
    stem = f"{prefix}_epsmax_{int(max_eps_order)}"
    return directory / f"{stem}.json", directory / f"{stem}.bin.gz"


def save_source_fused_assembler_evaluator(
    build: SourceFusedBuildResult,
    root: Path,
    *,
    sector_id: int,
    max_eps_order: int,
) -> None:
    """Persist the source-fused assembler evaluator and coefficient layout."""
    metadata_path, evaluator_path = _evaluator_cache_paths(root, "assembler", max_eps_order)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator_path.write_bytes(gzip.compress(build.evaluator.save(), compresslevel=6))
    metadata = {
        "kind": "psd2-source-fused-assembler",
        "sector_id": int(sector_id),
        "max_eps_order": int(max_eps_order),
        "laurent_orders": [int(order) for order in build.laurent_orders],
        "input_names": list(build.input_names),
        "coefficient_keys": [
            _coefficient_key_to_json(key) for key in build.coefficient_keys
        ],
        "expression_build_seconds": float(build.expression_build_seconds),
        "evaluator_build_seconds": float(build.evaluator_build_seconds),
        "expression_bytes": int(build.expression_bytes),
        "evaluator_bytes": int(evaluator_path.stat().st_size),
        "evaluator_file": evaluator_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_source_fused_assembler_evaluator(
    root: Path,
    prepared_bundle: Path,
    sector_id: int,
    *,
    max_eps_order: int,
) -> tuple[SourceFusedBuildResult, TopologyDefinition, SectorDefinition, SectorProcessor]:
    """Load a saved source-fused assembler evaluator for PSD2."""
    metadata_path, evaluator_path = _evaluator_cache_paths(root, "assembler", max_eps_order)
    if not metadata_path.is_file() or not evaluator_path.is_file():
        raise FileNotFoundError(f"missing source-fused assembler cache in {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "psd2-source-fused-assembler":
        raise ValueError(f"unexpected assembler cache kind in {metadata_path}")
    if int(metadata.get("sector_id")) != int(sector_id):
        raise ValueError(f"assembler cache sector mismatch in {metadata_path}")
    if int(metadata.get("max_eps_order")) != int(max_eps_order):
        raise ValueError(f"assembler cache epsilon-order mismatch in {metadata_path}")

    topology, sectors, _manifest = load_prepared_bundle(prepared_bundle, lru_size=0)
    topology.set_laurent_range(topology.laurent_min_order, int(max_eps_order))
    sector = sectors[int(sector_id)]
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")
    evaluator = Evaluator.load(gzip.decompress(evaluator_path.read_bytes()))
    build = SourceFusedBuildResult(
        evaluator=evaluator,
        output_expressions=[],
        input_names=[str(name) for name in metadata["input_names"]],
        coefficient_keys=[
            _coefficient_key_from_json(key) for key in metadata["coefficient_keys"]
        ],
        laurent_orders=[int(order) for order in metadata["laurent_orders"]],
        expression_build_seconds=float(metadata.get("expression_build_seconds", 0.0)),
        evaluator_build_seconds=float(metadata.get("evaluator_build_seconds", 0.0)),
        expression_bytes=int(metadata.get("expression_bytes", 0)),
        evaluator_bytes=int(metadata.get("evaluator_bytes", evaluator_path.stat().st_size)),
    )
    return build, topology, sector, processor


def save_source_coefficient_evaluator(
    build: SourceCoefficientBuildResult,
    root: Path,
    *,
    sector_id: int,
    max_eps_order: int,
) -> None:
    """Persist evaluator A from the two-stage PSD2 experiment."""
    metadata_path, evaluator_path = _evaluator_cache_paths(root, "source", max_eps_order)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    raw = build.evaluator.save()
    evaluator_path.write_bytes(gzip.compress(raw, compresslevel=6))
    metadata = {
        "kind": "psd2-source-coefficient-evaluator",
        "sector_id": int(sector_id),
        "max_eps_order": int(max_eps_order),
        "coefficient_keys": [
            _coefficient_key_to_json(key) for key in build.coefficient_keys
        ],
        "expression_build_seconds": float(build.expression_build_seconds),
        "evaluator_build_seconds": float(build.evaluator_build_seconds),
        "expression_bytes": int(build.expression_bytes),
        "evaluator_bytes": int(len(raw)),
        "evaluator_file": evaluator_path.name,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def load_source_coefficient_evaluator(
    root: Path,
    *,
    sector_id: int,
    max_eps_order: int,
    expected_coefficient_keys: list[
        tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], int]
    ],
) -> SourceCoefficientBuildResult:
    """Load evaluator A and verify that it matches the assembler layout."""
    metadata_path, evaluator_path = _evaluator_cache_paths(root, "source", max_eps_order)
    if not metadata_path.is_file() or not evaluator_path.is_file():
        raise FileNotFoundError(f"missing source coefficient evaluator cache in {root}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("kind") != "psd2-source-coefficient-evaluator":
        raise ValueError(f"unexpected source cache kind in {metadata_path}")
    if int(metadata.get("sector_id")) != int(sector_id):
        raise ValueError(f"source cache sector mismatch in {metadata_path}")
    if int(metadata.get("max_eps_order")) != int(max_eps_order):
        raise ValueError(f"source cache epsilon-order mismatch in {metadata_path}")
    coefficient_keys = [
        _coefficient_key_from_json(key) for key in metadata["coefficient_keys"]
    ]
    if coefficient_keys != list(expected_coefficient_keys):
        raise ValueError(
            "source coefficient evaluator layout does not match assembler layout"
        )
    evaluator = Evaluator.load(gzip.decompress(evaluator_path.read_bytes()))
    return SourceCoefficientBuildResult(
        evaluator=evaluator,
        coefficient_keys=coefficient_keys,
        expression_build_seconds=float(metadata.get("expression_build_seconds", 0.0)),
        evaluator_build_seconds=float(metadata.get("evaluator_build_seconds", 0.0)),
        expression_bytes=int(metadata.get("expression_bytes", 0)),
        evaluator_bytes=int(metadata.get("evaluator_bytes", evaluator_path.stat().st_size)),
    )


def _replace_sector_vars(sector: SectorDefinition, expr: Any, values: list[Any]) -> Any:
    """Replace sector variable names by the provided Symbolica expressions."""
    return replace_many(expr, list(zip(sector.variable_names, values)))


def _endpoint_coordinate_exprs(
    sector: SectorDefinition,
    boundary: tuple[int, ...],
    zero: tuple[int, ...],
    y_symbols: list[Any],
) -> list[Any]:
    """Return sector-coordinate expressions at one endpoint projector."""
    singular_position = {axis: pos for pos, axis in enumerate(sector.singular_axes)}
    boundary_set = {int(value) for value in boundary}
    zero_set = {int(value) for value in zero}
    out: list[Any] = []
    for axis, symbol in enumerate(y_symbols):
        position = singular_position.get(axis)
        if position is not None and position in boundary_set:
            out.append(E("1"))
        elif position is not None and position in zero_set:
            out.append(E("0"))
        else:
            out.append(symbol)
    return out


def _expr_derivative_coefficient(
    expr: Any,
    symbols: list[Any],
    multi_index: tuple[int, ...],
) -> Any:
    """Return the Taylor coefficient of ``expr`` for one multi-index."""
    out = expr
    denominator = 1
    for symbol, count in zip(symbols, multi_index):
        for _ in range(int(count)):
            out = out.derivative(symbol)
        denominator *= math.factorial(int(count))
    if denominator != 1:
        out = out / E(str(denominator))
    return out


def _monomial_taylor_series_expr(
    sector: SectorDefinition,
    monomial_powers: list[int],
    zero_positions: set[int],
    boundary_positions: set[int],
    max_orders: list[int],
    y_symbols: list[Any],
) -> dict[tuple[int, ...], Any]:
    """Expression analogue of the extracted-monomial denominator Taylor series."""
    axes = list(sector.singular_axes)
    axis_position = {axis: position for position, axis in enumerate(axes)}
    series = _expr_series_constant(E("1"), max_orders)
    for axis, power_value in enumerate(monomial_powers):
        power = int(power_value)
        if power == 0:
            continue
        position = axis_position.get(axis)
        if position is not None and position in zero_positions:
            continue
        factor: dict[tuple[int, ...], Any] = {}
        if position is None:
            factor = _expr_series_constant(y_symbols[axis] ** power, max_orders)
        else:
            base = E("1") if position in boundary_positions else y_symbols[axis]
            for order in range(min(power, int(max_orders[position])) + 1):
                multi = [0 for _ in max_orders]
                multi[position] = order
                factor[tuple(multi)] = E(str(math.comb(power, order))) * base ** (power - order)
        series = _expr_series_mul(series, factor, max_orders)
    return series


def _regular_monomial_base_log_expr(
    topology: TopologyDefinition,
    sector: SectorDefinition,
    y_symbols: list[Any],
) -> tuple[Any, Any]:
    """Expression analogue of regular non-singular monomial prefactor/log."""
    singular = set(sector.singular_axes)
    base_value = E("1")
    eps_log = E("0")
    for axis in range(sector.integration_dim):
        if axis in singular:
            continue
        endpoint_power = topology.endpoint_power(sector, axis)
        coord = y_symbols[axis]
        if abs(endpoint_power.base) > 1.0e-15:
            rounded = round(endpoint_power.base)
            if abs(endpoint_power.base - rounded) > 1.0e-12:
                raise ValueError(f"{sector.name}: non-integer regular base power {endpoint_power.base}")
            base_value = base_value * expression_power(coord, int(rounded))
        if abs(endpoint_power.eps_coeff) > 1.0e-15:
            eps_log = eps_log + E(repr(float(endpoint_power.eps_coeff))) * coord.log()
    return base_value, eps_log


def _residual_source_shape_expr(
    sector: SectorDefinition,
    zero_positions: set[int],
    residual_multis: set[tuple[int, ...]],
    monomial_powers: list[int],
) -> set[tuple[int, ...]]:
    """Return ancestor-closed source coefficient support for one residual."""
    axes = list(sector.singular_axes)
    axis_position = {axis: position for position, axis in enumerate(axes)}
    source: set[tuple[int, ...]] = {tuple(0 for _ in axes)}
    for residual_multi in residual_multis:
        shifted = list(int(value) for value in residual_multi)
        for axis, power in enumerate(monomial_powers):
            position = axis_position.get(axis)
            if position is not None and position in zero_positions:
                shifted[position] += int(power)
        source.add(tuple(shifted))
    return _ancestor_closed_multi_set(source, len(axes))


def _compose_polynomial_from_derivatives_expr(
    derivative_symbols: dict[tuple[int, ...], Any],
    h_series: list[dict[tuple[int, ...], Any]],
    allowed_multis: set[tuple[int, ...]],
) -> dict[tuple[int, ...], Any]:
    """Compose x-space derivative values with sector-map Taylor coefficients."""
    if not allowed_multis:
        return {}
    rank = len(next(iter(allowed_multis)))
    power_cache: dict[tuple[int, int], dict[tuple[int, ...], Any]] = {}

    def h_power(active_index: int, power: int) -> dict[tuple[int, ...], Any]:
        key = (int(active_index), int(power))
        cached = power_cache.get(key)
        if cached is not None:
            return cached
        if power == 0:
            cached = _expr_series_constant(E("1"), [0 for _ in range(rank)])
        elif power == 1:
            cached = {
                multi: value
                for multi, value in h_series[active_index].items()
                if multi in allowed_multis
            }
        else:
            cached = _expr_series_mul_allowed(
                h_power(active_index, power - 1),
                h_series[active_index],
                allowed_multis,
            )
        power_cache[key] = cached
        return cached

    product_cache: dict[tuple[int, ...], dict[tuple[int, ...], Any]] = {}

    def chain_product(alpha: tuple[int, ...]) -> dict[tuple[int, ...], Any]:
        cached = product_cache.get(alpha)
        if cached is not None:
            return cached
        term = _expr_series_constant(E("1"), [0 for _ in range(rank)])
        factorial = 1
        for active_index, power in enumerate(alpha):
            power_int = int(power)
            if power_int == 0:
                continue
            factorial *= math.factorial(power_int)
            term = _expr_series_mul_allowed(term, h_power(active_index, power_int), allowed_multis)
            if not term:
                break
        if term and factorial != 1:
            term = _expr_series_scale(term, E("1") / E(str(factorial)))
        product_cache[alpha] = term
        return term

    out: dict[tuple[int, ...], Any] = {}
    for alpha, symbol in derivative_symbols.items():
        product_series = chain_product(tuple(int(value) for value in alpha))
        if not product_series:
            continue
        out = _expr_series_add(out, _expr_series_scale(product_series, symbol))
    return out


def _expr_series_to_polynomial(
    series: dict[tuple[int, ...], Any],
    local_symbols: list[Any],
) -> Any:
    """Convert a sparse Taylor series dictionary to a formal polynomial."""
    out = E("0")
    for multi, coeff in series.items():
        term = coeff
        for symbol, power in zip(local_symbols, multi):
            if int(power):
                term = term * (symbol ** int(power))
        out = out + term
    return out


def _local_taylor_coefficient_expr(
    expr: Any,
    local_symbols: list[Any],
    multi_index: tuple[int, ...],
) -> Any:
    """Extract one formal Taylor coefficient by Symbolica differentiation."""
    out = expr
    denominator = 1
    for symbol, count in zip(local_symbols, multi_index):
        for _ in range(int(count)):
            out = out.derivative(symbol)
        denominator *= math.factorial(int(count))
    if denominator != 1:
        out = out / E(str(denominator))
    zero_replacements = [(str(symbol), E("0")) for symbol in local_symbols]
    return replace_many(out, zero_replacements)


def _g_coefficients_by_symbolic_diff(
    topology: TopologyDefinition,
    sector: SectorDefinition,
    u_residual: dict[tuple[int, ...], Any],
    f_residual: dict[tuple[int, ...], Any],
    j_series: dict[tuple[int, ...], Any],
    monomial_pref: Any,
    monomial_log: Any,
    requested_pairs: set[tuple[tuple[int, ...], int]],
    max_orders: list[int],
) -> list[dict[tuple[int, ...], Any]]:
    """Build requested regular coefficients through Symbolica derivatives.

    This avoids multiplying sparse expression series in Python for
    ``prefactor * log_power``.  The formula shape is topology-independent:
    topology information enters only through numeric exponent metadata and the
    derivative-symbol residual coefficients.
    """
    local_symbols = [S(f"loc{index}") for index in range(len(max_orders))]
    u_poly = _expr_series_to_polynomial(u_residual, local_symbols)
    f_poly = _expr_series_to_polynomial(f_residual, local_symbols)
    j_poly = _expr_series_to_polynomial(j_series, local_symbols)
    u_power = int(round(float(topology.u_power_base)))
    f_power = int(round(float(topology.f_power_base)))
    if abs(float(topology.u_power_base) - u_power) > 1.0e-12:
        raise ValueError(f"{sector.name}: symbolic-diff path requires integer U power")
    if abs(float(topology.f_power_base) - f_power) > 1.0e-12:
        raise ValueError(f"{sector.name}: symbolic-diff path requires integer F power")
    prefactor = (
        monomial_pref
        * j_poly
        * expression_power(u_poly, u_power)
        * expression_power(f_poly, -f_power)
    )
    eps_log = (
        monomial_log
        + E(repr(float(topology.eps_log_u_coeff))) * u_poly.log()
        + E(repr(float(topology.eps_log_f_coeff))) * f_poly.log()
    )
    out: list[dict[tuple[int, ...], Any]] = [
        {} for _ in range(topology.coefficient_count)
    ]
    for regular_order in range(topology.coefficient_count):
        pairs = [
            tuple(multi)
            for multi, order in requested_pairs
            if int(order) == int(regular_order)
        ]
        if not pairs:
            continue
        if regular_order == 0:
            g_expr = prefactor
        else:
            g_expr = prefactor * (eps_log ** int(regular_order)) / E(str(math.factorial(int(regular_order))))
        for multi in pairs:
            out[int(regular_order)][tuple(multi)] = _local_taylor_coefficient_expr(
                g_expr,
                local_symbols,
                tuple(multi),
            )
    return out


def build_derivative_fused_evaluators(
    data: dict[str, Any],
    source_fused: SourceFusedBuildResult,
    *,
    max_eps_order: int,
    evaluator_options: dict[str, Any],
    jit_compile: bool,
    max_groups: int | None = None,
    regular_method: str = "symbolic-diff",
) -> tuple[DerivativeFusedBuildResult, SectorDefinition]:
    """Build the generic derivative-value split for hard-coded PSD2."""
    topology, sector = instantiate_topology_and_sector(data)
    topology.set_laurent_range(topology.laurent_min_order, int(max_eps_order))
    top = data["topology"]
    y_symbols = [S(f"y{axis}") for axis in range(sector.integration_dim)]
    singular_symbols = [y_symbols[axis] for axis in sector.singular_axes]
    map_exprs = [E(str(expr)) for expr in data["sector"]["map_exprs"]]
    map_exprs_y = [
        _replace_sector_vars(sector, expr, y_symbols)
        for expr in map_exprs
    ]

    grouped_pairs: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        set[tuple[tuple[int, ...], int]],
    ] = {}
    for boundary, zero, multi_index, regular_order in source_fused.coefficient_keys:
        grouped_pairs.setdefault((tuple(boundary), tuple(zero)), set()).add(
            (tuple(multi_index), int(regular_order))
        )
    all_groups = sorted(grouped_pairs, key=lambda item: (len(item[0]), item[0], item[1]))
    groups = all_groups[: int(max_groups)] if max_groups is not None else all_groups
    included_group_set = set(groups)

    derivative_slot_names: dict[tuple[tuple[int, ...], tuple[int, ...], str, tuple[int, ...]], str] = {}
    derivative_slots: list[tuple[tuple[int, ...], tuple[int, ...], str, tuple[int, ...]]] = []

    def derivative_symbol(
        boundary: tuple[int, ...],
        zero: tuple[int, ...],
        polynomial: str,
        full_multi: tuple[int, ...],
    ) -> Any:
        key = (tuple(boundary), tuple(zero), str(polynomial), tuple(full_multi))
        name = derivative_slot_names.get(key)
        if name is None:
            name = f"d{len(derivative_slots)}"
            derivative_slot_names[key] = name
            derivative_slots.append(key)
        return S(name)

    derivative_candidates = {
        "u": set(topology._candidate_derivative_multi_indices(topology.u_expr, CHAIN_RULE_MAX_DERIVATIVE_DEGREE_1_TO_3_LOOPS)),
        "f": set(topology._candidate_derivative_multi_indices(topology.f_expr, CHAIN_RULE_MAX_DERIVATIVE_DEGREE_1_TO_3_LOOPS)),
    }

    g_expr_cache: dict[
        tuple[tuple[int, ...], tuple[int, ...]],
        list[dict[tuple[int, ...], Any]],
    ] = {}
    expression_start = time.perf_counter()
    for group_index, (boundary, zero) in enumerate(groups, start=1):
        group_start = time.perf_counter()
        print(
            f"  derivative-fused assembler group {group_index}/{len(groups)} "
            f"boundary={boundary} zero={zero} method={regular_method}",
            flush=True,
        )
        pairs = grouped_pairs[(boundary, zero)]
        requested_residual_multis = {tuple(multi) for multi, _order in pairs}
        residual_multis = _ancestor_closed_multi_set(
            requested_residual_multis,
            len(sector.singular_axes),
        )
        max_orders = [
            max(int(multi[position]) for multi in residual_multis)
            for position in range(len(sector.singular_axes))
        ]
        zero_set = {int(value) for value in zero}
        boundary_set = {int(value) for value in boundary}
        endpoint_values = _endpoint_coordinate_exprs(sector, boundary, zero, y_symbols)
        endpoint_map_exprs = [
            _replace_sector_vars(sector, expr, endpoint_values)
            for expr in map_exprs
        ]

        u_source_shape = _residual_source_shape_expr(
            sector,
            zero_set,
            residual_multis,
            sector.u_monomial_powers,
        )
        f_source_shape = _residual_source_shape_expr(
            sector,
            zero_set,
            residual_multis,
            sector.f_monomial_powers,
        )
        allowed_multis = set(u_source_shape) | set(f_source_shape) | {tuple(0 for _ in sector.singular_axes)}

        h_series_full: list[dict[tuple[int, ...], Any]] = []
        active_x_indices: list[int] = []
        for x_index, expr in enumerate(map_exprs_y):
            series: dict[tuple[int, ...], Any] = {}
            for multi in allowed_multis:
                if not any(multi):
                    continue
                coeff = _expr_derivative_coefficient(expr, singular_symbols, tuple(multi))
                coeff = _replace_sector_vars(sector, coeff, endpoint_values)
                if str(coeff) != "0":
                    series[tuple(multi)] = coeff
            h_series_full.append(series)
            if series:
                active_x_indices.append(int(x_index))

        h_series = [h_series_full[index] for index in active_x_indices]
        active_set = set(active_x_indices)
        polynomial_series_by_kind: dict[str, dict[tuple[int, ...], Any]] = {}
        for polynomial in ("u", "f"):
            derivative_symbols: dict[tuple[int, ...], Any] = {}
            max_total = min(
                CHAIN_RULE_MAX_DERIVATIVE_DEGREE_1_TO_3_LOOPS,
                max((sum(multi) for multi in allowed_multis), default=0),
            )
            for compressed in _dense_total_degree_multi_indices(len(active_x_indices), max_total):
                full = [0 for _ in topology.x_names]
                for active_position, x_index in enumerate(active_x_indices):
                    full[x_index] = int(compressed[active_position])
                full_multi = tuple(full)
                if full_multi not in derivative_candidates[polynomial]:
                    continue
                derivative_symbols[tuple(compressed)] = derivative_symbol(
                    boundary,
                    zero,
                    polynomial,
                    full_multi,
                )
            polynomial_series_by_kind[polynomial] = _compose_polynomial_from_derivatives_expr(
                derivative_symbols,
                h_series,
                allowed_multis,
            )

        def residual_series(
            polynomial: str,
            monomial_powers: list[int],
            source_shape: set[tuple[int, ...]],
        ) -> dict[tuple[int, ...], Any]:
            polynomial_series = polynomial_series_by_kind[polynomial]
            shifted_series: dict[tuple[int, ...], Any] = {}
            axis_position = {axis: position for position, axis in enumerate(sector.singular_axes)}
            for residual_multi in residual_multis:
                shifted = list(int(value) for value in residual_multi)
                for axis, power in enumerate(monomial_powers):
                    position = axis_position.get(axis)
                    if position is not None and position in zero_set:
                        shifted[position] += int(power)
                shifted_series[tuple(residual_multi)] = polynomial_series.get(tuple(shifted), E("0"))
            denominator = _monomial_taylor_series_expr(
                sector,
                monomial_powers,
                zero_set,
                boundary_set,
                max_orders,
                y_symbols,
            )
            if set(denominator) <= {tuple(0 for _ in max_orders)}:
                denom0 = denominator.get(tuple(0 for _ in max_orders), E("1"))
                return {multi: value / denom0 for multi, value in shifted_series.items()}
            return _expr_series_mul(
                shifted_series,
                _expr_series_pow_real(denominator, -1.0, max_orders),
                max_orders,
            )

        u_residual = residual_series("u", sector.u_monomial_powers, u_source_shape)
        f_residual = residual_series("f", sector.f_monomial_powers, f_source_shape)
        j_series = _expr_series_constant(E("1"), max_orders)
        monomial_pref, monomial_log = _regular_monomial_base_log_expr(topology, sector, y_symbols)
        if regular_method == "symbolic-diff":
            g_by_order = _g_coefficients_by_symbolic_diff(
                topology,
                sector,
                u_residual,
                f_residual,
                j_series,
                monomial_pref,
                monomial_log,
                pairs,
                max_orders,
            )
        else:
            prefactor = _expr_series_mul(
                _expr_series_mul(
                    j_series,
                    _expr_series_pow_real(u_residual, topology.u_power_base, max_orders),
                    max_orders,
                ),
                _expr_series_pow_real(f_residual, -topology.f_power_base, max_orders),
                max_orders,
            )
            prefactor = _expr_series_scale(prefactor, monomial_pref)
            eps_log_series = _expr_series_add(
                _expr_series_constant(monomial_log, max_orders),
                _expr_series_add(
                    _expr_series_scale(
                        _expr_series_log(u_residual, max_orders),
                        topology.eps_log_u_coeff,
                    ),
                    _expr_series_scale(
                        _expr_series_log(f_residual, max_orders),
                        topology.eps_log_f_coeff,
                    ),
                ),
            )
            g_by_order = []
            log_power = _expr_series_constant(E("1"), max_orders)
            factorial = 1
            for regular_order in range(topology.coefficient_count):
                if regular_order > 0:
                    log_power = _expr_series_mul(log_power, eps_log_series, max_orders)
                    factorial *= regular_order
                g_by_order.append(
                    _expr_series_scale(
                        _expr_series_mul(prefactor, log_power, max_orders),
                        E("1") / E(str(factorial)),
                    )
                )
        g_expr_cache[(boundary, zero)] = g_by_order
        print(
            f"    done group {group_index}/{len(groups)} "
            f"derivative-slots={len(derivative_slots)} "
            f"group={time.perf_counter() - group_start:.2f}s "
            f"elapsed={time.perf_counter() - expression_start:.1f}s",
            flush=True,
        )

    assembler_outputs = list(source_fused.output_expressions)
    replacements = []
    for index, key in enumerate(source_fused.coefficient_keys):
        boundary, zero, multi_index, regular_order = key
        if (tuple(boundary), tuple(zero)) not in included_group_set:
            expr = E("0")
        else:
            expr = g_expr_cache[(tuple(boundary), tuple(zero))][int(regular_order)].get(tuple(multi_index), E("0"))
        replacements.append((f"c{index}", expr))
    assembler_outputs = [replace_many(as_expression(expr), replacements) for expr in assembler_outputs]
    assembler_expression_seconds = time.perf_counter() - expression_start

    y_input_names = [f"y{axis}" for axis in range(sector.integration_dim)]
    derivative_input_names = [derivative_slot_names[key] for key in derivative_slots]
    assembler_input_names = [*y_input_names, *derivative_input_names]
    assembler_input_symbols = [S(name) for name in assembler_input_names]
    assembler_start = time.perf_counter()
    assembler = Expression.evaluator_multiple(
        assembler_outputs,
        assembler_input_symbols,
        jit_compile=jit_compile,
        **evaluator_options,
    )
    assembler_evaluator_seconds = time.perf_counter() - assembler_start

    source_expression_start = time.perf_counter()
    source_outputs: list[Any] = []
    x_replacements_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[str, Any]]] = {}
    derivative_expr_cache: dict[tuple[str, tuple[int, ...]], Any] = {}
    params_replacements = [
        (name, E(repr(float(value))))
        for name, value in zip(topology.parameter_names, topology.parameter_values)
    ]
    for slot_index, (boundary, zero, polynomial, multi_index) in enumerate(derivative_slots, start=1):
        endpoint_values = _endpoint_coordinate_exprs(sector, boundary, zero, y_symbols)
        endpoint_key = (tuple(boundary), tuple(zero))
        x_replacements = x_replacements_cache.get(endpoint_key)
        if x_replacements is None:
            endpoint_map_exprs = [
                _replace_sector_vars(sector, expr, endpoint_values)
                for expr in map_exprs
            ]
            x_replacements = list(zip(topology.x_names, endpoint_map_exprs))
            x_replacements_cache[endpoint_key] = x_replacements
        deriv_key = (str(polynomial), tuple(multi_index))
        deriv_expr = derivative_expr_cache.get(deriv_key)
        if deriv_expr is None:
            base_expr = topology.u_expr if polynomial == "u" else topology.f_expr
            deriv_expr = topology._differentiate_expr(base_expr, tuple(multi_index))
            derivative_expr_cache[deriv_key] = deriv_expr
        expr = replace_many(deriv_expr, x_replacements)
        if params_replacements:
            expr = replace_many(expr, params_replacements)
        source_outputs.append(expr)
        if slot_index % 250 == 0 or slot_index == len(derivative_slots):
            print(
                f"  derivative-fused source expression {slot_index}/{len(derivative_slots)} "
                f"elapsed={time.perf_counter() - source_expression_start:.1f}s",
                flush=True,
            )
    source_expression_seconds = time.perf_counter() - source_expression_start
    source_start = time.perf_counter()
    source = Expression.evaluator_multiple(
        source_outputs,
        y_symbols,
        jit_compile=jit_compile,
        **evaluator_options,
    )
    source_evaluator_seconds = time.perf_counter() - source_start
    return (
        DerivativeFusedBuildResult(
            source_evaluator=source,
            assembler_evaluator=assembler,
            derivative_slots=derivative_slots,
            groups=groups,
            laurent_orders=list(topology.laurent_orders),
            source_expression_build_seconds=float(source_expression_seconds),
            source_evaluator_build_seconds=float(source_evaluator_seconds),
            assembler_expression_build_seconds=float(assembler_expression_seconds),
            assembler_evaluator_build_seconds=float(assembler_evaluator_seconds),
            source_expression_bytes=int(sum(len(expr.format_plain().encode("utf-8")) for expr in source_outputs)),
            source_evaluator_bytes=int(len(source.save())),
            assembler_expression_bytes=int(sum(len(expr.format_plain().encode("utf-8")) for expr in assembler_outputs)),
            assembler_evaluator_bytes=int(len(assembler.save())),
        ),
        sector,
    )


def evaluate_derivative_fused(
    build: DerivativeFusedBuildResult,
    sector: SectorDefinition,
    coords: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate the derivative-fused PSD2 implementation."""
    rows = np.asarray(coords, dtype=float)
    total_start = time.perf_counter()
    source_start = time.perf_counter()
    derivative_values = np.asarray(build.source_evaluator.evaluate_complex(rows), dtype=np.complex128)
    source_seconds = time.perf_counter() - source_start
    input_matrix = np.zeros((rows.shape[0], sector.integration_dim + len(build.derivative_slots)), dtype=np.complex128)
    input_matrix[:, : sector.integration_dim] = rows.astype(np.complex128)
    input_matrix[:, sector.integration_dim :] = derivative_values
    assembler_start = time.perf_counter()
    coeffs = np.asarray(build.assembler_evaluator.evaluate_complex(input_matrix), dtype=np.complex128)
    assembler_seconds = time.perf_counter() - assembler_start
    wall = time.perf_counter() - total_start
    return coeffs, {
        "wall_seconds": float(wall),
        "source_eval_seconds": float(source_seconds),
        "assembler_eval_seconds": float(assembler_seconds),
        "eval_seconds": float(source_seconds + assembler_seconds),
        "python_seconds": float(max(wall - source_seconds - assembler_seconds, 0.0)),
    }


def run_fsd_reference(
    prepared_bundle: Path,
    sector_id: int,
    *,
    points: int,
    repeats: int,
    seed: int,
    sample: list[float] | None,
    max_eps_order: int | None,
) -> tuple[list[dict[str, Any]], Any, Any]:
    """Run the current prepared-bundle FSD-style PSD2 implementation."""
    load_start = time.perf_counter()
    topology, sectors, _manifest = load_prepared_bundle(prepared_bundle, lru_size=0)
    if max_eps_order is not None:
        if int(max_eps_order) < topology.laurent_min_order:
            raise ValueError(
                f"--fsd-max-eps-order {max_eps_order} is below prepared minimum "
                f"eps^{topology.laurent_min_order}"
            )
        if int(max_eps_order) > topology.laurent_max_order:
            raise ValueError(
                f"--fsd-max-eps-order {max_eps_order} is above prepared maximum "
                f"eps^{topology.laurent_max_order}"
            )
        topology.set_laurent_range(topology.laurent_min_order, int(max_eps_order))
    load_seconds = time.perf_counter() - load_start
    processor = SectorProcessor(topology, subtraction_backend="projector-formula")
    sector = sectors[int(sector_id)]
    rows: list[dict[str, Any]] = []
    for repeat in range(max(int(repeats), 1)):
        if sample is not None:
            coords = np.asarray([sample], dtype=float)
        else:
            rng = np.random.default_rng(int(seed) + 1_000_003 * int(sector_id) + repeat)
            coords = rng.random((max(int(points), 1), sector.integration_dim), dtype=float)
        start = time.perf_counter()
        coeffs, training, timing = processor.evaluate_batch(sector, coords)
        wall = time.perf_counter() - start
        rows.append(
            {
                "repeat": int(repeat),
                "points": int(coords.shape[0]),
                "coords": coords.tolist(),
                "wall_seconds": float(wall),
                "eval_seconds": float(timing.eval_seconds),
                "python_seconds": float(timing.python_seconds),
                "wall_us_per_sample": float(wall * 1.0e6 / coords.shape[0]),
                "eval_us_per_sample": float(timing.eval_seconds * 1.0e6 / coords.shape[0]),
                "python_us_per_sample": float(timing.python_seconds * 1.0e6 / coords.shape[0]),
                "precision_counts": timing.precision_counts,
                "coefficients": [complex_json(value) for value in coeffs[0]],
                "training": float(training[0]),
            }
        )
    for row in rows:
        row["bundle_load_seconds"] = float(load_seconds)
    return rows, topology, sector


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Return warm medians, skipping the first repeat when possible."""
    if not rows:
        return {}
    warm = rows[1:] if len(rows) > 1 else rows
    keys = [
        key
        for key in [
            "wall_seconds",
            "eval_seconds",
            "python_seconds",
            "wall_us_per_sample",
            "eval_us_per_sample",
            "python_us_per_sample",
            "source_wall_seconds",
            "source_eval_seconds",
            "source_python_seconds",
            "assembler_eval_seconds",
        ]
        if key in warm[0]
    ]
    return {key + "_median": float(np.median([float(row[key]) for row in warm])) for key in keys}


def main() -> int:
    """Run the PSD2 timing experiment."""
    args = parse_args()
    data = load_psd2_input(args.input)
    result: dict[str, Any] = {
        "input": str(args.input),
        "prepared_bundle": str(args.prepared_bundle),
        "sector": data["sector"],
        "topology": {
            key: data["topology"][key]
            for key in [
                "family",
                "x_names",
                "parameter_names",
                "parameter_values",
                "u_power_base",
                "f_power_base",
                "eps_log_u_coeff",
                "eps_log_f_coeff",
                "laurent_orders",
            ]
        },
    }

    fsd_rows: list[dict[str, Any]] = []
    fsd_topology = None
    fsd_sector = None
    if not args.skip_fsd:
        print("running FSD-style prepared-bundle reference...", flush=True)
        fsd_rows, fsd_topology, fsd_sector = run_fsd_reference(
            args.prepared_bundle,
            args.sector_id,
            points=args.points,
            repeats=args.repeats,
            seed=args.seed,
            sample=[float(value) for value in args.sample] if args.sample is not None else None,
            max_eps_order=args.fsd_max_eps_order,
        )
        result["fsd_style"] = {
            "rows": fsd_rows,
            "warm_medians": summarize_rows(fsd_rows),
            "laurent_orders": list(fsd_topology.laurent_orders) if fsd_topology is not None else None,
        }
        print("FSD-style warm medians:", result["fsd_style"]["warm_medians"], flush=True)

    evaluator_options = {
        "iterations": int(args.evaluator_iterations),
        "cpe_iterations": args.evaluator_cpe_iterations,
        "n_cores": int(args.evaluator_n_cores),
        "verbose": bool(args.evaluator_verbose),
        "direct_translation": bool(args.evaluator_direct_translation),
        "jit_direct_translation": bool(args.evaluator_jit_direct_translation),
        "max_horner_scheme_variables": int(args.evaluator_max_horner_vars),
        "max_common_pair_cache_entries": int(args.evaluator_max_common_pair_cache_entries),
        "max_common_pair_distance": int(args.evaluator_max_common_pair_distance),
    }

    source_fused: SourceFusedBuildResult | None = None
    source_topology: TopologyDefinition | None = None
    source_sector: SectorDefinition | None = None
    source_processor: SectorProcessor | None = None
    source_fused_cache_source: str | None = None
    if args.run_source_fused:
        source_fused_cache_source = "built"
        if args.load_evaluators_from is not None:
            try:
                print(
                    f"loading source-coefficient fused PSD2 assembler from {args.load_evaluators_from}...",
                    flush=True,
                )
                (
                    source_fused,
                    source_topology,
                    source_sector,
                    source_processor,
                ) = load_source_fused_assembler_evaluator(
                    args.load_evaluators_from,
                    args.prepared_bundle,
                    args.sector_id,
                    max_eps_order=int(args.source_fused_max_eps_order),
                )
                source_fused_cache_source = "loaded"
            except FileNotFoundError:
                print("source-fused assembler cache miss; building it", flush=True)
        if source_fused is None:
            print("building source-coefficient fused PSD2 assembler...", flush=True)
            source_fused, source_topology, source_sector, source_processor = build_source_fused_assembler(
                args.prepared_bundle,
                args.sector_id,
                max_eps_order=int(args.source_fused_max_eps_order),
                evaluator_options=evaluator_options,
                jit_compile=bool(args.jit_compile),
            )
            if args.save_evaluators_to is not None:
                save_source_fused_assembler_evaluator(
                    source_fused,
                    args.save_evaluators_to,
                    sector_id=int(args.sector_id),
                    max_eps_order=int(args.source_fused_max_eps_order),
                )
                source_fused_cache_source = "built-and-saved"
        print(
            "source-fused build:",
            {
                "laurent_orders": source_fused.laurent_orders,
                "inputs": len(source_fused.input_names),
                "coefficient_inputs": len(source_fused.coefficient_keys),
                "expression_build_seconds": source_fused.expression_build_seconds,
                "evaluator_build_seconds": source_fused.evaluator_build_seconds,
                "expression_bytes": source_fused.expression_bytes,
                "evaluator_bytes": source_fused.evaluator_bytes,
                "cache_source": source_fused_cache_source,
            },
            flush=True,
        )
        source_rows: list[dict[str, Any]] = []
        for repeat in range(max(int(args.repeats), 1)):
            if fsd_rows:
                coords = np.asarray(fsd_rows[repeat]["coords"], dtype=float)
            elif args.sample is not None:
                coords = np.asarray([[float(value) for value in args.sample]], dtype=float)
            else:
                rng = np.random.default_rng(int(args.seed) + 1_000_003 * int(args.sector_id) + repeat)
                coords = rng.random((max(int(args.points), 1), source_sector.integration_dim), dtype=float)
            coeffs, stats, timing = evaluate_source_fused(
                source_fused,
                source_processor,
                source_sector,
                coords,
            )
            row = {
                "repeat": int(repeat),
                "points": int(coords.shape[0]),
                "coords": coords.tolist(),
                "laurent_orders": list(source_fused.laurent_orders),
                "wall_us_per_sample": float(stats["wall_seconds"] * 1.0e6 / coords.shape[0]),
                "eval_us_per_sample": float(stats["eval_seconds"] * 1.0e6 / coords.shape[0]),
                "python_us_per_sample": float(stats["python_seconds"] * 1.0e6 / coords.shape[0]),
                "precision_counts": timing.precision_counts,
                "coefficients": [complex_json(value) for value in coeffs[0]],
                **stats,
            }
            if fsd_rows:
                fsd_orders = [
                    int(order)
                    for order in (
                        result.get("fsd_style", {}).get("laurent_orders")
                        or data["topology"]["laurent_orders"]
                    )
                ]
                order_to_fsd_index = {int(order): index for index, order in enumerate(fsd_orders)}
                fsd_coeffs = [
                    complex(item["re"], item["im"])
                    for item in fsd_rows[repeat]["coefficients"]
                ]
                diffs = [
                    complex(coeffs[0, index]) - fsd_coeffs[order_to_fsd_index[int(order)]]
                    for index, order in enumerate(source_fused.laurent_orders)
                ]
                row["max_abs_diff_vs_fsd_style"] = float(max(abs(value) for value in diffs))
                row["diffs_vs_fsd_style"] = [complex_json(value) for value in diffs]
            source_rows.append(row)
        result["source_fused"] = {
            "status": "ok",
            "laurent_orders": list(source_fused.laurent_orders),
            "cache_source": source_fused_cache_source,
            "load_evaluators_from": str(args.load_evaluators_from) if args.load_evaluators_from else None,
            "save_evaluators_to": str(args.save_evaluators_to) if args.save_evaluators_to else None,
            "input_count": len(source_fused.input_names),
            "coefficient_input_count": len(source_fused.coefficient_keys),
            "expression_build_seconds": float(source_fused.expression_build_seconds),
            "evaluator_build_seconds": float(source_fused.evaluator_build_seconds),
            "expression_bytes": int(source_fused.expression_bytes),
            "evaluator_bytes": int(source_fused.evaluator_bytes),
            "rows": source_rows,
            "warm_medians": summarize_rows(source_rows),
        }
        print("source-fused warm medians:", result["source_fused"]["warm_medians"], flush=True)

    if args.run_two_stage_fused:
        two_stage_assembler_cache_source = source_fused_cache_source
        if source_fused is None:
            two_stage_assembler_cache_source = "built"
            if args.load_evaluators_from is not None:
                try:
                    print(
                        f"loading two-stage assembler evaluator from {args.load_evaluators_from}...",
                        flush=True,
                    )
                    (
                        source_fused,
                        source_topology,
                        source_sector,
                        source_processor,
                    ) = load_source_fused_assembler_evaluator(
                        args.load_evaluators_from,
                        args.prepared_bundle,
                        args.sector_id,
                        max_eps_order=int(args.two_stage_max_eps_order),
                    )
                    two_stage_assembler_cache_source = "loaded"
                except FileNotFoundError:
                    print("two-stage assembler cache miss; building it", flush=True)
            if source_fused is None:
                print("building source-coefficient fused PSD2 assembler...", flush=True)
                source_fused, source_topology, source_sector, source_processor = build_source_fused_assembler(
                    args.prepared_bundle,
                    args.sector_id,
                    max_eps_order=int(args.two_stage_max_eps_order),
                    evaluator_options=evaluator_options,
                    jit_compile=bool(args.jit_compile),
                )
                if args.save_evaluators_to is not None:
                    save_source_fused_assembler_evaluator(
                        source_fused,
                        args.save_evaluators_to,
                        sector_id=int(args.sector_id),
                        max_eps_order=int(args.two_stage_max_eps_order),
                    )
                    two_stage_assembler_cache_source = "built-and-saved"
        assert source_sector is not None
        source_coefficient_cache_source = "built"
        source_coefficients: SourceCoefficientBuildResult | None = None
        if args.load_evaluators_from is not None:
            try:
                print(
                    f"loading two-stage source coefficient evaluator from {args.load_evaluators_from}...",
                    flush=True,
                )
                source_coefficients = load_source_coefficient_evaluator(
                    args.load_evaluators_from,
                    sector_id=int(args.sector_id),
                    max_eps_order=int(args.two_stage_max_eps_order),
                    expected_coefficient_keys=source_fused.coefficient_keys,
                )
                source_coefficient_cache_source = "loaded"
            except FileNotFoundError:
                print("two-stage source coefficient cache miss; building it", flush=True)
        if source_coefficients is None:
            print("building two-stage source coefficient evaluator...", flush=True)
            source_coefficients = build_source_coefficient_evaluator(
                data,
                source_fused.coefficient_keys,
                max_eps_order=int(args.two_stage_max_eps_order),
                evaluator_options=evaluator_options,
                jit_compile=bool(args.jit_compile),
            )
            if args.save_evaluators_to is not None:
                save_source_coefficient_evaluator(
                    source_coefficients,
                    args.save_evaluators_to,
                    sector_id=int(args.sector_id),
                    max_eps_order=int(args.two_stage_max_eps_order),
                )
                source_coefficient_cache_source = "built-and-saved"
        print(
            "two-stage source build:",
            {
                "coefficient_outputs": len(source_coefficients.coefficient_keys),
                "expression_build_seconds": source_coefficients.expression_build_seconds,
                "evaluator_build_seconds": source_coefficients.evaluator_build_seconds,
                "expression_bytes": source_coefficients.expression_bytes,
                "evaluator_bytes": source_coefficients.evaluator_bytes,
                "source_cache": source_coefficient_cache_source,
                "assembler_cache": two_stage_assembler_cache_source,
            },
            flush=True,
        )
        two_stage_rows: list[dict[str, Any]] = []
        for repeat in range(max(int(args.repeats), 1)):
            if fsd_rows:
                coords = np.asarray(fsd_rows[repeat]["coords"], dtype=float)
            elif args.sample is not None:
                coords = np.asarray([[float(value) for value in args.sample]], dtype=float)
            else:
                rng = np.random.default_rng(int(args.seed) + 1_000_003 * int(args.sector_id) + repeat)
                coords = rng.random((max(int(args.points), 1), source_sector.integration_dim), dtype=float)
            coeffs, stats = evaluate_two_stage_fused(
                source_coefficients,
                source_fused,
                source_sector,
                coords,
            )
            row = {
                "repeat": int(repeat),
                "points": int(coords.shape[0]),
                "coords": coords.tolist(),
                "laurent_orders": list(source_fused.laurent_orders),
                "wall_us_per_sample": float(stats["wall_seconds"] * 1.0e6 / coords.shape[0]),
                "eval_us_per_sample": float(stats["eval_seconds"] * 1.0e6 / coords.shape[0]),
                "python_us_per_sample": float(stats["python_seconds"] * 1.0e6 / coords.shape[0]),
                "source_eval_us_per_sample": float(stats["source_eval_seconds"] * 1.0e6 / coords.shape[0]),
                "assembler_eval_us_per_sample": float(stats["assembler_eval_seconds"] * 1.0e6 / coords.shape[0]),
                "coefficients": [complex_json(value) for value in coeffs[0]],
                **stats,
            }
            if fsd_rows:
                fsd_orders = [
                    int(order)
                    for order in (
                        result.get("fsd_style", {}).get("laurent_orders")
                        or data["topology"]["laurent_orders"]
                    )
                ]
                order_to_fsd_index = {int(order): index for index, order in enumerate(fsd_orders)}
                fsd_coeffs = [
                    complex(item["re"], item["im"])
                    for item in fsd_rows[repeat]["coefficients"]
                ]
                diffs = [
                    complex(coeffs[0, index]) - fsd_coeffs[order_to_fsd_index[int(order)]]
                    for index, order in enumerate(source_fused.laurent_orders)
                ]
                row["max_abs_diff_vs_fsd_style"] = float(max(abs(value) for value in diffs))
                row["diffs_vs_fsd_style"] = [complex_json(value) for value in diffs]
            two_stage_rows.append(row)
        result["two_stage_fused"] = {
            "status": "ok",
            "laurent_orders": list(source_fused.laurent_orders),
            "source_cache": source_coefficient_cache_source,
            "assembler_cache": two_stage_assembler_cache_source,
            "load_evaluators_from": str(args.load_evaluators_from) if args.load_evaluators_from else None,
            "save_evaluators_to": str(args.save_evaluators_to) if args.save_evaluators_to else None,
            "source_coefficient_count": len(source_coefficients.coefficient_keys),
            "source_expression_build_seconds": float(source_coefficients.expression_build_seconds),
            "source_evaluator_build_seconds": float(source_coefficients.evaluator_build_seconds),
            "source_expression_bytes": int(source_coefficients.expression_bytes),
            "source_evaluator_bytes": int(source_coefficients.evaluator_bytes),
            "assembler_expression_build_seconds": float(source_fused.expression_build_seconds),
            "assembler_evaluator_build_seconds": float(source_fused.evaluator_build_seconds),
            "assembler_expression_bytes": int(source_fused.expression_bytes),
            "assembler_evaluator_bytes": int(source_fused.evaluator_bytes),
            "rows": two_stage_rows,
            "warm_medians": summarize_rows(two_stage_rows),
        }
        print("two-stage fused warm medians:", result["two_stage_fused"]["warm_medians"], flush=True)

    if args.run_dual_envelope_source:
        if source_fused is None:
            print("building source-coefficient fused PSD2 assembler...", flush=True)
            source_fused, source_topology, source_sector, source_processor = build_source_fused_assembler(
                args.prepared_bundle,
                args.sector_id,
                max_eps_order=int(args.dual_envelope_max_eps_order),
                evaluator_options=evaluator_options,
                jit_compile=bool(args.jit_compile),
            )
        print("building dual-envelope black-box source context...", flush=True)
        dual_context = build_dual_envelope_source_context(
            data,
            source_fused,
            max_eps_order=int(args.dual_envelope_max_eps_order),
        )
        print(
            "dual-envelope source build:",
            {
                "laurent_orders": dual_context.source_fused.laurent_orders,
                "envelope_shape_len": len(dual_context.envelope_shape),
                "endpoint_groups": len(dual_context.grouped_pairs),
                "scalar_evaluator_build_seconds": dual_context.scalar_evaluator_build_seconds,
                "sector_evaluator_build_seconds": dual_context.sector_evaluator_build_seconds,
                "envelope_shape_build_seconds": dual_context.envelope_shape_build_seconds,
                "dual_evaluator_build_seconds": dual_context.dual_evaluator_build_seconds,
            },
            flush=True,
        )
        dual_rows: list[dict[str, Any]] = []
        for repeat in range(max(int(args.repeats), 1)):
            if fsd_rows:
                coords = np.asarray(fsd_rows[repeat]["coords"], dtype=float)
            elif args.sample is not None:
                coords = np.asarray([[float(value) for value in args.sample]], dtype=float)
            else:
                rng = np.random.default_rng(int(args.seed) + 1_000_003 * int(args.sector_id) + repeat)
                coords = rng.random((max(int(args.points), 1), dual_context.sector.integration_dim), dtype=float)
            coeffs, stats, timing = evaluate_dual_envelope_source(dual_context, coords)
            row = {
                "repeat": int(repeat),
                "points": int(coords.shape[0]),
                "coords": coords.tolist(),
                "laurent_orders": list(dual_context.source_fused.laurent_orders),
                "wall_us_per_sample": float(stats["wall_seconds"] * 1.0e6 / coords.shape[0]),
                "eval_us_per_sample": float(stats["eval_seconds"] * 1.0e6 / coords.shape[0]),
                "python_us_per_sample": float(stats["python_seconds"] * 1.0e6 / coords.shape[0]),
                "source_eval_us_per_sample": float(stats["source_eval_seconds"] * 1.0e6 / coords.shape[0]),
                "source_python_us_per_sample": float(stats["source_python_seconds"] * 1.0e6 / coords.shape[0]),
                "assembler_eval_us_per_sample": float(stats["assembler_eval_seconds"] * 1.0e6 / coords.shape[0]),
                "precision_counts": timing.precision_counts,
                "coefficients": [complex_json(value) for value in coeffs[0]],
                **stats,
            }
            if fsd_rows:
                fsd_orders = [
                    int(order)
                    for order in (
                        result.get("fsd_style", {}).get("laurent_orders")
                        or data["topology"]["laurent_orders"]
                    )
                ]
                order_to_fsd_index = {int(order): index for index, order in enumerate(fsd_orders)}
                fsd_coeffs = [
                    complex(item["re"], item["im"])
                    for item in fsd_rows[repeat]["coefficients"]
                ]
                diffs = [
                    complex(coeffs[0, index]) - fsd_coeffs[order_to_fsd_index[int(order)]]
                    for index, order in enumerate(dual_context.source_fused.laurent_orders)
                ]
                row["max_abs_diff_vs_fsd_style"] = float(max(abs(value) for value in diffs))
                row["diffs_vs_fsd_style"] = [complex_json(value) for value in diffs]
            dual_rows.append(row)
        result["dual_envelope_source"] = {
            "status": "ok",
            "laurent_orders": list(dual_context.source_fused.laurent_orders),
            "envelope_shape_len": len(dual_context.envelope_shape),
            "endpoint_group_count": len(dual_context.grouped_pairs),
            "coefficient_input_count": len(dual_context.source_fused.coefficient_keys),
            "scalar_evaluator_build_seconds": float(dual_context.scalar_evaluator_build_seconds),
            "sector_evaluator_build_seconds": float(dual_context.sector_evaluator_build_seconds),
            "envelope_shape_build_seconds": float(dual_context.envelope_shape_build_seconds),
            "dual_evaluator_build_seconds": float(dual_context.dual_evaluator_build_seconds),
            "assembler_expression_build_seconds": float(dual_context.source_fused.expression_build_seconds),
            "assembler_evaluator_build_seconds": float(dual_context.source_fused.evaluator_build_seconds),
            "assembler_expression_bytes": int(dual_context.source_fused.expression_bytes),
            "assembler_evaluator_bytes": int(dual_context.source_fused.evaluator_bytes),
            "rows": dual_rows,
            "warm_medians": summarize_rows(dual_rows),
        }
        print("dual-envelope source warm medians:", result["dual_envelope_source"]["warm_medians"], flush=True)

    if args.run_derivative_fused:
        if args.derivative_source != "symbolic":
            raise NotImplementedError(
                "PSD2 derivative-fused dual source adapter is not implemented yet; "
                "use --derivative-source symbolic for the generic split benchmark"
            )
        if source_fused is None:
            print("building source-coefficient fused PSD2 assembler seed...", flush=True)
            source_fused, source_topology, source_sector, source_processor = build_source_fused_assembler(
                args.prepared_bundle,
                args.sector_id,
                max_eps_order=int(args.derivative_fused_max_eps_order),
                evaluator_options=evaluator_options,
                jit_compile=bool(args.jit_compile),
            )
        print("building derivative-fused PSD2 evaluators...", flush=True)
        derivative_fused, derivative_sector = build_derivative_fused_evaluators(
            data,
            source_fused,
            max_eps_order=int(args.derivative_fused_max_eps_order),
            evaluator_options=evaluator_options,
            jit_compile=bool(args.jit_compile),
            max_groups=args.derivative_fused_max_groups,
            regular_method=str(args.derivative_fused_regular_method),
        )
        print(
            "derivative-fused build:",
            {
                "derivative_slots": len(derivative_fused.derivative_slots),
                "groups": len(derivative_fused.groups),
                "source_expression_build_seconds": derivative_fused.source_expression_build_seconds,
                "source_evaluator_build_seconds": derivative_fused.source_evaluator_build_seconds,
                "assembler_expression_build_seconds": derivative_fused.assembler_expression_build_seconds,
                "assembler_evaluator_build_seconds": derivative_fused.assembler_evaluator_build_seconds,
                "source_evaluator_bytes": derivative_fused.source_evaluator_bytes,
                "assembler_evaluator_bytes": derivative_fused.assembler_evaluator_bytes,
            },
            flush=True,
        )
        derivative_rows: list[dict[str, Any]] = []
        for repeat in range(max(int(args.repeats), 1)):
            if fsd_rows:
                coords = np.asarray(fsd_rows[repeat]["coords"], dtype=float)
            elif args.sample is not None:
                coords = np.asarray([[float(value) for value in args.sample]], dtype=float)
            else:
                rng = np.random.default_rng(int(args.seed) + 1_000_003 * int(args.sector_id) + repeat)
                coords = rng.random((max(int(args.points), 1), derivative_sector.integration_dim), dtype=float)
            coeffs, stats = evaluate_derivative_fused(derivative_fused, derivative_sector, coords)
            row = {
                "repeat": int(repeat),
                "points": int(coords.shape[0]),
                "coords": coords.tolist(),
                "laurent_orders": list(derivative_fused.laurent_orders),
                "wall_us_per_sample": float(stats["wall_seconds"] * 1.0e6 / coords.shape[0]),
                "eval_us_per_sample": float(stats["eval_seconds"] * 1.0e6 / coords.shape[0]),
                "python_us_per_sample": float(stats["python_seconds"] * 1.0e6 / coords.shape[0]),
                "source_eval_us_per_sample": float(stats["source_eval_seconds"] * 1.0e6 / coords.shape[0]),
                "assembler_eval_us_per_sample": float(stats["assembler_eval_seconds"] * 1.0e6 / coords.shape[0]),
                "coefficients": [complex_json(value) for value in coeffs[0]],
                **stats,
            }
            if fsd_rows:
                fsd_orders = [
                    int(order)
                    for order in (
                        result.get("fsd_style", {}).get("laurent_orders")
                        or data["topology"]["laurent_orders"]
                    )
                ]
                order_to_fsd_index = {int(order): index for index, order in enumerate(fsd_orders)}
                fsd_coeffs = [
                    complex(item["re"], item["im"])
                    for item in fsd_rows[repeat]["coefficients"]
                ]
                diffs = [
                    complex(coeffs[0, index]) - fsd_coeffs[order_to_fsd_index[int(order)]]
                    for index, order in enumerate(derivative_fused.laurent_orders)
                ]
                row["max_abs_diff_vs_fsd_style"] = float(max(abs(value) for value in diffs))
                row["diffs_vs_fsd_style"] = [complex_json(value) for value in diffs]
            derivative_rows.append(row)
        result["derivative_fused"] = {
            "status": "ok",
            "derivative_source": args.derivative_source,
            "laurent_orders": list(derivative_fused.laurent_orders),
            "derivative_slot_count": len(derivative_fused.derivative_slots),
            "group_count": len(derivative_fused.groups),
            "source_expression_build_seconds": derivative_fused.source_expression_build_seconds,
            "source_evaluator_build_seconds": derivative_fused.source_evaluator_build_seconds,
            "assembler_expression_build_seconds": derivative_fused.assembler_expression_build_seconds,
            "assembler_evaluator_build_seconds": derivative_fused.assembler_evaluator_build_seconds,
            "source_expression_bytes": derivative_fused.source_expression_bytes,
            "source_evaluator_bytes": derivative_fused.source_evaluator_bytes,
            "assembler_expression_bytes": derivative_fused.assembler_expression_bytes,
            "assembler_evaluator_bytes": derivative_fused.assembler_evaluator_bytes,
            "rows": derivative_rows,
            "warm_medians": summarize_rows(derivative_rows),
        }
        print("derivative-fused warm medians:", result["derivative_fused"]["warm_medians"], flush=True)

    if not args.skip_fused:
        print("building fused explicit PSD2 evaluator...", flush=True)
        fused: FusedBuildResult | None = None
        try:
            evaluator_orders = (
                {int(order) for order in args.fused_evaluator_orders}
                if args.fused_evaluator_orders is not None
                else None
            )
            if args.load_fused_expressions:
                fused = build_fused_evaluators_from_artifacts(
                    args.artifact_dir,
                    [int(order) for order in data["topology"]["laurent_orders"]],
                    [str(name) for name in data["sector"]["variable_names"]],
                    write_artifacts=not args.no_write_artifacts,
                    jit_compile=bool(args.jit_compile),
                    evaluator_orders=evaluator_orders,
                    evaluator_options=evaluator_options,
                )
            else:
                fused = build_fused_subtracted_evaluator(
                    data,
                    args.artifact_dir,
                    write_artifacts=not args.no_write_artifacts,
                    jit_compile=bool(args.jit_compile),
                    max_terms=int(args.fused_max_terms),
                    max_build_seconds=float(args.fused_max_build_seconds),
                    build_evaluators=not args.skip_fused_evaluator_build,
                    evaluator_orders=evaluator_orders,
                    evaluator_options=evaluator_options,
                )
        except Exception as exc:
            result["fused"] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            print(f"fused build failed: {type(exc).__name__}: {exc}", flush=True)
        if fused is not None:
            fused_rows: list[dict[str, Any]] = []
            if fused.evaluators:
                for repeat in range(max(int(args.repeats), 1)):
                    if fsd_rows:
                        coords = np.asarray(fsd_rows[repeat]["coords"], dtype=float)
                    elif args.sample is not None:
                        coords = np.asarray([[float(value) for value in args.sample]], dtype=float)
                    else:
                        rng = np.random.default_rng(int(args.seed) + 1_000_003 * int(args.sector_id) + repeat)
                        coords = rng.random((max(int(args.points), 1), data["sector"]["integration_dim"]), dtype=float)
                    coeffs, eval_seconds, active_orders = evaluate_fused(fused, coords)
                    row = {
                        "repeat": int(repeat),
                        "points": int(coords.shape[0]),
                        "laurent_orders": list(active_orders),
                        "eval_seconds": float(eval_seconds),
                        "eval_us_per_sample": float(eval_seconds * 1.0e6 / coords.shape[0]),
                        "coefficients": [complex_json(value) for value in coeffs[0]],
                    }
                    if fsd_rows:
                        order_to_fsd_index = {
                            int(order): index
                            for index, order in enumerate(data["topology"]["laurent_orders"])
                        }
                        fsd_coeffs = [
                            complex(item["re"], item["im"])
                            for item in fsd_rows[repeat]["coefficients"]
                        ]
                        diffs = [
                            complex(coeffs[0, index]) - fsd_coeffs[order_to_fsd_index[int(order)]]
                            for index, order in enumerate(active_orders)
                        ]
                        row["max_abs_diff_vs_fsd_style"] = float(max(abs(value) for value in diffs))
                        row["diffs_vs_fsd_style"] = [complex_json(value) for value in diffs]
                    fused_rows.append(row)
            result["fused"] = {
                "status": "ok",
                "generation_seconds": float(fused.generation_seconds),
                "expression_build_seconds": float(fused.expression_build_seconds),
                "evaluator_build_seconds": float(fused.evaluator_build_seconds),
                "compressed_expression_bytes": int(fused.expression_bytes),
                "compressed_evaluator_bytes": int(fused.evaluator_bytes),
                "artifact_dir": str(fused.artifact_dir),
                "evaluator_laurent_orders": list(fused.evaluator_laurent_orders),
                "rows": fused_rows,
                "eval_us_per_sample_median": (
                    float(np.median([row["eval_us_per_sample"] for row in fused_rows[1:] or fused_rows]))
                    if fused_rows
                    else None
                ),
            }
            print("fused summary:", {k: v for k, v in result["fused"].items() if k != "rows"}, flush=True)

    args.results_json.parent.mkdir(parents=True, exist_ok=True)
    args.results_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.results_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
