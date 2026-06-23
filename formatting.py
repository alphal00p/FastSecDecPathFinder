"""Summary, table, JSON, and convention-formatting helpers."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import textwrap
from typing import Any

from colorama import Fore, Style
from prettytable import PrettyTable
from symbolica import E

from cache_utils import formula_cache_dir
from definitions import (
    BenchmarkResult,
    EULER_GAMMA,
    EpsilonExpansion,
    IntegralRequest,
    JsonDict,
    ONELOOP_TO_FEYNMAN,
    SectorIntegrationResult,
    TargetDefinition,
)
from dot_topology import DotTopologyPrintout
from integrand import TopologyDefinition
from sectors_generator import SectorDefinition
from utils import format_complex_uncertainty, format_percent


def maybe_color(text: str, color: str, enabled: bool = True) -> str:
    """Apply ANSI color when enabled."""
    if not enabled:
        return text
    return f"{color}{text}{Style.RESET_ALL}"


def expression_text(expr: Any) -> str:
    """Format a Symbolica expression with graceful option fallback."""
    for kwargs in (
        {"terms_on_new_line": False, "color_top_level_sum": False},
        {"terms_on_new_line": False},
        {},
    ):
        try:
            return str(expr.format(**kwargs))
        except Exception:
            continue
    return str(expr)


def dual_shape_summary(sectors: list[SectorDefinition], preview_limit: int = 20) -> JsonDict:
    """Return compact Taylor-shape statistics without serializing every shape.

    DOT triple-box runs can have thousands of sectors and large sparse Taylor
    shapes.  Keeping every multi-index in ``result.json`` makes selected-sector
    diagnostics hundreds of megabytes even though the summary only needs the
    number and rough size of the distinct shapes.
    """
    counter: Counter[tuple[Any, ...]] = Counter()
    max_entries = 0
    max_rank = 0
    max_total_degree = 0
    shaped_sector_count = 0
    for sector in sectors:
        shape = sector.dual_shape
        if not shape:
            continue
        shaped_sector_count += 1
        max_entries = max(max_entries, len(shape))
        rank = len(shape[0]) if shape else 0
        max_orders = tuple(
            max((int(multi[position]) for multi in shape), default=0)
            for position in range(rank)
        )
        shape_total_degree = max(
            (sum(int(value) for value in multi) for multi in shape),
            default=0,
        )
        shape_key = (rank, len(shape), max_orders, shape_total_degree)
        counter[shape_key] += 1
        max_rank = max(max_rank, rank)
        max_total_degree = max(max_total_degree, shape_total_degree)
    most_common = [
        {
            "rank": shape[0],
            "entries": shape[1],
            "max_orders": list(shape[2]),
            "max_total_degree": shape[3],
            "sector_count": count,
        }
        for shape, count in counter.most_common(preview_limit)
    ]
    return {
        "sector_count_with_shape": shaped_sector_count,
        "unique_shape_count": len(counter),
        "max_entries": max_entries,
        "max_rank": max_rank,
        "max_total_degree": max_total_degree,
        "preview": most_common,
        "preview_limit": int(preview_limit),
    }


def terminal_math_text(expr: Any) -> str:
    """Format a Symbolica expression and use a readable terminal epsilon."""
    return expression_text(expr).replace("eps", "ε")


def affine_expression_text(expansion: EpsilonExpansion) -> str:
    """Format ``base + eps_coeff*eps`` through Symbolica."""
    expr = E(f"({expansion.base:.17g}) + ({expansion.eps_coeff:.17g})*eps")
    return terminal_math_text(expr)


def endpoint_power_text(variable: str, power: EpsilonExpansion) -> str:
    """Format one endpoint power through Symbolica, e.g. ``x^(-1-ε)``."""
    expr = E(
        f"({variable})^(({power.base:.17g}) + ({power.eps_coeff:.17g})*eps)"
    )
    return terminal_math_text(expr)


def monomial_power_text(variable_names: list[str], powers: list[int]) -> str:
    """Render a monomial from variable names and integer powers."""
    factors: list[str] = []
    for name, power in zip(variable_names, powers):
        if power == 0:
            continue
        factors.append(name if power == 1 else f"{name}^{power}")
    return "*".join(factors) if factors else "1"


def kinematic_restrictions(request: IntegralRequest) -> str:
    """Return the human-readable restriction enforced by validation."""
    if request.integral == "dot":
        return "scalar Euclidean DOT topology; unit propagator powers; no FSD contour deformation"
    if request.integral == "triangle":
        if request.mode == "massive":
            return "m > 0 and s < 4 m^2"
        return "m = 0 and s < 0"
    if request.mode == "massive":
        return "m > 0, s12 < 4 m^2, s23 < 4 m^2"
    return "m = 0, s12 < 0, s23 < 0"


def summary_data(
    request: IntegralRequest,
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    benchmark_available: bool,
) -> JsonDict:
    """Build the serializable run summary used for tables and JSON."""
    header = {
        "integral": request.integral,
        "mode": request.mode,
        "family": topology.family,
        "s": request.s,
        "s12": request.s12,
        "s23": request.s23,
        "m": request.m,
        "gamma_scheme": request.gamma_scheme,
        "prefactor_convention": request.prefactor_convention,
        "benchmark": "OneLOopBridge available" if benchmark_available else "unavailable",
        "seed": request.seed,
        "max_iter": request.max_iter,
        "samples_per_iter": request.samples_per_iter,
        "batch_size": request.batch_size,
        "sampling_mode": request.sampling_mode,
        "qmc_shifts": request.qmc_shifts if request.sampling_mode == "qmc" else "n/a",
        "qmc_korobov_alpha": request.qmc_korobov_alpha
        if request.sampling_mode == "qmc"
        else "n/a",
        "target_rel_accuracy_percent": request.target_rel_accuracy,
        "target_rel_error": request.target_rel_error,
        "target_abs_error": request.target_abs_error,
        "target_integration_time_s": request.target_integration_time,
        "bins": request.bins,
        "workers": request.workers,
        "jit_compile_evaluators": request.jit_compile_evaluators,
        "evaluator_compile_mode": request.evaluator_compile_mode,
        "real_evaluator": request.real_evaluator,
        "dual_evaluator_mode": request.dual_evaluator_mode,
        "subtraction_backend": request.subtraction_backend,
        "sector_evaluator_backend": request.sector_evaluator_backend,
        "IBP_reduce_to_log_endpoint": request.ibp_reduce_to_log_endpoint,
        "ibp_power_goal": request.ibp_power_goal,
        "direct_projector_cache_term_threshold": request.direct_projector_cache_term_threshold,
        "direct_projector_cache_override_sectors": getattr(
            topology, "endpoint_projector_direct_cache_override_sectors", 0
        ),
        "direct_projector_cache_override_signatures": getattr(
            topology, "endpoint_projector_direct_cache_override_signatures", 0
        ),
        "force_regular_taylor_formulas": request.force_regular_taylor_formulas,
        "regular_taylor_signature_limit": request.regular_taylor_signature_limit,
        "regular_taylor_formula_volume_limit": request.regular_taylor_formula_volume_limit,
        "regular_taylor_formula_axis_limit": request.regular_taylor_formula_axis_limit,
        "chain_rule_formula_signature_limit": request.chain_rule_formula_signature_limit,
        "chain_rule_formula_output_length_limit": request.chain_rule_formula_output_length_limit,
        "allow_fallback_for_missing_caches": request.allow_fallback_for_missing_caches,
        "runtime_ready": (
            "explicit sector evaluators pregenerated"
            if request.sector_evaluator_backend == "explicit"
            else "two-stage explicit sector evaluators pregenerated"
            if request.sector_evaluator_backend == "two-stage-explicit"
            else
            "recursive endpoint subtraction; coefficient dual evaluators lazy"
            if request.subtraction_backend == "recursive" and request.dual_evaluator_mode == "lazy"
            else "recursive endpoint subtraction; coefficient evaluators pregenerated"
            if request.subtraction_backend == "recursive"
            else "endpoint subtraction formulas pregenerated; coefficient dual evaluators lazy"
            if request.dual_evaluator_mode == "lazy"
            else "all endpoint subtraction and coefficient evaluators pregenerated"
        ),
        "sectors": list(request.sectors) if request.sectors is not None else "all",
        "stability_threshold": request.stability_threshold,
        "medium_precision_stability_threshold": request.medium_precision_stability_threshold,
        "high_precision_stability_threshold": request.high_precision_stability_threshold,
        "stability_precision": request.stability_precision,
        "medium_precision_stability_precision": request.medium_precision_stability_precision,
        "high_precision_stability_precision": request.high_precision_stability_precision,
    }
    if request.dot_file is not None:
        header["dot_file"] = request.dot_file
    parametric = topology.parametric_representation
    parametric_data = {}
    if parametric is not None:
        parametric_data = {
            "loops": parametric.loop_count,
            "propagator_powers": parametric.propagator_powers,
            "D": affine_expression_text(parametric.dimension),
            "Gamma argument": affine_expression_text(parametric.gamma_argument),
            "U exponent": affine_expression_text(parametric.u_exponent),
            "F exponent": affine_expression_text(parametric.f_exponent),
            "parameter weights": parametric.parameter_weight_powers,
            "prefactor": parametric.prefactor_description,
            "prefactor min order": int(getattr(topology, "global_prefactor_min_order", 0)),
            "prefactor coefficients": [
                complex(value) for value in (topology.global_prefactor_coeffs or [])
            ],
            "convention": parametric.convention_description,
        }

    sector_rows = [
        {
            "id": i,
            "name": sector.name,
            "variables": sector.variable_names,
            "map": [
                f"x{j}={expression_text(expr)}"
                for j, expr in enumerate(sector.map_exprs)
            ],
            "regular_jacobian": expression_text(sector.regular_jacobian_expr),
            "numerator": expression_text(sector.numerator_expr),
            "u_monomial": expression_text(sector.u_monomial_expr),
            "f_monomial": expression_text(sector.f_monomial_expr),
            "u_monomial_powers": sector.u_monomial_powers,
            "f_monomial_powers": sector.f_monomial_powers,
            "jacobian_monomial_powers": sector.jacobian_monomial_powers,
            "measure_monomial_powers": sector.measure_monomial_powers,
            "numerator_monomial_powers": sector.numerator_monomial_powers,
            "singular_axes": [sector.variable_names[axis] for axis in sector.singular_axes],
            "endpoint_powers": [
                endpoint_power_text(
                    sector.variable_names[axis],
                    topology.endpoint_power(sector, axis),
                )
                for axis in sector.singular_axes
            ],
            "subtraction": sector.subtraction_type,
            "description": sector.description,
        }
        for i, sector in enumerate(sectors)
    ]
    axis_counter = Counter(len(sector.singular_axes) for sector in sectors)
    f_monomial_counter = Counter(tuple(sector.f_monomial_powers) for sector in sectors)
    u_monomial_counter = Counter(tuple(sector.u_monomial_powers) for sector in sectors)
    pole_counter = Counter(len(sector.singular_axes) for sector in sectors)
    sector_stats = {
        "total_sector_count": len(sectors),
        "count_by_singular_axes": dict(sorted(axis_counter.items())),
        "count_by_f_monomial": {str(key): value for key, value in f_monomial_counter.items()},
        "count_by_u_monomial": {str(key): value for key, value in u_monomial_counter.items()},
        "count_by_endpoint_pole_depth": dict(sorted(pole_counter.items())),
        "max_integration_dimension": max((sector.integration_dim for sector in sectors), default=0),
        "max_laurent_pole_order": -topology.laurent_min_order,
    }
    max_endpoint_taylor_order = max(
        (max(sector.endpoint_taylor_orders) for sector in sectors if sector.endpoint_taylor_orders),
        default=0,
    )
    validation = {
        "sector_count": len(sectors),
        "continuous_dimension": sectors[0].integration_dim if sectors else 0,
        "expected_laurent_orders": topology.expected_laurent_orders,
        "benchmark_available": benchmark_available,
        "kinematic_restrictions": kinematic_restrictions(request),
        "max_endpoint_taylor_order": max_endpoint_taylor_order,
    }
    if request.integral == "dot" and max_endpoint_taylor_order > 0:
        validation["warning"] = (
            "DOT sectors with y^(-n+c*eps), n>1, use IBP-lowered endpoint projectors"
            if request.ibp_power_goal is not None
            else "DOT sectors with y^(-n+c*eps), n>1, use recursive Taylor endpoint projectors"
        )

    return {
        "header": header,
        "symanzik": {
            "U": expression_text(topology.u_expr),
            "F": expression_text(topology.f_expr),
            "parameter_order": topology.evaluator_parameter_order,
            "parameter_values": topology.parameter_values,
            "dual_shape_summary": dual_shape_summary(sectors),
            "dual_evaluator_mode": request.dual_evaluator_mode,
            "formula_cache_dir": str(formula_cache_dir()),
            "dual_evaluator_build_seconds": topology.dual_evaluator_build_seconds,
            "chain_rule_formula_build_seconds": getattr(
                topology,
                "chain_rule_formula_build_seconds",
                0.0,
            ),
            "chain_rule_formula_count": len(getattr(topology, "_chain_rule_formulas", {})),
            "chain_rule_formulas_from_cache": getattr(
                topology,
                "chain_rule_formulas_from_cache",
                0,
            ),
            "chain_rule_formulas_generated": getattr(
                topology,
                "chain_rule_formulas_generated",
                0,
            ),
            "chain_rule_formula_cache_seconds": getattr(
                topology,
                "chain_rule_formula_cache_seconds",
                0.0,
            ),
            "chain_rule_formula_generation_seconds": getattr(
                topology,
                "chain_rule_formula_generation_seconds",
                0.0,
            ),
            "chain_rule_formulas_skipped": getattr(
                topology,
                "chain_rule_formulas_skipped",
                0,
            ),
            "chain_rule_formula_output_length_limit": request.chain_rule_formula_output_length_limit,
            "subtraction_formula_count": len(topology._subtraction_formulas),
            "endpoint_projector_formula_count": len(topology._endpoint_projector_formulas),
            "endpoint_projector_formulas_from_cache": getattr(
                topology,
                "endpoint_projector_formulas_from_cache",
                0,
            ),
            "endpoint_projector_formulas_generated": getattr(
                topology,
                "endpoint_projector_formulas_generated",
                0,
            ),
            "endpoint_projector_formula_cache_seconds": getattr(
                topology,
                "endpoint_projector_formula_cache_seconds",
                0.0,
            ),
            "endpoint_projector_formula_generation_seconds": getattr(
                topology,
                "endpoint_projector_formula_generation_seconds",
                0.0,
            ),
            "direct_projector_cache_term_threshold": request.direct_projector_cache_term_threshold,
            "direct_projector_cache_override_sectors": getattr(
                topology, "endpoint_projector_direct_cache_override_sectors", 0
            ),
            "direct_projector_cache_override_signatures": getattr(
                topology, "endpoint_projector_direct_cache_override_signatures", 0
            ),
            "regular_taylor_formula_count": len(
                getattr(topology, "_regular_taylor_formulas", {})
            ),
            "regular_taylor_formulas_from_cache": getattr(
                topology,
                "regular_taylor_formulas_from_cache",
                0,
            ),
            "regular_taylor_formulas_generated": getattr(
                topology,
                "regular_taylor_formulas_generated",
                0,
            ),
            "regular_taylor_formula_cache_seconds": getattr(
                topology,
                "regular_taylor_formula_cache_seconds",
                0.0,
            ),
            "regular_taylor_formula_generation_seconds": getattr(
                topology,
                "regular_taylor_formula_generation_seconds",
                0.0,
            ),
            "regular_taylor_formulas_from_curated_cache": getattr(
                topology,
                "regular_taylor_formulas_from_curated_cache",
                0,
            ),
            "regular_taylor_formulas_skipped": getattr(
                topology,
                "regular_taylor_formulas_skipped",
                0,
            ),
            "regular_taylor_formula_policy": (
                "curated endpoint projectors and regular Taylor formulas default-on; uncached high-axis formulas guarded"
                if request.subtraction_backend == "projector-formula"
                else "not used by this subtraction backend"
            ),
            "two_stage_sector_formula_count": len(
                getattr(topology, "_two_stage_sector_formulas", {})
            ),
            "two_stage_sector_formula_build_seconds": getattr(
                topology,
                "two_stage_sector_formula_build_seconds",
                0.0,
            ),
            "explicit_sector_formula_count": len(
                getattr(topology, "_explicit_sector_formulas", {})
            ),
            "explicit_sector_formula_build_seconds": getattr(
                topology,
                "explicit_sector_formula_build_seconds",
                0.0,
            ),
            "subtraction_formula_build_seconds": topology.subtraction_formula_build_seconds,
            "parametric": parametric_data,
        },
        "sectors": sector_rows,
        "sector_stats": sector_stats,
        "validation": validation,
    }


def _short_table_text(value: Any, width: int = 72) -> str:
    """Keep generation-report cells single-line and terminal friendly."""
    text = str(value)
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    if width <= 1:
        return "…"
    return text[: width - 1] + "…"


def _ellipsis_table_text(value: Any, width: int = 50, marker: str = "[...]") -> str:
    """Clip one-line table content to ``width`` with an explicit marker."""
    text = " ".join(str(value).split())
    if len(text) <= width:
        return text
    if width <= len(marker):
        return marker[:width]
    return text[: width - len(marker)] + marker


def _wrapped_table_text(value: Any, width: int = 72) -> str:
    """Wrap generation-report cells without dropping diagnostic content."""
    text = " ".join(str(value).split())
    if not text:
        return "-"
    parts = text.split("; ")
    lines: list[str] = []
    for part in parts:
        wrapped = textwrap.wrap(part, width=width, break_long_words=False) or [part]
        lines.extend(wrapped)
    return "\n".join(lines)


def _color_table_cell_lines(value: str, color: str) -> str:
    """Color each physical table-cell line independently.

    PrettyTable renders multiline cells by printing row-continuation separator
    columns between the lines.  If a single ANSI color span crosses the embedded
    newline, those separators inherit the color.  Resetting per line keeps the
    table border and continuation columns uncolored.
    """
    return "\n".join(maybe_color(line, color) for line in str(value).splitlines())


def print_generation_report(request: IntegralRequest, data: JsonDict) -> None:
    """Print consolidated generation timings and cache/fallback statistics."""
    if request.integral != "dot":
        return
    prepared_bundle = data.get("prepared_bundle")
    generation_summary = data.get("generation_timings")
    timings = None
    if prepared_bundle is None:
        try:
            from dot_topology import get_dot_bundle

            timings = get_dot_bundle(request).timings
            generation_summary = timings.to_summary_dict()
        except Exception:
            return
    elif not isinstance(generation_summary, dict):
        return

    if isinstance(prepared_bundle, dict):
        bundle_path = request.output or prepared_bundle.get("output") or "."
        print(
            maybe_color("\nFSD prepared-bundle report", Fore.CYAN)
            + " "
            + maybe_color("(loaded from disk; timings are saved generation metadata)", Fore.YELLOW)
        )
        print(
            maybe_color("bundle:", Fore.CYAN)
            + " "
            + maybe_color(str(Path(bundle_path).expanduser()), Fore.MAGENTA)
        )
    else:
        print(maybe_color("\nFSD generation report", Fore.CYAN))

    headline_table = PrettyTable()
    headline_table.field_names = [
        maybe_color("headline bucket", Fore.CYAN),
        maybe_color("time", Fore.CYAN),
    ]
    if timings is not None:
        headline_items = list(timings.bucket_totals().items())
        total_seconds = timings.total()
    else:
        headline_items = [
            (str(record.get("name")), float(record.get("seconds", 0.0)))
            for record in generation_summary.get("headline", [])
            if isinstance(record, dict)
        ]
        total_seconds = float(generation_summary.get("total", 0.0))
    for name, seconds in headline_items:
        headline_table.add_row([maybe_color(name, Fore.MAGENTA), format_seconds(seconds)])
    headline_table.add_row(
        [
            maybe_color("total recorded generation", Fore.MAGENTA),
            format_seconds(total_seconds),
        ]
    )
    print(headline_table)

    details_table = PrettyTable()
    details_table.field_names = [
        maybe_color("#", Fore.CYAN),
        maybe_color("stage", Fore.CYAN),
        maybe_color("time", Fore.CYAN),
        maybe_color("detail", Fore.CYAN),
    ]
    details_table.align["stage"] = "l"
    details_table.align["detail"] = "l"
    details_table.max_width["stage"] = 34
    details_table.max_width["detail"] = 72
    if timings is not None:
        detail_records = [
            {
                "name": record.name,
                "seconds": record.seconds,
                "detail": record.detail or "-",
            }
            for record in timings.records
        ]
    else:
        detail_records = [
            record
            for record in generation_summary.get("records", generation_summary.get("details", []))
            if isinstance(record, dict)
        ]
    for index, record in enumerate(detail_records, start=1):
        details_table.add_row(
            [
                index,
                _color_table_cell_lines(_wrapped_table_text(record.get("name", "-"), 34), Fore.MAGENTA),
                format_seconds(float(record.get("seconds", 0.0))),
                _wrapped_table_text(record.get("detail") or "-", 72),
            ]
        )
    print(details_table)

    symanzik = data["symanzik"]
    cache_table = PrettyTable()
    cache_table.field_names = [
        maybe_color("artifact", Fore.CYAN),
        maybe_color("ready", Fore.CYAN),
        maybe_color("cache hit", Fore.CYAN),
        maybe_color("fallback gen", Fore.CYAN),
        maybe_color("cache retrieval", Fore.CYAN),
        maybe_color("fallback time", Fore.CYAN),
        maybe_color("skipped / notes", Fore.CYAN),
    ]
    cache_table.align["artifact"] = "l"
    cache_table.align["skipped / notes"] = "l"

    def add_cache_row(
        artifact: str,
        ready: int,
        cache_hits: int,
        generated: int,
        cache_seconds: float,
        generation_seconds: float,
        notes: str = "-",
    ) -> None:
        color = Fore.GREEN if generated == 0 else Fore.YELLOW
        cache_table.add_row(
            [
                maybe_color(artifact, Fore.MAGENTA),
                ready,
                maybe_color(str(cache_hits), Fore.GREEN),
                maybe_color(str(generated), color),
                format_seconds(cache_seconds),
                maybe_color(format_seconds(generation_seconds), color),
                _wrapped_table_text(notes, 64),
            ]
        )

    add_cache_row(
        "endpoint projectors",
        int(symanzik.get("endpoint_projector_formula_count", 0)),
        int(symanzik.get("endpoint_projector_formulas_from_cache", 0)),
        int(symanzik.get("endpoint_projector_formulas_generated", 0)),
        float(symanzik.get("endpoint_projector_formula_cache_seconds", 0.0)),
        float(symanzik.get("endpoint_projector_formula_generation_seconds", 0.0)),
        (
            f"direct overrides: {symanzik.get('direct_projector_cache_override_sectors', 0)} sectors / "
            f"{symanzik.get('direct_projector_cache_override_signatures', 0)} signatures"
            if symanzik.get("direct_projector_cache_override_sectors", 0)
            else "-"
        ),
    )
    add_cache_row(
        "regular Taylor",
        int(symanzik.get("regular_taylor_formula_count", 0)),
        int(symanzik.get("regular_taylor_formulas_from_cache", 0)),
        int(symanzik.get("regular_taylor_formulas_generated", 0)),
        float(symanzik.get("regular_taylor_formula_cache_seconds", 0.0)),
        float(symanzik.get("regular_taylor_formula_generation_seconds", 0.0)),
        (
            f"curated: {symanzik.get('regular_taylor_formulas_from_curated_cache', 0)}; "
            f"skipped: {symanzik.get('regular_taylor_formulas_skipped', 0)}"
        ),
    )
    add_cache_row(
        "chain rule",
        int(symanzik.get("chain_rule_formula_count", 0)),
        int(symanzik.get("chain_rule_formulas_from_cache", 0)),
        int(symanzik.get("chain_rule_formulas_generated", 0)),
        float(symanzik.get("chain_rule_formula_cache_seconds", 0.0)),
        float(symanzik.get("chain_rule_formula_generation_seconds", 0.0)),
        (
            f"skipped: {symanzik.get('chain_rule_formulas_skipped', 0)}; "
            f"output limit: {symanzik.get('chain_rule_formula_output_length_limit', 0)}"
        ),
    )
    print(cache_table)


def print_preintegration_summary(
    request: IntegralRequest,
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    benchmark_available: bool,
    data: JsonDict | None = None,
) -> None:
    """Print colored pre-integration tables for topology and sectors."""
    if data is None:
        data = summary_data(request, topology, sectors, benchmark_available)
    print(maybe_color("\nFSD run summary", Fore.CYAN))

    header = PrettyTable()
    header.field_names = [maybe_color("item", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    for key, value in data["header"].items():
        header.add_row([maybe_color(str(key), Fore.MAGENTA), value])
    print(header)

    symanzik = PrettyTable()
    symanzik.field_names = [maybe_color("object", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    symanzik.add_row([maybe_color("U", Fore.MAGENTA), data["symanzik"]["U"]])
    symanzik.add_row([maybe_color("F", Fore.MAGENTA), data["symanzik"]["F"]])
    symanzik.add_row(["evaluator order", ", ".join(data["symanzik"]["parameter_order"])])
    symanzik.add_row(["parameter values", data["symanzik"]["parameter_values"]])
    symanzik.add_row(["Taylor evaluator mode", data["symanzik"]["dual_evaluator_mode"]])
    symanzik.add_row(["formula cache dir", data["symanzik"].get("formula_cache_dir", "n/a")])
    symanzik.add_row(["Taylor evaluator build time", format_seconds(data["symanzik"]["dual_evaluator_build_seconds"])])
    symanzik.add_row([
        "chain-rule formula build time",
        format_seconds(data["symanzik"].get("chain_rule_formula_build_seconds", 0.0)),
    ])
    symanzik.add_row([
        "chain-rule formula count",
        data["symanzik"].get("chain_rule_formula_count", 0),
    ])
    symanzik.add_row([
        "chain-rule cache hit/generated",
        (
            f"{data['symanzik'].get('chain_rule_formulas_from_cache', 0)} / "
            f"{data['symanzik'].get('chain_rule_formulas_generated', 0)}"
        ),
    ])
    if data["symanzik"].get("chain_rule_formulas_skipped", 0):
        symanzik.add_row([
            "chain-rule formulas skipped",
            data["symanzik"].get("chain_rule_formulas_skipped", 0),
        ])
    if data["symanzik"].get("chain_rule_formula_output_length_limit", 0):
        symanzik.add_row([
            "chain-rule output limit",
            data["symanzik"].get("chain_rule_formula_output_length_limit", 0),
        ])
    symanzik.add_row(["subtraction formula count", data["symanzik"]["subtraction_formula_count"]])
    symanzik.add_row(["endpoint projector count", data["symanzik"]["endpoint_projector_formula_count"]])
    symanzik.add_row([
        "endpoint cache hit/generated",
        (
            f"{data['symanzik'].get('endpoint_projector_formulas_from_cache', 0)} / "
            f"{data['symanzik'].get('endpoint_projector_formulas_generated', 0)}"
        ),
    ])
    symanzik.add_row([
        "direct projector cache threshold",
        data["symanzik"].get("direct_projector_cache_term_threshold", 0),
    ])
    if data["symanzik"].get("direct_projector_cache_override_sectors", 0):
        symanzik.add_row([
            "direct cached projector overrides",
            (
                f"{data['symanzik'].get('direct_projector_cache_override_sectors', 0)} sectors; "
                f"{data['symanzik'].get('direct_projector_cache_override_signatures', 0)} signatures"
            ),
        ])
    symanzik.add_row([
        "regular Taylor formula count",
        data["symanzik"].get("regular_taylor_formula_count", 0),
    ])
    symanzik.add_row([
        "regular Taylor cache hit/generated",
        (
            f"{data['symanzik'].get('regular_taylor_formulas_from_cache', 0)} / "
            f"{data['symanzik'].get('regular_taylor_formulas_generated', 0)}"
        ),
    ])
    symanzik.add_row([
        "curated regular Taylor assets",
        data["symanzik"].get("regular_taylor_formulas_from_curated_cache", 0),
    ])
    if data["symanzik"].get("regular_taylor_formulas_skipped", 0):
        symanzik.add_row([
            "regular Taylor formulas skipped",
            data["symanzik"].get("regular_taylor_formulas_skipped", 0),
        ])
    symanzik.add_row([
        "regular Taylor policy",
        data["symanzik"].get("regular_taylor_formula_policy", "n/a"),
    ])
    symanzik.add_row([
        "subtraction formula build time",
        format_seconds(data["symanzik"]["subtraction_formula_build_seconds"]),
    ])
    for key, value in data["symanzik"]["parametric"].items():
        symanzik.add_row([key, value])
    shape_summary = data["symanzik"].get("dual_shape_summary")
    if isinstance(shape_summary, dict):
        shape_text = (
            f"{shape_summary.get('unique_shape_count', 0)} unique; "
            f"max entries={shape_summary.get('max_entries', 0)}, "
            f"rank={shape_summary.get('max_rank', 0)}, "
            f"degree={shape_summary.get('max_total_degree', 0)}"
        )
    else:
        unique_shapes = []
        for shape in data["symanzik"].get("dual_shapes", []):
            if shape not in unique_shapes:
                unique_shapes.append(shape)
        shape_text = unique_shapes if unique_shapes else "none"
    symanzik.add_row(["U/F Taylor shapes", shape_text])
    print(symanzik)

    print_generation_report(request, data)

    sector_table = PrettyTable()
    sector_table.field_names = [
        maybe_color("id", Fore.CYAN),
        maybe_color("sector", Fore.CYAN),
        maybe_color("vars", Fore.CYAN),
        maybe_color("map", Fore.CYAN),
        maybe_color("J_reg", Fore.CYAN),
        maybe_color("M_U", Fore.CYAN),
        maybe_color("M_F", Fore.CYAN),
        maybe_color("axes", Fore.CYAN),
        maybe_color("endpoint powers", Fore.CYAN),
        maybe_color("subtraction", Fore.CYAN),
    ]
    sector_table.align = "l"
    sector_cap = 20
    shown_sectors = data["sectors"][:sector_cap]
    if len(data["sectors"]) > sector_cap:
        print(maybe_color(f"showing {sector_cap}/{len(data['sectors'])} sectors", Fore.YELLOW))
    for sector in shown_sectors:
        sector_table.add_row(
            [
                sector["id"],
                maybe_color(sector["name"], Fore.MAGENTA),
                ", ".join(sector["variables"]),
                "\n".join(sector["map"]),
                sector["regular_jacobian"],
                sector["u_monomial"],
                sector["f_monomial"],
                ", ".join(sector["singular_axes"]) if sector["singular_axes"] else "-",
                ", ".join(sector["endpoint_powers"]) if sector["endpoint_powers"] else "-",
                sector["subtraction"],
            ]
        )
    print(sector_table)

    stats_table = PrettyTable()
    stats_table.field_names = [maybe_color("sector statistic", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    for key, value in data["sector_stats"].items():
        stats_table.add_row([maybe_color(str(key), Fore.MAGENTA), _ellipsis_table_text(value, 50)])
    print(stats_table)

    validation = PrettyTable()
    validation.field_names = [maybe_color("check", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    for key, value in data["validation"].items():
        validation.add_row([maybe_color(str(key), Fore.MAGENTA), value])
    print(validation)


def dot_placeholder_summary_data(printout: DotTopologyPrintout) -> JsonDict:
    """Build serializable summary data for a DOT topology placeholder."""
    return printout.to_dict()


def print_dot_placeholder_summary(printout: DotTopologyPrintout) -> None:
    """Print the generic DOT topology/sector summary scaffold."""
    print(maybe_color("\nFSD DOT topology placeholder", Fore.CYAN))

    header = PrettyTable()
    header.field_names = [maybe_color("item", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    for key, value in printout.header_rows():
        header.add_row([maybe_color(key, Fore.MAGENTA), value])
    print(header)

    topology = PrettyTable()
    topology.field_names = [maybe_color("topology field", Fore.CYAN), maybe_color("planned value", Fore.CYAN)]
    topology.align = "l"
    for key, value in printout.topology_rows():
        topology.add_row([maybe_color(key, Fore.MAGENTA), value])
    print(topology)

    sector = PrettyTable()
    sector.field_names = [
        maybe_color("sector field", Fore.CYAN),
        maybe_color("symbol", Fore.CYAN),
        maybe_color("purpose", Fore.CYAN),
    ]
    sector.align = "l"
    for name, symbol, purpose in printout.sector_schema_rows():
        sector.add_row([maybe_color(name, Fore.MAGENTA), symbol, purpose])
    print(sector)

    validation = PrettyTable()
    validation.field_names = [maybe_color("gate", Fore.CYAN), maybe_color("status", Fore.CYAN)]
    for key, value in printout.validation_rows():
        validation.add_row([maybe_color(key, Fore.MAGENTA), value])
    print(validation)


def error_component_norm(error: complex) -> float:
    """Use the larger real/imaginary standard error as a scalar norm."""
    err = complex(error)
    return max(abs(err.real), abs(err.imag))


def max_relative_error(coeffs: list[complex], errors: list[complex]) -> float:
    """Return the largest coefficient-wise relative error."""
    tiny = 1.0e-30
    rel_errors: list[float] = []
    for coeff, error in zip(coeffs, errors):
        err_norm = error_component_norm(error)
        coeff_norm = abs(complex(coeff))
        if err_norm == 0.0:
            rel_errors.append(0.0)
        elif coeff_norm > tiny:
            rel_errors.append(err_norm / coeff_norm)
        else:
            rel_errors.append(float("inf"))
    return max(rel_errors) if rel_errors else float("nan")


def summed_relative_error_percent(coeffs: list[complex], errors: list[complex]) -> float:
    """Return SUM(|errors|)/SUM(|coefficients|) as a percentage."""
    numerator = sum(abs(complex(error)) for error in errors)
    denominator = sum(abs(complex(coeff)) for coeff in coeffs)
    if numerator == 0.0:
        return 0.0
    if denominator <= 1.0e-30:
        return float("inf")
    return 100.0 * numerator / denominator


def relative_error_percent(coeff: complex, error: complex) -> float:
    """Return one coefficient's component-norm relative MC error in percent."""
    err_norm = error_component_norm(error)
    coeff_norm = abs(complex(coeff))
    if err_norm == 0.0:
        return 0.0
    if coeff_norm <= 1.0e-30:
        return float("inf")
    return 100.0 * err_norm / coeff_norm


def pull_value(diff: complex | None, err: complex) -> float | None:
    """Return max component pull between a result and benchmark."""
    if diff is None:
        return None
    tiny = 1.0e-12
    pulls: list[float] = []
    if err.real > 0.0:
        pulls.append(abs(diff.real) / err.real)
    elif abs(diff.real) > tiny:
        pulls.append(float("inf"))
    if err.imag > 0.0:
        pulls.append(abs(diff.imag) / err.imag)
    elif abs(diff.imag) > tiny:
        pulls.append(float("inf"))
    return max(pulls) if pulls else 0.0


def combine_uncorrelated_errors(left: complex, right: complex) -> complex:
    """Combine two complex component-wise one-sigma errors in quadrature."""
    return complex(
        math.hypot(float(complex(left).real), float(complex(right).real)),
        math.hypot(float(complex(left).imag), float(complex(right).imag)),
    )


def apply_global_convention(
    sector_coeffs: list[complex],
    sector_errors: list[complex],
    request: IntegralRequest,
    dot_global_prefactor_coeffs: list[complex] | tuple[complex, ...] | None = None,
    dot_global_prefactor_min_order: int | None = None,
) -> tuple[list[complex], list[complex]]:
    """Apply gamma-scheme and stripped-convention shifts to sector coefficients."""
    def convolve_regular_factor(
        coeffs_in: list[complex],
        errors_in: list[complex],
        factor_coeffs: list[complex],
    ) -> tuple[list[complex], list[complex]]:
        """Multiply a Laurent array by a regular epsilon series.

        Coefficients are stored from the deepest pole to the requested maximum
        order.  ``factor_coeffs[n]`` is the coefficient of ``eps^n``.
        """
        coeffs_out = [0.0 + 0.0j for _ in coeffs_in]
        errors_out = [0.0 + 0.0j for _ in errors_in]
        for coeff_index, coeff in enumerate(coeffs_in):
            for factor_index, factor_coeff in enumerate(factor_coeffs):
                out_index = coeff_index + factor_index
                if out_index >= len(coeffs_out):
                    break
                coeffs_out[out_index] += factor_coeff * coeff
                errors_out[out_index] += abs(factor_coeff) * errors_in[coeff_index]
        return coeffs_out, errors_out

    def convolve_laurent_factor(
        coeffs_in: list[complex],
        errors_in: list[complex],
        factor_min_order: int,
        factor_coeffs: list[complex],
    ) -> tuple[list[complex], list[complex]]:
        """Multiply by a Laurent prefactor and keep the displayed order window."""
        raw_min_order = (
            int(request.dot_sector_laurent_min_order)
            if request.dot_sector_laurent_min_order is not None
            else int(request.max_eps_order) - len(coeffs_in) + 1
        )
        raw_max_order = raw_min_order + len(coeffs_in) - 1
        display_min_order = raw_min_order + int(factor_min_order)
        display_max_order = int(request.max_eps_order)
        if (
            request.command == "integrate"
            and not request.max_eps_order_explicit
            and request.dot_sector_laurent_max_order is not None
        ):
            display_max_order = int(request.dot_sector_laurent_max_order) + int(factor_min_order)
        display_max_order = min(display_max_order, raw_max_order + int(factor_min_order) + len(factor_coeffs) - 1)
        if display_max_order < display_min_order:
            return [], []
        coeffs_out = [
            0.0 + 0.0j
            for _ in range(display_max_order - display_min_order + 1)
        ]
        errors_out = [
            0.0 + 0.0j
            for _ in range(display_max_order - display_min_order + 1)
        ]
        for coeff_index, coeff in enumerate(coeffs_in):
            raw_order = raw_min_order + coeff_index
            for factor_index, factor_coeff in enumerate(factor_coeffs):
                factor_order = int(factor_min_order) + factor_index
                out_order = raw_order + factor_order
                if out_order < display_min_order or out_order > display_max_order:
                    continue
                out_index = out_order - display_min_order
                coeffs_out[out_index] += factor_coeff * coeff
                errors_out[out_index] += abs(factor_coeff) * errors_in[coeff_index]
        return coeffs_out, errors_out

    if request.integral == "dot":
        if request.prefactor_convention == "pysecdec":
            prefactor = (
                list(dot_global_prefactor_coeffs)
                if dot_global_prefactor_coeffs is not None
                else list(request.dot_global_prefactor_coeffs or [])
            )
            if not prefactor:
                from dot_topology import get_dot_bundle

                bundle_topology = get_dot_bundle(request).topology
                prefactor = bundle_topology.global_prefactor_coeffs or [1.0 + 0.0j]
                prefactor_min_order = int(getattr(bundle_topology, "global_prefactor_min_order", 0))
            else:
                prefactor_min_order = (
                    int(dot_global_prefactor_min_order)
                    if dot_global_prefactor_min_order is not None
                    else int(request.dot_global_prefactor_min_order)
                )
            return convolve_laurent_factor(
                sector_coeffs,
                sector_errors,
                prefactor_min_order,
                prefactor,
            )
        return sector_coeffs[:], sector_errors[:]

    if request.integral == "box":
        if request.mode == "massless":
            return convolve_regular_factor(
                sector_coeffs,
                sector_errors,
                [1.0 + 0.0j, 1.0 + 0.0j, (math.pi * math.pi / 6.0) + 0.0j],
            )
        return sector_coeffs[:], sector_errors[:]

    if request.gamma_scheme == "full":
        g2 = 0.5 * EULER_GAMMA * EULER_GAMMA + math.pi * math.pi / 12.0
        signed = [-coeff for coeff in sector_coeffs]
        return convolve_regular_factor(
            signed,
            sector_errors,
            [1.0 + 0.0j, -EULER_GAMMA + 0.0j, g2 + 0.0j],
        )

    coeffs = [-coeff for coeff in sector_coeffs]
    errors = sector_errors[:]
    if request.mode == "massless":
        return convolve_regular_factor(
            coeffs,
            errors,
            [1.0 + 0.0j, 0.0 + 0.0j, (math.pi * math.pi / 6.0) + 0.0j],
        )
    return coeffs, errors


def selected_prefactor_values(
    request: IntegralRequest,
    raw_coeffs: list[complex],
    raw_errors: list[complex],
    benchmark: BenchmarkResult | None,
) -> tuple[list[complex], list[complex], list[complex], complex]:
    """Select raw or Feynman-normalized display coefficients."""
    factor = benchmark.factor if benchmark is not None else ONELOOP_TO_FEYNMAN
    bench_raw = list(benchmark.raw) if benchmark is not None else [0.0 + 0.0j for _ in raw_coeffs]
    if len(bench_raw) < len(raw_coeffs):
        bench_raw = [0.0 + 0.0j for _ in range(len(raw_coeffs) - len(bench_raw))] + bench_raw
    elif len(bench_raw) > len(raw_coeffs):
        bench_raw = bench_raw[-len(raw_coeffs) :]
    if request.prefactor_convention in {"sector", "pysecdec"}:
        return raw_coeffs, raw_errors, bench_raw, 1.0
    if request.prefactor_convention == "feynman":
        return (
            [factor * c for c in raw_coeffs],
            [abs(factor) * e for e in raw_errors],
            [factor * value for value in bench_raw],
            factor,
        )
    return raw_coeffs, raw_errors, bench_raw, factor


def laurent_labels_for_coefficients(request: IntegralRequest, count: int) -> list[str]:
    """Return contiguous Laurent labels ending at the requested max order."""
    min_order = request.max_eps_order - count + 1
    return [f"eps^{order}" for order in range(min_order, request.max_eps_order + 1)]


def display_laurent_labels(
    request: IntegralRequest,
    raw_count: int,
    summary: JsonDict | None = None,
) -> list[str]:
    """Return labels for coefficients after the selected prefactor convention."""
    if request.integral == "dot" and request.prefactor_convention == "pysecdec":
        raw_min_order = (
            int(request.dot_sector_laurent_min_order)
            if request.dot_sector_laurent_min_order is not None
            else int(request.max_eps_order) - raw_count + 1
        )
        display_min_order = raw_min_order + int(request.dot_global_prefactor_min_order)
        display_max_order = int(request.max_eps_order)
        if (
            request.command == "integrate"
            and not request.max_eps_order_explicit
            and request.dot_sector_laurent_max_order is not None
        ):
            display_max_order = (
                int(request.dot_sector_laurent_max_order)
                + int(request.dot_global_prefactor_min_order)
            )
        return [
            f"eps^{order}"
            for order in range(display_min_order, display_max_order + 1)
        ]
    labels = list((summary or {}).get("validation", {}).get("expected_laurent_orders", []))
    if len(labels) == raw_count:
        return labels
    return laurent_labels_for_coefficients(request, raw_count)


def format_complex(z: complex, digits: int = 8) -> str:
    """Format a complex number for benchmark columns."""
    c = complex(z)
    if abs(c.imag) < 5.0e-15:
        return f"{c.real:.{digits}e}"
    sign = "+" if c.imag >= 0 else "-"
    return f"{c.real:.{digits}e}{sign}{abs(c.imag):.{digits}e}i"


def format_complex_error(z: complex, digits: int = 2) -> str:
    """Format a complex error for verbose stats output."""
    c = complex(z)
    if c.imag == float("inf") or c.real == float("inf"):
        return "inf"
    if abs(c.imag) < 5.0e-15:
        return f"{c.real:.{digits}e}"
    return f"({c.real:.{digits}e}, {c.imag:.{digits}e})"


def format_complex_with_error(value: complex, error: complex) -> str:
    """Format a coefficient with MC uncertainty in parenthesis notation."""
    return format_complex_uncertainty(value, error)


def format_seconds(value: float) -> str:
    """Choose a readable unit for a timing value."""
    seconds = float(value)
    if not math.isfinite(seconds):
        return str(seconds)
    if abs(seconds) < 1.0e-3 and seconds != 0.0:
        return f"{seconds * 1.0e6:.2f} μs"
    if abs(seconds) < 1.0:
        return f"{seconds * 1.0e3:.2f} ms"
    if abs(seconds) < 10.0:
        return f"{seconds:.3f}s"
    if abs(seconds) < 100.0:
        return f"{seconds:.2f}s"
    return f"{seconds:.1f}s"


def colored_kv(label: str, value: str, color: str) -> str:
    """Return a colored ``label=value`` pair."""
    return maybe_color(label, color) + "=" + maybe_color(value, color)


def compare_pull(diff: complex | None, err: complex) -> tuple[str, str]:
    """Return formatted pull text and row color."""
    pull = pull_value(diff, err)
    if pull is None:
        return "N/A", Fore.WHITE
    if pull <= 2.0:
        return f"{pull:.2f}σ", Fore.GREEN
    if pull <= 5.0:
        return f"{pull:.2f}σ", Fore.YELLOW
    return f"{pull:.2f}σ", Fore.RED


def precision_stats_summary(
    counts: dict[str, int],
    total_samples: int,
    request: IntegralRequest,
) -> JsonDict:
    """Build JSON-friendly precision-escalation statistics."""
    total = max(int(total_samples), 0)

    def fraction(key: str) -> float:
        if total <= 0:
            return 0.0
        return float(counts.get(key, 0)) / float(total)

    return {
        "total_samples": total,
        "ordinary": {
            "samples": int(counts.get("ordinary", 0)),
            "fraction": fraction("ordinary"),
        },
        "stability": {
            "samples": int(counts.get("stability", 0)),
            "fraction": fraction("stability"),
            "threshold": request.stability_threshold,
            "precision_digits": request.stability_precision,
        },
        "medium_precision": {
            "samples": int(counts.get("medium_precision", 0)),
            "fraction": fraction("medium_precision"),
            "threshold": request.medium_precision_stability_threshold,
            "precision_digits": request.medium_precision_stability_precision,
        },
        "high_precision": {
            "samples": int(counts.get("high_precision", 0)),
            "fraction": fraction("high_precision"),
            "threshold": request.high_precision_stability_threshold,
            "precision_digits": request.high_precision_stability_precision,
        },
    }


def make_output(
    request: IntegralRequest,
    raw_coeffs: list[complex],
    raw_errors: list[complex],
    target: TargetDefinition | None,
    samples: int,
    elapsed_seconds: float,
    avg_eval_us_per_sample_per_worker: float,
    eval_seconds: float,
    python_seconds: float,
    havana_seconds: float,
    python_overhead_fraction: float,
    summary: JsonDict,
    precision_counts: dict[str, int] | None = None,
    sector_results: list[JsonDict] | None = None,
    interrupted: bool = False,
) -> JsonDict:
    """Assemble final output data before printing or JSON serialization."""
    display_coeffs, display_errors, _display_bench, factor = selected_prefactor_values(
        request, raw_coeffs, raw_errors, None
    )
    labels = display_laurent_labels(request, len(raw_coeffs), summary)
    benchmark_available = target is not None
    target_coeffs = target.coefficients if target is not None else [0.0 + 0.0j for _ in raw_coeffs]
    target_errors = target.errors if target is not None else [0.0 + 0.0j for _ in raw_coeffs]
    diffs = [
        coeff - ref if target is not None else None
        for coeff, ref in zip(display_coeffs, target_coeffs)
    ]
    comparison_errors = [
        combine_uncorrelated_errors(error, target_error)
        for error, target_error in zip(display_errors, target_errors)
    ]
    pulls = [
        pull_value(diff, err) if diff is not None else None
        for diff, err in zip(diffs, comparison_errors)
    ]
    return {
        "schema_version": 1,
        "integral": request.integral,
        "mode": request.mode,
        "gamma_scheme": request.gamma_scheme,
        "prefactor_convention": request.prefactor_convention,
        "to_feynman": factor,
        "samples": samples,
        "elapsed_seconds": elapsed_seconds,
        "avg_eval_us_per_sample_per_worker": avg_eval_us_per_sample_per_worker,
        "eval_seconds": eval_seconds,
        "python_seconds": python_seconds,
        "havana_seconds": havana_seconds,
        "integrator_seconds": havana_seconds,
        "dual_evaluator_build_seconds": summary.get("symanzik", {}).get(
            "dual_evaluator_build_seconds", 0.0
        ),
        "chain_rule_formula_build_seconds": summary.get("symanzik", {}).get(
            "chain_rule_formula_build_seconds", 0.0
        ),
        "chain_rule_formula_count": summary.get("symanzik", {}).get(
            "chain_rule_formula_count", 0
        ),
        "chain_rule_formulas_skipped": summary.get("symanzik", {}).get(
            "chain_rule_formulas_skipped", 0
        ),
        "python_overhead_fraction": python_overhead_fraction,
        "precision_stats": precision_stats_summary(precision_counts or {}, samples, request),
        "interrupted": interrupted,
        "benchmark_available": benchmark_available,
        "summary": summary,
        "laurent_labels": labels,
        "target": {
            "source": target.source if target is not None else "none",
            "convention": target.convention if target is not None else request.prefactor_convention,
            "coefficients": target_coeffs if target is not None else [],
            "errors": target_errors if target is not None else [],
            "metadata": target.metadata if target is not None else {},
        },
        "raw": {
            "coefficients": raw_coeffs,
            "errors": raw_errors,
        },
        "display": {
            "coefficients": display_coeffs,
            "errors": display_errors,
            "benchmark": target_coeffs,
        },
        "aggregate_results": {
            "labels": labels,
            "raw": {"coefficients": raw_coeffs, "errors": raw_errors},
            "display": {"coefficients": display_coeffs, "errors": display_errors},
            "target": {"coefficients": target_coeffs if target is not None else [], "errors": target_errors if target is not None else []},
            "comparison_errors": comparison_errors if target is not None else [],
            "diff": diffs,
            "pull": pulls,
        },
        "sector_results": sector_results or [],
    }


def build_sector_result_rows(
    request: IntegralRequest,
    sectors: list[SectorDefinition],
    per_sector: list[SectorIntegrationResult],
) -> list[JsonDict]:
    """Build JSON-ready per-sector coefficient summaries."""
    rows: list[JsonDict] = []
    sector_by_id = {i: sector for i, sector in enumerate(sectors)}
    for result in per_sector:
        sector = sector_by_id.get(result.sector_id)
        raw_coeffs, raw_errors = apply_global_convention(
            result.raw_sector_coeffs,
            result.raw_sector_errors,
            request,
        )
        display_coeffs, display_errors, _bench, _factor = selected_prefactor_values(
            request,
            raw_coeffs,
            raw_errors,
            None,
        )
        rows.append(
            {
                "sector_id": result.sector_id,
                "name": result.sector_name,
                "samples": result.samples,
                "precision_stats": precision_stats_summary(
                    result.precision_counts,
                    result.samples,
                    request,
                ),
                "singular_axes": [
                    sector.variable_names[axis] for axis in sector.singular_axes
                ] if sector is not None else [],
                "raw": {
                    "coefficients": raw_coeffs,
                    "errors": raw_errors,
                },
                "raw_sector": {
                    "coefficients": result.raw_sector_coeffs,
                    "errors": result.raw_sector_errors,
                },
                "display": {
                    "coefficients": display_coeffs,
                    "errors": display_errors,
                },
                "diagnostics": result.diagnostics,
                "sort_keys": {
                    "abs_central": max((abs(value) for value in display_coeffs), default=0.0),
                    "abs_error": max((abs(value) for value in display_errors), default=0.0),
                },
            }
        )
    return rows


def print_result_table(output: JsonDict) -> None:
    """Print the final coefficient comparison table and timing footer."""
    labels = output.get("laurent_labels", ["eps^-2", "eps^-1", "eps^0"])
    convention = output["prefactor_convention"]
    table = PrettyTable()
    table.field_names = [
        maybe_color("coeff", Fore.CYAN),
        maybe_color(f"FSD {convention}", Fore.CYAN),
        maybe_color("MC err", Fore.CYAN),
        maybe_color(f"benchmark {convention}", Fore.CYAN),
        maybe_color(f"diff {convention}", Fore.CYAN),
        maybe_color("pull", Fore.CYAN),
    ]
    display = output["display"]
    benchmark_available = bool(output.get("benchmark_available", True))
    for idx, label in enumerate(labels):
        fsd_value = display["coefficients"][idx]
        err = display["errors"][idx]
        bench_value = display["benchmark"][idx] if benchmark_available else None
        target_errors = output.get("target", {}).get("errors", [])
        target_err = target_errors[idx] if idx < len(target_errors) else 0.0 + 0.0j
        comparison_err = combine_uncorrelated_errors(err, target_err)
        diff = fsd_value - bench_value if bench_value is not None else None
        pull_text, row_color = compare_pull(diff, comparison_err)
        table.add_row(
            [
                maybe_color(label, Fore.MAGENTA),
                maybe_color(format_complex_with_error(fsd_value, err), row_color),
                format_percent(relative_error_percent(fsd_value, err)),
                format_complex(bench_value) if bench_value is not None else maybe_color("N/A", Fore.WHITE),
                maybe_color(format_complex_with_error(diff, comparison_err), row_color) if diff is not None else maybe_color("N/A", Fore.WHITE),
                maybe_color(pull_text, row_color),
            ]
        )
    print(table)
    if output.get("interrupted"):
        print(maybe_color("Run interrupted: partial accumulated result was written.", Fore.YELLOW))
    print(
        maybe_color("Legend:", Fore.CYAN)
        + " convention "
        + maybe_color(convention, Fore.MAGENTA)
        + "; "
        + maybe_color("FSD", Fore.GREEN)
        + " and "
        + maybe_color("diff", Fore.GREEN)
        + " use two-significant-digit MC parentheses; "
        + maybe_color("MC err", Fore.YELLOW)
        + " is relative 1σ in percent; "
        + maybe_color("pull", Fore.MAGENTA)
        + " uses combined FSD/reference 1σ errors when available; N/A means no reference backend was run."
    )
    total_timing = max(
        output["eval_seconds"] + output["python_seconds"] + output["havana_seconds"],
        1.0e-300,
    )
    eval_percent = 100.0 * output["eval_seconds"] / total_timing
    python_percent = 100.0 * output["python_seconds"] / total_timing
    integrator_percent = 100.0 * output["havana_seconds"] / total_timing
    print(
        maybe_color("Timing:", Fore.CYAN)
        + " "
        + colored_kv("EvalT", format_seconds(output["eval_seconds"]), Fore.GREEN)
        + "  "
        + colored_kv("PythonT", format_seconds(output["python_seconds"]), Fore.YELLOW)
        + "  "
        + colored_kv("IntegratorT", format_seconds(output["havana_seconds"]), Fore.BLUE)
        + "  "
        + colored_kv("TaylorGen", format_seconds(output["dual_evaluator_build_seconds"]), Fore.CYAN)
        + "  "
        + colored_kv("ChainGen", format_seconds(output.get("chain_rule_formula_build_seconds", 0.0)), Fore.CYAN)
        + "  "
        + colored_kv("avg", f"{output['avg_eval_us_per_sample_per_worker']:.3g} μs/smpl/wkr", Fore.MAGENTA)
    )
    print(
        maybe_color("Profile:", Fore.CYAN)
        + " "
        + maybe_color(f"{python_percent:.2f}% python", Fore.YELLOW)
        + " | "
        + maybe_color(f"{eval_percent:.2f}% evaluator", Fore.GREEN)
        + " | "
        + maybe_color(f"{integrator_percent:.2f}% integrator", Fore.BLUE)
        + "    "
        + maybe_color("Status:", Fore.CYAN)
        + " "
        + maybe_color("<=2σ", Fore.GREEN)
        + " / "
        + maybe_color("<=5σ", Fore.YELLOW)
        + " / "
        + maybe_color(">5σ", Fore.RED)
    )
    precision = output.get("precision_stats", {})
    if precision:
        stability = precision.get("stability", {})
        medium = precision.get("medium_precision", {})
        high = precision.get("high_precision", {})
        ordinary = precision.get("ordinary", {})
        print(
            maybe_color("Precision:", Fore.CYAN)
            + " "
            + colored_kv(
                "ordinary",
                f"{ordinary.get('samples', 0)} ({format_percent(100.0 * float(ordinary.get('fraction', 0.0)))})",
                Fore.GREEN,
            )
            + "  "
            + colored_kv(
                f"prec{stability.get('precision_digits', '?')}",
                f"{stability.get('samples', 0)} ({format_percent(100.0 * float(stability.get('fraction', 0.0)))})",
                Fore.YELLOW,
            )
            + "  "
            + colored_kv(
                f"prec{medium.get('precision_digits', '?')}",
                f"{medium.get('samples', 0)} ({format_percent(100.0 * float(medium.get('fraction', 0.0)))})",
                Fore.MAGENTA,
            )
            + "  "
            + colored_kv(
                f"prec{high.get('precision_digits', '?')}",
                f"{high.get('samples', 0)} ({format_percent(100.0 * float(high.get('fraction', 0.0)))})",
                Fore.RED,
            )
        )
    generation = output.get("summary", {}).get("generation_timings")
    if isinstance(generation, dict) and generation.get("headline"):
        parts = []
        for record in generation["headline"]:
            if isinstance(record, dict):
                parts.append(
                    colored_kv(str(record.get("name")), format_seconds(float(record.get("seconds", 0.0))), Fore.MAGENTA)
                )
        total = generation.get("total")
        if total is not None:
            parts.append(colored_kv("total", format_seconds(float(total)), Fore.CYAN))
        if parts:
            prepared_bundle = output.get("summary", {}).get("prepared_bundle")
            if isinstance(prepared_bundle, dict):
                label = "Precomputed generation"
                suffix = maybe_color(" (loaded bundle)", Fore.YELLOW)
            else:
                label = "Generation"
                suffix = ""
            print(maybe_color(f"{label}:", Fore.CYAN) + suffix + " " + "  ".join(parts))


def json_default(obj: Any) -> Any:
    """JSON serializer for complex numbers and fallback objects."""
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    return str(obj)


def output_json(output: JsonDict) -> str:
    """Return pretty JSON output for machine-readable runs."""
    return json.dumps(output, default=json_default, indent=2)
