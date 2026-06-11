"""Summary, table, JSON, and convention-formatting helpers."""

from __future__ import annotations

import json
import math
from typing import Any

from colorama import Fore, Style
from prettytable import PrettyTable

from definitions import BenchmarkResult, EULER_GAMMA, IntegralRequest, JsonDict, ONELOOP_TO_FEYNMAN
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
    return {
        "header": {
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
            "target_rel_accuracy_percent": request.target_rel_accuracy,
            "bins": request.bins,
            "workers": request.workers,
            "jit_compile_evaluators": request.jit_compile_evaluators,
        },
        "symanzik": {
            "U": expression_text(topology.u_expr),
            "F": expression_text(topology.f_expr),
            "parameter_order": topology.evaluator_parameter_order,
            "parameter_values": topology.parameter_values,
            "dual_shapes": [sector.dual_shape for sector in sectors if sector.dual_shape],
        },
        "sectors": [
            {
                "id": i,
                "name": sector.name,
                "variables": sector.variable_names,
                "map": [
                    f"x{j}={expression_text(expr)}"
                    for j, expr in enumerate(sector.map_exprs)
                ],
                "regular_jacobian": expression_text(sector.regular_jacobian_expr),
                "f_monomial": expression_text(sector.f_monomial_expr),
                "f_monomial_powers": sector.f_monomial_powers,
                "jacobian_monomial_powers": sector.jacobian_monomial_powers,
                "singular_axes": [sector.variable_names[axis] for axis in sector.singular_axes],
                "subtraction": sector.subtraction_type,
                "description": sector.description,
            }
            for i, sector in enumerate(sectors)
        ],
        "validation": {
            "sector_count": len(sectors),
            "continuous_dimension": sectors[0].integration_dim if sectors else 0,
            "expected_laurent_orders": topology.expected_laurent_orders,
            "benchmark_available": benchmark_available,
            "kinematic_restrictions": kinematic_restrictions(request),
        },
    }


def print_preintegration_summary(
    request: IntegralRequest,
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    benchmark_available: bool,
) -> None:
    """Print colored pre-integration tables for topology and sectors."""
    data = summary_data(request, topology, sectors, benchmark_available)
    print(maybe_color("\nFastSecDec v2 run summary", Fore.CYAN))

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
    unique_shapes = []
    for shape in data["symanzik"]["dual_shapes"]:
        if shape not in unique_shapes:
            unique_shapes.append(shape)
    symanzik.add_row(["F dual shapes", unique_shapes if unique_shapes else "none"])
    print(symanzik)

    sector_table = PrettyTable()
    sector_table.field_names = [
        maybe_color("id", Fore.CYAN),
        maybe_color("sector", Fore.CYAN),
        maybe_color("vars", Fore.CYAN),
        maybe_color("map", Fore.CYAN),
        maybe_color("J_reg", Fore.CYAN),
        maybe_color("M_F", Fore.CYAN),
        maybe_color("axes", Fore.CYAN),
        maybe_color("subtraction", Fore.CYAN),
    ]
    sector_table.align = "l"
    for sector in data["sectors"]:
        sector_table.add_row(
            [
                sector["id"],
                maybe_color(sector["name"], Fore.MAGENTA),
                ", ".join(sector["variables"]),
                "\n".join(sector["map"]),
                sector["regular_jacobian"],
                sector["f_monomial"],
                ", ".join(sector["singular_axes"]) if sector["singular_axes"] else "-",
                sector["subtraction"],
            ]
        )
    print(sector_table)

    validation = PrettyTable()
    validation.field_names = [maybe_color("check", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    for key, value in data["validation"].items():
        validation.add_row([maybe_color(str(key), Fore.MAGENTA), value])
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


def apply_global_convention(
    sector_coeffs: list[complex],
    sector_errors: list[complex],
    request: IntegralRequest,
) -> tuple[list[complex], list[complex]]:
    """Apply gamma-scheme and stripped-convention shifts to sector coefficients."""
    a, b, c = sector_coeffs
    ea, eb, ec = sector_errors
    if request.integral == "box":
        if request.mode == "massless":
            coeffs = [a, b + a, c + b + (math.pi * math.pi / 6.0) * a]
            errors = [
                ea,
                eb + ea,
                ec + eb + (math.pi * math.pi / 6.0) * ea,
            ]
            return coeffs, errors
        return [a, b, c], sector_errors

    if request.gamma_scheme == "full":
        g2 = 0.5 * EULER_GAMMA * EULER_GAMMA + math.pi * math.pi / 12.0
        coeffs = [-a, -b + EULER_GAMMA * a, -c + EULER_GAMMA * b - g2 * a]
        errors = [ea, eb + abs(EULER_GAMMA) * ea, ec + abs(EULER_GAMMA) * eb + abs(g2) * ea]
        return coeffs, errors

    coeffs = [-a, -b, -c]
    errors = sector_errors[:]
    if request.mode == "massless":
        coeffs[2] += (math.pi * math.pi / 6.0) * coeffs[0]
        errors[2] = errors[2] + (math.pi * math.pi / 6.0) * errors[0]
    return coeffs, errors


def selected_prefactor_values(
    request: IntegralRequest,
    raw_coeffs: list[complex],
    raw_errors: list[complex],
    benchmark: BenchmarkResult,
) -> tuple[list[complex], list[complex], list[complex], complex]:
    """Select raw or Feynman-normalized display coefficients."""
    factor = benchmark.factor if benchmark is not None else ONELOOP_TO_FEYNMAN
    if request.prefactor_convention == "feynman":
        return (
            [factor * c for c in raw_coeffs],
            [abs(factor) * e for e in raw_errors],
            [factor * value for value in benchmark.raw],
            factor,
        )
    return raw_coeffs, raw_errors, benchmark.raw, factor


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
        return "-", Fore.WHITE
    if pull <= 2.0:
        return f"{pull:.2f}σ", Fore.GREEN
    if pull <= 5.0:
        return f"{pull:.2f}σ", Fore.YELLOW
    return f"{pull:.2f}σ", Fore.RED


def make_output(
    request: IntegralRequest,
    raw_coeffs: list[complex],
    raw_errors: list[complex],
    benchmark: BenchmarkResult,
    samples: int,
    elapsed_seconds: float,
    avg_eval_us_per_sample_per_worker: float,
    eval_seconds: float,
    python_seconds: float,
    havana_seconds: float,
    python_overhead_fraction: float,
    summary: JsonDict,
) -> JsonDict:
    """Assemble final output data before printing or JSON serialization."""
    display_coeffs, display_errors, display_bench, factor = selected_prefactor_values(
        request, raw_coeffs, raw_errors, benchmark
    )
    return {
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
        "python_overhead_fraction": python_overhead_fraction,
        "summary": summary,
        "raw": {
            "epsilon_minus_2": raw_coeffs[0],
            "epsilon_minus_1": raw_coeffs[1],
            "epsilon_0": raw_coeffs[2],
            "errors": raw_errors,
        },
        "display": {
            "epsilon_minus_2": display_coeffs[0],
            "epsilon_minus_1": display_coeffs[1],
            "epsilon_0": display_coeffs[2],
            "errors": display_errors,
            "benchmark": display_bench,
        },
        "benchmark": {
            "raw": benchmark.raw,
            "feynman": benchmark.feynman,
            "factor": benchmark.factor,
        },
    }


def print_result_table(output: JsonDict) -> None:
    """Print the final coefficient comparison table and timing footer."""
    labels = ["eps^-2", "eps^-1", "eps^0"]
    keys = ["epsilon_minus_2", "epsilon_minus_1", "epsilon_0"]
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
    for idx, key in enumerate(keys):
        fsd_value = display[key]
        err = display["errors"][idx]
        bench_value = display["benchmark"][idx]
        diff = fsd_value - bench_value
        pull_text, row_color = compare_pull(diff, err)
        table.add_row(
            [
                maybe_color(labels[idx], Fore.MAGENTA),
                maybe_color(format_complex_with_error(fsd_value, err), row_color),
                format_percent(relative_error_percent(fsd_value, err)),
                format_complex(bench_value),
                maybe_color(format_complex_with_error(diff, err), row_color),
                maybe_color(pull_text, row_color),
            ]
        )
    print(table)
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
        + " uses the absolute 1σ error."
    )
    total_timing = max(
        output["eval_seconds"] + output["python_seconds"] + output["havana_seconds"],
        1.0e-300,
    )
    eval_percent = 100.0 * output["eval_seconds"] / total_timing
    python_percent = 100.0 * output["python_seconds"] / total_timing
    havana_percent = 100.0 * output["havana_seconds"] / total_timing
    print(
        maybe_color("Timing:", Fore.CYAN)
        + " "
        + colored_kv("EvalT", format_seconds(output["eval_seconds"]), Fore.GREEN)
        + "  "
        + colored_kv("PythonT", format_seconds(output["python_seconds"]), Fore.YELLOW)
        + "  "
        + colored_kv("HavanaT", format_seconds(output["havana_seconds"]), Fore.BLUE)
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
        + maybe_color(f"{havana_percent:.2f}% havana", Fore.BLUE)
        + "    "
        + maybe_color("Status:", Fore.CYAN)
        + " "
        + maybe_color("<=2σ", Fore.GREEN)
        + " / "
        + maybe_color("<=5σ", Fore.YELLOW)
        + " / "
        + maybe_color(">5σ", Fore.RED)
    )


def json_default(obj: Any) -> Any:
    """JSON serializer for complex numbers and fallback objects."""
    if isinstance(obj, complex):
        return {"re": obj.real, "im": obj.imag}
    return str(obj)


def output_json(output: JsonDict) -> str:
    """Return pretty JSON output for machine-readable runs."""
    return json.dumps(output, default=json_default, indent=2)
