"""Persistent result JSON and target-file helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
from typing import Any

from colorama import Fore
from prettytable import PrettyTable

from definitions import IntegralRequest, JsonDict, TargetDefinition
from formatting import (
    combine_uncorrelated_errors,
    compare_pull,
    format_complex,
    format_complex_with_error,
    format_percent,
    maybe_color,
    output_json,
    relative_error_percent,
)


def complex_from_json(value: Any) -> complex:
    """Decode a complex number emitted by ``formatting.output_json``."""
    if isinstance(value, complex):
        return value
    if isinstance(value, dict) and "re" in value and "im" in value:
        return complex(float(value["re"]), float(value["im"]))
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, str):
        return complex(value)
    raise ValueError(f"cannot decode complex value {value!r}")


def complex_list_from_json(values: Any) -> list[complex]:
    """Decode a list of JSON complex values."""
    if not isinstance(values, list):
        raise ValueError("expected a list of complex values")
    return [complex_from_json(value) for value in values]


def result_output_path(request: IntegralRequest) -> Path:
    """Return the default result.json path for one request."""
    return Path(request.result_path).expanduser()


def request_metadata(request: IntegralRequest) -> JsonDict:
    """Return a JSON-friendly request/config block."""
    data = asdict(request)
    data["target_args"] = list(request.target_args) if request.target_args is not None else None
    return data


def environment_metadata() -> JsonDict:
    """Return lightweight runtime metadata for result files."""
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
    }


def write_result_json(output: JsonDict, path: str | Path) -> Path:
    """Atomically write the final pretty JSON result file."""
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text(output_json(output) + "\n", encoding="utf-8")
    os.replace(tmp, destination)
    return destination


def _markdown_seconds(seconds: Any) -> str:
    """Format seconds compactly for markdown reports."""
    try:
        value = float(seconds)
    except Exception:
        return "n/a"
    if not math.isfinite(value):
        return "n/a"
    if value < 1.0e-3:
        return f"{1.0e6 * value:.3g} us"
    if value < 1.0:
        return f"{1.0e3 * value:.3g} ms"
    return f"{value:.3f} s"


def _markdown_complex(value: Any) -> str:
    """Format one real-dominant complex value for markdown."""
    try:
        z = complex_from_json(value)
    except Exception:
        return "n/a"
    if abs(z.imag) <= 1.0e-15 * max(abs(z.real), 1.0):
        return f"{z.real:.12g}"
    return f"{z.real:.12g}{z.imag:+.12g}i"


def _markdown_relerr(coeff: Any, error: Any) -> float | None:
    """Return a dimensionless relative error, or ``None`` when undefined."""
    try:
        c = complex_from_json(coeff)
        e = complex_from_json(error)
    except Exception:
        return None
    denom = abs(c)
    if denom <= 0.0 or not math.isfinite(denom):
        return None
    rel = abs(e) / denom
    return rel if math.isfinite(rel) else None


def _markdown_float(value: Any, digits: int = 4) -> str:
    """Format a finite float or return ``n/a``."""
    try:
        x = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    return f"{x:.{digits}g}"


def _markdown_count(value: Any) -> str:
    """Format an integer-like count with separators."""
    try:
        x = float(value)
    except Exception:
        return "n/a"
    if not math.isfinite(x):
        return "n/a"
    if abs(x) >= 1.0:
        return f"{int(round(x)):,}"
    return f"{x:,.4g}"


def _generation_details(output: JsonDict) -> tuple[list[dict[str, Any]], float]:
    """Return generation detail rows and total seconds from a result payload."""
    generation = output.get("summary", {}).get("generation_timings", {})
    if not isinstance(generation, dict):
        return [], 0.0
    details = generation.get("details", generation.get("records", []))
    if not isinstance(details, list):
        details = []
    total = generation.get("total", None)
    if total is None:
        total = sum(
            float(row.get("seconds", 0.0) or 0.0)
            for row in details
            if isinstance(row, dict)
        )
    return [row for row in details if isinstance(row, dict)], float(total or 0.0)


def _markdown_repo_relative(path: str | Path) -> str:
    """Render a path relative to the current repository when possible."""
    value = Path(path).expanduser()
    try:
        return value.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except Exception:
        return value.as_posix()


def write_markdown_report(output: JsonDict, path: str | Path) -> Path:
    """Write a compact markdown report for an FSD result payload.

    The report intentionally derives all numbers from the persisted result
    object.  For QMC runs with independent sector errors the aggregate errors
    are the quadrature-propagated sector errors stored by the integrator.
    """
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    labels = [str(label) for label in output.get("laurent_labels", [])]
    sector_rows = [
        row for row in output.get("sector_results", [])
        if isinstance(row, dict)
    ]
    active_rows = [row for row in sector_rows if int(row.get("samples", 0) or 0) > 0]
    diagnostics = output.get("integration_diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    request = output.get("request", {})
    if not isinstance(request, dict):
        request = {}
    try:
        worker_count = max(int(request.get("workers", 1) or 1), 1)
    except Exception:
        worker_count = 1
    generation_rows, generation_total = _generation_details(output)

    runtime_rows: list[tuple[float, str, int]] = []
    sample_rows: list[tuple[float, str, int]] = []
    wall_time_rows: list[tuple[float, str, int]] = []
    max_weight_rows: list[tuple[float, str, int]] = []
    sector_rel_rows: list[tuple[float, str, int, str]] = []
    threshold = 1.0e-3
    for row in active_rows:
        diag = row.get("diagnostics", {})
        if not isinstance(diag, dict):
            diag = {}
        sector_id = int(row.get("sector_id", -1))
        name = str(row.get("name", f"sector_{sector_id}"))
        try:
            samples = float(row.get("samples", 0) or 0)
            if math.isfinite(samples):
                sample_rows.append((samples, name, sector_id))
        except Exception:
            pass
        avg_eval = diag.get("avg_eval_us_per_sample")
        try:
            avg_value = float(avg_eval)
            if math.isfinite(avg_value):
                runtime_rows.append((avg_value, name, sector_id))
        except Exception:
            pass
        try:
            profiled_seconds = (
                float(diag.get("eval_seconds", 0.0) or 0.0)
                + float(diag.get("python_seconds", 0.0) or 0.0)
                + float(diag.get("havana_seconds", diag.get("integrator_seconds", 0.0)) or 0.0)
            )
            wall_equivalent = profiled_seconds / float(worker_count)
            if math.isfinite(wall_equivalent):
                wall_time_rows.append((wall_equivalent, name, sector_id))
        except Exception:
            pass
        try:
            max_weight = float(diag.get("max_abs_weight", 0.0) or 0.0)
            if math.isfinite(max_weight):
                max_weight_rows.append((max_weight, name, sector_id))
        except Exception:
            pass
        display = row.get("display", {})
        if not isinstance(display, dict):
            display = {}
        coeffs = display.get("coefficients", [])
        errors = display.get("errors", [])
        worst_rel = 0.0
        worst_label = "n/a"
        for label, coeff, error in zip(labels, coeffs, errors):
            rel = _markdown_relerr(coeff, error)
            if rel is not None and rel > worst_rel:
                worst_rel = rel
                worst_label = label
        sector_rel_rows.append((worst_rel, name, sector_id, worst_label))

    runtime_values = [value for value, _name, _sector_id in runtime_rows]
    sample_values = [value for value, _name, _sector_id in sample_rows]
    wall_time_values = [value for value, _name, _sector_id in wall_time_rows]
    rel_values = [value for value, _name, _sector_id, _label in sector_rel_rows]
    rel_failures = [row for row in sector_rel_rows if row[0] > threshold]
    precision = output.get("precision_stats", {})
    if not isinstance(precision, dict):
        precision = {}
    eval_seconds = float(output.get("eval_seconds", 0.0) or 0.0)
    python_seconds = float(output.get("python_seconds", 0.0) or 0.0)
    integrator_seconds = float(
        output.get(
            "integrator_seconds",
            output.get("havana_seconds", 0.0),
        )
        or 0.0
    )
    total_profiled_integration_seconds = (
        eval_seconds
        + python_seconds
        + integrator_seconds
    )

    lines: list[str] = []
    lines.append("# FSD Integration Report")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    run_file = request.get("run_file")
    lines.append(f"- result file: `{output.get('result_path', 'n/a')}`")
    if run_file:
        lines.append(f"- run card: `{run_file}`")
    lines.append(f"- integral: `{output.get('integral', 'n/a')}`")
    lines.append(f"- prefactor convention: `{output.get('prefactor_convention', 'n/a')}`")
    lines.append(f"- sampling mode: `{diagnostics.get('sampling_mode', 'n/a')}`")
    lines.append(f"- QMC backend: `{diagnostics.get('qmc_lattice_backend', 'n/a')}`")
    lines.append(f"- QMC support grouping: `{diagnostics.get('qmc_support_grouping', 'n/a')}`")
    lines.append(
        f"- QMC optimized evaluator groups: `{diagnostics.get('qmc_optimized_evaluator_groups_prepared', 'n/a')}`"
    )
    lines.append(
        f"- sectors sampled: `{len(active_rows)}` / `{len(sector_rows)}`"
    )
    lines.append(f"- total raw samples: `{int(output.get('samples', 0) or 0):,}`")
    lines.append(f"- total integration time (wall): `{_markdown_seconds(output.get('elapsed_seconds'))}`")
    lines.append(
        "- total profiled integration work "
        f"(EvalT + PythonT + IntegratorT): `{_markdown_seconds(total_profiled_integration_seconds)}`"
    )
    lines.append(f"- average eval time: `{_markdown_float(output.get('avg_eval_us_per_sample_per_worker'))} us/sample/worker`")
    patched_sectors = diagnostics.get("patched_nonfinite_qmc_optimized_sectors")
    if patched_sectors:
        lines.append(
            "- optimized-QMC fallback sectors: "
            f"`{', '.join(str(int(value)) for value in patched_sectors)}`"
        )
    refined_sectors = diagnostics.get("refined_sector_ids_for_1e_minus_3")
    if refined_sectors:
        lines.append(
            "- refined sectors for per-sector `1e-3` target: "
            f"`{', '.join(str(int(value)) for value in refined_sectors)}`"
        )
        if diagnostics.get("refinement_samples_per_selected_sector") is not None:
            lines.append(
                "- refinement raw samples per selected sector: "
                f"`{int(diagnostics.get('refinement_samples_per_selected_sector') or 0):,}`"
            )
    lines.append("")

    lines.append("## Reproduction Commands")
    lines.append("")
    single_sector_card = Path("examples/runs/four_loop_hard_psd2807_fsd_qmc.toml")
    all_sector_card = Path("examples/runs/four_loop_hard_all_sectors_fsd_qmc.toml")
    run_cards = [
        ("single-sector PSD2807 example", single_sector_card),
        ("full all-sector hard-polynomial example", all_sector_card),
    ]
    for label, card in run_cards:
        card_text = _markdown_repo_relative(card)
        lines.append(f"### {label}")
        lines.append("")
        lines.append("```bash")
        lines.append(f".venv/bin/python FSD.py --run {card_text}")
        lines.append("```")
        lines.append("")

    lines.append("## Generation")
    lines.append("")
    lines.append(f"- total recorded generation time: `{_markdown_seconds(generation_total)}`")
    lines.append("")
    lines.append("| stage | time | detail |")
    lines.append("|---|---:|---|")
    for row in generation_rows:
        detail = str(row.get("detail", "") or "-").replace("\n", " ")
        if len(detail) > 120:
            detail = detail[:117] + "..."
        lines.append(
            f"| {row.get('name', 'n/a')} | {_markdown_seconds(row.get('seconds', 0.0))} | {detail} |"
        )
    lines.append("")

    lines.append("## Aggregate Laurent Sum")
    lines.append("")
    lines.append("The aggregate coefficients below are the total sector sum stored in the result JSON.  In independent-sector QMC mode the reported errors are the quadrature propagation of per-sector errors.")
    lines.append("")
    lines.append("| order | value | MC error | relative error |")
    lines.append("|---|---:|---:|---:|")
    display = output.get("display", {})
    coeffs = display.get("coefficients", []) if isinstance(display, dict) else []
    errors = display.get("errors", []) if isinstance(display, dict) else []
    for label, coeff, error in zip(labels, coeffs, errors):
        rel = _markdown_relerr(coeff, error)
        rel_text = "n/a" if rel is None else f"{rel:.4g}"
        lines.append(
            f"| {label} | {_markdown_complex(coeff)} | {_markdown_complex(error)} | {rel_text} |"
        )
    lines.append("")

    lines.append("## Per-Sector Runtime")
    lines.append("")
    if runtime_values or sample_values or wall_time_values:
        lines.append(
            "Per-sector wall-equivalent time is computed from the profiled "
            f"sector work `(EvalT + PythonT + IntegratorT) / {worker_count}` "
            "using the run worker count."
        )
        lines.append("")
        min_runtime = min(runtime_rows, key=lambda item: item[0]) if runtime_rows else None
        max_runtime = max(runtime_rows, key=lambda item: item[0]) if runtime_rows else None
        min_samples = min(sample_rows, key=lambda item: item[0]) if sample_rows else None
        max_samples = max(sample_rows, key=lambda item: item[0]) if sample_rows else None
        min_wall = min(wall_time_rows, key=lambda item: item[0]) if wall_time_rows else None
        max_wall = max(wall_time_rows, key=lambda item: item[0]) if wall_time_rows else None
        def sector_text(row: tuple[float, str, int] | None) -> str:
            return f"{row[1]} ({row[2]})" if row is not None else "n/a"

        def eval_text(row: tuple[float, str, int] | None) -> str:
            return f"{row[0]:.4g} us/sample" if row is not None else "n/a"

        def sample_text(row: tuple[float, str, int] | None) -> str:
            return _markdown_count(row[0]) if row is not None else "n/a"

        def wall_text(row: tuple[float, str, int] | None) -> str:
            return _markdown_seconds(row[0]) if row is not None else "n/a"

        lines.append("| statistic | eval time | eval sector | samples | sample sector | wall-equivalent time | time sector |")
        lines.append("|---|---:|---|---:|---|---:|---|")
        lines.append(
            f"| min | {eval_text(min_runtime)} | {sector_text(min_runtime)} | "
            f"{sample_text(min_samples)} | {sector_text(min_samples)} | "
            f"{wall_text(min_wall)} | {sector_text(min_wall)} |"
        )
        lines.append(
            f"| max | {eval_text(max_runtime)} | {sector_text(max_runtime)} | "
            f"{sample_text(max_samples)} | {sector_text(max_samples)} | "
            f"{wall_text(max_wall)} | {sector_text(max_wall)} |"
        )
        lines.append(
            "| average | "
            f"{statistics.mean(runtime_values):.4g} us/sample | - | "
            f"{_markdown_count(statistics.mean(sample_values)) if sample_values else 'n/a'} | - | "
            f"{_markdown_seconds(statistics.mean(wall_time_values)) if wall_time_values else 'n/a'} | - |"
        )
        lines.append(
            "| median | "
            f"{statistics.median(runtime_values):.4g} us/sample | - | "
            f"{_markdown_count(statistics.median(sample_values)) if sample_values else 'n/a'} | - | "
            f"{_markdown_seconds(statistics.median(wall_time_values)) if wall_time_values else 'n/a'} | - |"
        )
    else:
        lines.append("No sampled-sector runtime rows were available.")
    lines.append("")

    lines.append("## Per-Sector Relative Accuracy")
    lines.append("")
    if rel_values:
        min_rel = min(sector_rel_rows, key=lambda item: item[0])
        max_rel = max(sector_rel_rows, key=lambda item: item[0])
        lines.append(f"- threshold checked: `{threshold:.1e}`")
        lines.append(f"- sectors above threshold: `{len(rel_failures)}` / `{len(active_rows)}`")
        lines.append("")
        lines.append("| statistic | max relative error | sector/order |")
        lines.append("|---|---:|---|")
        lines.append(f"| min sector max | {min_rel[0]:.4g} | {min_rel[1]} ({min_rel[2]}), {min_rel[3]} |")
        lines.append(f"| max sector max | {max_rel[0]:.4g} | {max_rel[1]} ({max_rel[2]}), {max_rel[3]} |")
        lines.append(f"| average sector max | {statistics.mean(rel_values):.4g} | - |")
        lines.append(f"| median sector max | {statistics.median(rel_values):.4g} | - |")
        lines.append("")
        lines.append("| worst sector | max relative error | order | samples |")
        lines.append("|---|---:|---|---:|")
        samples_by_id = {int(row.get("sector_id", -1)): int(row.get("samples", 0) or 0) for row in active_rows}
        for rel, name, sector_id, label in sorted(sector_rel_rows, reverse=True)[:20]:
            lines.append(f"| {name} ({sector_id}) | {rel:.4g} | {label} | {samples_by_id.get(sector_id, 0):,} |")
    else:
        lines.append("No finite per-sector relative errors were available.")
    lines.append("")

    lines.append("## Weight And Precision Diagnostics")
    lines.append("")
    if max_weight_rows:
        max_weight = max(max_weight_rows, key=lambda item: item[0])
        lines.append(f"- maximum absolute sampled weight: `{max_weight[0]:.6g}` in {max_weight[1]} ({max_weight[2]})")
    for key in ("ordinary", "stability", "medium_precision", "high_precision"):
        block = precision.get(key, {}) if isinstance(precision, dict) else {}
        if isinstance(block, dict):
            lines.append(
                f"- {key}: `{int(block.get('samples', 0) or 0):,}` samples "
                f"({_markdown_float(100.0 * float(block.get('fraction', 0.0) or 0.0), 3)}%)"
            )
    lines.append("")

    tmp = destination.with_name(destination.name + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, destination)
    return destination


def load_result_json(path: str | Path) -> JsonDict:
    """Load a result JSON file."""
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def target_from_result_file(path: str | Path, convention: str) -> TargetDefinition:
    """Build a target from a previous result JSON file."""
    data = load_result_json(path)
    file_convention = str(data.get("prefactor_convention", ""))
    if file_convention != convention:
        raise ValueError(
            f"target file convention {file_convention!r} does not match selected "
            f"prefactor convention {convention!r}"
        )

    # A result produced with an explicit target stores that reference under
    # ``target``.  Prefer it over the aggregate FSD estimate when the file is
    # later used as ``--target result.json``.  This makes a pySecDec comparison
    # reusable without rerunning pySecDec and also preserves explicit numeric
    # targets.
    stored_target = data.get("target", {})
    if (
        isinstance(stored_target, dict)
        and str(stored_target.get("source", "none")) != "none"
        and stored_target.get("coefficients")
    ):
        source = str(stored_target.get("source", "target"))
        coefficients = complex_list_from_json(stored_target.get("coefficients", []))
        errors = complex_list_from_json(
            stored_target.get("errors", [0.0 for _ in coefficients])
        )
        file_source = source if source.startswith("file:") else f"file:{source}"
        return TargetDefinition(
            source=file_source,
            convention=convention,
            coefficients=coefficients,
            errors=errors,
            metadata={
                "path": str(Path(path).expanduser()),
                "stored_source": "target",
                "orders": stored_target.get("metadata", {}).get("orders", []),
            },
        )

    # A result produced with ``--dot-engine pysecdec`` has no FSD aggregate.
    # Keep this legacy/simple block readable as a target too.
    stored_pysecdec = data.get("pysecdec", {})
    if (
        isinstance(stored_pysecdec, dict)
        and stored_pysecdec.get("coeffs")
    ):
        coefficients = complex_list_from_json(stored_pysecdec.get("coeffs", []))
        errors = complex_list_from_json(
            stored_pysecdec.get("errors", [0.0 for _ in coefficients])
        )
        return TargetDefinition(
            source="file:pysecdec",
            convention=convention,
            coefficients=coefficients,
            errors=errors,
            metadata={
                "path": str(Path(path).expanduser()),
                "stored_source": "pysecdec",
                "orders": stored_pysecdec.get("orders", []),
            },
        )

    aggregate = data.get("aggregate_results", {})
    display = aggregate.get("display", data.get("display", {}))
    coefficients = complex_list_from_json(display.get("coefficients", []))
    errors = complex_list_from_json(display.get("errors", [0.0 for _ in coefficients]))
    return TargetDefinition(
        source="file",
        convention=convention,
        coefficients=coefficients,
        errors=errors,
        metadata={"path": str(Path(path).expanduser())},
    )


def _sorted_sector_rows(rows: list[JsonDict], sort_mode: str) -> list[JsonDict]:
    """Sort sector rows according to the viewer option."""
    if sort_mode == "abs-central":
        return sorted(rows, key=lambda row: float(row.get("sort_keys", {}).get("abs_central", 0.0)), reverse=True)
    if sort_mode == "abs-error":
        return sorted(rows, key=lambda row: float(row.get("sort_keys", {}).get("abs_error", 0.0)), reverse=True)
    return sorted(rows, key=lambda row: int(row.get("sector_id", 0)))


def _short_laurent_label(label: Any, index: int) -> str:
    """Render ``eps^-4`` as ``-4`` to keep sector tables compact."""
    text = str(label)
    if text.startswith("eps^"):
        return text[4:]
    return str(index)


def _uncertainty_color(value: complex, error: complex) -> str:
    """Choose a table color from a coefficient's relative MC uncertainty."""
    rel = relative_error_percent(value, error)
    if rel <= 5.0:
        return Fore.GREEN
    if rel <= 25.0:
        return Fore.YELLOW
    return Fore.RED


def _compact_series_text(labels: list[Any], values: list[complex], errors: list[complex]) -> str:
    """Format one Laurent vector for compact colored sector tables."""
    pieces: list[str] = []
    for index, value in enumerate(values):
        label = _short_laurent_label(labels[index], index) if index < len(labels) else str(index)
        error = errors[index] if index < len(errors) else 0.0 + 0.0j
        color = _uncertainty_color(value, error)
        pieces.append(
            maybe_color(f"{label}:{format_complex(value, 3)}", color)
        )
    return "; ".join(pieces)


def _compact_error_text(labels: list[Any], values: list[complex], errors: list[complex]) -> str:
    """Format one Laurent error vector for compact colored sector tables."""
    pieces: list[str] = []
    for index, error in enumerate(errors):
        value = values[index] if index < len(values) else 0.0 + 0.0j
        label = _short_laurent_label(labels[index], index) if index < len(labels) else str(index)
        color = _uncertainty_color(value, error)
        pieces.append(
            maybe_color(f"{label}:{format_complex(error, 2)}", color)
        )
    return "; ".join(pieces)


def _precision_text(stats: JsonDict) -> str:
    """Compact precision-tier text for saved result tables."""
    if not isinstance(stats, dict):
        return "n/a"
    ordinary = stats.get("ordinary", {})
    stability = stats.get("stability", {})
    medium = stats.get("medium_precision", {})
    high = stats.get("high_precision", {})

    def piece(label: str, block: JsonDict) -> str:
        samples = int(block.get("samples", 0))
        fraction = 100.0 * float(block.get("fraction", 0.0))
        return f"{label}:{samples} ({format_percent(fraction)})"

    return "; ".join(
        [
            maybe_color(piece("ord", ordinary), Fore.GREEN),
            maybe_color(piece(f"p{stability.get('precision_digits', '?')}", stability), Fore.YELLOW),
            maybe_color(piece(f"p{medium.get('precision_digits', '?')}", medium), Fore.MAGENTA),
            maybe_color(piece(f"p{high.get('precision_digits', '?')}", high), Fore.RED),
        ]
    )


def print_saved_results(path: str | Path, sort_mode: str = "index") -> None:
    """Print a colored view of a stored result JSON file."""
    data = load_result_json(path)
    print(maybe_color(f"\nFSD result file: {path}", Fore.CYAN))

    header = PrettyTable()
    header.field_names = [maybe_color("item", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    for key in ("integral", "mode", "prefactor_convention", "samples", "elapsed_seconds", "interrupted"):
        header.add_row([maybe_color(key, Fore.MAGENTA), data.get(key)])
    target = data.get("target", {})
    header.add_row([maybe_color("target", Fore.MAGENTA), target.get("source", "none")])
    print(header)

    labels = data.get("laurent_labels", [])
    aggregate = data.get("aggregate_results", {})
    display = aggregate.get("display", data.get("display", {}))
    target_coeffs = target.get("coefficients", []) if target.get("source") != "none" else []
    target_error_values = target.get("errors", []) if target.get("source") != "none" else []
    table = PrettyTable()
    table.field_names = [
        maybe_color("coeff", Fore.CYAN),
        maybe_color("central", Fore.CYAN),
        maybe_color("MC err", Fore.CYAN),
        maybe_color("target", Fore.CYAN),
        maybe_color("diff", Fore.CYAN),
        maybe_color("pull", Fore.CYAN),
    ]
    coeffs = complex_list_from_json(display.get("coefficients", []))
    errors = complex_list_from_json(display.get("errors", []))
    targets = complex_list_from_json(target_coeffs) if target_coeffs else []
    target_errors = complex_list_from_json(target_error_values) if target_error_values else []
    for idx, coeff in enumerate(coeffs):
        error = errors[idx] if idx < len(errors) else 0.0 + 0.0j
        ref = targets[idx] if idx < len(targets) else None
        target_error = target_errors[idx] if idx < len(target_errors) else 0.0 + 0.0j
        comparison_error = combine_uncorrelated_errors(error, target_error)
        diff = coeff - ref if ref is not None else None
        pull, color = compare_pull(diff, comparison_error)
        table.add_row(
            [
                maybe_color(labels[idx] if idx < len(labels) else f"#{idx}", Fore.MAGENTA),
                maybe_color(format_complex_with_error(coeff, error), color),
                format_percent(relative_error_percent(coeff, error)),
                format_complex(ref) if ref is not None else maybe_color("N/A", Fore.WHITE),
                maybe_color(format_complex_with_error(diff, comparison_error), color) if diff is not None else maybe_color("N/A", Fore.WHITE),
                maybe_color(pull, color),
            ]
        )
    print(table)

    timing = PrettyTable()
    timing.field_names = [maybe_color("timing", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    timing_rows = 0
    timing_keys = [
        ("eval_seconds", "eval_seconds"),
        ("python_seconds", "python_seconds"),
        (
            "integrator_seconds",
            "integrator_seconds" if "integrator_seconds" in data else "havana_seconds",
        ),
        ("dual_evaluator_build_seconds", "dual_evaluator_build_seconds"),
        ("chain_rule_formula_build_seconds", "chain_rule_formula_build_seconds"),
        ("avg_eval_us_per_sample_per_worker", "avg_eval_us_per_sample_per_worker"),
    ]
    for label, key in timing_keys:
        if key in data:
            timing.add_row([maybe_color(label, Fore.MAGENTA), data.get(key)])
            timing_rows += 1
    if timing_rows:
        print(timing)

    summary = data.get("summary", {})
    symanzik = summary.get("symanzik", {}) if isinstance(summary, dict) else {}
    if isinstance(symanzik, dict):
        cache_rows = [
            ("endpoint projector formulas", symanzik.get("endpoint_projector_formula_count")),
            ("regular Taylor formulas", symanzik.get("regular_taylor_formula_count")),
            ("curated regular Taylor assets", symanzik.get("regular_taylor_formulas_from_curated_cache")),
            ("regular Taylor formulas skipped", symanzik.get("regular_taylor_formulas_skipped")),
            ("regular Taylor policy", symanzik.get("regular_taylor_formula_policy")),
            ("chain-rule formulas", symanzik.get("chain_rule_formula_count")),
            ("chain-rule formulas skipped", symanzik.get("chain_rule_formulas_skipped")),
        ]
        cache_rows = [(label, value) for label, value in cache_rows if value not in (None, 0, "0")]
        if cache_rows:
            cache_table = PrettyTable()
            cache_table.field_names = [
                maybe_color("formula asset", Fore.CYAN),
                maybe_color("count", Fore.CYAN),
            ]
            for label, value in cache_rows:
                color = Fore.GREEN if "curated" in label else Fore.MAGENTA
                cache_table.add_row([maybe_color(label, color), value])
            print(cache_table)

    precision = data.get("precision_stats", {})
    if precision:
        precision_table = PrettyTable()
        precision_table.field_names = [
            maybe_color("precision tier", Fore.CYAN),
            maybe_color("samples", Fore.CYAN),
            maybe_color("fraction", Fore.CYAN),
            maybe_color("threshold", Fore.CYAN),
            maybe_color("digits", Fore.CYAN),
        ]
        for key, label, color in (
            ("ordinary", "ordinary", Fore.GREEN),
            ("stability", "stability", Fore.YELLOW),
            ("medium_precision", "medium precision", Fore.MAGENTA),
            ("high_precision", "high precision", Fore.RED),
        ):
            block = precision.get(key, {})
            precision_table.add_row(
                [
                    maybe_color(label, color),
                    block.get("samples", 0),
                    format_percent(100.0 * float(block.get("fraction", 0.0))),
                    block.get("threshold", "-"),
                    block.get("precision_digits", "-"),
                ]
            )
        print(precision_table)

    sector_rows = _sorted_sector_rows(list(data.get("sector_results", [])), sort_mode)
    sampled_sector_rows = [
        row for row in sector_rows if int(row.get("samples", 0) or 0) > 0
    ]
    if sampled_sector_rows:
        sector_rows = sampled_sector_rows
    if sector_rows:
        sector_table = PrettyTable()
        sector_table.field_names = [
            maybe_color("id", Fore.CYAN),
            maybe_color("sector", Fore.CYAN),
            maybe_color("samples", Fore.CYAN),
            maybe_color("coefficients", Fore.CYAN),
            maybe_color("errors", Fore.CYAN),
            maybe_color("precision", Fore.CYAN),
            maybe_color("max |central|", Fore.CYAN),
            maybe_color("max |error|", Fore.CYAN),
        ]
        for row in sector_rows:
            keys = row.get("sort_keys", {})
            display = row.get("display", {})
            row_coeffs = complex_list_from_json(display.get("coefficients", []))
            row_errors = complex_list_from_json(display.get("errors", []))
            coeff_text = _compact_series_text(labels, row_coeffs, row_errors)
            error_text = _compact_error_text(labels, row_coeffs, row_errors)
            max_central = float(keys.get("abs_central", 0.0))
            max_error = float(keys.get("abs_error", 0.0))
            max_color = _uncertainty_color(max_central + 0.0j, max_error + 0.0j)
            sector_table.add_row(
                [
                    maybe_color(str(row.get("sector_id")), Fore.MAGENTA),
                    maybe_color(str(row.get("name")), Fore.MAGENTA),
                    row.get("samples"),
                    coeff_text,
                    error_text,
                    _precision_text(row.get("precision_stats", {})),
                    maybe_color(f"{max_central:.4e}", max_color),
                    maybe_color(f"{max_error:.2e}", max_color),
                ]
            )
        print(sector_table)
