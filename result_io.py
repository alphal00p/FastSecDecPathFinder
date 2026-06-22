"""Persistent result JSON and target-file helpers."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any

from colorama import Fore
from prettytable import PrettyTable

from definitions import IntegralRequest, JsonDict, TargetDefinition
from formatting import (
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
    for idx, coeff in enumerate(coeffs):
        error = errors[idx] if idx < len(errors) else 0.0 + 0.0j
        ref = targets[idx] if idx < len(targets) else None
        diff = coeff - ref if ref is not None else None
        pull, color = compare_pull(diff, error)
        table.add_row(
            [
                maybe_color(labels[idx] if idx < len(labels) else f"#{idx}", Fore.MAGENTA),
                maybe_color(format_complex_with_error(coeff, error), color),
                format_percent(relative_error_percent(coeff, error)),
                format_complex(ref) if ref is not None else maybe_color("N/A", Fore.WHITE),
                maybe_color(format_complex_with_error(diff, error), color) if diff is not None else maybe_color("N/A", Fore.WHITE),
                maybe_color(pull, color),
            ]
        )
    print(table)

    timing = PrettyTable()
    timing.field_names = [maybe_color("timing", Fore.CYAN), maybe_color("value", Fore.CYAN)]
    timing_rows = 0
    for key in (
        "eval_seconds",
        "python_seconds",
        "havana_seconds",
        "dual_evaluator_build_seconds",
        "chain_rule_formula_build_seconds",
        "avg_eval_us_per_sample_per_worker",
    ):
        if key in data:
            timing.add_row([maybe_color(key, Fore.MAGENTA), data.get(key)])
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
