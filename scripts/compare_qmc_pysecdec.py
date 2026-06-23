#!/usr/bin/env python3
"""Compare FSD's QMCPy/Korobov mode against pySecDec's QMC backend.

This is a developer benchmark for one-loop DOT examples.  FSD exposes its
sector sampling explicitly, whereas pySecDec hides sector/order integrands
behind the generated C++ package, so the table reports both the QMC lattice
points per random shift and FSD's resulting raw sector-sample count.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any

from prettytable import PrettyTable
from pySecDec.integral_interface import IntegralLibrary
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinematics import load_kinematics  # noqa: E402
from pysecdec_bridge import _parse_pysecdec_json_series  # noqa: E402
from result_io import complex_list_from_json, load_result_json, target_from_result_file  # noqa: E402


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


def _complex_json(value: complex) -> dict[str, float]:
    return {"re": float(value.real), "im": float(value.imag)}


def _jsonable(value: Any) -> Any:
    """Convert complex-valued nested comparison data to JSON-safe objects."""
    if isinstance(value, complex):
        return _complex_json(value)
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _complex_list_from_maybe_json(values: list[Any]) -> list[complex]:
    """Return complex numbers from plain complex values or result.json objects."""
    out: list[complex] = []
    for value in values:
        if isinstance(value, complex):
            out.append(value)
        elif isinstance(value, dict):
            out.append(complex(float(value.get("re", 0.0)), float(value.get("im", 0.0))))
        else:
            out.append(complex(value))
    return out


def _fmt_scientific(value: complex | float) -> str:
    scalar = abs(value) if isinstance(value, complex) else float(value)
    return f"{scalar:.3e}"


def _fmt_real(value: complex | float) -> str:
    scalar = complex(value).real if isinstance(value, complex) else float(value)
    return f"{scalar:.8g}"


def _labels_from_result(data: dict[str, Any]) -> list[str]:
    labels = data.get("laurent_labels")
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]
    coeffs = data.get("display", {}).get("coefficients", [])
    first = -len(coeffs) + 1
    return [f"eps^{order}" for order in range(first, first + len(coeffs))]


def _order_index(labels: list[str], requested: str) -> int:
    normalized = requested.strip()
    if normalized.startswith("eps^"):
        wanted = normalized
    else:
        wanted = f"eps^{normalized}"
    if wanted not in labels:
        raise ValueError(f"requested order {requested!r} not in Laurent labels {labels}")
    return labels.index(wanted)


def _target_cli_args(args: argparse.Namespace, target_coeffs: list[complex]) -> list[str]:
    """Return the CLI target arguments for the selected comparison convention."""
    if args.target_source == "file":
        return ["--target", str(args.target_file)]
    flattened: list[str] = []
    for value in target_coeffs:
        flattened.extend([f"{float(value.real):.17g}", f"{float(value.imag):.17g}"])
    return ["--target", *flattened]


def _run_fsd(
    args: argparse.Namespace,
    n_points: int,
    result_path: Path,
    target_coeffs: list[complex],
) -> dict[str, Any]:
    if args.fsd_prepared_output is None:
        cmd = [
            sys.executable,
            str(ROOT / "FSD.py"),
            "--run",
            str(args.run_file),
        ]
    else:
        cmd = [
            sys.executable,
            str(ROOT / "FSD.py"),
            "integrate",
            "--output",
            str(args.fsd_prepared_output),
        ]
    cmd.extend(
        [
            "--sampling-mode",
            "qmc",
            "--qmc-shifts",
            str(args.qmc_shifts),
            "--qmc-korobov-alpha",
            str(args.qmc_korobov_alpha),
            "--qmc-lattice-backend",
            str(args.qmc_lattice_backend),
            "--qmc-order",
            str(args.qmc_order),
            "--samples-per-iter",
            str(n_points),
            "--max-iter",
            "1",
            "--batch-size",
            str(args.batch_size or n_points),
            "--workers",
            str(args.workers),
            "--prefactor-convention",
            str(args.fsd_prefactor_convention),
            *list(args.fsd_extra_arg or []),
            *_target_cli_args(args, target_coeffs),
            "--no-progress",
            "--quiet-summary",
            "--json",
            "--result-path",
            str(result_path),
        ]
    )
    if args.stability_threshold is not None:
        cmd.extend(["--stability-threshold", str(args.stability_threshold)])
    if args.medium_precision_stability_threshold is not None:
        cmd.extend(
            [
                "--medium-precision-stability-threshold",
                str(args.medium_precision_stability_threshold),
            ]
        )
    if args.high_precision_stability_threshold is not None:
        cmd.extend(
            [
                "--high-precision-stability-threshold",
                str(args.high_precision_stability_threshold),
            ]
        )
    start = time.perf_counter()
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    elapsed = time.perf_counter() - start
    data = load_result_json(result_path)
    data["_comparison_elapsed_seconds"] = elapsed
    return data


def _oneloop_raw_target(args: argparse.Namespace) -> list[complex]:
    """Use OneLOopBridge for the built-in one-loop target at matching kinematics."""
    from benchmark import import_oneloop_bridge

    bridge = import_oneloop_bridge()
    m2 = complex(float(args.m) * float(args.m), 0.0)
    if args.oneloop_integral == "triangle":
        result = bridge.three_point(0.0, 0.0, float(args.s), m2, m2, m2)
    elif args.oneloop_integral == "box":
        result = bridge.four_point(
            0.0,
            0.0,
            0.0,
            0.0,
            float(args.s12),
            float(args.s23),
            m2,
            m2,
            m2,
            m2,
        )
    else:  # pragma: no cover - argparse restricts the choices.
        raise ValueError(f"unsupported OneLOop integral {args.oneloop_integral!r}")
    return [result.epsilon_minus_2, result.epsilon_minus_1, result.epsilon_0]


def _oneloop_raw_to_sector_target(args: argparse.Namespace, raw: list[complex]) -> list[complex]:
    """Convert OneLOop raw coefficients to the local DOT sector convention.

    This deliberately handles only the one-loop triangle/box kinematics used by
    the shipped DOT examples.  It is not a graph-to-OneLOop mapper.
    """
    if len(raw) != 3:
        raise ValueError("one-loop OneLOop target should have exactly three Laurent coefficients")
    if args.oneloop_integral == "box":
        if args.mode == "massless":
            a_m2 = raw[0]
            a_m1 = raw[1] - a_m2
            a_0 = raw[2] - a_m1 - (math.pi * math.pi / 6.0) * a_m2
            return [a_m2, a_m1, a_0]
        return raw[:]
    if args.oneloop_integral == "triangle":
        if args.mode == "massless":
            a_m2 = -raw[0]
            a_m1 = -raw[1]
            a_0 = -(raw[2] - (math.pi * math.pi / 6.0) * raw[0])
            return [a_m2, a_m1, a_0]
        return [-value for value in raw]
    raise ValueError(f"unsupported OneLOop integral {args.oneloop_integral!r}")


def _inverse_regular_factor(coeffs: list[complex], errors: list[complex], factor: list[complex]) -> tuple[list[complex], list[complex]]:
    """Divide a Laurent series by a regular prefactor with nonzero constant."""
    if not factor or factor[0] == 0:
        raise ValueError("cannot invert empty or singular regular prefactor")
    out = [0.0 + 0.0j for _ in coeffs]
    out_errors = [0.0 + 0.0j for _ in errors]
    for index, value in enumerate(coeffs):
        reduced = value
        reduced_error = abs(errors[index])
        for factor_index in range(1, min(len(factor), index + 1)):
            reduced -= factor[factor_index] * out[index - factor_index]
            reduced_error += abs(factor[factor_index]) * abs(out_errors[index - factor_index])
        out[index] = reduced / factor[0]
        out_errors[index] = reduced_error / abs(factor[0])
    return out, out_errors


_INTEGRAL_RE = re.compile(
    r"\bintegral\s+\d+/\d+:\s+\S+_sector_(?P<sector>\d+)_order_(?P<order>n?\-?\d+)"
)
_RESULT_RE = re.compile(
    r"\bres:\s*"
    r"\([^)]+\)\s*\+/-\s*\([^)]+\)\s*->\s*"
    r"\((?P<re>[^,]+),(?P<im>[^)]+)\)\s*\+/-\s*"
    r"\((?P<ere>[^,]+),(?P<eim>[^)]+)\),\s*"
    r"n:\s*(?P<oldn>\d+)\s*->\s*(?P<newn>\d+)"
)


def _pysecdec_order_token_to_int(token: str) -> int:
    if token.startswith("n"):
        return -int(token[1:])
    return int(token)


def _float_or_nan(value: str) -> float:
    return float(value.strip().replace("nan", "NaN"))


def _parse_pysecdec_integral_records(verbose_log: str) -> dict[tuple[int, int], dict[str, Any]]:
    """Return final pySecDec sector/order records parsed from verbose output.

    pySecDec prints individual generated sector/order integrals before the
    package-level Gamma/global prefactor convolution.  These are the correct
    objects to compare to FSD's raw sector coefficients.
    """
    records: dict[tuple[int, int], dict[str, Any]] = {}
    pending: tuple[int, int] | None = None
    for line in verbose_log.splitlines():
        header = _INTEGRAL_RE.search(line)
        if header:
            pending = (
                int(header.group("sector")) - 1,
                _pysecdec_order_token_to_int(header.group("order")),
            )
            continue
        if pending is None:
            continue
        result = _RESULT_RE.search(line)
        if not result:
            continue
        sector_id, order = pending
        records[(sector_id, order)] = {
            "sector_id": sector_id,
            "sector_name": f"PSD{sector_id}",
            "order": order,
            "label": f"eps^{order}",
            "coefficient": complex(
                _float_or_nan(result.group("re")),
                _float_or_nan(result.group("im")),
            ),
            "error": complex(
                abs(_float_or_nan(result.group("ere"))),
                abs(_float_or_nan(result.group("eim"))),
            ),
            "old_n": int(result.group("oldn")),
            "new_n": int(result.group("newn")),
        }
        pending = None
    return records


def _sector_order_rows(
    fsd: dict[str, Any],
    pysecdec_records: dict[tuple[int, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    labels = _labels_from_result(fsd)
    order_by_label = {
        label: int(label.replace("eps^", ""))
        for label in labels
        if label.startswith("eps^")
    }
    rows: list[dict[str, Any]] = []
    sector_results = fsd.get("sector_results", [])
    py_sector_ids = {int(sector_id) for sector_id, _order in pysecdec_records}
    fsd_sector_ids = {
        int(sector.get("sector_id", -1))
        for sector in sector_results
        if int(sector.get("sector_id", -1)) >= 0
    }
    if py_sector_ids and fsd_sector_ids and py_sector_ids != fsd_sector_ids:
        return (
            [],
            (
                "pySecDec verbose sector ids do not match FSD sector ids "
                f"({len(py_sector_ids)} pySecDec sectors vs {len(fsd_sector_ids)} "
                "FSD sectors). Raw sector/order rows are suppressed because "
                "the generated pySecDec package used a different sector enumeration."
            ),
        )
    for sector in sector_results:
        if int(sector.get("samples", 0)) <= 0:
            continue
        sector_id = int(sector.get("sector_id", -1))
        raw = sector.get("raw_sector") or sector.get("raw") or {}
        coeffs = complex_list_from_json(raw.get("coefficients", []))
        errors = complex_list_from_json(raw.get("errors", []))
        for coeff_index, label in enumerate(labels):
            order = order_by_label.get(label)
            if order is None:
                continue
            record = pysecdec_records.get((sector_id, order))
            if record is None:
                continue
            fsd_value = coeffs[coeff_index]
            fsd_error = errors[coeff_index]
            py_value = complex(record["coefficient"])
            py_error = complex(record["error"])
            combined_error = math.hypot(abs(fsd_error), abs(py_error))
            diff = fsd_value - py_value
            rows.append(
                {
                    "sector_id": sector_id,
                    "sector_name": str(sector.get("name", f"PSD{sector_id}")),
                    "order": order,
                    "label": label,
                    "fsd_coeff": fsd_value,
                    "fsd_error": fsd_error,
                    "pysecdec_coeff": py_value,
                    "pysecdec_error": py_error,
                    "diff": diff,
                    "pull": abs(diff) / combined_error if combined_error > 0.0 else math.inf,
                    "pysecdec_n": int(record["new_n"]),
                    "fsd_samples": int(sector.get("samples", 0)),
                }
            )
    return rows, None


def _resolve_target(args: argparse.Namespace) -> tuple[list[complex], str]:
    """Return target coefficients in the FSD display convention."""
    if args.target_source == "file":
        target = target_from_result_file(args.target_file, args.fsd_prefactor_convention)
        return list(target.coefficients), f"file:{args.target_file}"
    raw = _oneloop_raw_target(args)
    if args.fsd_prefactor_convention != "sector":
        raise ValueError("--target-source oneloop-sector currently requires --fsd-prefactor-convention sector")
    return _oneloop_raw_to_sector_target(args, raw), f"OneLOopBridge {args.oneloop_integral} -> sector"


def _pysecdec_coefficients_in_fsd_convention(
    args: argparse.Namespace,
    fsd: dict[str, Any],
    pysecdec: dict[str, Any],
) -> tuple[list[complex], list[complex]]:
    """Convert pySecDec's generated output to the FSD comparison convention."""
    coeffs = list(pysecdec["coefficients"])
    errors = list(pysecdec["errors"])
    if args.fsd_prefactor_convention != "sector":
        return coeffs, errors

    request = fsd.get("request", {})
    prefactor_min = int(request.get("dot_global_prefactor_min_order", 0))
    if prefactor_min != 0:
        raise ValueError("sector-convention pySecDec comparison currently expects regular DOT prefactors")
    prefactor = _complex_list_from_maybe_json(request.get("dot_global_prefactor_coeffs", []))
    if not prefactor:
        raise ValueError("FSD result did not record DOT global prefactor coefficients")
    return _inverse_regular_factor(coeffs, errors, prefactor)


def _run_pysecdec(args: argparse.Namespace, n_points: int) -> dict[str, Any]:
    kinematics = load_kinematics(args.kinematics_file)
    library = IntegralLibrary(str(args.pysecdec_shared.resolve()))
    library.use_Qmc(
        transform=f"korobov{args.qmc_korobov_alpha}",
        generatingvectors=str(args.pysecdec_generatingvectors),
        minn=int(n_points),
        minm=int(args.qmc_shifts),
        maxeval=int(n_points) * int(args.qmc_shifts),
        cputhreads=int(args.workers),
        seed=int(args.seed),
        epsrel=1.0e-99,
        epsabs=1.0e-99,
    )
    start = time.perf_counter()
    verbose_log = ""
    if bool(args.pysecdec_verbose):
        sys.stdout.flush()
        sys.stderr.flush()
        saved_stdout_fd = os.dup(1)
        saved_stderr_fd = os.dup(2)
        with tempfile.TemporaryFile(mode="w+b") as capture:
            try:
                os.dup2(capture.fileno(), 1)
                os.dup2(capture.fileno(), 2)
                series = library(
                    real_parameters=kinematics.parameter_values,
                    format="json",
                    verbose=True,
                    number_of_presamples=0,
                )
                sys.stdout.flush()
                sys.stderr.flush()
            finally:
                os.dup2(saved_stdout_fd, 1)
                os.dup2(saved_stderr_fd, 2)
                os.close(saved_stdout_fd)
                os.close(saved_stderr_fd)
            capture.seek(0)
            verbose_log = capture.read().decode("utf-8", errors="replace")
    else:
        series = library(
            real_parameters=kinematics.parameter_values,
            format="json",
            verbose=False,
            number_of_presamples=0,
        )
    elapsed = time.perf_counter() - start
    orders, coeffs, errors = _parse_pysecdec_json_series(series)
    refinement_pairs = [
        (int(start_n), int(stop_n))
        for start_n, stop_n in re.findall(r"\bn:\s*(\d+)\s*->\s*(\d+)", verbose_log)
    ]
    direct_nm = [
        (int(n_value), int(m_value))
        for n_value, m_value in re.findall(r"\bn\s+(\d+),\s*m\s+(\d+)", verbose_log)
    ]
    all_n_values = [n for n, _m in direct_nm] + [stop_n for _start_n, stop_n in refinement_pairs]
    max_observed_n = max(all_n_values) if all_n_values else None
    integral_records = _parse_pysecdec_integral_records(verbose_log)
    record_new_n_values = [
        int(record["new_n"])
        for record in integral_records.values()
        if int(record.get("new_n", 0)) > 0
    ]
    effective_record_samples = int(sum(record_new_n_values) * int(args.qmc_shifts))
    return {
        "orders": orders,
        "coefficients": coeffs,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "requested_maxeval_budget": int(n_points) * int(args.qmc_shifts),
        "requested_n_points": int(n_points),
        "observed_max_n_points": max_observed_n,
        "observed_refinement_count": len(refinement_pairs),
        "observed_refinements": refinement_pairs[:20],
        "per_integral_records": integral_records,
        "per_integral_record_count": len(record_new_n_values),
        "effective_record_samples": effective_record_samples,
        "average_record_n_points": (
            float(sum(record_new_n_values) / len(record_new_n_values))
            if record_new_n_values
            else None
        ),
        "captured_verbose_log": verbose_log if bool(args.keep_pysecdec_verbose_log) else "",
    }


def _make_row(
    n_points: int,
    args: argparse.Namespace,
    fsd: dict[str, Any],
    pysecdec: dict[str, Any],
    target_coeffs: list[complex],
    order_index: int,
) -> dict[str, Any]:
    fsd_coeffs = complex_list_from_json(fsd.get("display", {}).get("coefficients", []))
    fsd_errors = complex_list_from_json(fsd.get("display", {}).get("errors", []))
    py_coeffs, py_errors = _pysecdec_coefficients_in_fsd_convention(args, fsd, pysecdec)
    target = target_coeffs[order_index]
    fsd_diagnostics = fsd.get("integration_diagnostics") or fsd.get("diagnostics") or {}
    fsd_group_count = int(fsd_diagnostics.get("qmc_sector_group_count", 0) or 0)
    fsd_raw_samples = int(fsd.get("samples", 0))
    fsd_observed_n = (
        fsd_raw_samples // (fsd_group_count * int(args.qmc_shifts))
        if fsd_group_count > 0 and int(args.qmc_shifts) > 0
        else None
    )
    return {
        "n_points_per_shift": int(n_points),
        "qmc_shifts": int(args.qmc_shifts),
        "fsd_support_groups": fsd_group_count,
        "fsd_observed_n_points": fsd_observed_n,
        "fsd_raw_sector_samples": fsd_raw_samples,
        "fsd_elapsed_seconds": float(fsd.get("_comparison_elapsed_seconds", fsd.get("elapsed_seconds", 0.0))),
        "laurent_labels": list(_labels_from_result(fsd)),
        "target_coefficients": list(target_coeffs),
        "fsd_coefficients": list(fsd_coeffs),
        "fsd_errors": list(fsd_errors),
        "pysecdec_coefficients": list(py_coeffs),
        "pysecdec_errors": list(py_errors),
        "fsd_coeff": fsd_coeffs[order_index],
        "fsd_error": fsd_errors[order_index],
        "fsd_diff": fsd_coeffs[order_index] - target,
        "fsd_minus_pysecdec": fsd_coeffs[order_index] - py_coeffs[order_index],
        "pysecdec_requested_eval_budget": int(pysecdec["requested_maxeval_budget"]),
        "pysecdec_observed_max_n": pysecdec.get("observed_max_n_points"),
        "pysecdec_observed_refinements": int(pysecdec.get("observed_refinement_count", 0)),
        "pysecdec_integral_records": int(pysecdec.get("per_integral_record_count", 0)),
        "pysecdec_effective_record_samples": int(pysecdec.get("effective_record_samples", 0)),
        "pysecdec_average_record_n": pysecdec.get("average_record_n_points"),
        "pysecdec_elapsed_seconds": float(pysecdec["elapsed_seconds"]),
        "pysecdec_coeff": py_coeffs[order_index],
        "pysecdec_error": py_errors[order_index],
        "pysecdec_diff": py_coeffs[order_index] - target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-file", type=Path, default=ROOT / "examples/runs/dot_triangle.yaml")
    parser.add_argument(
        "--fsd-prepared-output",
        type=Path,
        default=None,
        help=(
            "Prepared FSD bundle directory. When set, the FSD side is run as "
            "'FSD.py integrate --output DIR' instead of the single-shot "
            "'FSD.py --run RUN_FILE' path."
        ),
    )
    parser.add_argument(
        "--kinematics-file",
        type=Path,
        default=None,
        help=(
            "Kinematics YAML for the pySecDec side. Defaults to the "
            "'kinematics' entry in --run-file, falling back to the triangle "
            "example when absent."
        ),
    )
    parser.add_argument("--target-file", type=Path, default=ROOT / "examples/outputs/dot_triangle_pysecdec_target.json")
    parser.add_argument(
        "--target-source",
        choices=["file", "oneloop-sector"],
        default="file",
        help=(
            "Comparison target source. 'oneloop-sector' uses OneLOopBridge for "
            "the matching one-loop triangle/box kinematics and converts the "
            "result to the DOT sector convention."
        ),
    )
    parser.add_argument(
        "--oneloop-integral",
        choices=["triangle", "box"],
        default="triangle",
        help="One-loop integral used when --target-source oneloop-sector is selected.",
    )
    parser.add_argument(
        "--fsd-prefactor-convention",
        choices=["sector", "pysecdec"],
        default="pysecdec",
        help="FSD display convention used in the comparison table.",
    )
    parser.add_argument(
        "--pysecdec-shared",
        type=Path,
        default=ROOT / ".pysecdec_build/fsd_psd_triangle/fsd_psd_triangle_pylink.so",
    )
    parser.add_argument("--sample-counts", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--qmc-shifts", type=int, default=16)
    parser.add_argument("--qmc-korobov-alpha", type=int, default=3)
    parser.add_argument(
        "--qmc-order",
        choices=["linear", "radical-inverse", "gray"],
        default="linear",
        help="QMCPy lattice order used when --qmc-lattice-backend qmcpy is selected.",
    )
    parser.add_argument(
        "--qmc-lattice-backend",
        choices=["qmcpy", "cbcpt-dn1-100"],
        default="cbcpt-dn1-100",
        help="Independent FSD lattice backend used for the comparison. Default: cbcpt-dn1-100.",
    )
    parser.add_argument(
        "--pysecdec-generatingvectors",
        choices=["default", "cbcpt_dn1_100", "cbcpt_dn2_6", "cbcpt_cfftw1_6", "cbcpt_cfftw2_10"],
        default="default",
        help="pySecDec generated-integrator generating vector table.",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument(
        "--fsd-extra-arg",
        action="append",
        default=[],
        help=(
            "Additional single argument forwarded verbatim to FSD.py. Repeat "
            "for flag-only options such as --explicit; valued options are best "
            "kept in the run YAML used by --run-file."
        ),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--s", type=float, default=-1.0)
    parser.add_argument("--s12", type=float, default=-1.0)
    parser.add_argument("--s23", type=float, default=-1.0)
    parser.add_argument("--m", type=float, default=0.0)
    parser.add_argument("--mode", choices=["massless", "massive"], default="massless")
    parser.add_argument("--order", default="0", help="Laurent order to print, e.g. 0 or eps^0.")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--sector-order-limit",
        type=int,
        default=0,
        help=(
            "Print a raw sector/order comparison table using pySecDec verbose "
            "records. 0 disables it; -1 prints every matched row."
        ),
    )
    parser.add_argument(
        "--sector-order-sort",
        choices=["index", "abs-diff", "pull"],
        default="index",
        help="Ordering for --sector-order-limit rows.",
    )
    parser.add_argument(
        "--pysecdec-verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Capture pySecDec's verbose integration log so the table can report "
            "whether it refined sector/order lattices beyond the requested minn."
        ),
    )
    parser.add_argument(
        "--keep-pysecdec-verbose-log",
        action="store_true",
        help="Store the captured pySecDec verbose text in --output-json.",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=None,
        help="Optional override forwarded to FSD; defaults come from --run-file/FSD.py.",
    )
    parser.add_argument(
        "--medium-precision-stability-threshold",
        type=float,
        default=None,
        help="Optional override forwarded to FSD; defaults come from --run-file/FSD.py.",
    )
    parser.add_argument(
        "--high-precision-stability-threshold",
        type=float,
        default=None,
        help="Optional override forwarded to FSD; defaults come from --run-file/FSD.py.",
    )
    args = parser.parse_args()

    args.run_file = args.run_file.expanduser().resolve()
    if args.fsd_prepared_output is not None:
        args.fsd_prepared_output = args.fsd_prepared_output.expanduser().resolve()
        if not args.fsd_prepared_output.exists():
            raise FileNotFoundError(f"prepared FSD bundle not found: {args.fsd_prepared_output}")
    if args.kinematics_file is None:
        args.kinematics_file = _path_from_run_file(
            args.run_file,
            "kinematics",
            ROOT / "examples/graphs/triangle_kinematics.yaml",
        )
    args.kinematics_file = args.kinematics_file.expanduser().resolve()
    args.target_file = args.target_file.expanduser().resolve()
    args.pysecdec_shared = args.pysecdec_shared.expanduser().resolve()
    if not args.pysecdec_shared.exists():
        raise FileNotFoundError(
            f"pySecDec shared library not found: {args.pysecdec_shared}. "
            "Generate it first with --target pysecdec or --dot-engine pysecdec."
        )

    target_coeffs, target_label = _resolve_target(args)
    first_result_labels: list[str] | None = None
    rows: list[dict[str, Any]] = []
    sector_order_rows: list[dict[str, Any]] = []
    sector_order_warnings: list[str] = []
    with tempfile.TemporaryDirectory(prefix="fsd-qmc-compare-") as tmp:
        tmpdir = Path(tmp)
        for n_points in args.sample_counts:
            result_path = tmpdir / f"fsd_qmc_{n_points}.json"
            fsd = _run_fsd(args, n_points, result_path, target_coeffs)
            labels = _labels_from_result(fsd)
            if first_result_labels is None:
                first_result_labels = labels
            order_index = _order_index(labels, args.order)
            pysecdec = _run_pysecdec(args, n_points)
            rows.append(_make_row(n_points, args, fsd, pysecdec, target_coeffs, order_index))
            if int(args.sector_order_limit) != 0:
                parsed_sector_rows, warning = _sector_order_rows(
                    fsd,
                    pysecdec.get("per_integral_records", {}),
                )
                if warning is not None:
                    sector_order_warnings.append(f"N/shift {n_points}: {warning}")
                for sector_row in parsed_sector_rows:
                    sector_row["n_points_per_shift"] = int(n_points)
                    sector_order_rows.append(sector_row)

    table = PrettyTable()
    order_label = first_result_labels[_order_index(first_result_labels, args.order)] if first_result_labels else args.order
    table.field_names = [
        "N/shift",
        "FSD groups",
        "FSD max n",
        "FSD raw smpl",
        "FSD t [s]",
        f"FSD {order_label} diff",
        f"FSD {order_label} err",
        "pySecDec req eval",
        "pySecDec max n",
        "pySecDec refs",
        "pySecDec eff smpl",
        "pySecDec t [s]",
        f"FSD-pySecDec {order_label}",
        f"pySecDec {order_label} diff",
        f"pySecDec {order_label} err",
    ]
    for row in rows:
        table.add_row(
            [
                row["n_points_per_shift"],
                row["fsd_support_groups"],
                row["fsd_observed_n_points"] if row["fsd_observed_n_points"] is not None else "n/a",
                row["fsd_raw_sector_samples"],
                f"{row['fsd_elapsed_seconds']:.3f}",
                _fmt_scientific(row["fsd_diff"]),
                _fmt_scientific(row["fsd_error"]),
                row["pysecdec_requested_eval_budget"],
                row["pysecdec_observed_max_n"] if row["pysecdec_observed_max_n"] is not None else "n/a",
                row["pysecdec_observed_refinements"],
                row["pysecdec_effective_record_samples"] or "n/a",
                f"{row['pysecdec_elapsed_seconds']:.3f}",
                _fmt_scientific(row["fsd_minus_pysecdec"]),
                _fmt_scientific(row["pysecdec_diff"]),
                _fmt_scientific(row["pysecdec_error"]),
            ]
        )
    print(table)
    print(
        "Note: FSD raw samples = support groups * observed lattice n * shifts. "
        "Support groups split a sector by Laurent orders with different "
        "effective endpoint support. "
        "pySecDec req eval is the requested QMC budget, while pySecDec max n "
        "and refs are parsed from its verbose refinement log when available; "
        "pySecDec may refine hard sector/order integrals beyond the nominal "
        "minn/maxeval request. Target: "
        f"{target_label}; displayed convention: {args.fsd_prefactor_convention}."
    )

    if int(args.sector_order_limit) != 0:
        for warning in sector_order_warnings:
            print(f"Sector/order comparison warning: {warning}")
        if args.sector_order_sort == "abs-diff":
            sorted_sector_rows = sorted(
                sector_order_rows,
                key=lambda row: abs(row["diff"]),
                reverse=True,
            )
        elif args.sector_order_sort == "pull":
            sorted_sector_rows = sorted(
                sector_order_rows,
                key=lambda row: float(row["pull"]),
                reverse=True,
            )
        else:
            sorted_sector_rows = sorted(
                sector_order_rows,
                key=lambda row: (
                    int(row["n_points_per_shift"]),
                    int(row["sector_id"]),
                    int(row["order"]),
                ),
            )
        if int(args.sector_order_limit) > 0:
            sorted_sector_rows = sorted_sector_rows[: int(args.sector_order_limit)]
        if sorted_sector_rows:
            sector_table = PrettyTable()
            sector_table.field_names = [
                "N/shift",
                "sector",
                "order",
                "FSD raw",
                "FSD err",
                "pySecDec raw",
                "pySecDec err",
                "diff",
                "py n",
                "FSD smpl",
            ]
            for row in sorted_sector_rows:
                sector_table.add_row(
                    [
                        row["n_points_per_shift"],
                        row["sector_name"],
                        row["label"].replace("eps^", ""),
                        _fmt_real(row["fsd_coeff"]),
                        _fmt_scientific(row["fsd_error"]),
                        _fmt_real(row["pysecdec_coeff"]),
                        _fmt_scientific(row["pysecdec_error"]),
                        _fmt_scientific(row["diff"]),
                        row["pysecdec_n"],
                        row["fsd_samples"],
                    ]
                )
            print()
            print("Raw sector/order comparison before global prefactor convolution:")
            print(sector_table)
            print(
                "Sector/order pySecDec values are parsed from its verbose text log, "
                "which prints coefficients with limited decimal precision.  Treat "
                "these rows as structural and variance checks; aggregate rows use "
                "the full JSON series returned by pySecDec."
            )

    if args.output_json is not None:
        payload = {
            "run_file": str(args.run_file),
            "fsd_prepared_output": str(args.fsd_prepared_output) if args.fsd_prepared_output is not None else None,
            "target_file": str(args.target_file),
            "target_source": str(args.target_source),
            "target_label": target_label,
            "pysecdec_shared": str(args.pysecdec_shared),
            "order": order_label,
            "qmc_shifts": int(args.qmc_shifts),
            "qmc_korobov_alpha": int(args.qmc_korobov_alpha),
            "qmc_lattice_backend": str(args.qmc_lattice_backend),
            "qmc_order": str(args.qmc_order),
            "fsd_prefactor_convention": str(args.fsd_prefactor_convention),
            "pysecdec_generatingvectors": str(args.pysecdec_generatingvectors),
            "rows": [
                {key: _jsonable(value) for key, value in row.items()}
                for row in rows
            ],
            "sector_order_rows": [
                {key: _jsonable(value) for key, value in row.items()}
                for row in sector_order_rows
            ],
            "sector_order_warnings": sector_order_warnings,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
