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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kinematics import load_kinematics  # noqa: E402
from pysecdec_bridge import _parse_pysecdec_json_series  # noqa: E402
from result_io import complex_list_from_json, load_result_json, target_from_result_file  # noqa: E402


def _complex_json(value: complex) -> dict[str, float]:
    return {"re": float(value.real), "im": float(value.imag)}


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
    cmd = [
        sys.executable,
        str(ROOT / "FSD.py"),
        "--run",
        str(args.run_file),
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
        "--stability-threshold",
        str(args.stability_threshold),
        "--medium-precision-stability-threshold",
        str(args.medium_precision_stability_threshold),
        "--high-precision-stability-threshold",
        str(args.high_precision_stability_threshold),
        "--no-progress",
        "--quiet-summary",
        "--json",
        "--result-path",
        str(result_path),
    ]
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
        "fsd_coeff": fsd_coeffs[order_index],
        "fsd_error": fsd_errors[order_index],
        "fsd_diff": fsd_coeffs[order_index] - target,
        "fsd_minus_pysecdec": fsd_coeffs[order_index] - py_coeffs[order_index],
        "pysecdec_requested_eval_budget": int(pysecdec["requested_maxeval_budget"]),
        "pysecdec_observed_max_n": pysecdec.get("observed_max_n_points"),
        "pysecdec_observed_refinements": int(pysecdec.get("observed_refinement_count", 0)),
        "pysecdec_elapsed_seconds": float(pysecdec["elapsed_seconds"]),
        "pysecdec_coeff": py_coeffs[order_index],
        "pysecdec_error": py_errors[order_index],
        "pysecdec_diff": py_coeffs[order_index] - target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-file", type=Path, default=ROOT / "examples/runs/dot_triangle.yaml")
    parser.add_argument("--kinematics-file", type=Path, default=ROOT / "examples/graphs/triangle_kinematics.yaml")
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
        help="QMCPy lattice order used by FSD. Linear is closest to pySecDec's direct lattice loop.",
    )
    parser.add_argument(
        "--qmc-lattice-backend",
        choices=["qmcpy", "cbcpt-dn1-100"],
        default="qmcpy",
        help="Independent FSD lattice backend used for the comparison.",
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
    parser.add_argument("--stability-threshold", type=float, default=1.0e-12)
    parser.add_argument("--medium-precision-stability-threshold", type=float, default=1.0e-14)
    parser.add_argument("--high-precision-stability-threshold", type=float, default=0.0)
    args = parser.parse_args()

    args.run_file = args.run_file.expanduser().resolve()
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

    if args.output_json is not None:
        payload = {
            "run_file": str(args.run_file),
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
                {
                    key: (_complex_json(value) if isinstance(value, complex) else value)
                    for key, value in row.items()
                }
                for row in rows
            ],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
