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
from pathlib import Path
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


def _run_fsd(args: argparse.Namespace, n_points: int, result_path: Path) -> dict[str, Any]:
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
        "--samples-per-iter",
        str(n_points),
        "--max-iter",
        "1",
        "--batch-size",
        str(args.batch_size or n_points),
        "--workers",
        str(args.workers),
        "--target",
        str(args.target_file),
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


def _run_pysecdec(args: argparse.Namespace, n_points: int) -> dict[str, Any]:
    kinematics = load_kinematics(args.kinematics_file)
    library = IntegralLibrary(str(args.pysecdec_shared.resolve()))
    library.use_Qmc(
        transform=f"korobov{args.qmc_korobov_alpha}",
        minn=int(n_points),
        minm=int(args.qmc_shifts),
        maxeval=int(n_points) * int(args.qmc_shifts),
        cputhreads=int(args.workers),
        seed=int(args.seed),
        epsrel=1.0e-99,
        epsabs=1.0e-99,
    )
    start = time.perf_counter()
    series = library(
        real_parameters=kinematics.parameter_values,
        format="json",
        verbose=False,
        number_of_presamples=0,
    )
    elapsed = time.perf_counter() - start
    orders, coeffs, errors = _parse_pysecdec_json_series(series)
    return {
        "orders": orders,
        "coefficients": coeffs,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "maxeval_budget": int(n_points) * int(args.qmc_shifts),
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
    py_coeffs = pysecdec["coefficients"]
    py_errors = pysecdec["errors"]
    target = target_coeffs[order_index]
    return {
        "n_points_per_shift": int(n_points),
        "qmc_shifts": int(args.qmc_shifts),
        "fsd_raw_sector_samples": int(fsd.get("samples", 0)),
        "fsd_elapsed_seconds": float(fsd.get("_comparison_elapsed_seconds", fsd.get("elapsed_seconds", 0.0))),
        "fsd_coeff": fsd_coeffs[order_index],
        "fsd_error": fsd_errors[order_index],
        "fsd_diff": fsd_coeffs[order_index] - target,
        "pysecdec_eval_budget": int(pysecdec["maxeval_budget"]),
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
        "--pysecdec-shared",
        type=Path,
        default=ROOT / ".pysecdec_build/fsd_psd_triangle/fsd_psd_triangle_pylink.so",
    )
    parser.add_argument("--sample-counts", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--qmc-shifts", type=int, default=16)
    parser.add_argument("--qmc-korobov-alpha", type=int, default=3)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--order", default="0", help="Laurent order to print, e.g. 0 or eps^0.")
    parser.add_argument("--output-json", type=Path, default=None)
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

    target = target_from_result_file(args.target_file, "pysecdec")
    first_result_labels: list[str] | None = None
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="fsd-qmc-compare-") as tmp:
        tmpdir = Path(tmp)
        for n_points in args.sample_counts:
            result_path = tmpdir / f"fsd_qmc_{n_points}.json"
            fsd = _run_fsd(args, n_points, result_path)
            labels = _labels_from_result(fsd)
            if first_result_labels is None:
                first_result_labels = labels
            order_index = _order_index(labels, args.order)
            pysecdec = _run_pysecdec(args, n_points)
            rows.append(_make_row(n_points, args, fsd, pysecdec, target.coefficients, order_index))

    table = PrettyTable()
    order_label = first_result_labels[_order_index(first_result_labels, args.order)] if first_result_labels else args.order
    table.field_names = [
        "N/shift",
        "FSD raw smpl",
        "FSD t [s]",
        f"FSD {order_label} diff",
        f"FSD {order_label} err",
        "pySecDec maxeval",
        "pySecDec t [s]",
        f"pySecDec {order_label} diff",
        f"pySecDec {order_label} err",
    ]
    for row in rows:
        table.add_row(
            [
                row["n_points_per_shift"],
                row["fsd_raw_sector_samples"],
                f"{row['fsd_elapsed_seconds']:.3f}",
                _fmt_scientific(row["fsd_diff"]),
                _fmt_scientific(row["fsd_error"]),
                row["pysecdec_eval_budget"],
                f"{row['pysecdec_elapsed_seconds']:.3f}",
                _fmt_scientific(row["pysecdec_diff"]),
                _fmt_scientific(row["pysecdec_error"]),
            ]
        )
    print(table)
    print(
        "Note: FSD raw samples = sectors * N/shift * shifts. "
        "pySecDec maxeval is the public QMC budget; pySecDec does not expose "
        "the same per-sector sample accounting through the pylink API."
    )

    if args.output_json is not None:
        payload = {
            "run_file": str(args.run_file),
            "target_file": str(args.target_file),
            "pysecdec_shared": str(args.pysecdec_shared),
            "order": order_label,
            "qmc_shifts": int(args.qmc_shifts),
            "qmc_korobov_alpha": int(args.qmc_korobov_alpha),
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
