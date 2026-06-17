"""Universal formula-cache warmup and verification helpers.

The cache warmed here contains endpoint projector, regular Taylor, and
chain-rule formula assets.  These signatures are topology-independent in the
projector backend, so the generated JSON/evaluator sidecars can be distributed
with FSD and reused by later DOT topologies that expose the same endpoint
structure.  Topology-specific prepared bundles and two-stage sector evaluators
are deliberately outside this module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any

from colorama import Fore, Style
from prettytable import PrettyTable

from cache_utils import formula_cache_dir
from definitions import IntegralRequest
from dot_topology import clear_dot_bundle_cache, get_dot_bundle
from generation_timing import GenerationProgress
from integrand import build_topology
from integrator import integrate
from sectors_generator import generate_sectors


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CacheWarmCase:
    """One DOT example used to exercise universal formula-cache signatures."""

    name: str
    loop_count: int
    dot_file: str
    kinematics_file: str


EXAMPLE_CASES: tuple[CacheWarmCase, ...] = (
    CacheWarmCase("triangle", 1, "examples/graphs/triangle.dot", "examples/graphs/triangle_kinematics.yaml"),
    CacheWarmCase("box", 1, "examples/graphs/box.dot", "examples/graphs/box_kinematics.yaml"),
    CacheWarmCase("kite_2loop", 2, "examples/graphs/kite_2loop.dot", "examples/graphs/kite_2loop_kinematics.yaml"),
    CacheWarmCase("three_point_2loop", 2, "examples/graphs/three_point_2loop.dot", "examples/graphs/three_point_2loop_kinematics.yaml"),
    CacheWarmCase("three_point_2loop_6line", 2, "examples/graphs/three_point_2loop_6line.dot", "examples/graphs/three_point_2loop_6line_kinematics.yaml"),
    CacheWarmCase("double_box", 2, "examples/graphs/double_box.dot", "examples/graphs/double_box_kinematics.yaml"),
    CacheWarmCase("self_energy_3loop", 3, "examples/graphs/self_energy_3loop.dot", "examples/graphs/self_energy_3loop_kinematics.yaml"),
    CacheWarmCase("three_point_3loop", 3, "examples/graphs/three_point_3loop.dot", "examples/graphs/three_point_3loop_kinematics.yaml"),
    CacheWarmCase("three_point_3loop_8line", 3, "examples/graphs/three_point_3loop_8line.dot", "examples/graphs/three_point_3loop_8line_kinematics.yaml"),
    CacheWarmCase("triple_box", 3, "examples/graphs/triple_box.dot", "examples/graphs/triple_box_kinematics.yaml"),
)


def _deepest_order(loop_count: int) -> int:
    """Return the expected deepest pole order for a scalar L-loop integral."""

    return -2 * int(loop_count)


def _count_cache_files() -> dict[str, int]:
    """Count generated universal cache files by family."""

    root = formula_cache_dir().expanduser()
    patterns = {
        "endpoint_projector_json": "endpoint_projector_*.json",
        "endpoint_projector_evaluator": "endpoint_projector_*.eval_*.bin.gz",
        "regular_taylor_json": "regular_taylor_*.json",
        "regular_taylor_evaluator": "regular_taylor_*.eval_*.bin.gz",
        "chain_rule_json": "chain_rule_*.json",
        "chain_rule_evaluator": "chain_rule_*.eval_*.bin.gz",
    }
    if not root.exists():
        return {key: 0 for key in patterns}
    return {key: len(list(root.glob(pattern))) for key, pattern in patterns.items()}


def _cache_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    """Return cache-file count differences."""

    keys = set(before) | set(after)
    return {key: int(after.get(key, 0) - before.get(key, 0)) for key in sorted(keys)}


def _select_cases(request: IntegralRequest) -> list[CacheWarmCase]:
    """Select examples from CLI cache filters."""

    wanted_loops = set(int(loop) for loop in request.cache_loop_counts)
    by_name = {case.name: case for case in EXAMPLE_CASES}
    if request.cache_cases:
        selected: list[CacheWarmCase] = []
        for name in request.cache_cases:
            if name not in by_name:
                known = ", ".join(sorted(by_name))
                raise ValueError(f"unknown cache case {name!r}; known cases: {known}")
            selected.append(by_name[name])
        return selected
    return [case for case in EXAMPLE_CASES if case.loop_count in wanted_loops]


def _case_request(base: IntegralRequest, case: CacheWarmCase, workdir: Path) -> IntegralRequest:
    """Create an ordinary DOT request for one cache-warm case."""

    case_workdir = workdir / case.name
    result_path = case_workdir / "result.json"
    return replace(
        base,
        command="run",
        integral="dot",
        dot_file=str((ROOT / case.dot_file).resolve()),
        kinematics_file=str((ROOT / case.kinematics_file).resolve()),
        graph_name=None,
        dot_engine="fsd",
        output=None,
        result_path=str(result_path),
        prefactor_convention="pysecdec",
        target_args=None,
        refresh_target=False,
        sampling_mode="democratic",
        democratic_samples_per_sector=int(base.cache_verify_samples_per_sector),
        max_iter=1,
        min_iter=1,
        samples_per_iter=max(1, int(base.cache_verify_samples_per_sector)),
        batch_size=max(1, int(base.batch_size or base.cache_verify_samples_per_sector)),
        workers=1,
        no_progress=True,
        quiet_summary=True,
        json=True,
        max_eps_order=int(base.max_eps_order),
        max_eps_order_explicit=True,
        subtraction_backend="projector-formula",
        sector_evaluator_backend="projector",
        dual_evaluator_mode="symbolic-derivatives",
    )


def _configure_laurent_range(topology: Any, sectors: list[Any], max_eps_order: int) -> None:
    """Apply the scalar ``eps^(-2L)..eps^N`` range used by cache warmup."""

    parametric = topology.parametric_representation
    loop_count = int(parametric.loop_count if parametric is not None else 1)
    min_order = _deepest_order(loop_count)
    if max_eps_order < min_order:
        raise ValueError(f"cache max epsilon order {max_eps_order} is below eps^{min_order}")
    max_depth = max((len(sector.singular_axes) for sector in sectors), default=0)
    if max_depth > -min_order:
        examples = [
            sector.name for sector in sectors if len(sector.singular_axes) == max_depth
        ][:5]
        raise ValueError(
            f"sector endpoint pole depth {max_depth} exceeds 2L={-min_order}; "
            f"examples: {', '.join(examples)}"
        )
    topology.set_laurent_range(min_order, max_eps_order)


def _prepare_universal_formulas(
    request: IntegralRequest,
    topology: Any,
    sectors: list[Any],
    progress: GenerationProgress | None,
) -> float:
    """Prepare universal projector formula caches for a case."""

    topology.regular_taylor_formula_signature_limit = request.regular_taylor_signature_limit
    topology.regular_taylor_formula_volume_limit = request.regular_taylor_formula_volume_limit
    topology.regular_taylor_formula_axis_limit = request.regular_taylor_formula_axis_limit
    topology.chain_rule_formula_signature_limit = request.chain_rule_formula_signature_limit
    topology.chain_rule_formula_output_length_limit = request.chain_rule_formula_output_length_limit
    topology.direct_projector_cache_term_threshold = request.direct_projector_cache_term_threshold
    start = time.perf_counter()
    topology.prepare_dual_evaluators(sectors, request.dual_evaluator_mode, progress=progress)
    topology.prepare_endpoint_projector_formulas(sectors, progress=progress)
    topology.prepare_regular_taylor_formulas(sectors, progress=progress)
    topology.prepare_chain_rule_formulas(sectors, progress=progress)
    return time.perf_counter() - start


def _verify_case(
    request: IntegralRequest,
    topology: Any,
    sectors: list[Any],
) -> dict[str, Any]:
    """Run low-stat democratic sampling to prove the warmed formulas are usable."""

    start = time.perf_counter()
    result = integrate(request, topology, sectors, target=None)
    elapsed = time.perf_counter() - start
    diagnostics = dict(result.diagnostics)
    return {
        "samples": int(result.samples),
        "elapsed_seconds": elapsed,
        "min_avg_eval_us_per_sample_sector": diagnostics.get("min_avg_eval_us_per_sample_sector"),
        "max_avg_eval_us_per_sample_sector": diagnostics.get("max_avg_eval_us_per_sample_sector"),
        "max_abs_weight_sector": diagnostics.get("max_abs_weight_sector"),
        "sector_timing_count": len(diagnostics.get("sector_timing_summary", [])),
        "interrupted": bool(diagnostics.get("interrupted", False)),
    }


def _sector_stats(sectors: list[Any]) -> dict[str, Any]:
    """Summarize sector endpoint structure for reports."""

    axis_counts = Counter(len(sector.singular_axes) for sector in sectors)
    endpoint_inputs = Counter()
    for sector in sectors:
        endpoint_inputs[
            (
                tuple(int(axis) for axis in sector.singular_axes),
                tuple(int(power) for power in sector.f_monomial_powers),
                tuple(int(power) for power in sector.u_monomial_powers),
                tuple(float(power) for power in sector.measure_monomial_powers),
                tuple(int(order) for order in sector.endpoint_taylor_orders),
            )
        ] += 1
    return {
        "sector_count": len(sectors),
        "max_integration_dim": max((sector.integration_dim for sector in sectors), default=0),
        "max_singular_axes": max((len(sector.singular_axes) for sector in sectors), default=0),
        "axis_count_histogram": {str(key): value for key, value in sorted(axis_counts.items())},
        "distinct_declarative_endpoint_inputs": len(endpoint_inputs),
    }


def _three_loop_estimate(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate 3L universal-cache cost from measured lower-loop warmup."""

    two_loop_rows = [row for row in case_rows if row.get("loop_count") == 2 and row.get("status") == "ok"]
    if not two_loop_rows:
        return {
            "status": "insufficient_data",
            "message": "No successful 2L rows were available for extrapolation.",
        }
    max_per_signature = 0.0
    total_new = 0
    for row in two_loop_rows:
        new_files = row.get("cache_delta", {})
        new_json = (
            int(new_files.get("endpoint_projector_json", 0))
            + int(new_files.get("regular_taylor_json", 0))
            + int(new_files.get("chain_rule_json", 0))
        )
        if new_json > 0:
            total_new += new_json
            max_per_signature = max(
                max_per_signature,
                float(row.get("universal_cache_seconds", 0.0)) / float(new_json),
            )
    if max_per_signature <= 0.0:
        max_per_signature = max(
            float(row.get("universal_cache_seconds", 0.0)) for row in two_loop_rows
        )
    # The hard 3L signatures observed in the triple-box work are dominated by
    # six-axis source/formula families.  Treat the lower-loop extrapolation as
    # a lower bound only.  The calibrated timings below come from the PSD2 and
    # all-sector triple-box experiments: one direct six-axis regular formula
    # reached 823 s locally, and one PSD2 source-expression group exceeded 90 s
    # before being stopped.  These numbers are deliberately kept in the report
    # so cluster planning is based on the hard family rather than the easy 2L
    # cache-hit path.
    estimated_signature_counts = {
        "small_3l_examples": 250,
        "triple_box_hard_six_axis_family": 2000,
    }
    optimistic_estimates = {
        name: count * max_per_signature
        for name, count in estimated_signature_counts.items()
    }
    hard_signature_seconds = {
        "observed_direct_six_axis_regular_formula_seconds": 823.0,
        "observed_psd2_source_group_lower_bound_seconds": 90.0,
    }
    calibrated_serial = {
        name: {
            "90s_group_lower_bound_seconds": count * hard_signature_seconds[
                "observed_psd2_source_group_lower_bound_seconds"
            ],
            "823s_direct_formula_seconds": count * hard_signature_seconds[
                "observed_direct_six_axis_regular_formula_seconds"
            ],
        }
        for name, count in estimated_signature_counts.items()
    }
    return {
        "status": "rough_extrapolation",
        "basis": (
            "Uses the slowest measured 2L universal-cache seconds per newly "
            "written JSON formula as an optimistic lower bound, plus hard "
            "six-axis calibration timings from the triple-box PSD2 studies."
        ),
        "measured_2l_new_formula_count": total_new,
        "max_2l_seconds_per_new_formula": max_per_signature,
        "optimistic_lower_bound_seconds": optimistic_estimates,
        "optimistic_lower_bound_hours": {
            key: value / 3600.0 for key, value in optimistic_estimates.items()
        },
        "hard_signature_calibration_seconds": hard_signature_seconds,
        "calibrated_serial_seconds": calibrated_serial,
        "calibrated_serial_hours": {
            key: {
                label: seconds / 3600.0 for label, seconds in values.items()
            }
            for key, values in calibrated_serial.items()
        },
        "cluster_note": (
            "If the hard family is embarrassingly parallel and disk artifacts are "
            "written one signature at a time, divide the calibrated serial hours "
            "approximately by the number of independent workers, subject to RAM "
            "limits per worker."
        ),
    }


def _print_cache_summary(report: dict[str, Any]) -> None:
    """Render a compact terminal summary for cache warmup."""

    title = f"{Fore.CYAN}Universal FSD Formula Cache Warmup{Style.RESET_ALL}"
    print(title)
    print(f"cache root: {Fore.BLUE}{report['cache_root']}{Style.RESET_ALL}")
    table = PrettyTable()
    table.field_names = [
        "case",
        "L",
        "sectors",
        "cache gen [s]",
        "verify smpl",
        "min μs/smpl",
        "max μs/smpl",
        "status",
    ]
    for row in report["cases"]:
        verify = row.get("verification") or {}
        min_row = verify.get("min_avg_eval_us_per_sample_sector") or {}
        max_row = verify.get("max_avg_eval_us_per_sample_sector") or {}
        status = row.get("status", "unknown")
        color = Fore.GREEN if status == "ok" else Fore.RED
        table.add_row(
            [
                row.get("case"),
                row.get("loop_count"),
                row.get("sector_stats", {}).get("sector_count", "n/a"),
                f"{float(row.get('universal_cache_seconds', 0.0)):.3f}",
                verify.get("samples", "n/a"),
                f"{float(min_row.get('avg_eval_us_per_sample', 0.0)):.3g}"
                if min_row
                else "n/a",
                f"{float(max_row.get('avg_eval_us_per_sample', 0.0)):.3g}"
                if max_row
                else "n/a",
                f"{color}{status}{Style.RESET_ALL}",
            ]
        )
    print(table)
    estimate = report.get("three_loop_estimate", {})
    if estimate:
        print(f"{Fore.YELLOW}3L estimate:{Style.RESET_ALL} {estimate.get('status')}")
        for key, seconds in (estimate.get("optimistic_lower_bound_seconds") or {}).items():
            print(f"  optimistic {key}: {seconds:.3g}s ({seconds / 3600.0:.3g}h)")
        for key, values in (estimate.get("calibrated_serial_hours") or {}).items():
            parts = ", ".join(f"{label}={hours:.3g}h" for label, hours in values.items())
            print(f"  calibrated {key}: {parts}")
    print(f"report: {Fore.BLUE}{report['report_path']}{Style.RESET_ALL}")


def run_universal_cache_mode(
    request: IntegralRequest,
    logger: Any,
) -> dict[str, Any]:
    """Warm universal formula caches and verify them on selected DOT examples."""

    cases = _select_cases(request)
    workdir = Path(request.cache_workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    report_path = Path(request.cache_report_path).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_root": str(formula_cache_dir().expanduser().resolve()),
        "workdir": str(workdir),
        "selected_loop_counts": list(request.cache_loop_counts),
        "selected_cases": [case.name for case in cases],
        "max_eps_order": int(request.max_eps_order),
        "verification_samples_per_sector": int(request.cache_verify_samples_per_sector),
        "cases": [],
    }

    for case in cases:
        logger.info("warming universal cache for %s (L=%s)", case.name, case.loop_count)
        clear_dot_bundle_cache()
        case_request = _case_request(request, case, workdir)
        row: dict[str, Any] = {
            "case": case.name,
            "loop_count": case.loop_count,
            "dot_file": case.dot_file,
            "kinematics_file": case.kinematics_file,
        }
        cache_before = _count_cache_files()
        start = time.perf_counter()
        progress = None
        if not request.no_progress and not request.json:
            progress = GenerationProgress(logger, label=f"cache warm {case.name}")
        try:
            bundle = get_dot_bundle(case_request, progress=progress)
            topology = build_topology(case_request)
            sectors = generate_sectors(case_request)
            _configure_laurent_range(topology, sectors, int(request.max_eps_order))
            universal_seconds = _prepare_universal_formulas(
                case_request,
                topology,
                sectors,
                progress,
            )
            verify = _verify_case(case_request, topology, sectors)
            row.update(
                {
                    "status": "ok",
                    "total_seconds": time.perf_counter() - start,
                    "universal_cache_seconds": universal_seconds,
                    "generation_timings": bundle.timings.to_summary_dict(),
                    "sector_stats": _sector_stats(sectors),
                    "verification": verify,
                    "cache_before": cache_before,
                    "cache_after": _count_cache_files(),
                }
            )
            row["cache_delta"] = _cache_delta(row["cache_before"], row["cache_after"])
        except Exception as exc:
            row.update(
                {
                    "status": "error",
                    "error": str(exc),
                    "total_seconds": time.perf_counter() - start,
                    "cache_before": cache_before,
                    "cache_after": _count_cache_files(),
                }
            )
            row["cache_delta"] = _cache_delta(row["cache_before"], row["cache_after"])
            logger.exception("cache warm case %s failed", case.name)
        finally:
            if progress is not None:
                progress.close()
        report["cases"].append(row)

    if request.cache_estimate_3l:
        report["three_loop_estimate"] = _three_loop_estimate(report["cases"])

    tmp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, report_path)
    report["report_path"] = str(report_path)
    if request.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_cache_summary(report)
    return report
