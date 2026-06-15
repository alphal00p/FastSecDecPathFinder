#!/usr/bin/env python3
"""Command-line entry point for the modular FastSecDec v2 prototype.

This file intentionally contains only steering code: argument parsing,
kinematic validation, summary rendering, benchmark setup, integration launch,
and output formatting.  The sector declarations, black-box integrand
construction, and Havana sampling logic live in separate modules.
"""

from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import sys

from colorama import Fore, Style, init as colorama_init
import yaml

from benchmark import check_oneloop_bridge, compute_benchmark
from definitions import IntegralRequest, TargetDefinition
from formatting import (
    apply_global_convention,
    build_sector_result_rows,
    make_output,
    output_json,
    print_preintegration_summary,
    print_result_table,
    selected_prefactor_values,
    summary_data,
)
from dot_topology import get_dot_bundle
from generation_timing import GenerationProgress
from integrand import build_topology
from integrator import integrate
from pysecdec_bridge import run_pysecdec_package
from result_io import (
    environment_metadata,
    print_saved_results,
    request_metadata,
    result_output_path,
    target_from_result_file,
    write_result_json,
)
from sectors_generator import generate_sectors


def resolve_mode(m: float, requested: str) -> str:
    """Resolve the user-facing ``auto`` mode into massive or massless mode."""
    if requested != "auto":
        return requested
    return "massless" if abs(m) == 0.0 else "massive"


def validate_request(request: IntegralRequest) -> None:
    """Reject unsupported kinematics before building sectors or benchmarks."""
    if request.max_iter != -1 and request.max_iter <= 0:
        raise ValueError("--max-iter must be positive, or -1 for an unbounded run")
    if request.samples_per_iter <= 0:
        raise ValueError("--samples-per-iter must be positive")
    if request.batch_size < 0:
        raise ValueError("--batch-size must be >= 0, where 0 means one batch per worker chunk")
    if request.target_rel_accuracy is not None and request.target_rel_accuracy <= 0.0:
        raise ValueError("--target-rel-accuracy must be > 0 and is interpreted as a percent")
    if request.stability_threshold < 0.0:
        raise ValueError("--stability-threshold must be non-negative")
    if request.high_precision_stability_threshold < 0.0:
        raise ValueError("--high-precision-stability-threshold must be non-negative")
    if request.high_precision_stability_threshold > request.stability_threshold:
        raise ValueError(
            "--high-precision-stability-threshold must be <= --stability-threshold"
        )
    if request.stability_precision <= 0:
        raise ValueError("--stability-precision must be positive")
    if request.high_precision_stability_precision <= 0:
        raise ValueError("--high-precision-stability-precision must be positive")
    if (
        request.ibp_reduce_to_log_endpoint
        and request.subtraction_backend != "projector-formula"
    ):
        raise ValueError(
            "--IBP_reduce_to_log_endpoint is only supported with "
            "--subtraction-backend projector-formula"
        )
    if (
        request.force_regular_taylor_formulas
        and request.subtraction_backend != "projector-formula"
    ):
        raise ValueError(
            "--force-regular-taylor-formulas is only supported with "
            "--subtraction-backend projector-formula"
        )
    if request.regular_taylor_signature_limit < 0:
        raise ValueError("--regular-taylor-signature-limit must be >= 0")
    if request.regular_taylor_formula_volume_limit < 0:
        raise ValueError("--regular-taylor-formula-volume-limit must be >= 0")
    if request.regular_taylor_formula_axis_limit < 0:
        raise ValueError("--regular-taylor-formula-axis-limit must be >= 0")
    if request.chain_rule_formula_signature_limit < 0:
        raise ValueError("--chain-rule-formula-signature-limit must be >= 0")
    if request.direct_projector_cache_term_threshold < 0:
        raise ValueError("--direct-projector-cache-term-threshold must be >= 0")

    if request.integral == "dot":
        if request.dot_file is None:
            raise ValueError("DOT-file topology mode requires --dot-file")
        dot_path = Path(request.dot_file).expanduser()
        if not dot_path.is_file():
            raise ValueError(f"DOT-file topology does not exist: {dot_path}")
        if dot_path.suffix.lower() != ".dot":
            raise ValueError(f"DOT-file topology input must use a .dot suffix: {dot_path}")
        if request.kinematics_file is None:
            raise ValueError("DOT-file topology mode requires --kinematics")
        kin_path = Path(request.kinematics_file).expanduser()
        if not kin_path.is_file():
            raise ValueError(f"DOT kinematics YAML does not exist: {kin_path}")
        if request.sector_method == "geometric":
            normaliz = request.normaliz_executable or shutil.which("normaliz") or shutil.which("Normaliz")
            if normaliz is None:
                raise ValueError(
                    "--sector-method geometric requires Normaliz on PATH or --normaliz-executable; "
                    "use --sector-method geometric_ku or iterative when Normaliz is unavailable"
                )
        return

    if request.integral == "triangle":
        if request.s is None:
            raise ValueError("triangle integral requires --s")
        if request.mode == "massive":
            if request.m <= 0.0:
                raise ValueError("massive triangle mode requires --m > 0")
            if not (request.s < 4.0 * request.m * request.m):
                raise ValueError("massive triangle mode currently requires s < 4 m^2")
        elif request.mode == "massless":
            if abs(request.m) > 0.0:
                raise ValueError("massless triangle mode requires --m 0")
            if not (request.s < 0.0):
                raise ValueError(
                    "massless triangle mode currently requires Euclidean s < 0; "
                    "timelike or threshold kinematics require contour deformation or threshold regularization"
                )
        else:
            raise ValueError(f"unknown mode {request.mode!r}")
        return

    if request.integral == "box":
        if request.s12 is None or request.s23 is None:
            raise ValueError("box integral requires --s12 and --s23")
        if request.mode == "massive":
            if request.m <= 0.0:
                raise ValueError("massive box mode requires --m > 0")
            if not (
                request.s12 < 4.0 * request.m * request.m
                and request.s23 < 4.0 * request.m * request.m
            ):
                raise ValueError("massive box mode currently requires s12 < 4 m^2 and s23 < 4 m^2")
        elif request.mode == "massless":
            if abs(request.m) > 0.0:
                raise ValueError("massless box mode requires --m 0")
            if not (request.s12 < 0.0 and request.s23 < 0.0):
                raise ValueError(
                    "massless box mode currently requires Euclidean s12 < 0 and s23 < 0; "
                    "timelike or threshold kinematics require contour deformation or threshold regularization"
                )
        else:
            raise ValueError(f"unknown mode {request.mode!r}")
        return

    raise ValueError(f"unsupported integral {request.integral!r}")


def _normalise_run_key(key: str) -> str:
    """Map YAML option keys to argparse destination names."""
    return str(key).strip().replace("-", "_").lower()


def _is_numeric_token(token: object) -> bool:
    """Return whether a target token should be interpreted as a number."""
    try:
        float(str(token))
        return True
    except Exception:
        return False


def _resolve_yaml_path(value: object, base_dir: Path) -> str:
    """Resolve one YAML path relative to the run-file directory."""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _load_run_defaults(run_file: str | None) -> dict[str, object]:
    """Load CLI defaults from a YAML run preset.

    YAML keys mirror long CLI options without the leading ``--`` and may use
    either kebab-case or snake_case.  Path-like values in the YAML are resolved
    relative to the YAML file location; explicit CLI paths remain untouched by
    this helper because argparse applies them after defaults are installed.
    """
    if run_file is None:
        return {}
    run_path = Path(run_file).expanduser().resolve()
    data = yaml.safe_load(run_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{run_path}: run preset must contain a YAML mapping")

    base_dir = run_path.parent
    defaults: dict[str, object] = {"run_file": str(run_path)}
    path_keys = {
        "dot_file",
        "kinematics",
        "normaliz_executable",
        "pysecdec_workdir",
        "result_path",
        "log_file",
        "show_results",
    }
    for raw_key, raw_value in data.items():
        key = _normalise_run_key(raw_key)
        value = raw_value
        if key in {
            "pregenerate_dual_evaluators",
            "lazy_dual_evaluators_generation",
            "pregenerate_single_overall_dual_evaluator",
            "symbolic_derivatives",
        }:
            if bool(value):
                defaults["dual_evaluator_mode"] = {
                    "pregenerate_dual_evaluators": "pregenerate",
                    "lazy_dual_evaluators_generation": "lazy",
                    "pregenerate_single_overall_dual_evaluator": "single-overall",
                    "symbolic_derivatives": "symbolic-derivatives",
                }[key]
            continue
        if key in path_keys and value is not None:
            value = _resolve_yaml_path(value, base_dir)
        elif key == "target" and value is not None:
            tokens = value if isinstance(value, list) else [value]
            resolved_tokens: list[str] = []
            for token in tokens:
                if str(token) == "pysecdec" or _is_numeric_token(token):
                    resolved_tokens.append(str(token))
                else:
                    resolved_tokens.append(_resolve_yaml_path(token, base_dir))
            value = resolved_tokens
        elif key == "sectors" and value is not None:
            value = [int(item) for item in (value if isinstance(value, list) else [value])]
        defaults[key] = value
    return defaults


def build_parser(defaults: dict[str, object] | None = None) -> argparse.ArgumentParser:
    """Build the CLI parser, optionally seeded with YAML defaults."""
    defaults = defaults or {}
    parser = argparse.ArgumentParser(
        description="FSD modular black-box sector-decomposition prototype."
    )
    parser.set_defaults(**defaults)
    parser.add_argument(
        "--run",
        dest="run_file",
        default=defaults.get("run_file"),
        help="YAML run preset. Explicit CLI flags override values from the file.",
    )
    parser.add_argument(
        "--integral",
        choices=["triangle", "box"],
        default="triangle",
        help="Built-in example integral. Ignored when --dot-file is supplied.",
    )
    parser.add_argument(
        "--dot-file",
        default=None,
        help="Path to a GammaLoop-convention DOT file describing the integral.",
    )
    parser.add_argument("--kinematics", default=None, help="YAML values/replacements for DOT mode.")
    parser.add_argument("--graph-name", default=None, help="DOT graph name when a file contains multiple graphs.")
    parser.add_argument(
        "--sector-method",
        choices=["geometric", "geometric_ku", "iterative"],
        default="iterative",
        help="pySecDec decomposition method used for DOT sector generation. Default: iterative.",
    )
    parser.add_argument("--normaliz-executable", default=None, help="Normaliz command for geometric pySecDec decomposition.")
    parser.add_argument(
        "--dot-engine",
        choices=["fsd", "pysecdec", "both"],
        default="fsd",
        help="DOT execution engine: FSD/Havana, pySecDec generated integrator, or both.",
    )
    parser.add_argument(
        "--sectors",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Integrate only the listed canonical sector ids from the sector summary table. "
            "Inactive sectors are still recorded in result.json with zero samples."
        ),
    )
    parser.add_argument("--s", type=float, default=None, help="Triangle invariant s=p0^2.")
    parser.add_argument("--s12", type=float, default=None, help="Box invariant (p1+p2)^2.")
    parser.add_argument("--s23", type=float, default=None, help="Box invariant (p2+p3)^2.")
    parser.add_argument(
        "--m",
        type=float,
        default=0.0,
        help="Internal physical mass m for built-in examples. Defaults to 0 for DOT-file scaffolding.",
    )
    parser.add_argument("--mode", choices=["auto", "massive", "massless"], default="auto")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-iter", type=int, default=8, help="Maximum Havana iterations, or -1 for unbounded.")
    parser.add_argument("--min-iter", type=int, default=2)
    parser.add_argument("--samples-per-iter", type=int, default=50000)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Maximum number of Monte Carlo samples per batched processor task. "
            "0 keeps the default unbounded per-worker iteration chunk."
        ),
    )
    parser.add_argument(
        "--target-rel-accuracy",
        "--target-relative-accuracy",
        dest="target_rel_accuracy",
        type=float,
        default=None,
        help=(
            "Optional target for the displayed summed relative MC error in percent. "
            "When enabled, progress and ETA are extrapolated with err ~ 1/sqrt(N)."
        ),
    )
    parser.add_argument("--min-error", type=float, default=2.0e-4)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=1.0e-8,
        help=(
            "Endpoint-distance threshold on dimensionless sector coordinates below "
            "which Symbolica evaluators use evaluate_with_prec(..., --stability-precision)."
        ),
    )
    parser.add_argument(
        "--high-precision-stability-threshold",
        type=float,
        default=1.0e-12,
        help=(
            "Stronger endpoint-distance threshold below which Symbolica evaluators "
            "use evaluate_with_prec(..., --high-precision-stability-precision)."
        ),
    )
    parser.add_argument(
        "--stability-precision",
        type=int,
        default=100,
        help="Decimal digits used for Symbolica evaluator calls below --stability-threshold.",
    )
    parser.add_argument(
        "--high-precision-stability-precision",
        type=int,
        default=1000,
        help=(
            "Decimal digits used for Symbolica evaluator calls below "
            "--high-precision-stability-threshold."
        ),
    )
    parser.add_argument(
        "--jit-compile-evaluators",
        action="store_true",
        help=(
            "Enable Symbolica jit_compile=True for generated evaluators. "
            "Disabled by default because current Symbolica batch JIT can mis-evaluate simple row-wise expressions."
        ),
    )
    dual_group = parser.add_mutually_exclusive_group()
    dual_group.add_argument(
        "--pregenerate-dual-evaluators",
        dest="dual_evaluator_mode",
        action="store_const",
        const="pregenerate",
        default="pregenerate",
        help="Pregenerate one U/F dual evaluator per unique sector dual shape before integration.",
    )
    dual_group.add_argument(
        "--lazy-dual-evaluators-generation",
        dest="dual_evaluator_mode",
        action="store_const",
        const="lazy",
        help="Build U/F dual evaluators on first use during sector processing.",
    )
    dual_group.add_argument(
        "--pregenerate-single-overall-dual-evaluator",
        dest="dual_evaluator_mode",
        action="store_const",
        const="single-overall",
        help="Pregenerate one envelope U/F dual evaluator per integration dimension.",
    )
    dual_group.add_argument(
        "--symbolic-derivatives",
        dest="dual_evaluator_mode",
        action="store_const",
        const="symbolic-derivatives",
        help=(
            "Build shared symbolic U/F partial-derivative evaluators and use "
            "explicit chain rules instead of dualizing U/F evaluators."
        ),
    )
    parser.add_argument(
        "--subtraction-backend",
        choices=["formula", "projector-formula", "recursive"],
        default="projector-formula",
        help=(
            "Endpoint subtraction evaluator. 'formula' uses pregenerated Symbolica "
            "subtraction-formula evaluators; 'projector-formula' uses smaller "
            "endpoint-only Symbolica projectors fed by black-box Taylor data; "
            "'recursive' uses the vectorized recursive Taylor subtraction "
            "implementation and skips formula generation."
        ),
    )
    parser.add_argument(
        "--regular-taylor-signature-limit",
        type=int,
        default=256,
        help=(
            "Maximum regular-Taylor source-request workload to pregenerate for "
            "the projector-formula backend. Larger values can use the "
            "regular_taylor_* JSON cache but may be slow on a cold cache."
        ),
    )
    parser.add_argument(
        "--regular-taylor-formula-volume-limit",
        type=int,
        default=64,
        help=(
            "Maximum product of (Taylor order + 1) for a universal "
            "regular-Taylor formula prepared in all-sector projector-formula "
            "mode. Larger values prepare more formulas but can make cold "
            "Symbolica evaluator builds slow."
        ),
    )
    parser.add_argument(
        "--regular-taylor-formula-axis-limit",
        type=int,
        default=5,
        help=(
            "Maximum number of singular Taylor axes for universal "
            "regular-Taylor formulas prepared by default. Higher-axis "
            "signatures are left to the fallback path to avoid long cold "
            "Symbolica builds."
        ),
    )
    parser.add_argument(
        "--force-regular-taylor-formulas",
        action="store_true",
        help=(
            "Lift the regular-Taylor formula guards for projector-formula mode. "
            "This is intended for cache-warming and viability studies of the "
            "expensive universal high-axis formulas; cold generation can be slow."
        ),
    )
    parser.add_argument(
        "--chain-rule-formula-signature-limit",
        type=int,
        default=256,
        help=(
            "Maximum mapped-derivative chain-rule formula count to pregenerate "
            "in symbolic-derivative projector-formula mode. Larger values move "
            "more chain-rule composition into Symbolica but can make cold "
            "generation slow for all-sector high-loop runs."
        ),
    )
    parser.add_argument(
        "--direct-projector-cache-term-threshold",
        type=int,
        default=54,
        help=(
            "When IBP endpoint lowering is enabled, use a shipped direct "
            "endpoint-projector cache asset instead for sectors whose IBP "
            "compound projector would require at least this many child terms. "
            "Set to 0 to always prefer IBP."
        ),
    )
    parser.add_argument("--show-stats", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable the integration progress bar.")
    parser.add_argument("--quiet-summary", action="store_true", help="Suppress the pre-integration summary.")
    parser.add_argument("--mu", type=float, default=None)
    parser.add_argument("--onshell-threshold", type=float, default=None)
    parser.add_argument(
        "--gamma-scheme",
        choices=["oneloop", "full"],
        default="oneloop",
        help="Global epsilon prefactor convention. 'oneloop' strips Gamma/Euler factors.",
    )
    parser.add_argument(
        "--prefactor-convention",
        choices=["raw", "feynman", "sector", "pysecdec"],
        default=None,
        help=(
            "Displayed scalar-integral normalization. 'raw' uses OneLOopBridge raw "
            "coefficients; 'feynman' multiplies by TO_FEYNMAN = -1/(16*pi^2). "
            "DOT mode also accepts 'sector' and 'pysecdec'."
        ),
    )
    parser.add_argument("--progress-value-order", default="eps^0", help="Laurent order shown in DOT progress value.")
    parser.add_argument("--pysecdec-workdir", default=".pysecdec_build")
    parser.add_argument("--pysecdec-epsrel", type=float, default=1.0e-2)
    parser.add_argument("--pysecdec-maxeval", type=int, default=100000)
    parser.add_argument("--keep-pysecdec-workdir", action="store_true")
    parser.add_argument(
        "--max-eps-order",
        type=int,
        default=0,
        help="Highest epsilon order to integrate; deepest pole is eps^(-2*loop_count).",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        default=defaults.get("target_args", defaults.get("target")),
        help=(
            "Reference target: numeric re/im pairs from deepest pole upward, "
            "'pysecdec' for DOT mode, or a previous result.json path."
        ),
    )
    parser.add_argument(
        "--refresh-target",
        action="store_true",
        help="Regenerate file-backed pySecDec targets instead of reusing an existing file.",
    )
    parser.add_argument("--show-results", default=None, help="Show a stored result.json and exit.")
    parser.add_argument(
        "--result-path",
        default=None,
        help=(
            "Path for the persistent result JSON written after integration. "
            "Defaults to ./result.json for built-ins and result.json next to the DOT file."
        ),
    )
    parser.add_argument(
        "--sort-sector-results",
        choices=["index", "abs-central", "abs-error"],
        default="index",
        help="Sorting mode used by --show-results sector tables.",
    )
    parser.add_argument(
        "--IBP_reduce_to_log_endpoint",
        "--ibp-reduce-to-log-endpoint",
        dest="ibp_reduce_to_log_endpoint",
        action="store_true",
        help=(
            "For --subtraction-backend projector-formula, lower y^(-n+c eps) "
            "endpoints to logarithmic y^(-1+c eps) endpoints by integration by parts."
        ),
    )
    parser.add_argument(
        "--no-IBP_reduce_to_log_endpoint",
        "--no-ibp-reduce-to-log-endpoint",
        dest="ibp_reduce_to_log_endpoint",
        action="store_false",
        help=(
            "Disable IBP endpoint lowering, overriding a run YAML file that "
            "enabled --ibp-reduce-to-log-endpoint."
        ),
    )
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of tables.")
    parser.set_defaults(**defaults)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Build the CLI parser and return parsed command-line options."""
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run", dest="run_file", default=None)
    pre_args, _unknown = pre_parser.parse_known_args(argv)
    try:
        defaults = _load_run_defaults(pre_args.run_file)
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc
    parser = build_parser(defaults)
    return parser.parse_args(argv)


def build_request(args: argparse.Namespace) -> IntegralRequest:
    """Convert argparse output into the immutable request object used below."""
    mode = resolve_mode(args.m, args.mode)
    dot_file = str(Path(args.dot_file).expanduser()) if args.dot_file is not None else None
    kinematics_file = str(Path(args.kinematics).expanduser()) if args.kinematics is not None else None
    prefactor_convention = args.prefactor_convention
    if prefactor_convention is None:
        prefactor_convention = "pysecdec" if dot_file is not None else "raw"
    if args.result_path is not None:
        result_path = str(Path(args.result_path).expanduser())
    else:
        result_path = (
            str(Path(dot_file).expanduser().resolve().parent / "result.json")
            if dot_file is not None
            else str(Path.cwd() / "result.json")
        )
    target_args: tuple[str, ...] | None
    if args.target is None:
        target_args = None
    elif isinstance(args.target, (list, tuple)):
        target_args = tuple(str(item) for item in args.target)
    else:
        target_args = (str(args.target),)
    force_regular_taylor_formulas = bool(args.force_regular_taylor_formulas)
    regular_taylor_signature_limit = int(args.regular_taylor_signature_limit)
    regular_taylor_formula_volume_limit = int(args.regular_taylor_formula_volume_limit)
    regular_taylor_formula_axis_limit = int(args.regular_taylor_formula_axis_limit)
    chain_rule_formula_signature_limit = int(args.chain_rule_formula_signature_limit)
    if force_regular_taylor_formulas:
        regular_taylor_signature_limit = max(regular_taylor_signature_limit, 1_000_000)
        regular_taylor_formula_volume_limit = max(regular_taylor_formula_volume_limit, 1_000_000)
        regular_taylor_formula_axis_limit = max(regular_taylor_formula_axis_limit, 32)

    return IntegralRequest(
        run_file=str(Path(args.run_file).expanduser().resolve()) if args.run_file is not None else None,
        integral="dot" if dot_file is not None else args.integral,
        dot_file=dot_file,
        kinematics_file=kinematics_file,
        graph_name=args.graph_name,
        sector_method=args.sector_method,
        normaliz_executable=args.normaliz_executable,
        dot_engine=args.dot_engine,
        sectors=tuple(args.sectors) if args.sectors is not None else None,
        pysecdec_workdir=args.pysecdec_workdir,
        pysecdec_epsrel=args.pysecdec_epsrel,
        pysecdec_maxeval=args.pysecdec_maxeval,
        keep_pysecdec_workdir=args.keep_pysecdec_workdir,
        progress_value_order=args.progress_value_order,
        max_eps_order=args.max_eps_order,
        target_args=target_args,
        refresh_target=args.refresh_target,
        show_results=args.show_results,
        sort_sector_results=args.sort_sector_results,
        result_path=result_path,
        log_level=args.log_level,
        log_file=args.log_file,
        mode=mode,
        s=args.s,
        s12=args.s12,
        s23=args.s23,
        m=args.m,
        gamma_scheme=args.gamma_scheme,
        prefactor_convention=prefactor_convention,
        seed=args.seed,
        max_iter=args.max_iter,
        min_iter=args.min_iter,
        samples_per_iter=args.samples_per_iter,
        batch_size=args.batch_size,
        target_rel_accuracy=args.target_rel_accuracy,
        min_error=args.min_error,
        bins=args.bins,
        workers=args.workers,
        jit_compile_evaluators=args.jit_compile_evaluators,
        dual_evaluator_mode=args.dual_evaluator_mode,
        subtraction_backend=args.subtraction_backend,
        ibp_reduce_to_log_endpoint=args.ibp_reduce_to_log_endpoint,
        direct_projector_cache_term_threshold=args.direct_projector_cache_term_threshold,
        force_regular_taylor_formulas=force_regular_taylor_formulas,
        regular_taylor_signature_limit=regular_taylor_signature_limit,
        regular_taylor_formula_volume_limit=regular_taylor_formula_volume_limit,
        regular_taylor_formula_axis_limit=regular_taylor_formula_axis_limit,
        chain_rule_formula_signature_limit=chain_rule_formula_signature_limit,
        stability_threshold=args.stability_threshold,
        high_precision_stability_threshold=args.high_precision_stability_threshold,
        stability_precision=args.stability_precision,
        high_precision_stability_precision=args.high_precision_stability_precision,
        show_stats=args.show_stats,
        no_progress=args.no_progress,
        quiet_summary=args.quiet_summary,
        json=args.json,
        mu=args.mu,
        onshell_threshold=args.onshell_threshold,
    )


def deepest_laurent_order(topology) -> int:
    """Return the universal scalar-integral deepest pole order ``-2L``."""
    parametric = topology.parametric_representation
    loop_count = int(parametric.loop_count if parametric is not None else 1)
    return -2 * loop_count


def configure_laurent_range(request: IntegralRequest, topology, sectors) -> None:
    """Apply the CLI Laurent range and validate sector endpoint depth."""
    min_order = deepest_laurent_order(topology)
    if request.max_eps_order < min_order:
        raise ValueError(
            f"--max-eps-order must be >= eps^{min_order}; got eps^{request.max_eps_order}"
        )
    max_sector_depth = max((len(sector.singular_axes) for sector in sectors), default=0)
    if max_sector_depth > -min_order:
        worst = [
            sector.name for sector in sectors if len(sector.singular_axes) == max_sector_depth
        ][:5]
        raise ValueError(
            f"sector endpoint pole depth {max_sector_depth} exceeds universal "
            f"2L depth {-min_order}; examples: {', '.join(worst)}"
        )
    topology.set_laurent_range(min_order, request.max_eps_order)


def validate_sector_selection(request: IntegralRequest, sectors) -> None:
    """Validate canonical sector ids requested through ``--sectors``."""
    if request.sectors is None:
        return
    if not request.sectors:
        raise ValueError("--sectors requires at least one sector id")
    if len(set(request.sectors)) != len(request.sectors):
        raise ValueError("--sectors must not contain duplicate ids")
    invalid = [sector_id for sector_id in request.sectors if sector_id < 0 or sector_id >= len(sectors)]
    if invalid:
        raise ValueError(
            f"--sectors contains invalid sector ids {invalid}; valid range is 0..{len(sectors)-1}"
        )


def _align_coefficients(values: list[complex], count: int) -> list[complex]:
    """Align Laurent coefficients to the current deepest-pole-first range.

    Targets are always ordered from the universal deepest pole upward.  When a
    run truncates at ``--max-eps-order < 0``, extra target entries are higher
    epsilon orders and must be dropped from the end, not from the beginning.
    Missing trailing entries are interpreted as zero by the CLI contract.
    """
    if len(values) < count:
        return list(values) + [0.0 + 0.0j for _ in range(count - len(values))]
    if len(values) > count:
        return list(values[:count])
    return list(values)


def _numeric_target(request: IntegralRequest, topology, args: tuple[str, ...]) -> TargetDefinition:
    """Parse numeric re/im target pairs in the selected display convention."""
    if len(args) % 2 != 0:
        raise ValueError("--target numeric form requires re/im pairs")
    pair_count = len(args) // 2
    if pair_count > topology.coefficient_count:
        raise ValueError(
            f"--target supplies {pair_count} coefficients but current Laurent range has "
            f"{topology.coefficient_count}"
        )
    coeffs = [
        complex(float(args[2 * index]), float(args[2 * index + 1]))
        for index in range(pair_count)
    ]
    coeffs.extend([0.0 + 0.0j for _ in range(topology.coefficient_count - pair_count)])
    return TargetDefinition(
        source="numeric",
        convention=request.prefactor_convention,
        coefficients=coeffs,
        errors=[0.0 + 0.0j for _ in coeffs],
        metadata={"entries": list(args)},
    )


def _oneloop_target(request: IntegralRequest, topology) -> TargetDefinition:
    """Resolve the built-in OneLOopBridge target in display convention."""
    benchmark = compute_benchmark_quietly(request)
    zeros = [0.0 + 0.0j for _ in range(topology.coefficient_count)]
    display_coeffs, display_errors, display_bench, _factor = selected_prefactor_values(
        request,
        zeros,
        zeros,
        benchmark,
    )
    return TargetDefinition(
        source="oneloop",
        convention=request.prefactor_convention,
        coefficients=display_bench,
        errors=display_errors,
        metadata={"factor": benchmark.factor},
    )


def _generation_progress(
    request: IntegralRequest,
    logger: logging.Logger,
    label: str,
) -> GenerationProgress:
    """Create a generation reporter respecting machine-readable output modes."""
    return GenerationProgress(
        enabled=not request.json and not request.no_progress,
        logger=logger,
        label=label,
    )


def _pysecdec_target(
    request: IntegralRequest,
    summary: dict,
    logger: logging.Logger | None = None,
) -> TargetDefinition:
    """Run pySecDec and return its coefficients as a DOT target."""
    if request.integral != "dot":
        raise ValueError("--target pysecdec is only available in DOT mode")
    if request.prefactor_convention != "pysecdec":
        raise ValueError("--target pysecdec requires --prefactor-convention pysecdec")
    progress = (
        _generation_progress(request, logger, "pySecDec target")
        if logger is not None
        else None
    )
    try:
        result = run_pysecdec_package(get_dot_bundle(request), request, progress=progress)
    finally:
        if progress is not None:
            progress.close()
    summary["pysecdec_timings"] = result.timings.to_dict()
    return TargetDefinition(
        source="pysecdec",
        convention=request.prefactor_convention,
        coefficients=result.coeffs,
        errors=result.errors,
        metadata={"workdir": request.pysecdec_workdir},
    )


def _ensure_target_parent(path: Path) -> None:
    """Create at most the immediate target parent directory."""
    parent = path.expanduser().parent
    if parent.exists():
        return
    if parent.parent.exists():
        parent.mkdir()
        return
    raise ValueError(
        f"cannot create target parent {parent}; create deeper parent directories explicitly"
    )


def _write_pysecdec_target_file(
    request: IntegralRequest,
    path: Path,
    summary: dict,
    logger: logging.Logger | None = None,
) -> None:
    """Generate a DOT pySecDec target and persist it as a reusable result file."""
    if request.integral != "dot":
        raise ValueError("missing file-backed targets can only be generated in DOT mode")
    if request.prefactor_convention != "pysecdec":
        raise ValueError("file-backed pySecDec targets require --prefactor-convention pysecdec")
    _ensure_target_parent(path)
    target = _pysecdec_target(request, summary, logger=logger)
    output = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "integral": request.integral,
        "mode": request.mode,
        "prefactor_convention": request.prefactor_convention,
        "request": request_metadata(request),
        "environment": environment_metadata(),
        "summary": summary,
        "target": {
            "source": target.source,
            "convention": target.convention,
            "coefficients": target.coefficients,
            "errors": target.errors,
            "metadata": {**target.metadata, "generated_path": str(path.expanduser())},
        },
        "pysecdec": {
            "coeffs": target.coefficients,
            "errors": target.errors,
            "timings": summary.get("pysecdec_timings", {}),
        },
    }
    write_result_json(output, path)


def resolve_target(
    request: IntegralRequest,
    topology,
    summary: dict,
    logger: logging.Logger | None = None,
) -> TargetDefinition | None:
    """Resolve explicit or implicit comparison target."""
    args = request.target_args
    if args is not None:
        if len(args) == 1:
            token = args[0]
            if token == "pysecdec":
                target = _pysecdec_target(request, summary, logger=logger)
                target = TargetDefinition(
                    source=target.source,
                    convention=target.convention,
                    coefficients=_align_coefficients(target.coefficients, topology.coefficient_count),
                    errors=_align_coefficients(target.errors, topology.coefficient_count),
                    metadata=target.metadata,
                )
                return target
            candidate_path = Path(token).expanduser()
            if candidate_path.is_file() and not request.refresh_target:
                target = target_from_result_file(candidate_path, request.prefactor_convention)
                return TargetDefinition(
                    source=target.source,
                    convention=target.convention,
                    coefficients=_align_coefficients(target.coefficients, topology.coefficient_count),
                    errors=_align_coefficients(target.errors, topology.coefficient_count),
                    metadata=target.metadata,
                )
            if not _is_numeric_token(token):
                _write_pysecdec_target_file(request, candidate_path, summary, logger=logger)
                target = target_from_result_file(candidate_path, request.prefactor_convention)
                return TargetDefinition(
                    source=target.source,
                    convention=target.convention,
                    coefficients=_align_coefficients(target.coefficients, topology.coefficient_count),
                    errors=_align_coefficients(target.errors, topology.coefficient_count),
                    metadata=target.metadata,
                )
        return _numeric_target(request, topology, args)

    if request.integral != "dot":
        return _oneloop_target(request, topology)
    if request.dot_engine == "both":
        target = _pysecdec_target(request, summary, logger=logger)
        return TargetDefinition(
            source=target.source,
            convention=target.convention,
            coefficients=_align_coefficients(target.coefficients, topology.coefficient_count),
            errors=_align_coefficients(target.errors, topology.coefficient_count),
            metadata=target.metadata,
        )
    return None


def configure_logging(request: IntegralRequest) -> logging.Logger:
    """Configure stdlib logging for generation timing and backend diagnostics."""
    level = getattr(logging, request.log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if not request.json:
        handlers.append(logging.StreamHandler(sys.stderr))
    if request.log_file is not None:
        handlers.append(logging.FileHandler(request.log_file, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers, format="%(levelname)s:%(name)s:%(message)s", force=True)
    return logging.getLogger("FSD")


def compute_benchmark_quietly(request: IntegralRequest):
    """Call OneLOopBridge while suppressing any stdout emitted by the bridge."""
    with open(os.devnull, "w") as devnull:
        saved_stdout = os.dup(1)
        try:
            os.dup2(devnull.fileno(), 1)
            with contextlib.redirect_stdout(devnull):
                return compute_benchmark(request)
        finally:
            os.dup2(saved_stdout, 1)
            os.close(saved_stdout)


def main() -> int:
    """Run the complete CLI workflow and return a process exit code."""
    colorama_init(strip=False)
    args = parse_args()
    if args.show_results is not None:
        try:
            print_saved_results(args.show_results, args.sort_sector_results)
            return 0
        except Exception as exc:
            print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
            return 2

    request = build_request(args)
    logger = configure_logging(request)
    generation_progress: GenerationProgress | None = None

    try:
        validate_request(request)
        if request.integral == "dot":
            generation_progress = _generation_progress(request, logger, "FSD generation")
            bundle = get_dot_bundle(request, progress=generation_progress)
            bundle.timings.log(logger)
        elif request.target_args is None:
            check_oneloop_bridge()
    except Exception as exc:
        if generation_progress is not None:
            generation_progress.close()
        print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
        return 2

    topology = build_topology(request)
    sectors = generate_sectors(request)
    try:
        validate_sector_selection(request, sectors)
        configure_laurent_range(request, topology, sectors)
    except Exception as exc:
        if generation_progress is not None:
            generation_progress.close()
        print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
        return 2
    active_sectors = (
        [sectors[sector_id] for sector_id in request.sectors]
        if request.sectors is not None
        else sectors
    )
    topology.regular_taylor_formula_signature_limit = request.regular_taylor_signature_limit
    topology.regular_taylor_formula_volume_limit = request.regular_taylor_formula_volume_limit
    topology.regular_taylor_formula_axis_limit = request.regular_taylor_formula_axis_limit
    topology.chain_rule_formula_signature_limit = request.chain_rule_formula_signature_limit
    topology.direct_projector_cache_term_threshold = request.direct_projector_cache_term_threshold
    topology.prepare_dual_evaluators(active_sectors, request.dual_evaluator_mode)
    extra_dual_build_before = topology.dual_evaluator_build_seconds
    if request.subtraction_backend == "formula":
        topology.prepare_subtraction_formulas(active_sectors, progress=generation_progress)
    elif request.subtraction_backend == "projector-formula":
        topology.prepare_endpoint_projector_formulas(active_sectors, progress=generation_progress)
        topology.prepare_regular_taylor_formulas(active_sectors, progress=generation_progress)
        topology.prepare_chain_rule_formulas(active_sectors, progress=generation_progress)
    if generation_progress is not None:
        generation_progress.close()
        generation_progress = None
    if request.integral == "dot" and topology.subtraction_formula_build_seconds > 0.0:
        if request.subtraction_backend == "projector-formula":
            signature_count = len(topology._endpoint_projector_formulas)
            regular_count = len(getattr(topology, "_regular_taylor_formulas", {}))
            skipped_regular = getattr(topology, "regular_taylor_formulas_skipped", 0)
            curated_regular = getattr(
                topology, "regular_taylor_formulas_from_curated_cache", 0
            )
            detail = (
                f"{signature_count} endpoint projector signature(s), "
                f"{regular_count} regular Taylor signature(s)"
            )
            if curated_regular:
                detail += f", {curated_regular} curated regular Taylor asset(s)"
            if skipped_regular:
                detail += f", skipped {skipped_regular} regular Taylor signature(s)"
        else:
            signature_count = len(topology._subtraction_formulas)
            detail = f"{signature_count} formula signature(s)"
        get_dot_bundle(request).timings.add(
            "Symbolica subtraction formula build",
            topology.subtraction_formula_build_seconds,
            detail=detail,
        )
    if request.integral == "dot":
        extra_dual_build = topology.dual_evaluator_build_seconds - extra_dual_build_before
        if extra_dual_build > 0.0:
            get_dot_bundle(request).timings.add(
                "Symbolica dual evaluator build",
                extra_dual_build,
                detail="regular Taylor source shapes",
            )
        if topology.chain_rule_formula_build_seconds > 0.0:
            get_dot_bundle(request).timings.add(
                "Symbolica chain-rule formula build",
                topology.chain_rule_formula_build_seconds,
                detail=(
                    f"{len(getattr(topology, '_chain_rule_formulas', {}))} "
                    "mapped derivative formula(s)"
                ),
            )
        elif getattr(topology, "chain_rule_formulas_skipped", 0):
            get_dot_bundle(request).timings.add(
                "Symbolica chain-rule formula build",
                0.0,
                detail=(
                    f"skipped {topology.chain_rule_formulas_skipped} "
                    "mapped derivative formula(s)"
                ),
            )
    summary = summary_data(request, topology, sectors, benchmark_available=False)
    if request.integral == "dot":
        summary["generation_timings"] = get_dot_bundle(request).timings.to_summary_dict()
    try:
        target = resolve_target(request, topology, summary, logger=logger)
    except Exception as exc:
        if generation_progress is not None:
            generation_progress.close()
        print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
        return 2
    summary["validation"]["benchmark_available"] = target is not None
    summary["header"]["benchmark"] = target.source if target is not None else "unavailable"

    if not request.json and not request.quiet_summary:
        print_preintegration_summary(request, topology, sectors, benchmark_available=target is not None)

    try:
        if request.integral == "dot":
            integration = None
            if request.dot_engine in {"fsd", "both"}:
                integration = integrate(request, topology, sectors, target)
            pysecdec_result = None
            if request.dot_engine == "pysecdec":
                pysecdec_progress = _generation_progress(request, logger, "pySecDec")
                try:
                    pysecdec_result = run_pysecdec_package(
                        get_dot_bundle(request),
                        request,
                        progress=pysecdec_progress,
                    )
                finally:
                    pysecdec_progress.close()
                pysecdec_result.timings.log(logger)
                summary["pysecdec_timings"] = pysecdec_result.timings.to_dict()
            if integration is None:
                output = {
                    "schema_version": 1,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "command": sys.argv,
                    "integral": request.integral,
                    "mode": request.mode,
                    "prefactor_convention": request.prefactor_convention,
                    "request": request_metadata(request),
                    "environment": environment_metadata(),
                    "summary": summary,
                    "pysecdec": {
                        "coeffs": pysecdec_result.coeffs if pysecdec_result else [],
                        "errors": pysecdec_result.errors if pysecdec_result else [],
                    },
                }
                write_result_json(output, result_output_path(request))
                print(output_json(output) if request.json else output)
                return 0
        else:
            integration = integrate(request, topology, sectors, target)
    except Exception as exc:
        print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
        return 1

    raw_coeffs, raw_errors = apply_global_convention(
        integration.raw_sector_coeffs,
        integration.raw_sector_errors,
        request,
    )
    summary.setdefault("symanzik", {})["dual_evaluator_build_seconds"] = topology.dual_evaluator_build_seconds
    summary.setdefault("symanzik", {})[
        "chain_rule_formula_build_seconds"
    ] = topology.chain_rule_formula_build_seconds
    summary.setdefault("symanzik", {})[
        "chain_rule_formula_count"
    ] = len(getattr(topology, "_chain_rule_formulas", {}))
    summary.setdefault("symanzik", {})[
        "chain_rule_formulas_skipped"
    ] = getattr(topology, "chain_rule_formulas_skipped", 0)
    sector_rows = build_sector_result_rows(request, sectors, integration.per_sector)
    output = make_output(
        request=request,
        raw_coeffs=raw_coeffs,
        raw_errors=raw_errors,
        target=target,
        samples=integration.samples,
        elapsed_seconds=integration.elapsed_seconds,
        avg_eval_us_per_sample_per_worker=integration.avg_eval_us_per_sample_per_worker,
        eval_seconds=integration.eval_seconds,
        python_seconds=integration.python_seconds,
        havana_seconds=integration.havana_seconds,
        python_overhead_fraction=integration.python_overhead_fraction,
        precision_counts=integration.precision_counts,
        summary=summary,
        sector_results=sector_rows,
        interrupted=integration.interrupted,
    )
    output["request"] = request_metadata(request)
    output["environment"] = environment_metadata()
    output["created_utc"] = datetime.now(timezone.utc).isoformat()
    output["command"] = sys.argv
    output["result_path"] = str(result_output_path(request))
    if request.integral == "dot":
        bundle = get_dot_bundle(request)
        output["input_metadata"] = {
            "dot_file": request.dot_file,
            "kinematics_file": request.kinematics_file,
            "graph_name": bundle.parsed_graph.graph_name,
            "kinematics": {
                "values": bundle.kinematics.values,
                "replacements": bundle.kinematics.replacement_expressions,
            },
        }

    write_result_json(output, result_output_path(request))

    if request.json:
        print(output_json(output))
    else:
        print_result_table(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
