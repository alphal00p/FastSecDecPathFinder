#!/usr/bin/env python3
"""Command-line entry point for the modular FastSecDec v2 prototype.

This file intentionally contains only steering code: argument parsing,
kinematic validation, summary rendering, benchmark setup, integration launch,
and output formatting.  The sector declarations, black-box integrand
construction, and Havana sampling logic live in separate modules.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
from dataclasses import replace
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
import signal
import sys
import time

from colorama import Fore, Style, init as colorama_init
import yaml

try:
    import psutil
except ImportError:  # pragma: no cover - only relevant in very small envs.
    psutil = None

from benchmark import check_oneloop_bridge, compute_benchmark
from cache_warm import run_universal_cache_mode
from definitions import IntegralRequest, TargetDefinition
from formatting import (
    apply_global_convention,
    build_sector_result_rows,
    make_output,
    output_json,
    print_generation_report,
    print_preintegration_summary,
    print_result_table,
    selected_prefactor_values,
    summary_data,
)
from dot_topology import get_dot_bundle
from generation_timing import GenerationProgress
from integrand import build_topology
from integrator import integrate
from prepared_bundle import load_prepared_bundle, save_prepared_bundle
from pysecdec_bridge import ensure_pysecdec_package, run_pysecdec_package
from pysecdec_kernel_benchmark import (
    benchmark_pysecdec_generated_kernels,
    output_pysecdec_kernel_benchmark_json,
    print_pysecdec_kernel_benchmark_report,
)
from result_io import (
    environment_metadata,
    print_saved_results,
    request_metadata,
    result_output_path,
    target_from_result_file,
    write_result_json,
)
from runtime_benchmark import run_sector_runtime_benchmark
from sectors_generator import generate_sectors


_INTERRUPT_CLEANUP_ACTIVE = False
_INTERRUPT_CLEANUP_RUNNING = False


def _live_descendants() -> list[object]:
    """Return live child processes of this CLI process, if psutil is available."""
    if psutil is None:
        return []
    try:
        current = psutil.Process(os.getpid())
        out = []
        for child in current.children(recursive=True):
            if not child.is_running():
                continue
            try:
                if child.status() == psutil.STATUS_ZOMBIE:
                    continue
            except Exception:
                pass
            out.append(child)
        return out
    except Exception:
        return []


def _terminate_generation_descendants(reason: str, *, quiet: bool = False) -> None:
    """Best-effort cleanup for subprocess trees created during generation.

    DOT generation can enter pySecDec/FORM/Normaliz/make subprocesses.  If the
    user interrupts the main CLI while one of those tools is active, Python does
    not automatically reap every descendant.  This cleanup intentionally targets
    only descendants of the current process, never unrelated user processes.
    """

    global _INTERRUPT_CLEANUP_RUNNING
    if _INTERRUPT_CLEANUP_RUNNING:
        return
    _INTERRUPT_CLEANUP_RUNNING = True
    try:
        children = _live_descendants()
        if not children:
            return
        if not quiet:
            print(
                f"{Fore.YELLOW}interrupt:{Style.RESET_ALL} {reason}; "
                f"terminating {len(children)} child process(es)",
                file=sys.stderr,
                flush=True,
            )
        own_pgid = os.getpgrp()

        def signal_children(sig: signal.Signals) -> None:
            seen_groups: set[int] = set()
            for child in list(children):
                try:
                    pid = int(child.pid)
                    pgid = os.getpgid(pid)
                except Exception:
                    continue
                if pgid != own_pgid and pgid not in seen_groups:
                    try:
                        os.killpg(pgid, sig)
                        seen_groups.add(pgid)
                    except ProcessLookupError:
                        pass
                    except Exception:
                        # Fall back to the individual PID below.
                        pass
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
                except Exception:
                    pass

        signal_children(signal.SIGINT)
        if psutil is not None:
            try:
                psutil.wait_procs(children, timeout=0.5)
            except Exception:
                pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not _live_descendants():
                return
            time.sleep(0.1)
        children = _live_descendants()
        signal_children(signal.SIGTERM)
        if psutil is not None:
            try:
                psutil.wait_procs(children, timeout=0.5)
            except Exception:
                pass
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if not _live_descendants():
                return
            time.sleep(0.1)
        children = _live_descendants()
        signal_children(signal.SIGKILL)
    finally:
        _INTERRUPT_CLEANUP_RUNNING = False


class InterruptCleanupGuard:
    """Install SIGINT/SIGTERM handlers that reap generation subprocesses."""

    def __init__(self) -> None:
        self._previous: dict[int, object] = {}

    def __enter__(self) -> "InterruptCleanupGuard":
        global _INTERRUPT_CLEANUP_ACTIVE
        _INTERRUPT_CLEANUP_ACTIVE = True
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[int(sig)] = signal.getsignal(sig)
            signal.signal(sig, self._handle_signal)
        atexit.register(self._atexit_cleanup)
        return self

    def __exit__(self, exc_type, _exc, _tb) -> bool:
        global _INTERRUPT_CLEANUP_ACTIVE
        _INTERRUPT_CLEANUP_ACTIVE = False
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        if exc_type is KeyboardInterrupt:
            _terminate_generation_descendants("keyboard interrupt")
        return False

    def _atexit_cleanup(self) -> None:
        if _INTERRUPT_CLEANUP_ACTIVE:
            _terminate_generation_descendants("process exit", quiet=True)

    def _handle_signal(self, signum: int, _frame) -> None:
        name = signal.Signals(signum).name
        _terminate_generation_descendants(name)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        raise SystemExit(128 + int(signum))


def resolve_mode(m: float, requested: str) -> str:
    """Resolve the user-facing ``auto`` mode into massive or massless mode."""
    if requested != "auto":
        return requested
    return "massless" if abs(m) == 0.0 else "massive"


def validate_request(request: IntegralRequest) -> None:
    """Reject unsupported kinematics before building sectors or benchmarks."""
    if request.command not in {"run", "generate", "integrate", "cache", "test", "benchmark"}:
        raise ValueError(f"unsupported command {request.command!r}")
    if request.command == "cache":
        if not request.cache_loop_counts:
            raise ValueError("cache mode requires at least one --cache-loop-counts entry")
        if any(loop <= 0 for loop in request.cache_loop_counts):
            raise ValueError("--cache-loop-counts entries must be positive")
        if request.cache_verify_samples_per_sector <= 0:
            raise ValueError("--cache-verify-samples-per-sector must be positive")
        return
    if request.evaluator_lru_size < 0:
        raise ValueError("--evaluator-lru-size must be >= 0")
    if request.command == "test":
        if request.workers <= 0:
            raise ValueError("--workers must be positive in test mode")
        if not request.test_boundary_distances:
            raise ValueError("--test-boundary-distances requires at least one distance")
        invalid_distances = [
            value for value in request.test_boundary_distances
            if not (0.0 < float(value) < 0.5)
        ]
        if invalid_distances:
            raise ValueError(
                "--test-boundary-distances entries must lie in (0, 0.5); "
                f"got {invalid_distances}"
            )
        invalid_retry_scales = [
            value for value in request.test_boundary_retry_scales if float(value) <= 0.0
        ]
        if invalid_retry_scales:
            raise ValueError(f"--retries entries must be positive; got {invalid_retry_scales}")
        if (
            request.test_boundary_max_simultaneous_endpoint_approaches is not None
            and request.test_boundary_max_simultaneous_endpoint_approaches <= 0
        ):
            raise ValueError("--max-simultaneous-endpoint-approaches must be positive")
        if (
            request.test_boundary_growth_power_tolerance is not None
            and request.test_boundary_growth_power_tolerance < 0.0
        ):
            raise ValueError("--test-boundary-growth-power-tolerance must be non-negative")
        if (
            request.test_boundary_growth_power_tolerance is not None
            and len(request.test_boundary_distances) < 2
        ):
            raise ValueError(
                "--test-boundary-growth-power-tolerance requires at least two "
                "--test-boundary-distances"
            )
    if request.command == "benchmark" and request.benchmark_samples_per_sector <= 0:
        raise ValueError("--benchmark-samples-per-sector must be positive")
    if request.command in {"generate", "integrate"} and request.output is None:
        raise ValueError(f"{request.command} requires --output DIR")
    if request.command == "generate":
        if request.dot_file is None:
            raise ValueError("generate currently supports DOT topologies only and requires --dot-file")
        if request.kinematics_file is None:
            raise ValueError("generate requires --kinematics")
        if request.dual_evaluator_mode == "lazy":
            raise ValueError(
                "generate cannot use --lazy-dual-evaluators-generation because "
                "prepared integrate mode is strict and cannot create missing evaluators"
            )
    if request.command == "integrate":
        output = Path(request.output or "").expanduser()
        if not output.is_dir():
            raise ValueError(f"prepared bundle directory does not exist: {output}")
        if request.target_args == ("pysecdec",):
            raise ValueError("integrate from a prepared bundle cannot use --target pysecdec")

    if request.max_iter != -1 and request.max_iter <= 0:
        raise ValueError("--max-iter must be positive, or -1 for an unbounded run")
    if request.samples_per_iter <= 0:
        raise ValueError("--samples-per-iter must be positive")
    if request.batch_size < 0:
        raise ValueError("--batch-size must be >= 0, where 0 means one batch per worker chunk")
    if request.evaluator_compile_mode not in {"jit", "eager", "compile"}:
        raise ValueError("--evaluator compile mode must be one of: jit, eager, compile")
    if request.evaluator_compile_mode == "compile" and not request.real_evaluator:
        raise ValueError("--compile currently requires --real-evaluator")
    if request.sampling_mode not in {"havana", "democratic", "qmc"}:
        raise ValueError("--sampling-mode must be 'havana', 'democratic', or 'qmc'")
    if request.democratic_samples_per_sector <= 0:
        raise ValueError("--democratic-samples-per-sector must be positive")
    if request.qmc_shifts <= 1:
        raise ValueError("--qmc-shifts must be greater than 1 to estimate a randomized QMC error")
    if request.qmc_korobov_alpha < 1:
        raise ValueError("--qmc-korobov-alpha must be a positive integer")
    if request.qmc_lattice_backend not in {"qmcpy", "cbcpt-dn1-100"}:
        raise ValueError("--qmc-lattice-backend supports 'qmcpy' and 'cbcpt-dn1-100'")
    if request.qmc_order not in {"linear", "radical-inverse", "gray"}:
        raise ValueError("--qmc-order must be 'linear', 'radical-inverse', or 'gray'")
    if (
        request.sampling_mode == "qmc"
        and request.qmc_lattice_backend == "qmcpy"
        and request.samples_per_iter & (request.samples_per_iter - 1)
    ):
        raise ValueError("--sampling-mode qmc requires --samples-per-iter to be a power of two")
    if request.target_rel_accuracy is not None and request.target_rel_accuracy <= 0.0:
        raise ValueError("--target-rel-accuracy must be > 0 and is interpreted as a percent")
    if request.target_rel_error is not None and request.target_rel_error <= 0.0:
        raise ValueError("--target-rel-error must be > 0 and is interpreted as a dimensionless ratio")
    if request.target_abs_error is not None and request.target_abs_error <= 0.0:
        raise ValueError("--target-abs-error must be > 0 in the selected prefactor convention")
    if request.target_integration_time is not None and request.target_integration_time <= 0.0:
        raise ValueError("--target-integration-time must be > 0 seconds")
    if request.stability_threshold < 0.0:
        raise ValueError("--stability-threshold must be non-negative")
    if request.medium_precision_stability_threshold < 0.0:
        raise ValueError("--medium-precision-stability-threshold must be non-negative")
    if request.high_precision_stability_threshold < 0.0:
        raise ValueError("--high-precision-stability-threshold must be non-negative")
    if request.medium_precision_stability_threshold > request.stability_threshold:
        raise ValueError(
            "--medium-precision-stability-threshold must be <= --stability-threshold"
        )
    if request.high_precision_stability_threshold > request.medium_precision_stability_threshold:
        raise ValueError(
            "--high-precision-stability-threshold must be <= "
            "--medium-precision-stability-threshold"
        )
    if request.stability_precision <= 0:
        raise ValueError("--stability-precision must be positive")
    if request.medium_precision_stability_precision <= 0:
        raise ValueError("--medium-precision-stability-precision must be positive")
    if request.high_precision_stability_precision <= 0:
        raise ValueError("--high-precision-stability-precision must be positive")
    if not (0.0 <= request.max_weight_precision_xi <= 1.0):
        raise ValueError("--max-weight-precision-xi must be between 0 and 1; use 0 to disable")
    if (
        request.sector_evaluator_backend in {"two-stage-explicit", "explicit"}
        and request.subtraction_backend != "projector-formula"
    ):
        raise ValueError(
            "--sector-evaluator-backend two-stage-explicit/explicit requires "
            "--subtraction-backend projector-formula"
        )
    if (
        (request.ibp_power_goal is not None or request.ibp_reduce_to_log_endpoint)
        and request.subtraction_backend != "projector-formula"
        and request.sector_evaluator_backend != "explicit"
    ):
        raise ValueError(
            "--ibp-power-goal/--IBP_reduce_to_log_endpoint is only supported with "
            "--subtraction-backend projector-formula or --explicit"
        )
    if request.ibp_power_goal is not None and request.ibp_power_goal > -1:
        raise ValueError("--ibp-power-goal must be <= -1; use no IBP flag to disable lowering")
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
    if request.chain_rule_formula_output_length_limit < 0:
        raise ValueError("--chain-rule-formula-output-length-limit must be >= 0")
    if request.direct_projector_cache_term_threshold < 0:
        raise ValueError("--direct-projector-cache-term-threshold must be >= 0")

    if request.integral == "dot":
        if request.command == "integrate":
            return
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
        "output",
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
        "command",
        nargs="?",
        choices=["run", "generate", "integrate", "cache", "test", "benchmark"],
        default=defaults.get("command", "run"),
        help=(
            "Two-stage DOT workflow command. Omit for the legacy single-shot "
            "generate+integrate path; use 'generate' to prepare a bundle and "
            "'integrate' to run strictly from a prepared bundle. Use 'cache' "
            "to warm topology-independent formula caches on example DOT cases, "
            "'test' to probe selected sectors near all endpoint corners, or "
            "'benchmark' to time ordinary f64 sector-integrand evaluation."
        ),
    )
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
        "--numerator-reducer",
        choices=["symbolica", "pysecdec"],
        default="symbolica",
        help=(
            "Reducer used to turn DOT momentum-space dot-product numerators into "
            "Feynman-parameter numerator polynomials for the FSD path."
        ),
    )
    parser.add_argument(
        "--sectors",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Restrict sector-oriented commands to the listed canonical sector ids "
            "from the sector summary table. In integration output, inactive sectors "
            "are still recorded in result.json with zero samples."
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
        "--sampling-mode",
        choices=["havana", "democratic", "qmc"],
        default="havana",
        help=(
            "Sector sampling policy. 'havana' uses the adaptive discrete sector "
            "dimension; 'democratic' gives every active sector the same number "
            "of uniform samples and records per-sector diagnostics; 'qmc' "
            "uses randomized shifted QMCPy rank-1 lattices per sector with "
            "Korobov periodization."
        ),
    )
    parser.add_argument(
        "--democratic-sampling",
        dest="sampling_mode",
        action="store_const",
        const="democratic",
        help="Shortcut for --sampling-mode democratic.",
    )
    parser.add_argument(
        "--democratic-samples-per-sector",
        type=int,
        default=1000,
        help="Uniform sample count assigned to each active sector in democratic mode.",
    )
    parser.add_argument(
        "--qmc-shifts",
        type=int,
        default=int(defaults.get("qmc_shifts", 16)),
        help=(
            "Number of independent random shifts for --sampling-mode qmc. "
            "The QMC one-sigma error is estimated from these shifted lattice "
            "replicates. Default: 16."
        ),
    )
    parser.add_argument(
        "--qmc-korobov-alpha",
        type=int,
        default=int(defaults.get("qmc_korobov_alpha", 3)),
        help=(
            "Korobov periodizing transform exponent used in QMC mode. "
            "The 2016 pySecDec-inspired setup uses alpha=3. Default: 3."
        ),
    )
    parser.add_argument(
        "--qmc-lattice-backend",
        choices=["qmcpy", "cbcpt-dn1-100"],
        default=str(defaults.get("qmc_lattice_backend", "cbcpt-dn1-100")),
        help=(
            "Rank-1 lattice source for --sampling-mode qmc. QMC integration "
            "is implemented independently of pySecDec. 'qmcpy' uses QMCPy's "
            "base-two lattice; 'cbcpt-dn1-100' uses a bundled CBC/PT rank-1 "
            "vector table with pySecDec-like prime rule sizes. Default: "
            "cbcpt-dn1-100."
        ),
    )
    parser.add_argument(
        "--qmc-order",
        choices=["linear", "radical-inverse", "gray"],
        default=str(defaults.get("qmc_order", "linear")),
        help=(
            "QMCPy lattice ordering. 'linear' is closest to pySecDec's direct "
            "rank-1 lattice loop; 'radical-inverse' is QMCPy's historical "
            "default. Default: linear."
        ),
    )
    qmc_correlate_default = bool(defaults.get("qmc_correlate_sectors", True))
    parser.add_argument(
        "--qmc-correlate-sectors",
        dest="qmc_correlate_sectors",
        action="store_true",
        default=qmc_correlate_default,
        help=(
            "Use the same randomized lattice shifts for all sectors and "
            "estimate the total QMC error from the shift-by-shift sector sum. "
            "This is the default and is closest to pySecDec's grouped QMC "
            "estimator."
        ),
    )
    parser.add_argument(
        "--qmc-independent-sector-errors",
        dest="qmc_correlate_sectors",
        action="store_false",
        help=(
            "Legacy QMC error mode: integrate sectors with independent "
            "random shifts and combine sector errors in quadrature."
        ),
    )
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
        default=defaults.get("target_rel_accuracy", None),
        help=(
            "Optional target for the displayed summed relative MC error in percent. "
            "When enabled, progress and ETA are extrapolated with err ~ 1/sqrt(N)."
        ),
    )
    parser.add_argument(
        "--target-rel-error",
        "--target-relative-error",
        dest="target_rel_error",
        type=float,
        default=defaults.get("target_rel_error", None),
        help=(
            "Optional stop target for SUM(|MC errors|)/SUM(|coefficients|) as a "
            "dimensionless ratio. For example, 3e-4 is equivalent to "
            "--target-rel-accuracy 0.03."
        ),
    )
    parser.add_argument(
        "--target-abs-error",
        "--target-absolute-error",
        dest="target_abs_error",
        type=float,
        default=defaults.get("target_abs_error", None),
        help=(
            "Optional stop target for SUM(|MC errors|), in the selected "
            "prefactor convention."
        ),
    )
    parser.add_argument(
        "--target-integration-time",
        type=float,
        default=defaults.get("target_integration_time", None),
        help=(
            "Requested integration wall time in seconds. FSD performs a short "
            "same-worker-count warm-up and adjusts sampling statistics to get "
            "close to this budget; the elapsed-time stop remains active as a guard."
        ),
    )
    parser.add_argument("--min-error", type=float, default=2.0e-4)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=1.0e-3,
        help=(
            "Endpoint-distance threshold on dimensionless sector coordinates below "
            "which Symbolica evaluators use evaluate_with_prec(..., --stability-precision)."
        ),
    )
    parser.add_argument(
        "--medium-precision-stability-threshold",
        type=float,
        default=1.0e-6,
        help=(
            "Intermediate endpoint-distance threshold below which Symbolica evaluators "
            "use evaluate_with_prec(..., --medium-precision-stability-precision)."
        ),
    )
    parser.add_argument(
        "--high-precision-stability-threshold",
        type=float,
        default=1.0e-8,
        help=(
            "Stronger endpoint-distance threshold below which Symbolica evaluators "
            "use evaluate_with_prec(..., --high-precision-stability-precision)."
        ),
    )
    parser.add_argument(
        "--stability-precision",
        type=int,
        default=32,
        help="Decimal digits used for Symbolica evaluator calls below --stability-threshold.",
    )
    parser.add_argument(
        "--medium-precision-stability-precision",
        type=int,
        default=100,
        help=(
            "Decimal digits used for Symbolica evaluator calls below "
            "--medium-precision-stability-threshold."
        ),
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
        "--max-weight-precision-xi",
        type=float,
        default=float(defaults.get("max_weight_precision_xi", 0.9)),
        help=(
            "Safety guard for rare large weights.  If a weighted Laurent row is "
            "at least xi times the current per-sector maximum in any coefficient, "
            "the row is recomputed with --high-precision-stability-precision before "
            "being accumulated.  Set to 0 to disable."
        ),
    )
    default_evaluator_mode = str(
        defaults.get(
            "evaluator_compile_mode",
            "jit" if bool(defaults.get("jit_compile_evaluators", True)) else "eager",
        )
    )
    evaluator_mode_group = parser.add_mutually_exclusive_group()
    evaluator_mode_group.add_argument(
        "--jit-compile",
        "--jit-compile-evaluators",
        dest="evaluator_compile_mode",
        action="store_const",
        const="jit",
        default=default_evaluator_mode,
        help=(
            "Build Symbolica evaluators with jit_compile=True. This is the "
            "default outside QMC; QMC uses eager complex evaluators unless an "
            "evaluator mode is supplied explicitly."
        ),
    )
    evaluator_mode_group.add_argument(
        "--compile",
        dest="evaluator_compile_mode",
        action="store_const",
        const="compile",
        help=(
            "Build Symbolica evaluators and compile their f64 hot path to a shared "
            "library. Multiprecision rescue keeps a source-evaluator fallback."
        ),
    )
    evaluator_mode_group.add_argument(
        "--eager-evaluator",
        "--no-jit-compile",
        dest="evaluator_compile_mode",
        action="store_const",
        const="eager",
        help="Build eager Symbolica evaluators without JIT or compiled f64 code.",
    )
    real_default = bool(defaults.get("real_evaluator", True))
    real_group = parser.add_mutually_exclusive_group()
    real_group.add_argument(
        "--real-evaluator",
        dest="real_evaluator",
        action="store_true",
        default=real_default,
        help=(
            "Use real-valued Symbolica evaluator APIs for f64 real kinematics. "
            "Default outside QMC."
        ),
    )
    real_group.add_argument(
        "--complex-evaluator",
        dest="real_evaluator",
        action="store_false",
        help="Force complex-valued Symbolica evaluator APIs for f64 batches.",
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
        "--sector-evaluator-backend",
        choices=["projector", "two-stage-explicit", "explicit"],
        default=str(defaults.get("sector_evaluator_backend", "explicit")),
        help=(
            "Singular-sector runtime evaluator layout. 'explicit' builds one "
            "fully substituted multi-output Symbolica evaluator per sector and "
            "is the default. 'projector' keeps the fast-generation parametric "
            "black-box U/F path. 'two-stage-explicit' prepares one "
            "source-coefficient evaluator and one assembler evaluator per "
            "singular sector."
        ),
    )
    parser.add_argument(
        "--explicit",
        dest="sector_evaluator_backend",
        action="store_const",
        const="explicit",
        help="Shortcut for --sector-evaluator-backend explicit; this is the default.",
    )
    parser.add_argument(
        "--projector-generation",
        dest="sector_evaluator_backend",
        action="store_const",
        const="projector",
        help=(
            "Use the fast-generation projector backend: keep U/F as black-box "
            "evaluators and assemble sector weights from universal projector "
            "formulae. This is slower at evaluation time than --explicit."
        ),
    )
    parser.add_argument(
        "--parametric-generation",
        dest="sector_evaluator_backend",
        action="store_const",
        const="projector",
        help=argparse.SUPPRESS,
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
        default=8192,
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
        default=6,
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
        "--chain-rule-formula-output-length-limit",
        type=int,
        default=0,
        help=(
            "Skip cold chain-rule formula generation when the requested output "
            "coefficient count exceeds this value. Cached formulas above the "
            "limit are still reused. 0 disables the per-signature guard."
        ),
    )
    parser.add_argument(
        "--allow-fallback-for-missing-caches",
        action="store_true",
        help=(
            "Allow generation of missing universal subtraction formula cache entries "
            "during this run. Disabled by default so generation benchmarks do not "
            "silently include cold cache construction."
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
        "--output",
        default=None,
        help=(
            "Prepared bundle directory. Required by 'generate' and 'integrate'. "
            "Integrate writes result.json inside this directory unless --result-path is supplied."
        ),
    )
    parser.add_argument(
        "--evaluator-lru-size",
        type=int,
        default=128,
        help=(
            "Prepared integrate-mode evaluator LRU size, counted in artifact groups. "
            "Use 0 for an unlimited cache."
        ),
    )
    parser.add_argument(
        "--cache-loop-counts",
        nargs="+",
        type=int,
        default=defaults.get("cache_loop_counts", (1, 2)),
        help="Loop counts to cover in 'cache' mode. Default: 1 2.",
    )
    parser.add_argument(
        "--cache-cases",
        nargs="+",
        default=defaults.get("cache_cases"),
        help=(
            "Named example DOT cases to warm instead of selecting by loop count. "
            "Known examples include triangle, box, kite_2loop, "
            "three_point_2loop, three_point_2loop_6line, double_box, and the "
            "available 3L examples."
        ),
    )
    parser.add_argument(
        "--cache-report-path",
        default=defaults.get("cache_report_path", "docs/universal_cache_report.json"),
        help="JSON report written by 'cache' mode.",
    )
    parser.add_argument(
        "--cache-workdir",
        default=defaults.get("cache_workdir", ".cache_warm"),
        help="Scratch directory for cache-mode low-stat verification outputs.",
    )
    parser.add_argument(
        "--cache-verify-samples-per-sector",
        type=int,
        default=int(defaults.get("cache_verify_samples_per_sector", 16)),
        help="Democratic samples per sector used to verify warmed cache assets.",
    )
    parser.add_argument(
        "--test-boundary-distances",
        nargs="+",
        type=float,
        default=defaults.get("test_boundary_distances", (1.0e-6, 1.0e-8)),
        help=(
            "Endpoint distances used by the 'test' subcommand. For each distance "
            "the test probes all low/high hypercube corners plus diagnostic faces."
        ),
    )
    parser.add_argument(
        "--test-boundary-growth-power-tolerance",
        "--test-boundary-relative-tolerance",
        dest="test_boundary_growth_power_tolerance",
        type=float,
        default=defaults.get(
            "test_boundary_growth_power_tolerance",
            defaults.get("test_boundary_relative_tolerance", 0.5),
        ),
        help=(
            "Per-endpoint-distance-coordinate allowed effective power-law growth exponent in the Laurent "
            "weight vector or training weight between matched endpoint probes at "
            "consecutive --test-boundary-distances. This flags power-like endpoint "
            "growth when p exceeds tolerance times the number of coordinates tending to zero. "
            "Set to a negative value to disable this stability check."
        ),
    )
    parser.add_argument(
        "--test-report-path",
        default=defaults.get("test_report_path"),
        help="Optional JSON report path written by the 'test' subcommand.",
    )
    parser.add_argument(
        "--retries",
        dest="test_boundary_retry_scales",
        nargs="+",
        type=float,
        default=defaults.get("test_boundary_retry_scales", defaults.get("retries", (1.0e-2,))),
        help=(
            "Retry failed endpoint-test sectors with all "
            "--test-boundary-distances multiplied by each listed positive scale. "
            "Default: 1e-2. Example: --retries 1e-2 retries 1e-4 1e-6 1e-8 "
            "as 1e-6 1e-8 1e-10."
        ),
    )
    parser.add_argument(
        "--max-simultaneous-endpoint-approaches",
        dest="test_boundary_max_simultaneous_endpoint_approaches",
        type=int,
        default=defaults.get("test_boundary_max_simultaneous_endpoint_approaches"),
        help=(
            "In endpoint-test mode, probe only endpoint approaches where at most "
            "N sector variables approach lower/upper limits simultaneously. "
            "Unselected variables stay at deterministic bulk values and are "
            "labelled with 'x' in probe names."
        ),
    )
    parser.add_argument(
        "--benchmark-samples-per-sector",
        type=int,
        default=int(defaults.get("benchmark_samples_per_sector", 5)),
        help=(
            "Number of ordinary f64 interior sample points evaluated per active "
            "sector by the 'benchmark' subcommand. Default: 5."
        ),
    )
    parser.add_argument(
        "--no-cache-estimate-3l",
        dest="cache_estimate_3l",
        action="store_false",
        help="Skip the rough 3L cache-generation estimate in 'cache' mode.",
    )
    parser.set_defaults(cache_estimate_3l=defaults.get("cache_estimate_3l", True))
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
            "For endpoint-projector generation, lower y^(-n+c eps) endpoints "
            "to logarithmic y^(-1+c eps) endpoints by integration by parts. "
            "Equivalent to --ibp-power-goal -1."
        ),
    )
    parser.add_argument(
        "--ibp-power-goal",
        type=int,
        default=defaults.get("ibp_power_goal"),
        help=(
            "Numeric IBP endpoint lowering goal. For example -3 lowers "
            "endpoints only until their base power is >= -3; -1 reproduces "
            "--ibp-reduce-to-log-endpoint. Omit to disable IBP lowering."
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
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--run", dest="run_file", default=None)
    pre_args, _unknown = pre_parser.parse_known_args(raw_argv)
    try:
        defaults = _load_run_defaults(pre_args.run_file)
    except Exception as exc:
        raise SystemExit(f"error: {exc}") from exc
    parser = build_parser(defaults)
    parsed = parser.parse_args(raw_argv)
    parsed.max_eps_order_explicit = (
        "--max-eps-order" in raw_argv
        or "max_eps_order" in defaults
        or "max-eps-order" in defaults
    )
    parsed.ibp_power_goal_explicit = "--ibp-power-goal" in raw_argv
    parsed.ibp_reduce_disabled_explicit = (
        "--no-IBP_reduce_to_log_endpoint" in raw_argv
        or "--no-ibp-reduce-to-log-endpoint" in raw_argv
    )
    parsed.evaluator_mode_explicit = (
        "--jit-compile" in raw_argv
        or "--jit-compile-evaluators" in raw_argv
        or "--compile" in raw_argv
        or "--eager-evaluator" in raw_argv
        or "--no-jit-compile" in raw_argv
        or "evaluator_compile_mode" in defaults
        or "jit_compile_evaluators" in defaults
    )
    parsed.real_evaluator_explicit = (
        "--real-evaluator" in raw_argv
        or "--complex-evaluator" in raw_argv
        or "real_evaluator" in defaults
    )
    return parsed


def build_request(args: argparse.Namespace) -> IntegralRequest:
    """Convert argparse output into the immutable request object used below."""
    mode = resolve_mode(args.m, args.mode)
    command = getattr(args, "command", "run") or "run"
    dot_file = str(Path(args.dot_file).expanduser()) if args.dot_file is not None else None
    kinematics_file = str(Path(args.kinematics).expanduser()) if args.kinematics is not None else None
    prefactor_convention = args.prefactor_convention
    if prefactor_convention is None:
        prefactor_convention = "pysecdec" if dot_file is not None or command in {"generate", "integrate", "cache"} else "raw"
    output = str(Path(args.output).expanduser()) if args.output is not None else None
    if args.result_path is not None:
        result_path = str(Path(args.result_path).expanduser())
    elif command == "integrate" and output is not None:
        result_path = str(Path(output).expanduser() / "result.json")
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
    chain_rule_formula_output_length_limit = int(args.chain_rule_formula_output_length_limit)
    if force_regular_taylor_formulas:
        regular_taylor_signature_limit = max(regular_taylor_signature_limit, 1_000_000)
        regular_taylor_formula_volume_limit = max(regular_taylor_formula_volume_limit, 1_000_000)
        regular_taylor_formula_axis_limit = max(regular_taylor_formula_axis_limit, 32)
    if bool(getattr(args, "ibp_power_goal_explicit", False)):
        ibp_power_goal = None if args.ibp_power_goal is None else int(args.ibp_power_goal)
    elif bool(getattr(args, "ibp_reduce_disabled_explicit", False)):
        ibp_power_goal = None
    elif args.ibp_power_goal is not None:
        ibp_power_goal = int(args.ibp_power_goal)
    elif bool(args.ibp_reduce_to_log_endpoint):
        ibp_power_goal = -1
    else:
        ibp_power_goal = None
    ibp_reduce_to_log_endpoint = ibp_power_goal == -1
    evaluator_compile_mode = str(args.evaluator_compile_mode)
    real_evaluator = bool(args.real_evaluator)
    # The current Symbolica real-JIT path is intentionally still exposed when
    # requested explicitly, because it has standalone MREs in this repository.
    # For automatic QMC steering, however, the validated parity path with
    # pySecDec/OneLOop is eager complex evaluation; using it by default avoids
    # silently producing biased QMC coefficients.
    if str(args.sampling_mode) == "qmc":
        if not bool(getattr(args, "evaluator_mode_explicit", False)):
            evaluator_compile_mode = "eager"
            if not bool(getattr(args, "real_evaluator_explicit", False)):
                real_evaluator = False

    return IntegralRequest(
        run_file=str(Path(args.run_file).expanduser().resolve()) if args.run_file is not None else None,
        integral="dot" if dot_file is not None or command in {"generate", "integrate"} else args.integral,
        dot_file=dot_file,
        kinematics_file=kinematics_file,
        graph_name=args.graph_name,
        sector_method=args.sector_method,
        normaliz_executable=args.normaliz_executable,
        dot_engine=args.dot_engine,
        numerator_reducer=args.numerator_reducer,
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
        sampling_mode=args.sampling_mode,
        democratic_samples_per_sector=args.democratic_samples_per_sector,
        qmc_shifts=int(args.qmc_shifts),
        qmc_korobov_alpha=int(args.qmc_korobov_alpha),
        qmc_lattice_backend=str(args.qmc_lattice_backend),
        qmc_order=str(args.qmc_order),
        qmc_correlate_sectors=bool(args.qmc_correlate_sectors),
        target_rel_accuracy=(
            None if args.target_rel_accuracy is None else float(args.target_rel_accuracy)
        ),
        target_abs_error=None if args.target_abs_error is None else float(args.target_abs_error),
        target_rel_error=None if args.target_rel_error is None else float(args.target_rel_error),
        target_integration_time=(
            None
            if args.target_integration_time is None
            else float(args.target_integration_time)
        ),
        min_error=args.min_error,
        bins=args.bins,
        workers=args.workers,
        jit_compile_evaluators=evaluator_compile_mode == "jit",
        evaluator_compile_mode=evaluator_compile_mode,
        real_evaluator=real_evaluator,
        dual_evaluator_mode=args.dual_evaluator_mode,
        subtraction_backend=args.subtraction_backend,
        sector_evaluator_backend=args.sector_evaluator_backend,
        ibp_reduce_to_log_endpoint=ibp_reduce_to_log_endpoint,
        ibp_power_goal=ibp_power_goal,
        direct_projector_cache_term_threshold=args.direct_projector_cache_term_threshold,
        allow_fallback_for_missing_caches=bool(
            args.allow_fallback_for_missing_caches or command == "cache"
        ),
        force_regular_taylor_formulas=force_regular_taylor_formulas,
        regular_taylor_signature_limit=regular_taylor_signature_limit,
        regular_taylor_formula_volume_limit=regular_taylor_formula_volume_limit,
        regular_taylor_formula_axis_limit=regular_taylor_formula_axis_limit,
        chain_rule_formula_signature_limit=chain_rule_formula_signature_limit,
        chain_rule_formula_output_length_limit=chain_rule_formula_output_length_limit,
        stability_threshold=args.stability_threshold,
        medium_precision_stability_threshold=args.medium_precision_stability_threshold,
        high_precision_stability_threshold=args.high_precision_stability_threshold,
        stability_precision=args.stability_precision,
        medium_precision_stability_precision=args.medium_precision_stability_precision,
        high_precision_stability_precision=args.high_precision_stability_precision,
        max_weight_precision_xi=args.max_weight_precision_xi,
        show_stats=args.show_stats,
        no_progress=args.no_progress,
        quiet_summary=args.quiet_summary,
        json=args.json,
        mu=args.mu,
        onshell_threshold=args.onshell_threshold,
        command=command,
        output=output,
        evaluator_lru_size=int(args.evaluator_lru_size),
        max_eps_order_explicit=bool(getattr(args, "max_eps_order_explicit", False)),
        cache_loop_counts=tuple(int(loop) for loop in args.cache_loop_counts),
        cache_cases=tuple(str(case) for case in args.cache_cases) if args.cache_cases is not None else None,
        cache_report_path=str(Path(args.cache_report_path).expanduser()),
        cache_workdir=str(Path(args.cache_workdir).expanduser()),
        cache_verify_samples_per_sector=int(args.cache_verify_samples_per_sector),
        cache_estimate_3l=bool(args.cache_estimate_3l),
        test_boundary_distances=tuple(float(value) for value in args.test_boundary_distances),
        test_boundary_growth_power_tolerance=(
            None
            if args.test_boundary_growth_power_tolerance is not None
            and float(args.test_boundary_growth_power_tolerance) < 0.0
            else float(args.test_boundary_growth_power_tolerance)
        ),
        test_report_path=(
            str(Path(args.test_report_path).expanduser())
            if args.test_report_path is not None
            else None
        ),
        test_boundary_retry_scales=tuple(
            float(value) for value in (args.test_boundary_retry_scales or ())
        ),
        test_boundary_max_simultaneous_endpoint_approaches=(
            int(args.test_boundary_max_simultaneous_endpoint_approaches)
            if args.test_boundary_max_simultaneous_endpoint_approaches is not None
            else None
        ),
        benchmark_samples_per_sector=int(args.benchmark_samples_per_sector),
    )


def deepest_laurent_order(topology) -> int:
    """Return the universal scalar-integral deepest pole order ``-2L``."""
    parametric = topology.parametric_representation
    loop_count = int(parametric.loop_count if parametric is not None else 1)
    return -2 * loop_count


def _sector_max_order_for_display(request: IntegralRequest, topology, min_order: int) -> int:
    """Return the raw sector order needed for the requested displayed order.

    In DOT pySecDec convention the global Gamma prefactor is multiplied after
    sector integration.  If that prefactor starts at eps^p with p < 0, a final
    coefficient through eps^M needs raw sector coefficients through eps^(M-p).
    """
    if request.integral == "dot" and request.prefactor_convention == "pysecdec":
        prefactor_min_order = int(getattr(topology, "global_prefactor_min_order", 0))
        display_min_order = int(min_order) + prefactor_min_order
        if request.max_eps_order < display_min_order:
            raise ValueError(
                f"--max-eps-order must be >= eps^{display_min_order}; "
                f"got eps^{request.max_eps_order}"
            )
        return int(request.max_eps_order) - prefactor_min_order
    if request.max_eps_order < min_order:
        raise ValueError(
            f"--max-eps-order must be >= eps^{min_order}; got eps^{request.max_eps_order}"
        )
    return int(request.max_eps_order)


def configure_laurent_range(request: IntegralRequest, topology, sectors) -> None:
    """Apply the CLI Laurent range and validate sector endpoint depth."""
    min_order = deepest_laurent_order(topology)
    sector_max_order = _sector_max_order_for_display(request, topology, min_order)
    max_sector_depth = max((len(sector.singular_axes) for sector in sectors), default=0)
    if max_sector_depth > -min_order:
        worst = [
            sector.name for sector in sectors if len(sector.singular_axes) == max_sector_depth
        ][:5]
        raise ValueError(
            f"sector endpoint pole depth {max_sector_depth} exceeds universal "
            f"2L depth {-min_order}; examples: {', '.join(worst)}"
        )
    topology.set_laurent_range(min_order, sector_max_order)


def _display_laurent_orders(request: IntegralRequest, topology) -> list[int]:
    """Return the Laurent orders shown in the selected prefactor convention."""
    if request.command == "integrate" and not request.max_eps_order_explicit:
        display_max_order = int(topology.laurent_max_order)
    else:
        display_max_order = int(request.max_eps_order)
    if request.integral == "dot" and request.prefactor_convention == "pysecdec":
        prefactor_min_order = int(getattr(topology, "global_prefactor_min_order", 0))
        display_min_order = int(topology.laurent_min_order) + prefactor_min_order
        if request.command == "integrate" and not request.max_eps_order_explicit:
            display_max_order = int(topology.laurent_max_order) + prefactor_min_order
        return list(range(display_min_order, display_max_order + 1))
    return list(range(int(topology.laurent_min_order), display_max_order + 1))


def _prepared_output_count(request: IntegralRequest, topology) -> int:
    """Return how many Laurent coefficients should be shown for integrate mode."""
    if request.command != "integrate" or not request.max_eps_order_explicit:
        return len(_display_laurent_orders(request, topology))
    return len(_display_laurent_orders(request, topology))


def _trim_sequence(values: list, count: int) -> list:
    """Return the deepest-pole-first prefix used by the displayed Laurent range."""
    return list(values[:count])


def _trim_target(target: TargetDefinition | None, count: int) -> TargetDefinition | None:
    """Trim an optional comparison target to the displayed prepared subrange."""
    if target is None:
        return None
    return TargetDefinition(
        source=target.source,
        convention=target.convention,
        coefficients=_trim_sequence(target.coefficients, count),
        errors=_trim_sequence(target.errors, count),
        metadata=target.metadata,
    )


def _trim_sector_rows(rows: list[dict], count: int) -> list[dict]:
    """Trim per-sector coefficient arrays to the displayed prepared subrange."""
    trimmed_rows: list[dict] = []
    for row in rows:
        copied = dict(row)
        for key in ("raw", "raw_sector", "display"):
            if key in copied:
                block = dict(copied[key])
                block["coefficients"] = _trim_sequence(block.get("coefficients", []), count)
                block["errors"] = _trim_sequence(block.get("errors", []), count)
                copied[key] = block
        display = copied.get("display", {})
        coeffs = display.get("coefficients", []) if isinstance(display, dict) else []
        errors = display.get("errors", []) if isinstance(display, dict) else []
        copied["sort_keys"] = {
            "abs_central": max((abs(value) for value in coeffs), default=0.0),
            "abs_error": max((abs(value) for value in errors), default=0.0),
        }
        trimmed_rows.append(copied)
    return trimmed_rows


def _trim_summary_laurent(summary: dict, topology, count: int) -> None:
    """Update summary Laurent labels after prepared subrange output trimming."""
    labels = topology.expected_laurent_orders[:count]
    summary.setdefault("validation", {})["expected_laurent_orders"] = labels
    summary.setdefault("symanzik", {})["expected_laurent_orders"] = labels


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


def _align_coefficients_by_order(
    values: list[complex],
    source_orders: list[int] | tuple[int, ...] | None,
    target_orders: list[int],
) -> list[complex]:
    """Align coefficients by their integer epsilon powers when known.

    pySecDec may return a reduced Laurent range when numerator factors cancel
    the deepest universal pole.  In that case positional alignment would shift
    the target by one or more epsilon powers.  Order-aware alignment keeps
    explicit zero coefficients in the missing deepest-pole rows.
    """
    if not source_orders:
        return _align_coefficients(values, len(target_orders))
    by_order = {
        int(order): values[index]
        for index, order in enumerate(source_orders)
        if index < len(values)
    }
    return [by_order.get(int(order), 0.0 + 0.0j) for order in target_orders]


def _align_target_to_topology(
    target: TargetDefinition,
    topology,
    request: IntegralRequest,
) -> TargetDefinition:
    """Return a target aligned to the displayed Laurent order range."""
    source_orders = target.metadata.get("orders")
    target_orders = _display_laurent_orders(request, topology)
    return TargetDefinition(
        source=target.source,
        convention=target.convention,
        coefficients=_align_coefficients_by_order(
            target.coefficients,
            source_orders,
            target_orders,
        ),
        errors=_align_coefficients_by_order(target.errors, source_orders, target_orders),
        metadata=target.metadata,
    )


def _numeric_target(request: IntegralRequest, topology, args: tuple[str, ...]) -> TargetDefinition:
    """Parse numeric re/im target pairs in the selected display convention."""
    if len(args) % 2 != 0:
        raise ValueError("--target numeric form requires re/im pairs")
    pair_count = len(args) // 2
    target_count = len(_display_laurent_orders(request, topology))
    if pair_count > target_count:
        raise ValueError(
            f"--target supplies {pair_count} coefficients but current Laurent range has "
            f"{target_count}"
        )
    coeffs = [
        complex(float(args[2 * index]), float(args[2 * index + 1]))
        for index in range(pair_count)
    ]
    coeffs.extend([0.0 + 0.0j for _ in range(target_count - pair_count)])
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
        metadata={"workdir": request.pysecdec_workdir, "orders": result.orders},
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
            "orders": target.metadata.get("orders", []),
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
                return _align_target_to_topology(target, topology, request)
            candidate_path = Path(token).expanduser()
            if candidate_path.is_file() and not request.refresh_target:
                target = target_from_result_file(candidate_path, request.prefactor_convention)
                return _align_target_to_topology(target, topology, request)
            if not _is_numeric_token(token):
                if request.command == "integrate":
                    raise ValueError(
                        "prepared integrate mode accepts only numeric targets or an existing result.json file"
                    )
                _write_pysecdec_target_file(request, candidate_path, summary, logger=logger)
                target = target_from_result_file(candidate_path, request.prefactor_convention)
                return _align_target_to_topology(target, topology, request)
        return _numeric_target(request, topology, args)

    if request.integral != "dot":
        return _oneloop_target(request, topology)
    if request.dot_engine == "both":
        target = _pysecdec_target(request, summary, logger=logger)
        return _align_target_to_topology(target, topology, request)
    return None


def configure_logging(request: IntegralRequest) -> logging.Logger:
    """Configure stdlib logging for generation timing and backend diagnostics."""
    level = getattr(logging, request.log_level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if not request.json:
        console_handler = logging.StreamHandler(sys.stderr)
        # INFO logging and progressbar redraws both target stderr.  During live
        # progress, keep INFO records out of the terminal to avoid the
        # ``... detail: <large padding> INFO:FSD:...`` suffix seen in wrapped
        # generation output.  Full INFO detail is still available with
        # ``--no-progress`` or in ``--log-file``.
        if not request.no_progress and level <= logging.INFO:
            console_handler.setLevel(logging.WARNING)
        else:
            console_handler.setLevel(level)
        handlers.append(console_handler)
    if request.log_file is not None:
        file_handler = logging.FileHandler(request.log_file, encoding="utf-8")
        file_handler.setLevel(level)
        handlers.append(file_handler)
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


def _prepare_sector_runtime_artifacts(
    request: IntegralRequest,
    topology,
    active_sectors,
    generation_progress: GenerationProgress | None,
) -> float:
    """Apply CLI generation options and build all runtime evaluator artifacts."""
    topology.generation_workers = max(int(request.workers), 1)
    topology.regular_taylor_formula_signature_limit = request.regular_taylor_signature_limit
    topology.regular_taylor_formula_volume_limit = request.regular_taylor_formula_volume_limit
    topology.regular_taylor_formula_axis_limit = request.regular_taylor_formula_axis_limit
    topology.chain_rule_formula_signature_limit = request.chain_rule_formula_signature_limit
    topology.chain_rule_formula_output_length_limit = request.chain_rule_formula_output_length_limit
    topology.direct_projector_cache_term_threshold = request.direct_projector_cache_term_threshold
    topology.allow_fallback_for_missing_caches = request.allow_fallback_for_missing_caches
    bridge_already_prepared_dot_duals = (
        request.command == "generate" and request.integral == "dot"
        and request.sector_evaluator_backend not in {"two-stage-explicit", "explicit"}
    )
    if (
        request.sector_evaluator_backend not in {"two-stage-explicit", "explicit"}
        and not bridge_already_prepared_dot_duals
    ):
        topology.prepare_dual_evaluators(
            active_sectors,
            request.dual_evaluator_mode,
            progress=generation_progress,
        )
    extra_dual_build_before = topology.dual_evaluator_build_seconds
    if request.subtraction_backend == "formula":
        topology.prepare_subtraction_formulas(active_sectors, progress=generation_progress)
    elif request.subtraction_backend == "projector-formula":
        topology.prepare_endpoint_projector_formulas(active_sectors, progress=generation_progress)
        if request.sector_evaluator_backend == "explicit":
            topology.prepare_explicit_sector_formulas(active_sectors, progress=generation_progress)
        elif request.sector_evaluator_backend == "two-stage-explicit":
            topology.prepare_two_stage_sector_formulas(active_sectors, progress=generation_progress)
        else:
            topology.prepare_regular_taylor_formulas(active_sectors, progress=generation_progress)
            topology.prepare_chain_rule_formulas(active_sectors, progress=generation_progress)
    return extra_dual_build_before


def _record_generation_artifact_timings(
    request: IntegralRequest,
    topology,
    extra_dual_build_before: float,
) -> None:
    """Append evaluator/formula build timings to the DOT generation timeline."""
    if request.integral != "dot" or request.command == "integrate":
        return
    if topology.subtraction_formula_build_seconds > 0.0:
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
            endpoint_cache_hits = getattr(topology, "endpoint_projector_formulas_from_cache", 0)
            endpoint_generated = getattr(topology, "endpoint_projector_formulas_generated", 0)
            regular_cache_hits = getattr(topology, "regular_taylor_formulas_from_cache", 0)
            regular_generated = getattr(topology, "regular_taylor_formulas_generated", 0)
            if endpoint_cache_hits or endpoint_generated:
                detail += (
                    f", endpoint cache hits/generated="
                    f"{endpoint_cache_hits}/{endpoint_generated}"
                )
            if regular_cache_hits or regular_generated:
                detail += (
                    f", regular cache hits/generated="
                    f"{regular_cache_hits}/{regular_generated}"
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
                "mapped derivative formula(s), "
                f"cache hits/generated="
                f"{getattr(topology, 'chain_rule_formulas_from_cache', 0)}/"
                f"{getattr(topology, 'chain_rule_formulas_generated', 0)}"
            ),
        )
    elif getattr(topology, "chain_rule_formulas_skipped", 0):
        get_dot_bundle(request).timings.add(
            "Symbolica chain-rule formula build",
            0.0,
            detail=f"skipped {topology.chain_rule_formulas_skipped} mapped derivative formula(s)",
        )
    if getattr(topology, "two_stage_sector_formula_build_seconds", 0.0) > 0.0:
        get_dot_bundle(request).timings.add(
            "Symbolica two-stage sector build",
            topology.two_stage_sector_formula_build_seconds,
            detail=(
                f"{getattr(topology, 'two_stage_sector_formulas_generated', 0)} "
                "source+assembler sector evaluator pair(s)"
            ),
        )
    if getattr(topology, "explicit_sector_formula_build_seconds", 0.0) > 0.0:
        get_dot_bundle(request).timings.add(
            "Symbolica explicit sector build",
            topology.explicit_sector_formula_build_seconds,
            detail=(
                f"{getattr(topology, 'explicit_sector_formulas_generated', 0)} "
                "single-evaluator sector integrand(s)"
            ),
        )


def _main_impl() -> int:
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
    prepared_manifest: dict | None = None

    try:
        validate_request(request)
        if request.command == "cache":
            run_universal_cache_mode(request, logger)
            return 0
        legacy_prepared_output = (
            request.command == "run"
            and request.integral == "dot"
            and request.output is not None
            and (Path(request.output).expanduser() / "manifest.json").is_file()
        )
        if request.command == "integrate" or legacy_prepared_output:
            topology, sectors, prepared_manifest = load_prepared_bundle(
                request.output or "",
                lru_size=request.evaluator_lru_size,
            )
            generation_options = prepared_manifest.get("generation_options", {})
            prepared_min_order = topology.laurent_min_order
            prepared_max_order = topology.laurent_max_order
            if request.max_eps_order_explicit:
                display_min_order = (
                    prepared_min_order
                    + int(getattr(topology, "global_prefactor_min_order", 0))
                    if request.prefactor_convention == "pysecdec"
                    else prepared_min_order
                )
                required_sector_max_order = _sector_max_order_for_display(
                    request,
                    topology,
                    prepared_min_order,
                )
                if request.max_eps_order < display_min_order:
                    raise ValueError(
                        "--max-eps-order is below the deepest pole prepared in "
                        f"the bundle: got eps^{request.max_eps_order}, prepared "
                        f"starts at eps^{display_min_order} in the selected convention"
                    )
                if required_sector_max_order > prepared_max_order:
                    raise ValueError(
                        "--max-eps-order cannot exceed the prepared bundle range: "
                        f"got displayed eps^{request.max_eps_order}, which requires "
                        f"raw sector eps^{required_sector_max_order}, but the bundle "
                        f"is prepared only through eps^{prepared_max_order}"
                    )
            request = replace(
                request,
                dot_global_prefactor_coeffs=tuple(topology.global_prefactor_coeffs or []),
                dot_global_prefactor_min_order=int(getattr(topology, "global_prefactor_min_order", 0)),
                dot_sector_laurent_min_order=int(topology.laurent_min_order),
                dot_sector_laurent_max_order=int(topology.laurent_max_order),
                dual_evaluator_mode=topology.dual_evaluator_mode,
                ibp_reduce_to_log_endpoint=topology.ibp_reduce_to_log_endpoint,
                ibp_power_goal=topology.ibp_power_goal,
                subtraction_backend=str(
                    generation_options.get("subtraction_backend", request.subtraction_backend)
                ),
                sector_evaluator_backend=str(
                    generation_options.get(
                        "sector_evaluator_backend",
                        request.sector_evaluator_backend,
                    )
                ),
                direct_projector_cache_term_threshold=int(
                    generation_options.get(
                        "direct_projector_cache_term_threshold",
                        request.direct_projector_cache_term_threshold,
                    )
                ),
            )
        elif request.integral == "dot" and request.command != "generate":
            generation_progress = _generation_progress(request, logger, "FSD generation")
            bundle = get_dot_bundle(request, progress=generation_progress)
            request = replace(
                request,
                dot_global_prefactor_coeffs=tuple(bundle.topology.global_prefactor_coeffs or []),
                dot_global_prefactor_min_order=int(
                    getattr(bundle.topology, "global_prefactor_min_order", 0)
                ),
                dot_sector_laurent_min_order=int(bundle.topology.laurent_min_order),
                dot_sector_laurent_max_order=int(bundle.topology.laurent_max_order),
            )
        elif request.target_args is None:
            check_oneloop_bridge()
    except Exception as exc:
        if generation_progress is not None:
            generation_progress.close()
        print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
        return 2

    if request.command != "integrate" and prepared_manifest is None:
        if request.command == "generate" and request.integral == "dot" and generation_progress is None:
            generation_progress = _generation_progress(request, logger, "FSD generation")
        topology = build_topology(request)
        if request.command == "generate":
            topology.chain_rule_metadata_only = True
            if request.output is not None:
                topology.streaming_evaluator_cache_dir = str(
                    Path(request.output).expanduser().resolve() / ".stream_evaluator_cache"
                )
        sectors = generate_sectors(request)
    try:
        validate_sector_selection(request, sectors)
        if request.command != "integrate" and prepared_manifest is None:
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

    skip_fsd_runtime_prepare = (
        request.command == "benchmark"
        and request.integral == "dot"
        and request.dot_engine == "pysecdec"
    )
    if request.command != "integrate" and prepared_manifest is None and not skip_fsd_runtime_prepare:
        extra_dual_build_before = _prepare_sector_runtime_artifacts(
            request,
            topology,
            active_sectors,
            generation_progress,
        )
        if generation_progress is not None:
            generation_progress.close()
            generation_progress = None
        _record_generation_artifact_timings(request, topology, extra_dual_build_before)
    elif generation_progress is not None:
        generation_progress.close()
        generation_progress = None
    summary = summary_data(request, topology, sectors, benchmark_available=False)
    if prepared_manifest is not None:
        summary["prepared_bundle"] = prepared_manifest
        timings_file = Path(request.output or "") / "generation_timings.json"
        if timings_file.is_file():
            try:
                import json as _json

                summary["generation_timings"] = _json.loads(timings_file.read_text(encoding="utf-8"))
            except Exception:
                summary["generation_timings"] = {}
    elif request.integral == "dot":
        summary["generation_timings"] = get_dot_bundle(request).timings.to_summary_dict()

    if request.command == "generate":
        try:
            manifest = save_prepared_bundle(
                request.output or "",
                request,
                topology,
                sectors,
                generation_timings=summary.get("generation_timings", {}),
            )
        except Exception as exc:
            print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
            return 1
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
            "prepared_bundle": manifest,
            "output": str(Path(request.output or "").expanduser().resolve()),
        }
        if request.json:
            print(output_json(output))
        else:
            if not request.quiet_summary:
                print_generation_report(request, summary)
            print(
                f"{Fore.GREEN}prepared bundle written:{Style.RESET_ALL} "
                f"{Path(request.output or '').expanduser().resolve()}"
            )
            print(
                f"{Fore.CYAN}artifact counts:{Style.RESET_ALL} "
                f"{manifest.get('artifact_counts', {})}"
            )
        return 0

    if (
        request.command == "run"
        and request.integral == "dot"
        and request.output is not None
        and request.dot_engine in {"fsd", "both"}
        and prepared_manifest is None
    ):
        try:
            prepared_manifest = save_prepared_bundle(
                request.output,
                request,
                topology,
                sectors,
                generation_timings=summary.get("generation_timings", {}),
            )
            summary["prepared_bundle"] = prepared_manifest
            summary["prepared_bundle"]["saved_during_run"] = True
        except Exception as exc:
            print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
            return 1
        if not request.json and not request.quiet_summary:
            print(
                f"{Fore.GREEN}prepared bundle written:{Style.RESET_ALL} "
                f"{Path(request.output).expanduser().resolve()}"
            )

    if request.command == "benchmark":
        try:
            if request.integral == "dot" and request.dot_engine == "pysecdec":
                pysecdec_progress = _generation_progress(
                    request,
                    logger,
                    "pySecDec benchmark package",
                )
                try:
                    paths, pysecdec_timings = ensure_pysecdec_package(
                        get_dot_bundle(request),
                        request,
                        progress=pysecdec_progress,
                    )
                finally:
                    pysecdec_progress.close()
                pysecdec_timings.log(logger)
                summary["pysecdec_timings"] = pysecdec_timings.to_summary_dict()
                if not request.json and not request.quiet_summary:
                    print_generation_report(request, summary)
                report = benchmark_pysecdec_generated_kernels(
                    paths.integral_dir,
                    samples_per_sector=int(request.benchmark_samples_per_sector),
                    sectors=list(request.sectors) if request.sectors is not None else None,
                    real_parameters=list(get_dot_bundle(request).kinematics.parameter_values),
                    repeats=1,
                    seed=int(request.seed),
                )
                if request.json:
                    print(output_pysecdec_kernel_benchmark_json(report))
                else:
                    print_pysecdec_kernel_benchmark_report(
                        report,
                        show_all=bool(request.show_stats),
                    )
            else:
                if not request.json and not request.quiet_summary:
                    print_generation_report(request, summary)
                run_sector_runtime_benchmark(request, topology, sectors, summary)
        except Exception as exc:
            print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
            return 1
        return 0

    if request.command == "test":
        try:
            from boundary_test import run_endpoint_test_mode

            if not request.json and not request.quiet_summary:
                print_generation_report(request, summary)
            report = run_endpoint_test_mode(request, topology, sectors, summary)
        except Exception as exc:
            print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
            return 1
        return 0 if report.get("status") == "ok" else 1

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
        print_preintegration_summary(
            request,
            topology,
            sectors,
            benchmark_available=target is not None,
            data=summary,
        )

    try:
        if request.integral == "dot" and request.command != "integrate":
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
                        "orders": pysecdec_result.orders if pysecdec_result else [],
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
    displayed_count = _prepared_output_count(request, topology)
    if displayed_count != topology.coefficient_count:
        raw_coeffs = _trim_sequence(raw_coeffs, displayed_count)
        raw_errors = _trim_sequence(raw_errors, displayed_count)
        target = _trim_target(target, displayed_count)
        _trim_summary_laurent(summary, topology, displayed_count)
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
    if displayed_count != topology.coefficient_count:
        sector_rows = _trim_sector_rows(sector_rows, displayed_count)
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
    if integration.diagnostics:
        output["integration_diagnostics"] = integration.diagnostics
    output["request"] = request_metadata(request)
    output["environment"] = environment_metadata()
    output["created_utc"] = datetime.now(timezone.utc).isoformat()
    output["command"] = sys.argv
    output["result_path"] = str(result_output_path(request))
    if request.command == "integrate":
        output["input_metadata"] = {
            "prepared_bundle": str(Path(request.output or "").expanduser().resolve()),
            "manifest": prepared_manifest or {},
        }
    elif request.integral == "dot":
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


def main() -> int:
    """Run the CLI with signal cleanup for generation subprocess trees."""
    with InterruptCleanupGuard():
        try:
            return _main_impl()
        except KeyboardInterrupt:
            print(f"{Fore.YELLOW}interrupted by user{Style.RESET_ALL}", file=sys.stderr)
            return 130


if __name__ == "__main__":
    raise SystemExit(main())
