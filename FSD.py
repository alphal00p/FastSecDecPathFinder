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
import os
from pathlib import Path
import sys

from colorama import Fore, Style, init as colorama_init

from benchmark import check_oneloop_bridge, compute_benchmark
from definitions import IntegralRequest
from formatting import (
    apply_global_convention,
    dot_placeholder_summary_data,
    make_output,
    output_json,
    print_dot_placeholder_summary,
    print_preintegration_summary,
    print_result_table,
    summary_data,
)
from dot_topology import GammaLoopDotTopologyBuilder
from integrand import build_topology
from integrator import integrate
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

    if request.integral == "dot":
        if request.dot_file is None:
            raise ValueError("DOT-file topology mode requires --dot-file")
        dot_path = Path(request.dot_file).expanduser()
        if not dot_path.is_file():
            raise ValueError(f"DOT-file topology does not exist: {dot_path}")
        if dot_path.suffix.lower() != ".dot":
            raise ValueError(f"DOT-file topology input must use a .dot suffix: {dot_path}")
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


def parse_args() -> argparse.Namespace:
    """Build the CLI parser and return parsed command-line options."""
    parser = argparse.ArgumentParser(
        description="FSD modular black-box sector-decomposition prototype."
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
        help=(
            "Path to a GammaLoop-convention DOT file describing the integral. "
            "This selects the DOT-backed topology path, which is scaffolded but "
            "not implemented yet."
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
        "--jit-compile-evaluators",
        action="store_true",
        help=(
            "Enable Symbolica jit_compile=True for generated evaluators. "
            "Disabled by default because current Symbolica batch JIT can mis-evaluate simple row-wise expressions."
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
        choices=["raw", "feynman"],
        default="raw",
        help=(
            "Displayed scalar-integral normalization. 'raw' uses OneLOopBridge raw "
            "coefficients; 'feynman' multiplies by TO_FEYNMAN = -1/(16*pi^2)."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of tables.")
    return parser.parse_args()


def build_request(args: argparse.Namespace) -> IntegralRequest:
    """Convert argparse output into the immutable request object used below."""
    mode = resolve_mode(args.m, args.mode)
    dot_file = str(Path(args.dot_file).expanduser()) if args.dot_file is not None else None
    return IntegralRequest(
        integral="dot" if dot_file is not None else args.integral,
        dot_file=dot_file,
        mode=mode,
        s=args.s,
        s12=args.s12,
        s23=args.s23,
        m=args.m,
        gamma_scheme=args.gamma_scheme,
        prefactor_convention=args.prefactor_convention,
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
        show_stats=args.show_stats,
        no_progress=args.no_progress,
        quiet_summary=args.quiet_summary,
        json=args.json,
        mu=args.mu,
        onshell_threshold=args.onshell_threshold,
    )


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
    request = build_request(args)

    try:
        validate_request(request)
        if request.integral == "dot":
            # The DOT path is intentionally wired through the same public
            # topology/sector builders as implemented examples.  Those builders
            # currently raise NotImplementedError at the future GammaLoop parser
            # and sector-generation hooks.
            dot_printout = GammaLoopDotTopologyBuilder.from_request(request).printout_placeholder()
            if request.json:
                print(
                    output_json(
                        {
                            "status": "not_implemented",
                            "dot_placeholder": dot_placeholder_summary_data(dot_printout),
                        }
                    )
                )
            elif not request.quiet_summary:
                print_dot_placeholder_summary(dot_printout)
            sys.stdout.flush()
            build_topology(request)
            generate_sectors(request)
            raise NotImplementedError("DOT-file topology path unexpectedly returned concrete objects")
        check_oneloop_bridge()
    except Exception as exc:
        print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
        return 2

    topology = build_topology(request)
    sectors = generate_sectors(request)
    summary = summary_data(request, topology, sectors, benchmark_available=True)

    if not request.json and not request.quiet_summary:
        print_preintegration_summary(request, topology, sectors, benchmark_available=True)

    try:
        benchmark = compute_benchmark_quietly(request)
        integration = integrate(request, topology, sectors, benchmark)
    except Exception as exc:
        print(f"{Fore.RED}error:{Style.RESET_ALL} {exc}", file=sys.stderr)
        return 1

    raw_coeffs, raw_errors = apply_global_convention(
        integration.raw_sector_coeffs,
        integration.raw_sector_errors,
        request,
    )
    output = make_output(
        request=request,
        raw_coeffs=raw_coeffs,
        raw_errors=raw_errors,
        benchmark=benchmark,
        samples=integration.samples,
        elapsed_seconds=integration.elapsed_seconds,
        avg_eval_us_per_sample_per_worker=integration.avg_eval_us_per_sample_per_worker,
        eval_seconds=integration.eval_seconds,
        python_seconds=integration.python_seconds,
        havana_seconds=integration.havana_seconds,
        python_overhead_fraction=integration.python_overhead_fraction,
        summary=summary,
    )

    if request.json:
        print(output_json(output))
    else:
        print_result_table(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
