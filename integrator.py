"""Havana sampling driver and manual Laurent-coefficient accumulation.

Symbolica's Havana object is used as an adaptive sampler and grid owner.  The
code evaluates all Laurent coefficients itself, fills running statistics
manually, and trains Havana with a scalar finite-part importance function.
"""

from __future__ import annotations

import copy
import math
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

import numpy as np
from colorama import Fore, Style
from symbolica import NumericalIntegrator

from definitions import BenchmarkResult, HotPathTiming, IntegralRequest, IntegrationResult
from formatting import (
    apply_global_convention,
    format_complex,
    format_complex_error,
    pull_value,
    selected_prefactor_values,
    summed_relative_error_percent,
)
from integrand import SectorProcessor, TopologyDefinition, build_topology
from sectors_generator import SectorDefinition, generate_sectors

try:
    import progressbar
except ImportError:  # pragma: no cover - requirements.txt includes progressbar2.
    progressbar = None


TARGET_PROGRESS_UNITS = 10_000


@dataclass
class RunningStats:
    """Streaming complex mean/error estimator for one Laurent coefficient."""

    count: int = 0
    sum_re: float = 0.0
    sum_im: float = 0.0
    sumsq_re: float = 0.0
    sumsq_im: float = 0.0

    def add(self, value: complex) -> None:
        """Add one weighted sample."""
        z = complex(value)
        self.count += 1
        self.sum_re += z.real
        self.sum_im += z.imag
        self.sumsq_re += z.real * z.real
        self.sumsq_im += z.imag * z.imag

    def add_many(self, values: np.ndarray) -> None:
        """Add a vector of weighted samples using NumPy reductions."""
        array = np.asarray(values, dtype=np.complex128)
        if array.size == 0:
            return
        self.count += int(array.size)
        self.sum_re += float(np.sum(array.real))
        self.sum_im += float(np.sum(array.imag))
        self.sumsq_re += float(np.sum(array.real * array.real))
        self.sumsq_im += float(np.sum(array.imag * array.imag))

    @property
    def mean(self) -> complex:
        """Return the current Monte Carlo mean."""
        if self.count == 0:
            return 0.0 + 0.0j
        return complex(self.sum_re / self.count, self.sum_im / self.count)

    @property
    def error(self) -> complex:
        """Return component-wise standard errors of the mean."""
        if self.count < 2:
            return complex(float("inf"), float("inf"))
        mean_re = self.sum_re / self.count
        mean_im = self.sum_im / self.count
        var_re = max(self.sumsq_re / self.count - mean_re * mean_re, 0.0)
        var_im = max(self.sumsq_im / self.count - mean_im * mean_im, 0.0)
        return complex(math.sqrt(var_re / self.count), math.sqrt(var_im / self.count))


@dataclass(frozen=True)
class EvaluationBatch:
    """Batch of Havana samples sent through the vectorized sector processor."""

    indices: np.ndarray
    sector_indices: np.ndarray
    coords: np.ndarray
    weights: np.ndarray


_WORKER_SECTORS: list[SectorDefinition] | None = None
_WORKER_PROCESSOR: SectorProcessor | None = None


def _init_worker(request: IntegralRequest) -> None:
    """Build per-process topology/sector state for worker reuse."""
    global _WORKER_SECTORS, _WORKER_PROCESSOR
    topology = build_topology(request)
    _WORKER_SECTORS = generate_sectors(request)
    _WORKER_PROCESSOR = SectorProcessor(topology)


def _evaluate_records_worker(
    batch: EvaluationBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, HotPathTiming]:
    """Evaluate one batch in a worker process."""
    if _WORKER_PROCESSOR is None or _WORKER_SECTORS is None:
        raise RuntimeError("worker processor not initialized")
    return _evaluate_records(_WORKER_PROCESSOR, _WORKER_SECTORS, batch)


def _evaluate_records(
    processor: SectorProcessor,
    sectors: list[SectorDefinition],
    batch: EvaluationBatch,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, HotPathTiming]:
    """Evaluate a batch and return weighted coefficients and training values."""
    hot_start = time.perf_counter()
    timing = HotPathTiming()
    if batch.indices.size == 0:
        return (
            np.empty(0, dtype=int),
            np.empty((0, 3), dtype=np.complex128),
            np.empty(0, dtype=float),
            timing,
        )

    indices = batch.indices
    sector_indices = batch.sector_indices
    coords = batch.coords
    weights = batch.weights
    weighted = np.zeros((indices.size, 3), dtype=np.complex128)
    training = np.zeros(indices.size, dtype=float)

    for sector_index, sector in enumerate(sectors):
        mask = sector_indices == sector_index
        if not np.any(mask):
            continue
        coeffs, train, sector_timing = processor.evaluate_batch(sector, coords[mask])
        timing.absorb(sector_timing)
        weighted[mask, :] = coeffs * weights[mask, np.newaxis]
        training[mask] = train

    hot_elapsed = time.perf_counter() - hot_start
    timing.add_python(hot_elapsed - timing.total_seconds)
    return indices, weighted, training, timing


def split_evaluation_batches(
    indices: np.ndarray,
    sector_indices: np.ndarray,
    coords: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    default_batches: int,
) -> list[EvaluationBatch]:
    """Split one Havana iteration into vectorized processor tasks."""
    n_samples = int(indices.size)
    if n_samples == 0:
        return []
    if batch_size <= 0:
        n_batches = max(int(default_batches), 1)
        step = max(math.ceil(n_samples / n_batches), 1)
    else:
        step = batch_size
    return [
        EvaluationBatch(
            indices=indices[start : start + step],
            sector_indices=sector_indices[start : start + step],
            coords=coords[start : start + step, :],
            weights=weights[start : start + step],
        )
        for start in range(0, n_samples, step)
    ]


def format_progress_scalar(value: float | None) -> str:
    """Compact scalar formatter for live progress fields."""
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-2 or abs(value) >= 1.0e3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def format_progress_percent(value: float | None) -> str:
    """Format a live percent with scientific notation below 0.001%."""
    if value is None:
        return "n/a"
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf"
    if value == 0.0:
        return "0.00"
    abs_value = abs(value)
    if abs_value < 1.0e-3:
        return f"{value:.2e}"
    if abs_value < 0.01:
        return f"{value:.5f}"
    if abs_value < 0.1:
        return f"{value:.4f}"
    if abs_value < 1.0:
        return f"{value:.3f}"
    if abs_value < 10.0:
        return f"{value:.2f}"
    if abs_value < 100.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def format_sample_count(value: int) -> str:
    """Format sample counts with 3-significant-digit K/M/B units."""
    count = int(value)
    if abs(count) < 1000:
        return str(count)
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if abs(count) >= scale:
            scaled = count / scale
            if abs(scaled) >= 100.0:
                return f"{scaled:.0f}{suffix}"
            if abs(scaled) >= 10.0:
                return f"{scaled:.1f}{suffix}"
            return f"{scaled:.2f}{suffix}"
    return str(count)


def max_sample_budget(request: IntegralRequest) -> int | None:
    """Return the finite maximum sample budget, or None when unbounded."""
    if request.max_iter < 0:
        return None
    return request.max_iter * request.samples_per_iter


def format_max_iter(request: IntegralRequest) -> str:
    """Format the iteration budget for the progress bar."""
    return "∞" if request.max_iter < 0 else str(request.max_iter)


def format_sample_target(request: IntegralRequest) -> str:
    """Format the sample budget for the progress bar."""
    budget = max_sample_budget(request)
    return "∞" if budget is None else format_sample_count(budget)


def format_duration(seconds: float) -> str:
    """Format a non-negative duration as HH:MM:SS."""
    total = max(int(seconds), 0)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_eta(seconds: float | None) -> str:
    """Format an ETA, preserving non-finite sentinel values."""
    if seconds is None:
        return "n/a"
    if math.isnan(seconds):
        return "nan"
    if math.isinf(seconds):
        return "inf"
    return format_duration(seconds)


def progress_color(text: str, color: str) -> str:
    """Apply ANSI color to a progress-bar label."""
    return f"{color}{text}{Style.RESET_ALL}"


def profile_text(timing: HotPathTiming) -> str:
    """Render the live profile tuple ``(python | evaluator | havana)``."""
    return (
        f"({100.0 * timing.python_overhead_fraction:.2f}% | "
        f"{100.0 * timing.evaluator_fraction:.2f}% | "
        f"{100.0 * timing.havana_fraction:.2f}%)"
    )


def avg_eval_us_per_sample_per_worker(timing: HotPathTiming, sample_count: int) -> float:
    """Normalize worker-summed evaluator time by accepted samples."""
    if sample_count <= 0:
        return 0.0
    # EvalT is already worker-summed: each worker reports its evaluator wall time
    # and the master absorbs all worker timings. Dividing by total samples gives
    # the average evaluator time for one sample on one worker.
    return timing.eval_seconds * 1.0e6 / sample_count


def make_progress_bar(request: IntegralRequest) -> Any | None:
    """Create the colored progress bar unless the output mode disables it."""
    if request.json or request.no_progress or progressbar is None:
        return None
    sample_budget = max_sample_budget(request)
    target_mode = request.target_rel_accuracy is not None
    unbounded_mode = sample_budget is None and not target_mode
    max_value = (
        progressbar.UnknownLength
        if unbounded_mode
        else TARGET_PROGRESS_UNITS
        if target_mode
        else sample_budget
    )
    live_widget = progressbar.FormatCustomText(
        (
            f"{progress_color('it', Fore.CYAN)}:%(iteration)s/%(max_iter)s "
            f"{progress_color('smpl', Fore.CYAN)}:%(samples)s/%(target_samples)s "
            f"{progress_color('err%%', Fore.GREEN)}:%(relerr)s "
            f"{progress_color('pull', Fore.YELLOW)}: %(pull)s "
            f"{progress_color('t', Fore.BLUE)}:%(elapsed)s "
            f"{progress_color('eta', Fore.BLUE)}:%(eta)s "
            f"{progress_color('eval μs/smpl/wkr', Fore.MAGENTA)}:%(avg_us)s "
            f"{progress_color('prof py|eval|hav', Fore.CYAN)}:%(profile)s"
        ),
        {
            "iteration": "0",
            "max_iter": format_max_iter(request),
            "samples": "0",
            "target_samples": format_sample_target(request),
            "relerr": "n/a",
            "pull": "n/a",
            "elapsed": "00:00:00",
            "eta": "n/a",
            "avg_us": "n/a",
            "profile": "n/a",
        },
    )
    widgets = [
        progress_color(f"integrating {request.integral} ", Fore.CYAN),
        progressbar.AnimatedMarker() if unbounded_mode else progressbar.Percentage(),
        " ",
        "" if unbounded_mode else progressbar.Bar(),
        "" if unbounded_mode else " ",
        live_widget,
        " ",
    ]
    bar = progressbar.ProgressBar(max_value=max_value, widgets=widgets)
    bar.fsd_live_widget = live_widget
    return bar


def live_progress_metrics(
    request: IntegralRequest,
    stats: list[RunningStats],
    benchmark: BenchmarkResult,
) -> tuple[float, float]:
    """Compute live summed relative error and max pull for display."""
    sector_coeffs = [stat.mean for stat in stats]
    sector_errors = [stat.error for stat in stats]
    raw_coeffs, raw_errors = apply_global_convention(sector_coeffs, sector_errors, request)
    display_coeffs, display_errors, display_bench, _ = selected_prefactor_values(
        request, raw_coeffs, raw_errors, benchmark
    )
    relerr = summed_relative_error_percent(display_coeffs, display_errors)
    pulls = [
        pull_value(coeff - ref, err)
        for coeff, err, ref in zip(display_coeffs, display_errors, display_bench)
    ]
    numeric_pulls = [pull for pull in pulls if pull is not None]
    return relerr, max(numeric_pulls) if numeric_pulls else 0.0


def target_progress_fraction(relerr_percent: float, target_percent: float | None) -> float | None:
    """Estimate target completion fraction using error proportional to 1/sqrt(N)."""
    if target_percent is None:
        return None
    if target_percent <= 0.0:
        return None
    if relerr_percent == 0.0:
        return 1.0
    if not math.isfinite(relerr_percent):
        return 0.0
    return min((target_percent / relerr_percent) ** 2, 1.0)


def sample_budget_progress_fraction(request: IntegralRequest, samples_done: int) -> float | None:
    """Return the fraction of a finite sample budget already consumed."""
    budget = max_sample_budget(request)
    if budget is None or budget <= 0:
        return None
    return min(max(samples_done / budget, 0.0), 1.0)


def estimate_eta_seconds(
    request: IntegralRequest,
    samples_done: int,
    elapsed_seconds: float,
    relerr_percent: float,
) -> float | None:
    """Estimate ETA from target accuracy and finite sample budget candidates."""
    if samples_done <= 0 or elapsed_seconds <= 0.0:
        return None
    sample_rate = samples_done / elapsed_seconds
    if sample_rate <= 0.0:
        return None
    target = request.target_rel_accuracy
    budget = max_sample_budget(request)
    budget_eta = None
    if budget is not None:
        remaining_budget_samples = max(budget - samples_done, 0)
        budget_eta = remaining_budget_samples / sample_rate
    if target is None:
        return budget_eta
    if target <= 0.0:
        return budget_eta
    target_eta = None
    if relerr_percent == 0.0:
        target_eta = 0.0
    elif math.isfinite(relerr_percent):
        estimated_target_samples = samples_done * (relerr_percent / target) ** 2
        remaining = max(estimated_target_samples - samples_done, 0.0)
        target_eta = remaining / sample_rate

    candidates = [eta for eta in (target_eta, budget_eta) if eta is not None]
    return min(candidates) if candidates else None


def format_relerr_with_target(request: IntegralRequest, relerr_percent: float) -> str:
    """Format live relative error and append a target if configured."""
    relerr_text = format_progress_percent(relerr_percent)
    if request.target_rel_accuracy is None:
        return relerr_text
    target_text = format_progress_percent(request.target_rel_accuracy)
    return relerr_text + " " + maybe_blue_slash_target(target_text)


def maybe_blue_slash_target(target_text: str) -> str:
    """Color the target suffix shown next to ``err%``."""
    return progress_color(f"/ {target_text}", Fore.BLUE)


def update_progress_bar(
    bar: Any | None,
    request: IntegralRequest,
    stats: list[RunningStats],
    benchmark: BenchmarkResult,
    iteration: int,
    elapsed_seconds: float,
    avg_eval_us_per_sample_per_worker: float,
    timing: HotPathTiming,
) -> None:
    """Refresh progress widgets from the current accumulators."""
    if bar is None:
        return
    relerr, pull = live_progress_metrics(request, stats, benchmark)
    eta = estimate_eta_seconds(request, stats[0].count, elapsed_seconds, relerr)
    live_widget = getattr(bar, "fsd_live_widget", None)
    if live_widget is not None:
        live_widget.update_mapping(
            iteration=str(iteration),
            samples=format_sample_count(stats[0].count),
            relerr=format_relerr_with_target(request, relerr),
            pull=f"{pull:.2f}σ",
            elapsed=format_duration(elapsed_seconds),
            eta=format_eta(eta),
            avg_us=format_progress_scalar(avg_eval_us_per_sample_per_worker),
            profile=profile_text(timing),
        )
    if request.target_rel_accuracy is None:
        if max_sample_budget(request) is None:
            progress_value = stats[0].count
        else:
            progress_value = min(stats[0].count, max_sample_budget(request) or stats[0].count)
    else:
        target_fraction = target_progress_fraction(relerr, request.target_rel_accuracy)
        sample_fraction = sample_budget_progress_fraction(request, stats[0].count)
        finite_fractions = [
            fraction
            for fraction in (target_fraction, sample_fraction)
            if fraction is not None and math.isfinite(fraction)
        ]
        fraction = max(finite_fractions) if finite_fractions else 0.0
        progress_value = int(TARGET_PROGRESS_UNITS * (fraction if fraction is not None else 0.0))
    bar.update(progress_value, force=True)


def update_progress_bar_timed(
    bar: Any | None,
    request: IntegralRequest,
    stats: list[RunningStats],
    benchmark: BenchmarkResult,
    iteration: int,
    elapsed_seconds: float,
    avg_eval_us_per_sample_per_worker: float,
    timing: HotPathTiming,
) -> None:
    """Refresh the progress bar and charge rendering work to PythonT."""
    if bar is None:
        return
    progress_start = time.perf_counter()
    update_progress_bar(
        bar,
        request,
        stats,
        benchmark,
        iteration,
        elapsed_seconds,
        avg_eval_us_per_sample_per_worker,
        timing,
    )
    timing.add_python(time.perf_counter() - progress_start)


def target_accuracy_reached(
    request: IntegralRequest,
    stats: list[RunningStats],
    benchmark: BenchmarkResult,
    iteration: int,
) -> bool:
    """Return whether the requested relative target has been reached."""
    if request.target_rel_accuracy is None or iteration < request.min_iter:
        return False
    relerr, _ = live_progress_metrics(request, stats, benchmark)
    return math.isfinite(relerr) and relerr <= request.target_rel_accuracy


def integrate(
    request: IntegralRequest,
    topology: TopologyDefinition,
    sectors: list[SectorDefinition],
    benchmark: BenchmarkResult,
) -> IntegrationResult:
    """Run the adaptive Monte Carlo integration and return raw coefficients."""
    if not sectors:
        raise ValueError("no sectors generated")
    continuous_dim = sectors[0].integration_dim
    if any(sector.integration_dim != continuous_dim for sector in sectors):
        raise ValueError("all sectors must have the same integration dimension for the current Havana driver")

    # One discrete dimension chooses the sector, and each sector owns an
    # adaptive continuous Havana grid of the same dimension.  Lower-support
    # subtraction terms are localized into this same dimension by the processor.
    grid = NumericalIntegrator.discrete(
        [NumericalIntegrator.continuous(continuous_dim, n_bins=request.bins) for _ in sectors]
    )
    rng = grid.rng(request.seed, 0)
    stats = [RunningStats(), RunningStats(), RunningStats()]

    processor: SectorProcessor | None = None
    executor: ProcessPoolExecutor | None = None
    if request.workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=request.workers,
            initializer=_init_worker,
            initargs=(request,),
        )
    else:
        processor = SectorProcessor(topology)

    bar = make_progress_bar(request)
    if bar is not None:
        bar.start()

    start_time = time.perf_counter()
    hot_timing = HotPathTiming()

    try:
        iteration = 0
        while request.max_iter < 0 or iteration < request.max_iter:
            iteration += 1
            havana_start = time.perf_counter()
            # Train a cloned grid during the iteration.  The live grid is only
            # merged/updated after the iteration completes, so mid-iteration
            # progress updates do not perturb the sampler state.
            training_grid = copy.copy(grid)
            samples = grid.sample(request.samples_per_iter, rng)
            hot_timing.add_havana(time.perf_counter() - havana_start)

            master_start = time.perf_counter()
            n_samples = len(samples)
            sample_indices = np.arange(n_samples, dtype=int)
            sector_indices = np.empty(n_samples, dtype=int)
            coords = np.empty((n_samples, continuous_dim), dtype=float)
            weights = np.empty(n_samples, dtype=float)
            for sample_index, sample in enumerate(samples):
                # Havana samples contain one discrete coordinate d[0] and one
                # continuous coordinate vector c.  The first weight is the full
                # MC weight used in the manual Laurent accumulators.
                sector_indices[sample_index] = int(sample.d[0])
                coords[sample_index, :] = sample.c
                weights[sample_index] = float(sample.weights[0])
            batches = split_evaluation_batches(
                sample_indices,
                sector_indices,
                coords,
                weights,
                request.batch_size,
                request.workers,
            )
            hot_timing.add_python(time.perf_counter() - master_start)

            def register_batch(
                indices: np.ndarray,
                w_part: np.ndarray,
                t_part: np.ndarray,
            ) -> bool:
                # This is the point where a batch becomes part of the returned
                # result.  Target-accuracy termination is checked only after
                # these samples have been accumulated.
                aggregate_start = time.perf_counter()
                for coeff_index, stat in enumerate(stats):
                    stat.add_many(w_part[:, coeff_index])
                start_index = int(indices[0])
                stop_index = int(indices[-1]) + 1
                if stop_index - start_index == indices.size:
                    batch_samples = samples[start_index:stop_index]
                else:
                    batch_samples = [samples[int(index)] for index in indices]
                hot_timing.add_python(time.perf_counter() - aggregate_start)

                havana_batch_start = time.perf_counter()
                # The training observable is scalar and finite-part based.
                # It steers the adaptive grid, while the Laurent coefficients
                # themselves are accumulated in RunningStats above.
                training_grid.add_training_samples(batch_samples, t_part)
                hot_timing.add_havana(time.perf_counter() - havana_batch_start)

                elapsed_seconds = time.perf_counter() - start_time
                avg_eval_us = avg_eval_us_per_sample_per_worker(hot_timing, stats[0].count)
                update_progress_bar_timed(
                    bar,
                    request,
                    stats,
                    benchmark,
                    iteration,
                    elapsed_seconds,
                    avg_eval_us,
                    hot_timing,
                )
                return target_accuracy_reached(request, stats, benchmark, iteration)

            stop_requested = False
            if executor is None:
                assert processor is not None
                for batch in batches:
                    indices, w_part, t_part, worker_timing = _evaluate_records(processor, sectors, batch)
                    hot_timing.absorb(worker_timing)
                    if register_batch(indices, w_part, t_part):
                        stop_requested = True
                        break
            else:
                pending: set[Future[tuple[np.ndarray, np.ndarray, np.ndarray, HotPathTiming]]] = set()
                next_batch_index = 0

                def submit_until_full() -> None:
                    nonlocal next_batch_index
                    submit_start = time.perf_counter()
                    # Keep only a worker-sized window in flight.  This allows
                    # target-accuracy termination to stop inside an iteration
                    # without submitting the whole iteration up front.
                    while (
                        next_batch_index < len(batches)
                        and len(pending) < max(request.workers, 1)
                    ):
                        pending.add(executor.submit(_evaluate_records_worker, batches[next_batch_index]))
                        next_batch_index += 1
                    hot_timing.add_python(time.perf_counter() - submit_start)

                submit_until_full()
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        if stop_requested:
                            future.cancel()
                            continue
                        result_start = time.perf_counter()
                        indices, w_part, t_part, worker_timing = future.result()
                        hot_timing.add_python(time.perf_counter() - result_start)
                        hot_timing.absorb(worker_timing)
                        if register_batch(indices, w_part, t_part):
                            stop_requested = True
                            # Futures that have not started can be cancelled.
                            # Running futures may finish in the background, but
                            # their samples are intentionally not accumulated.
                            for pending_future in pending:
                                pending_future.cancel()
                            pending.clear()
                            break
                    if stop_requested:
                        break
                    submit_until_full()

            if stop_requested:
                # A target-accuracy stop returns the already accumulated
                # partial iteration.  The cloned training grid is deliberately
                # not merged because the live grid will not be sampled again.
                break

            havana_update_start = time.perf_counter()
            # Full completed iterations update the live Havana grid from the
            # cloned training grid.  This preserves stable sampling during the
            # iteration while still adapting between iterations.
            grid.merge(training_grid)
            live_avg, live_err, live_chi = grid.update(1.5, 1.5)
            hot_timing.add_havana(time.perf_counter() - havana_update_start)

            finite_err = stats[2].error
            if request.show_stats:
                print(
                    f"iter {iteration:3d}: finite raw sector-sum "
                    f"{format_complex(stats[2].mean)} +- {format_complex_error(finite_err)} "
                    f"(havana train {live_avg:.6g} +- {live_err:.3g}, chi={live_chi:.3g})"
                )

            elapsed_seconds = time.perf_counter() - start_time
            avg_eval_us = avg_eval_us_per_sample_per_worker(hot_timing, stats[0].count)
            update_progress_bar_timed(
                bar,
                request,
                stats,
                benchmark,
                iteration,
                elapsed_seconds,
                avg_eval_us,
                hot_timing,
            )

            if (
                request.target_rel_accuracy is None
                and request.max_iter >= 0
                and iteration >= request.min_iter
                and max(finite_err.real, finite_err.imag) < request.min_error
            ):
                break
            if target_accuracy_reached(request, stats, benchmark, iteration):
                break

        return IntegrationResult(
            raw_sector_coeffs=[stat.mean for stat in stats],
            raw_sector_errors=[stat.error for stat in stats],
            samples=stats[0].count,
            elapsed_seconds=time.perf_counter() - start_time,
            avg_eval_us_per_sample_per_worker=avg_eval_us_per_sample_per_worker(
                hot_timing, stats[0].count
            ),
            eval_seconds=hot_timing.eval_seconds,
            python_seconds=hot_timing.python_seconds,
            havana_seconds=hot_timing.havana_seconds,
            python_overhead_fraction=hot_timing.python_overhead_fraction,
        )
    finally:
        if bar is not None:
            bar.finish(dirty=True)
        if executor is not None:
            executor.shutdown()
