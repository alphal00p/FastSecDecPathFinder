"""Sector-level f64 runtime benchmark for prepared FSD integrands."""

from __future__ import annotations

from dataclasses import dataclass, is_dataclass, replace
from datetime import datetime, timezone
import json
import statistics
import time
from typing import Any
from types import SimpleNamespace

from colorama import Fore, Style
import numpy as np
from prettytable import PrettyTable

from definitions import IntegralRequest
from integrator import _make_sector_processor

try:
    import progressbar
except ImportError:  # pragma: no cover - requirements.txt includes progressbar2.
    progressbar = None


@dataclass(frozen=True)
class SectorRuntimeRecord:
    """Measured ordinary double-precision runtime for one sector."""

    sector_id: int
    sector_name: str
    samples: int
    wall_seconds: float
    eval_seconds: float
    python_seconds: float
    ordinary_samples: int
    stability_samples: int
    high_precision_samples: int

    @property
    def wall_us_per_sample(self) -> float:
        """Return elapsed wall time per sample in microseconds."""
        return 1.0e6 * self.wall_seconds / max(int(self.samples), 1)

    @property
    def eval_percent(self) -> float:
        """Return profiled evaluator fraction in percent."""
        total = self.eval_seconds + self.python_seconds
        return 100.0 * self.eval_seconds / total if total > 0.0 else 0.0

    @property
    def python_percent(self) -> float:
        """Return profiled Python fraction in percent."""
        total = self.eval_seconds + self.python_seconds
        return 100.0 * self.python_seconds / total if total > 0.0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record for JSON output."""
        return {
            "sector_id": self.sector_id,
            "sector_name": self.sector_name,
            "samples": self.samples,
            "wall_seconds": self.wall_seconds,
            "wall_us_per_sample": self.wall_us_per_sample,
            "eval_seconds": self.eval_seconds,
            "python_seconds": self.python_seconds,
            "eval_percent": self.eval_percent,
            "python_percent": self.python_percent,
            "ordinary_samples": self.ordinary_samples,
            "stability_samples": self.stability_samples,
            "high_precision_samples": self.high_precision_samples,
        }


def _color(text: str, color: str) -> str:
    return f"{color}{text}{Style.RESET_ALL}"


def _sample_points(seed: int, sector_id: int, samples: int, dimension: int) -> np.ndarray:
    """Return deterministic interior points for ordinary f64 timing probes."""
    rng = np.random.default_rng(int(seed) + 104729 * (int(sector_id) + 1))
    # Stay away from endpoint-rescue thresholds so this command measures the
    # ordinary f64 path, not arbitrary-precision stabilization.
    return 0.125 + 0.75 * rng.random((int(samples), int(dimension)))


def _selected_sector_ids(request: IntegralRequest, sectors: list[Any]) -> list[int]:
    if request.sectors is None:
        return list(range(len(sectors)))
    return [int(sector_id) for sector_id in request.sectors]


def _f64_request(request: IntegralRequest) -> IntegralRequest:
    """Return a request-like object with precision-rescue thresholds disabled."""
    changes = {
        "stability_threshold": 0.0,
        "high_precision_stability_threshold": 0.0,
    }
    if is_dataclass(request):
        return replace(request, **changes)
    data = dict(vars(request))
    data.update(changes)
    return SimpleNamespace(**data)  # type: ignore[return-value]


def _fixed_text(value: object, width: int) -> str:
    """Return a fixed-width single-line field for progress output."""
    text = " ".join(str(value).split())
    if len(text) > width:
        text = text[: max(width - 1, 0)] + "…"
    return text.ljust(width)


def _runtime_stats(records: list[SectorRuntimeRecord]) -> dict[str, Any]:
    """Return current min/max/average/median runtime statistics."""
    values = [record.wall_us_per_sample for record in records]
    min_record = min(records, key=lambda item: item.wall_us_per_sample) if records else None
    max_record = max(records, key=lambda item: item.wall_us_per_sample) if records else None
    return {
        "min": min(values) if values else None,
        "min_sector": (
            {"id": min_record.sector_id, "name": min_record.sector_name}
            if min_record is not None
            else None
        ),
        "max": max(values) if values else None,
        "max_sector": (
            {"id": max_record.sector_id, "name": max_record.sector_name}
            if max_record is not None
            else None
        ),
        "average": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def _make_progress_bar(request: IntegralRequest, total: int) -> Any | None:
    """Create the benchmark progressbar unless disabled."""
    if request.json or request.no_progress or progressbar is None:
        return None
    live_widget = progressbar.FormatCustomText(
        (
            f"{_color('sector', Fore.CYAN)}:%(sector)s "
            f"{_color('min', Fore.GREEN)}:%(min)s "
            f"{_color('max', Fore.YELLOW)}:%(max)s "
            f"{_color('med', Fore.MAGENTA)}:%(median)s "
            f"{_color('avg', Fore.BLUE)}:%(average)s"
        ),
        {
            "sector": _fixed_text("starting", 24),
            "min": _fixed_text("n/a", 10),
            "max": _fixed_text("n/a", 10),
            "median": _fixed_text("n/a", 10),
            "average": _fixed_text("n/a", 10),
        },
    )
    widgets = [
        _color("benchmark sectors ", Fore.CYAN),
        progressbar.Percentage(),
        " ",
        progressbar.Bar(),
        " ",
        live_widget,
        " ",
    ]
    bar = progressbar.ProgressBar(max_value=max(int(total), 1), widgets=widgets)
    bar.fsd_live_widget = live_widget
    return bar


def _update_progress(
    bar: Any | None,
    done: int,
    total: int,
    current_sector: str,
    records: list[SectorRuntimeRecord],
) -> None:
    """Update benchmark progress with live aggregate timing stats."""
    if bar is None:
        return
    stats = _runtime_stats(records)
    bar.fsd_live_widget.update_mapping(
        sector=_fixed_text(current_sector, 24),
        min=_fixed_text(_fmt_us(stats["min"]), 10),
        max=_fixed_text(_fmt_us(stats["max"]), 10),
        median=_fixed_text(_fmt_us(stats["median"]), 10),
        average=_fixed_text(_fmt_us(stats["average"]), 10),
    )
    bar.update(min(int(done), max(int(total), 1)), force=True)


def _build_report(
    *,
    status: str,
    samples: int,
    records: list[SectorRuntimeRecord],
    total_seconds: float,
    requested_sector_count: int,
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the JSON-friendly benchmark report."""
    return {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "interrupted": status == "interrupted",
        "samples_per_sector": samples,
        "requested_sector_count": int(requested_sector_count),
        "sector_count": len(records),
        "completed_sector_count": len(records),
        "total_seconds": total_seconds,
        "summary": summary or {},
        "runtime_us_per_sample": _runtime_stats(records),
        "sectors": [record.to_dict() for record in sorted(records, key=lambda item: item.sector_id)],
    }


def run_sector_runtime_benchmark(
    request: IntegralRequest,
    topology: Any,
    sectors: list[Any],
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure ordinary f64 sector evaluation runtimes and print a compact report."""
    selected_ids = _selected_sector_ids(request, sectors)
    samples = int(request.benchmark_samples_per_sector)
    # Force ordinary f64 evaluation for this diagnostic.  The sample points are
    # interior already, but zero thresholds make the intent explicit.
    benchmark_request = _f64_request(request)
    processor = _make_sector_processor(topology, benchmark_request)
    records: list[SectorRuntimeRecord] = []
    start = time.perf_counter()
    interrupted = False
    progress = _make_progress_bar(request, len(selected_ids))
    try:
        _update_progress(progress, 0, len(selected_ids), "starting", records)
        for done, sector_id in enumerate(selected_ids, start=1):
            sector = sectors[sector_id]
            current_sector = f"{sector_id}:{sector.name}"
            _update_progress(progress, done - 1, len(selected_ids), current_sector, records)
            coords = _sample_points(request.seed, sector_id, samples, sector.integration_dim)
            wall_start = time.perf_counter()
            _coeffs, _training, timing = processor.evaluate_batch(sector, coords)
            wall_seconds = time.perf_counter() - wall_start
            records.append(
                SectorRuntimeRecord(
                    sector_id=sector_id,
                    sector_name=sector.name,
                    samples=samples,
                    wall_seconds=wall_seconds,
                    eval_seconds=timing.eval_seconds,
                    python_seconds=timing.python_seconds,
                    ordinary_samples=timing.ordinary_precision_samples,
                    stability_samples=timing.stability_precision_samples,
                    high_precision_samples=timing.high_precision_samples,
                )
            )
            _update_progress(progress, done, len(selected_ids), current_sector, records)
    except KeyboardInterrupt:
        interrupted = True
        _update_progress(progress, len(records), len(selected_ids), "interrupted", records)
    finally:
        if progress is not None:
            progress.finish()
    report = _build_report(
        status="interrupted" if interrupted else "ok",
        samples=samples,
        records=records,
        total_seconds=time.perf_counter() - start,
        requested_sector_count=len(selected_ids),
        summary=summary,
    )
    if request.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_runtime_report(report, records, show_all=bool(request.show_stats))
    return report


def _fmt_us(value: float | None) -> str:
    if value is None:
        return "n/a"
    if abs(value) >= 1000.0:
        return f"{value / 1000.0:.3g} ms"
    return f"{value:.3g} μs"


def _print_runtime_report(
    report: dict[str, Any],
    records: list[SectorRuntimeRecord],
    *,
    show_all: bool = False,
) -> None:
    """Print colored benchmark summary tables."""
    stats = report["runtime_us_per_sample"]
    print(_color("\nFSD f64 sector runtime benchmark", Fore.CYAN))
    if report.get("interrupted"):
        print(
            f"{Fore.YELLOW}benchmark interrupted:{Style.RESET_ALL} "
            f"showing {report['completed_sector_count']}/"
            f"{report['requested_sector_count']} measured sector(s)"
        )
    summary_table = PrettyTable()
    summary_table.field_names = [_color("statistic", Fore.CYAN), _color("value", Fore.CYAN)]
    summary_table.add_row(["status", report["status"]])
    summary_table.add_row([
        "sectors measured",
        f"{report['completed_sector_count']}/{report['requested_sector_count']}",
    ])
    summary_table.add_row(["samples / sector", report["samples_per_sector"]])
    summary_table.add_row(["total wall time", f"{float(report['total_seconds']):.3g}s"])
    summary_table.add_row(["min μs/sample", _fmt_us(stats["min"])])
    summary_table.add_row(["max μs/sample", _fmt_us(stats["max"])])
    summary_table.add_row(["average μs/sample", _fmt_us(stats["average"])])
    summary_table.add_row(["median μs/sample", _fmt_us(stats["median"])])
    print(summary_table)

    extrema = [
        ("min", min(records, key=lambda item: item.wall_us_per_sample, default=None)),
        ("max", max(records, key=lambda item: item.wall_us_per_sample, default=None)),
    ]
    extrema_table = PrettyTable()
    extrema_table.field_names = [
        _color("kind", Fore.CYAN),
        _color("sector", Fore.CYAN),
        _color("μs/sample", Fore.CYAN),
        _color("eval%", Fore.CYAN),
        _color("python%", Fore.CYAN),
        _color("prec O/S/H", Fore.CYAN),
    ]
    for kind, record in extrema:
        if record is None:
            continue
        color = Fore.GREEN if kind == "min" else Fore.YELLOW
        extrema_table.add_row(
            [
                _color(kind, color),
                f"{record.sector_id}:{record.sector_name}",
                _fmt_us(record.wall_us_per_sample),
                f"{record.eval_percent:.2f}",
                f"{record.python_percent:.2f}",
                (
                    f"{record.ordinary_samples}/"
                    f"{record.stability_samples}/"
                    f"{record.high_precision_samples}"
                ),
            ]
        )
    print(extrema_table)

    if show_all:
        table = PrettyTable()
        table.field_names = [
            _color("sector", Fore.CYAN),
            _color("μs/sample", Fore.CYAN),
            _color("wall", Fore.CYAN),
            _color("eval%", Fore.CYAN),
            _color("python%", Fore.CYAN),
            _color("prec O/S/H", Fore.CYAN),
        ]
        for record in sorted(records, key=lambda item: item.sector_id):
            table.add_row(
                [
                    f"{record.sector_id}:{record.sector_name}",
                    _fmt_us(record.wall_us_per_sample),
                    f"{record.wall_seconds:.3g}s",
                    f"{record.eval_percent:.2f}",
                    f"{record.python_percent:.2f}",
                    (
                        f"{record.ordinary_samples}/"
                        f"{record.stability_samples}/"
                        f"{record.high_precision_samples}"
                    ),
                ]
            )
        print(table)
