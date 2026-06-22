"""Structured generation timing records for DOT and pySecDec workflows."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import shutil
import time
from typing import Iterator

from colorama import Fore, Style

try:
    import progressbar
except ImportError:  # pragma: no cover - requirements.txt includes progressbar2.
    progressbar = None


UF_TIMING_BUCKET = "Generation U and F polynomial"
SECTOR_TIMING_BUCKET = "Generating sectors"
SYMBOLICA_TIMING_BUCKET = "Generating Symbolica evaluators"


@dataclass
class TimingRecord:
    """One named generation or setup timing measurement."""

    name: str
    seconds: float
    detail: str = ""


@dataclass
class GenerationTimings:
    """Mutable collection of generation timings with logging helpers."""

    records: list[TimingRecord] = field(default_factory=list)

    def add(self, name: str, seconds: float, detail: str = "") -> None:
        """Append one timing record."""
        self.records.append(TimingRecord(name=name, seconds=max(float(seconds), 0.0), detail=detail))

    @contextmanager
    def measure(
        self,
        name: str,
        detail: str = "",
        progress: "GenerationProgress | None" = None,
    ) -> Iterator[None]:
        """Context manager recording elapsed wall time under ``name``."""
        if progress is not None:
            progress.start_stage(name, detail=detail)
        start = time.perf_counter()
        try:
            yield
        finally:
            seconds = time.perf_counter() - start
            self.add(name, seconds, detail)
            if progress is not None:
                progress.finish_stage(name, seconds, detail=detail)

    def log(self, logger: logging.Logger) -> None:
        """Emit all records at INFO level."""
        for record in self.records:
            suffix = f" ({record.detail})" if record.detail else ""
            logger.info("generation timing: %-42s %.6fs%s", record.name, record.seconds, suffix)

    def to_dict(self) -> list[dict[str, object]]:
        """Return JSON-friendly records."""
        return [
            {"name": record.name, "seconds": record.seconds, "detail": record.detail}
            for record in self.records
        ]

    def total(self) -> float:
        """Return the sum of recorded generation/setup timings."""
        return sum(record.seconds for record in self.records)

    def bucket_totals(self) -> dict[str, float]:
        """Return the headline generation buckets used in CLI summaries."""
        buckets = {
            UF_TIMING_BUCKET: 0.0,
            SECTOR_TIMING_BUCKET: 0.0,
            SYMBOLICA_TIMING_BUCKET: 0.0,
        }
        for record in self.records:
            if record.name in {
                "DOT parse",
                "kinematics load/evaluation",
                "pySecDec LoopIntegralFromGraph",
                "pySecDec LoopIntegralFromPropagators",
                "U/F extraction",
                "FSD Symbolica numerator reduction",
                "FSD numerator Polynomial conversion",
            }:
                buckets[UF_TIMING_BUCKET] += record.seconds
            elif record.name in {
                "pySecDec sector decomposition",
                "FSD SectorDefinition conversion",
            }:
                buckets[SECTOR_TIMING_BUCKET] += record.seconds
            elif record.name in {
                "Symbolica scalar evaluator build",
                "Symbolica sector evaluator build",
                "Symbolica dual evaluator build",
                "Symbolica Taylor evaluator build",
                "Symbolica subtraction formula build",
                "Symbolica two-stage sector build",
                "Symbolica explicit sector build",
            }:
                buckets[SYMBOLICA_TIMING_BUCKET] += record.seconds
        return buckets

    def to_summary_dict(self) -> dict[str, object]:
        """Return headline and detailed timing records for JSON output."""
        return {
            "headline": [
                {"name": name, "seconds": seconds}
                for name, seconds in self.bucket_totals().items()
            ],
            "details": self.to_dict(),
            "total": self.total(),
        }


def _duration_text(seconds: float | None) -> str:
    """Return a compact wall-time string."""
    if seconds is None:
        return "n/a"
    total = max(int(seconds), 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _color(text: str, color: str) -> str:
    """Apply ANSI color to a progress label."""
    return f"{color}{text}{Style.RESET_ALL}"


def _clip(text: object, width: int) -> str:
    """Return ``text`` truncated to one terminal-friendly field."""
    value = str(text)
    if width <= 0 or len(value) <= width:
        return value
    if width <= 1:
        return "…"
    return value[: width - 1] + "…"


_STAGE_SHORT_NAMES = {
    "DOT parse": "DOT parse",
    "kinematics load/evaluation": "kinematics",
    "pySecDec LoopIntegralFromGraph": "pySecDec graph",
    "U/F extraction": "U/F",
    "Symbolica scalar evaluator build": "scalar evals",
    "pySecDec sector decomposition": "sector decomp",
    "FSD SectorDefinition conversion": "sector convert",
    "Symbolica sector evaluator build": "sector evals",
    "Symbolica Taylor evaluator build": "Taylor evals",
    "Symbolica endpoint projector build": "endpoint proj",
    "Symbolica regular Taylor build": "regular Taylor",
    "Symbolica chain-rule build": "chain rule",
    "Symbolica subtraction formula build": "subtraction",
    "Symbolica two-stage sector build": "two-stage",
    "Symbolica explicit sector build": "explicit",
}


def _stage_label(name: str) -> str:
    """Return a compact label for live generation progress."""
    return _STAGE_SHORT_NAMES.get(name, name)


class GenerationProgress:
    """Live status reporter for long topology/package generation phases."""

    def __init__(
        self,
        *,
        enabled: bool,
        logger: logging.Logger | None = None,
        label: str = "generation",
    ) -> None:
        """Create a progress reporter.

        The progressbar is intentionally unknown-length because pySecDec and
        Symbolica generation stages do not expose stable internal progress.
        Stages with a known item count can still provide an ETA through
        ``update``.
        """
        self.enabled = bool(enabled and progressbar is not None)
        self.logger = logger
        self.label = label
        self._overall_start = time.perf_counter()
        self._stage_start = self._overall_start
        self._stage_name = "starting"
        self._stage_detail = ""
        self._stage_current = 0
        self._stage_total: int | None = None
        self._tick = 0
        self._closed = False
        self._live_widget = None
        self._bar = None
        if self.enabled:
            self._live_widget = progressbar.FormatCustomText(
                (
                    f"{_color('stg', Fore.CYAN)}:%(stage)s "
                    f"{_color('step', Fore.GREEN)}:%(step)s "
                    f"{_color('t', Fore.BLUE)}:%(elapsed)s "
                    f"{_color('st', Fore.BLUE)}:%(stage_elapsed)s "
                    f"{_color('eta', Fore.BLUE)}:%(eta)s "
                    f"{_color('msg', Fore.MAGENTA)}:%(detail)s"
                ),
                {
                    "stage": self._stage_name,
                    "step": "n/a",
                    "elapsed": "00:00:00",
                    "stage_elapsed": "00:00:00",
                    "eta": "n/a",
                    "detail": "",
                },
            )
            display_label = "FSD gen" if label == "FSD generation" else _clip(label, 10)
            self._bar = progressbar.ProgressBar(
                max_value=progressbar.UnknownLength,
                widgets=[
                    _color(f"{display_label} ", Fore.CYAN),
                    progressbar.AnimatedMarker(),
                    " ",
                    self._live_widget,
                    " ",
                ],
            )

    def start_stage(self, name: str, detail: str = "", total: int | None = None) -> None:
        """Start a new named generation stage."""
        self._stage_name = name
        self._stage_detail = detail
        self._stage_current = 0
        self._stage_total = int(total) if total is not None else None
        self._stage_start = time.perf_counter()
        if self.logger is not None:
            suffix = f" ({detail})" if detail else ""
            self.logger.info("generation stage started: %s%s", name, suffix)
        self.update(0, total=total, detail=detail)

    def update(
        self,
        current: int | None = None,
        *,
        total: int | None = None,
        detail: str | None = None,
    ) -> None:
        """Refresh the progressbar for a known-count or unknown-count stage."""
        if current is not None:
            self._stage_current = int(current)
        if total is not None:
            self._stage_total = int(total)
        if detail is not None:
            self._stage_detail = detail
        if not self.enabled or self._bar is None or self._live_widget is None or self._closed:
            return
        now = time.perf_counter()
        elapsed = now - self._overall_start
        stage_elapsed = now - self._stage_start
        eta: float | None = None
        if self._stage_total is not None and self._stage_current > 0:
            remaining = max(self._stage_total - self._stage_current, 0)
            eta = stage_elapsed * remaining / float(self._stage_current)
        step = (
            f"{self._stage_current}/{self._stage_total}"
            if self._stage_total is not None
            else "n/a"
        )
        terminal_width = max(shutil.get_terminal_size((120, 20)).columns, 80)
        # Keep the live line safely below common terminal widths.  The progress
        # bar itself accounts for ANSI escapes imperfectly, so leave margin.
        detail_width = max(min(terminal_width - 78, 56), 16)
        self._live_widget.update_mapping(
            stage=_clip(_stage_label(self._stage_name), 20),
            step=step,
            elapsed=_duration_text(elapsed),
            stage_elapsed=_duration_text(stage_elapsed),
            eta=_duration_text(eta),
            detail=_clip(self._stage_detail or "", detail_width),
        )
        self._tick += 1
        self._bar.update(self._tick, force=True)

    def finish_stage(self, name: str, seconds: float, detail: str = "") -> None:
        """Mark the current generation stage complete."""
        if self._stage_total is not None:
            self._stage_current = self._stage_total
        self._stage_detail = f"done in {seconds:.3g}s"
        if self.logger is not None:
            suffix = f" ({detail})" if detail else ""
            self.logger.info("generation stage finished: %s %.6fs%s", name, seconds, suffix)
        self.update(self._stage_current, total=self._stage_total, detail=self._stage_detail)

    @contextmanager
    def stage(self, name: str, detail: str = "", total: int | None = None) -> Iterator[None]:
        """Context manager for untimed stages outside ``GenerationTimings``."""
        self.start_stage(name, detail=detail, total=total)
        start = time.perf_counter()
        try:
            yield
        finally:
            self.finish_stage(name, time.perf_counter() - start, detail=detail)

    def close(self) -> None:
        """Finish the live progressbar if one was created."""
        if self._closed:
            return
        self._closed = True
        if self._bar is not None:
            self._bar.finish(dirty=True)
