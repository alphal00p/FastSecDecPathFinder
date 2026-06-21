#!/usr/bin/env python3
"""Run 3L universal-cache warmup in resumable sector shards."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from FSD import build_request, parse_args, validate_request  # noqa: E402
from cache_warm import EXAMPLE_CASES, _case_request  # noqa: E402
from dot_topology import clear_dot_bundle_cache  # noqa: E402
from sectors_generator import generate_sectors  # noqa: E402


DIRECT_SMALL_CASES = (
    "self_energy_3loop",
    "three_point_3loop",
    "three_point_3loop_8line",
)
TRIPLE_BOX_CASES = ("triple_box",)
ALL_3L_CASES = (*DIRECT_SMALL_CASES, *TRIPLE_BOX_CASES)
DRAINED_RC = 75


@dataclass(frozen=True)
class Phase:
    name: str
    cases: tuple[str, ...]
    extra_args: tuple[str, ...] = ()


PHASES: dict[str, Phase] = {
    "direct-small": Phase("direct-small", DIRECT_SMALL_CASES),
    "triple-box-direct": Phase("triple-box-direct", TRIPLE_BOX_CASES),
    "ibp-3l": Phase(
        "ibp-3l",
        ALL_3L_CASES,
        (
            "--ibp-reduce-to-log-endpoint",
            "--direct-projector-cache-term-threshold",
            "0",
        ),
    ),
}


@dataclass(frozen=True)
class ShardTask:
    phase: str
    case: str
    index: int
    total: int
    sectors: tuple[int, ...]
    report_path: Path
    workdir: Path
    log_path: Path
    command: tuple[str, ...]

    @property
    def task_id(self) -> str:
        return f"{self.phase}:{self.case}:{self.index:04d}/{self.total:04d}"


@dataclass
class RunningTask:
    task: ShardTask
    proc: subprocess.Popen[bytes]
    log_handle: Any
    start: float


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, default=_json_default))
        handle.write("\n")


def _optional_rooted_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def _drain_requested(args: argparse.Namespace) -> bool:
    drain_path = _optional_rooted_path(getattr(args, "drain_file", None))
    return drain_path is not None and drain_path.exists()


def _base_cache_args(samples_per_sector: int) -> list[str]:
    return [
        "cache",
        "--cache-loop-counts",
        "3",
        "--cache-verify-samples-per-sector",
        str(samples_per_sector),
        "--max-eps-order",
        "0",
        "--force-regular-taylor-formulas",
        "--chain-rule-formula-signature-limit",
        "1000000",
        "--chain-rule-formula-output-length-limit",
        "0",
        "--no-cache-estimate-3l",
        "--no-progress",
        "--json",
    ]


def _case_by_name() -> dict[str, Any]:
    return {case.name: case for case in EXAMPLE_CASES}


def _discover_sector_counts(
    cases: tuple[str, ...],
    samples_per_sector: int,
    workdir_root: Path,
) -> dict[str, int]:
    request_args = _base_cache_args(samples_per_sector) + [
        "--cache-report-path",
        str(workdir_root / "sector_count_probe.json"),
        "--cache-workdir",
        str(workdir_root / "sector_count_probe"),
    ]
    request = build_request(parse_args(request_args))
    validate_request(request)
    known_cases = _case_by_name()
    counts: dict[str, int] = {}
    for case_name in cases:
        clear_dot_bundle_cache()
        case = known_cases[case_name]
        case_request = _case_request(request, case, workdir_root / "sector_count_probe")
        counts[case_name] = len(generate_sectors(case_request))
    return counts


def _chunk_sector_ids(sector_count: int, max_shards: int) -> list[tuple[int, ...]]:
    if sector_count <= 0:
        return []
    shard_count = max(1, min(int(max_shards), sector_count))
    chunk_size = int(math.ceil(sector_count / shard_count))
    ids = list(range(sector_count))
    return [
        tuple(ids[start:start + chunk_size])
        for start in range(0, sector_count, chunk_size)
    ]


def _phase_sequence(selected: str) -> list[Phase]:
    if selected == "all":
        return [PHASES["direct-small"], PHASES["triple-box-direct"], PHASES["ibp-3l"]]
    return [PHASES[selected]]


def _build_tasks(
    args: argparse.Namespace,
    phase: Phase,
    counts: dict[str, int],
) -> list[ShardTask]:
    python = Path(args.python)
    if not python.is_absolute():
        python = ROOT / python
    report_root = Path(args.report_dir) / phase.name
    log_root = Path(args.log_dir) / phase.name
    workdir_root = Path(args.workdir_root) / phase.name
    tasks: list[ShardTask] = []
    for case_name in phase.cases:
        chunks = _chunk_sector_ids(counts[case_name], args.shards_per_case)
        total = len(chunks)
        for index, sector_ids in enumerate(chunks, start=1):
            stem = f"{case_name}_shard_{index:04d}_of_{total:04d}"
            report_path = report_root / f"{stem}.json"
            workdir = workdir_root / stem
            log_path = log_root / f"{stem}.log"
            command = [
                str(python),
                str(ROOT / "FSD.py"),
                *_base_cache_args(args.cache_verify_samples_per_sector),
                *phase.extra_args,
                "--cache-cases",
                case_name,
                "--sectors",
                *(str(sector_id) for sector_id in sector_ids),
                "--cache-report-path",
                str(report_path),
                "--cache-workdir",
                str(workdir),
            ]
            tasks.append(
                ShardTask(
                    phase=phase.name,
                    case=case_name,
                    index=index,
                    total=total,
                    sectors=sector_ids,
                    report_path=report_path,
                    workdir=workdir,
                    log_path=log_path,
                    command=tuple(command),
                )
            )
    return tasks


def _report_status(task: ShardTask) -> tuple[bool, str]:
    if not task.report_path.is_file():
        return False, "missing report"
    try:
        report = json.loads(task.report_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"unreadable report: {exc}"
    rows = report.get("cases")
    if not isinstance(rows, list) or len(rows) != 1:
        return False, "report does not contain exactly one case row"
    row = rows[0]
    if row.get("case") != task.case:
        return False, f"report case mismatch: {row.get('case')!r}"
    if row.get("status") == "deferred":
        return False, f"deferred: {row.get('error', '')}"
    if row.get("status") != "ok":
        return False, f"case status is {row.get('status')!r}: {row.get('error', '')}"
    selected = tuple(int(sector_id) for sector_id in row.get("selected_sector_ids", []))
    if selected != task.sectors:
        return False, "selected sector ids do not match shard"
    return True, "ok"


def _snapshot(
    args: argparse.Namespace,
    phase: Phase,
    tasks: list[ShardTask],
    pending_count: int,
    running: dict[int, RunningTask],
    completed: set[str],
    skipped: set[str],
    failures: list[dict[str, Any]],
    deferred_until: dict[str, float],
    start: float,
) -> dict[str, Any]:
    elapsed = time.monotonic() - start
    now = time.monotonic()
    return {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "phase": phase.name,
        "jobs": int(args.jobs),
        "elapsed_seconds": elapsed,
        "task_count": len(tasks),
        "pending": pending_count,
        "running": len(running),
        "completed": len(completed),
        "skipped": len(skipped),
        "failed": len(failures),
        "deferred_pending": sum(
            1
            for task in tasks
            if deferred_until.get(task.task_id, 0.0) > now
        ),
        "running_tasks": [
            {
                "task_id": item.task.task_id,
                "pid": item.proc.pid,
                "elapsed_seconds": time.monotonic() - item.start,
                "log_path": item.task.log_path,
            }
            for item in running.values()
        ],
        "failures": failures[-10:],
    }


def _print_progress(snapshot: dict[str, Any]) -> None:
    running_ids = ", ".join(
        str(item["task_id"]) for item in snapshot.get("running_tasks", [])[:5]
    )
    if len(snapshot.get("running_tasks", [])) > 5:
        running_ids += ", ..."
    print(
        "[cache-shards] "
        f"phase={snapshot['phase']} elapsed={snapshot['elapsed_seconds']:.0f}s "
        f"done={snapshot['completed']} skipped={snapshot['skipped']} "
        f"running={snapshot['running']} pending={snapshot['pending']} "
        f"deferred={snapshot.get('deferred_pending', 0)} "
        f"failed={snapshot['failed']}"
        + (f" active={running_ids}" if running_ids else ""),
        flush=True,
    )


def _terminate_running(running: dict[int, RunningTask], reason: str) -> None:
    print(f"[cache-shards] terminating {len(running)} running shard(s): {reason}", flush=True)
    for item in running.values():
        try:
            item.proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 60.0
    while running and time.monotonic() < deadline:
        for pid, item in list(running.items()):
            if item.proc.poll() is not None:
                item.log_handle.close()
                del running[pid]
        time.sleep(0.5)
    for item in running.values():
        try:
            item.proc.terminate()
        except ProcessLookupError:
            pass
    time.sleep(5.0)
    for item in running.values():
        if item.proc.poll() is None:
            try:
                item.proc.kill()
            except ProcessLookupError:
                pass


def _pop_next_launchable(
    pending: list[ShardTask],
    deferred_until: dict[str, float],
) -> ShardTask | None:
    """Pop the next pending task whose cold-lock backoff has expired."""

    now = time.monotonic()
    for index, task in enumerate(pending):
        if deferred_until.get(task.task_id, 0.0) <= now:
            return pending.pop(index)
    return None


def _is_deferred_reason(reason: str) -> bool:
    """Return whether a shard report represents an intentional cold-lock defer."""

    return reason.startswith("deferred:") or "cold chain-rule formula deferred" in reason


def _clear_previous_report(task: ShardTask) -> None:
    """Remove stale shard report files before launching a fresh attempt."""

    for path in (task.report_path, task.report_path.with_suffix(task.report_path.suffix + ".tmp")):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _run_phase(args: argparse.Namespace, phase: Phase) -> int:
    all_cases = tuple(dict.fromkeys(phase.cases))
    print(f"[cache-shards] discovering sector counts for {phase.name}", flush=True)
    counts = _discover_sector_counts(
        all_cases,
        args.cache_verify_samples_per_sector,
        Path(args.workdir_root) / phase.name,
    )
    print(f"[cache-shards] sector counts for {phase.name}: {counts}", flush=True)
    tasks = _build_tasks(args, phase, counts)

    pending: list[ShardTask] = []
    skipped: set[str] = set()
    for task in tasks:
        ok, _reason = _report_status(task)
        if args.resume and ok:
            skipped.add(task.task_id)
        else:
            pending.append(task)

    _append_event(
        Path(args.event_path),
        {
            "event": "phase_start",
            "phase": phase.name,
            "task_count": len(tasks),
            "pending": len(pending),
            "skipped": len(skipped),
            "sector_counts": counts,
        },
    )
    if args.dry_run:
        for task in pending:
            print(" ".join(task.command))
        return 0

    running: dict[int, RunningTask] = {}
    completed: set[str] = set()
    failures: list[dict[str, Any]] = []
    attempts: dict[str, int] = {}
    deferred_until: dict[str, float] = {}
    start = time.monotonic()
    next_progress = 0.0

    try:
        while pending or running:
            drain_requested = _drain_requested(args)
            while pending and not drain_requested and len(running) < args.jobs:
                task = _pop_next_launchable(pending, deferred_until)
                if task is None:
                    break
                deferred_until.pop(task.task_id, None)
                task.log_path.parent.mkdir(parents=True, exist_ok=True)
                task.workdir.mkdir(parents=True, exist_ok=True)
                _clear_previous_report(task)
                log_handle = task.log_path.open("ab")
                attempts[task.task_id] = attempts.get(task.task_id, 0) + 1
                proc = subprocess.Popen(
                    task.command,
                    cwd=ROOT,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running[proc.pid] = RunningTask(task, proc, log_handle, time.monotonic())
                _append_event(
                    Path(args.event_path),
                    {
                        "event": "task_start",
                        "phase": phase.name,
                        "task_id": task.task_id,
                        "pid": proc.pid,
                        "attempt": attempts[task.task_id],
                        "case": task.case,
                        "sectors": task.sectors,
                        "report_path": task.report_path,
                        "log_path": task.log_path,
                    },
                )

            for pid, item in list(running.items()):
                rc = item.proc.poll()
                if rc is None:
                    continue
                item.log_handle.close()
                del running[pid]
                report_ok, reason = _report_status(item.task)
                if rc == 0 and report_ok:
                    deferred_until.pop(item.task.task_id, None)
                    completed.add(item.task.task_id)
                    _append_event(
                        Path(args.event_path),
                        {
                            "event": "task_ok",
                            "phase": phase.name,
                            "task_id": item.task.task_id,
                            "returncode": rc,
                            "attempt": attempts.get(item.task.task_id, 1),
                            "elapsed_seconds": time.monotonic() - item.start,
                            "report_path": item.task.report_path,
                            "log_path": item.task.log_path,
                        },
                    )
                else:
                    failure = {
                        "task_id": item.task.task_id,
                        "case": item.task.case,
                        "returncode": rc,
                        "attempt": attempts.get(item.task.task_id, 1),
                        "reason": reason,
                        "report_path": item.task.report_path,
                        "log_path": item.task.log_path,
                    }
                    if _is_deferred_reason(reason):
                        attempts[item.task.task_id] = max(
                            attempts.get(item.task.task_id, 1) - 1,
                            0,
                        )
                        deferred_until[item.task.task_id] = (
                            time.monotonic() + float(args.deferred_retry_seconds)
                        )
                        pending.append(item.task)
                        _append_event(
                            Path(args.event_path),
                            {
                                "event": "task_deferred",
                                "phase": phase.name,
                                "retry_after_seconds": float(args.deferred_retry_seconds),
                                **failure,
                            },
                        )
                        continue
                    _append_event(
                        Path(args.event_path),
                        {"event": "task_failed", "phase": phase.name, **failure},
                    )
                    if attempts.get(item.task.task_id, 1) < args.max_task_attempts:
                        pending.append(item.task)
                        _append_event(
                            Path(args.event_path),
                            {
                                "event": "task_retry",
                                "phase": phase.name,
                                "next_attempt": attempts.get(item.task.task_id, 1) + 1,
                                **failure,
                            },
                        )
                        continue
                    failures.append(failure)
                    if not args.fail_fast:
                        continue
                    _terminate_running(running, f"{item.task.task_id} failed")
                    snapshot = _snapshot(
                        args,
                        phase,
                        tasks,
                        len(pending),
                        running,
                        completed,
                        skipped,
                        failures,
                        deferred_until,
                        start,
                    )
                    _write_json_atomic(Path(args.status_path), snapshot)
                    _print_progress(snapshot)
                    return 1

            if drain_requested and not running:
                snapshot = _snapshot(
                    args,
                    phase,
                    tasks,
                    len(pending),
                    running,
                    completed,
                    skipped,
                    failures,
                    deferred_until,
                    start,
                )
                _write_json_atomic(Path(args.status_path), snapshot)
                _append_event(
                    Path(args.event_path),
                    {"event": "phase_drained", "phase": phase.name, **snapshot},
                )
                _print_progress(snapshot)
                return DRAINED_RC

            now = time.monotonic()
            if now >= next_progress:
                snapshot = _snapshot(
                    args,
                    phase,
                    tasks,
                    len(pending),
                    running,
                    completed,
                    skipped,
                    failures,
                    deferred_until,
                    start,
                )
                _write_json_atomic(Path(args.status_path), snapshot)
                _print_progress(snapshot)
                next_progress = now + float(args.progress_seconds)
            time.sleep(1.0)
    except KeyboardInterrupt:
        _terminate_running(running, "received interrupt")
        raise

    snapshot = _snapshot(
        args,
        phase,
        tasks,
        0,
        running,
        completed,
        skipped,
        failures,
        deferred_until,
        start,
    )
    _write_json_atomic(Path(args.status_path), snapshot)
    if failures:
        _append_event(
            Path(args.event_path),
            {"event": "phase_failed", "phase": phase.name, **snapshot},
        )
        _print_progress(snapshot)
        return 1
    _append_event(Path(args.event_path), {"event": "phase_ok", "phase": phase.name, **snapshot})
    _print_progress(snapshot)
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("all", *PHASES.keys()),
        default="all",
        help="3L cache phase to run. Default: all phases in the project handoff.",
    )
    parser.add_argument("--jobs", type=int, default=100)
    parser.add_argument(
        "--shards-per-case",
        type=int,
        default=100,
        help="Maximum number of sector shards to make for each selected case.",
    )
    parser.add_argument(
        "--cache-verify-samples-per-sector",
        type=int,
        default=1,
        help="Low-stat verification samples per active sector in each shard.",
    )
    parser.add_argument("--python", default=".venv/bin/python")
    parser.add_argument("--report-dir", default="docs/cache_shards/reports")
    parser.add_argument("--log-dir", default="docs/cache_shards/logs")
    parser.add_argument("--workdir-root", default=".cache_warm_cluster_shards")
    parser.add_argument("--status-path", default="docs/cluster_cache_3l_shards_status.json")
    parser.add_argument("--event-path", default="docs/cluster_cache_3l_shards_events.jsonl")
    parser.add_argument(
        "--drain-file",
        default="docs/cluster_cache_3l_drain.order",
        help=(
            "When this file exists, stop launching new shards and exit after "
            "currently running shards finish.  Relative paths are rooted at "
            "the repository."
        ),
    )
    parser.add_argument("--progress-seconds", type=float, default=30.0)
    parser.add_argument(
        "--deferred-retry-seconds",
        type=float,
        default=300.0,
        help="seconds to wait before retrying a shard deferred by a cold formula lock",
    )
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--max-task-attempts",
        type=int,
        default=2,
        help="maximum attempts per shard before recording a terminal failure",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="terminate other running shards immediately after a terminal shard failure",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    if args.shards_per_case <= 0:
        parser.error("--shards-per-case must be positive")
    if args.max_task_attempts <= 0:
        parser.error("--max-task-attempts must be positive")
    if args.deferred_retry_seconds < 0.0:
        parser.error("--deferred-retry-seconds must be non-negative")
    return args


def main() -> int:
    args = _parse_args()
    for phase in _phase_sequence(args.variant):
        rc = _run_phase(args, phase)
        if rc == DRAINED_RC:
            return 0
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
