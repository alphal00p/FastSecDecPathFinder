#!/usr/bin/env python3
"""Print a compact status snapshot for the cluster 3L cache run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _read_pid(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    status_path = Path("/proc") / str(pid) / "status"
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("State:"):
                fields = line.split()
                return len(fields) < 2 or fields[1] != "Z"
    except OSError:
        return False
    return True


def _pid_environ(pid: int | None) -> dict[str, str]:
    """Return a process environment snapshot without exposing secret values."""

    if pid is None:
        return {}
    env_path = Path("/proc") / str(pid) / "environ"
    try:
        raw = env_path.read_bytes()
    except OSError:
        return {}
    values: dict[str, str] = {}
    for item in raw.split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        values[key.decode("utf-8", errors="replace")] = value.decode(
            "utf-8",
            errors="replace",
        )
    return values


def _pid_cmdline(pid: int | None) -> str:
    if pid is None:
        return ""
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    try:
        raw = cmdline_path.read_bytes()
    except OSError:
        return ""
    parts = [part.decode("utf-8", errors="replace") for part in raw.split(b"\0") if part]
    return shlex.join(parts)


def _process_count_matching(pattern: str) -> int:
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except (OSError, subprocess.SubprocessError):
        return 0
    self_pid = os.getpid()
    count = 0
    for line in output.splitlines():
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid != self_pid and pattern in parts[1]:
            count += 1
    return count


def _process_rows_matching(patterns: tuple[str, ...]) -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,stat=,etime=,rss=,cmd="],
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    self_pid = os.getpid()
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = line.split(None, 4)
        if len(parts) != 5:
            continue
        pid_text, stat, elapsed, rss_kib, command = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == self_pid or stat.startswith("Z"):
            continue
        if not any(pattern in command for pattern in patterns):
            continue
        rows.append(
            {
                "pid": str(pid),
                "stat": stat,
                "elapsed": elapsed,
                "rss_kib": rss_kib,
                "command": command,
            }
        )
    return rows


def _command_arg(command: str, option: str) -> str | None:
    try:
        parts = shlex.split(command)
    except ValueError:
        return None
    for index, part in enumerate(parts[:-1]):
        if part == option:
            return parts[index + 1]
    prefix = f"{option}="
    for part in parts:
        if part.startswith(prefix):
            return part[len(prefix) :]
    return None


def _guard_kind(command: str) -> str:
    for name, kind in (
        ("guard_chain_rule_expression_checkpoint.py", "checkpoint_guard"),
        ("protect_chain_rule_checkpoint_memory.py", "memory_guard"),
        ("release_cache_drain_after_checkpoint.py", "drain_release"),
        ("restart_cache_after_chain_rule_checkpoint.py", "checkpoint_restart"),
    ):
        if name in command:
            return kind
    return "unknown_guard"


def _print_active_guard_processes() -> None:
    patterns = (
        "guard_chain_rule_expression_checkpoint.py",
        "protect_chain_rule_checkpoint_memory.py",
        "release_cache_drain_after_checkpoint.py",
        "restart_cache_after_chain_rule_checkpoint.py",
    )
    rows = _process_rows_matching(patterns)
    print(f"active_guard_processes count={len(rows)}")
    for row in sorted(rows, key=lambda item: (_guard_kind(item["command"]), int(item["pid"]))):
        command = row["command"]
        log_file = _command_arg(command, "--log-file")
        digest = _command_arg(command, "--digest")
        fields = [
            f"pid={row['pid']}",
            f"kind={_guard_kind(command)}",
            f"stat={row['stat']}",
            f"elapsed={row['elapsed']}",
            f"rss_kib={row['rss_kib']}",
        ]
        if digest:
            fields.append(f"digest={digest}")
        if log_file:
            fields.append(f"log={log_file}")
            latest = _latest_nonempty_line(Path(log_file))
            if latest:
                fields.append(f"latest={latest}")
        print("active_guard " + " ".join(fields))


def _print_pid_log_status(label: str, pid_file: Path, log_file: Path) -> None:
    pid = _read_pid(pid_file)
    print(
        f"{label} "
        f"pid={pid if pid is not None else 'unknown'} "
        f"alive={_pid_alive(pid)} "
        f"pid_file={pid_file}"
    )
    latest = _latest_nonempty_line(log_file)
    if latest:
        print(f"{label}_latest path={log_file} line={latest}")
    else:
        print(f"{label}_latest unavailable path={log_file}")


def _latest_nonempty_line(path: Path, *, max_bytes: int = 8_000) -> str | None:
    lines = [line for line in _tail_text(path, max_bytes=max_bytes).splitlines() if line.strip()]
    return lines[-1] if lines else None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _tail_text(path: Path, max_bytes: int = 256_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - int(max_bytes)))
            return handle.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _latest_matching_line(
    log_dir: Path,
    pattern: str | tuple[str, ...],
    *,
    include: tuple[str, ...] = (),
    exclude: tuple[str, ...] = (),
) -> tuple[Path | None, str]:
    best_path: Path | None = None
    best_line = ""
    best_mtime = -1.0
    patterns = (pattern,) if isinstance(pattern, str) else pattern
    for path in sorted(log_dir.glob("*.log")):
        text = _tail_text(path)
        lines = [
            line
            for line in text.splitlines()
            if any(item in line for item in patterns)
            and (not include or any(token in line for token in include))
            and not any(token in line for token in exclude)
        ]
        if not lines:
            continue
        mtime = path.stat().st_mtime
        if mtime >= best_mtime:
            best_mtime = mtime
            best_path = path
            best_line = lines[-1]
    return best_path, best_line


def _utc_after(seconds: float) -> str:
    eta_timestamp = time.time() + max(float(seconds), 0.0)
    return datetime.fromtimestamp(eta_timestamp, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _dense_total_degree_multi_indices(rank: int, max_total: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def visit(prefix: list[int], remaining_rank: int, remaining_total: int) -> None:
        if remaining_rank == 0:
            out.append(tuple(prefix))
            return
        for value in range(remaining_total + 1):
            prefix.append(value)
            visit(prefix, remaining_rank - 1, remaining_total - value)
            prefix.pop()

    visit([], int(rank), int(max_total))
    return sorted(out, key=lambda item: (sum(item), item))


def _dense_total_degree_from_count(rank: int, count: int) -> int | None:
    for degree in range(0, 16):
        if math.comb(int(rank) + degree, degree) == int(count):
            return degree
    return None


def _chain_product_alpha_from_label(label: str) -> tuple[int, ...] | None:
    match = re.search(r"\bchain_product\.(?P<alpha>[0-9_]+)\.x\d+\^\d+\b", label)
    if match is None:
        return None
    try:
        return tuple(int(part) for part in match.group("alpha").split("_"))
    except ValueError:
        return None


def _chain_rule_inferred_expression_progress(
    *,
    alpha: tuple[int, ...] | None,
    total: int | None,
) -> str | None:
    if alpha is None or total is None:
        return None
    degree = _dense_total_degree_from_count(len(alpha), int(total))
    if degree is None:
        return None
    derivative_indices = _dense_total_degree_multi_indices(len(alpha), degree)
    try:
        current_index = derivative_indices.index(tuple(alpha)) + 1
    except ValueError:
        return None
    remaining_after_current = max(len(derivative_indices) - current_index, 0)
    alpha_text = "_".join(str(value) for value in alpha)
    return (
        f"inferred_current_expression={current_index}/{len(derivative_indices)} "
        f"current_alpha={alpha_text} remaining_after_current={remaining_after_current}"
    )


def _chain_rule_trend(line_source: str, latest_age_seconds: float | None = None) -> str | None:
    progress_pattern = re.compile(r"compose_progress\s+(?P<done>\d+)/(?P<total>\d+)")
    evaluator_start_pattern = re.compile(r"evaluator_start\b")
    evaluator_done_pattern = re.compile(
        r"evaluator_done\s+seconds=(?P<seconds>[0-9.]+)"
    )
    sidecar_load_progress_pattern = re.compile(
        r"expression_binary_sidecar_load_progress\s+(?P<done>\d+)/(?P<total>\d+)\s+"
        r"file=(?P<file>\S+)"
    )
    sidecar_load_done_pattern = re.compile(
        r"expression_binary_sidecar_load_done\s+seconds=(?P<seconds>[0-9.]+)"
    )
    series_mul_pattern = re.compile(
        r"series_mul_(?P<event>progress|done)\s+label=(?P<label>\S+)\s+"
        r"pairs=(?P<done>\d+)/(?P<total>\d+)"
        r"(?:\s+original_pairs=(?P<original_total>\d+))?\s+"
        r"kept_terms=(?P<kept>\d+)"
    )
    series_mul_start_pattern = re.compile(
        r"series_mul_start\s+label=(?P<label>\S+)\s+"
        r"left_terms=(?P<left>\d+)\s+right_terms=(?P<right>\d+)\s+"
        r"total_pairs=(?P<total>\d+)"
        r"(?:\s+original_pairs=(?P<original_total>\d+))?"
    )
    elapsed_pattern = re.compile(r"\belapsed=(?P<elapsed>[0-9.]+)")
    rss_pattern = re.compile(r"\brss_gib=(?P<rss>[0-9.]+)")
    points: list[tuple[int, int, float | None, float | None]] = []
    evaluator_points: list[tuple[str, dict[str, str], float | None, float | None]] = []
    sidecar_load_points: list[tuple[str, int, int, str | None, float | None, float | None]] = []
    inner_points: list[tuple[str, str, int, int, int, int | None]] = []

    def log_field(line: str, name: str) -> str | None:
        match = re.search(rf"\b{re.escape(name)}=(?P<value>\S+)", line)
        return match.group("value") if match is not None else None

    for line in line_source.splitlines():
        if "[fsd-chain-rule]" not in line:
            continue
        if "evaluator_start" in line:
            match = evaluator_start_pattern.search(line)
            if match is None:
                continue
            elapsed_match = elapsed_pattern.search(line)
            rss_match = rss_pattern.search(line)
            settings = {
                name: value
                for name in (
                    "source",
                    "n_cores",
                    "verbose",
                    "iterations",
                    "cpe_iterations",
                    "max_horner_scheme_variables",
                    "max_common_pair_cache_entries",
                    "max_common_pair_distance",
                    "jit_compile",
                )
                if (value := log_field(line, name)) is not None
            }
            evaluator_points.append(
                (
                    "start",
                    settings,
                    float(elapsed_match.group("elapsed")) if elapsed_match is not None else None,
                    float(rss_match.group("rss")) if rss_match is not None else None,
                )
            )
            continue
        if "evaluator_done" in line:
            match = evaluator_done_pattern.search(line)
            elapsed_match = elapsed_pattern.search(line)
            rss_match = rss_pattern.search(line)
            evaluator_points.append(
                (
                    "done",
                    {},
                    float(elapsed_match.group("elapsed")) if elapsed_match is not None else (
                        float(match.group("seconds")) if match is not None else None
                    ),
                    float(rss_match.group("rss")) if rss_match is not None else None,
                )
            )
            continue
        if "expression_binary_sidecar_load_progress" in line:
            match = sidecar_load_progress_pattern.search(line)
            if match is None:
                continue
            elapsed_match = elapsed_pattern.search(line)
            rss_match = rss_pattern.search(line)
            sidecar_load_points.append(
                (
                    "progress",
                    int(match.group("done")),
                    int(match.group("total")),
                    match.group("file"),
                    float(elapsed_match.group("elapsed")) if elapsed_match is not None else None,
                    float(rss_match.group("rss")) if rss_match is not None else None,
                )
            )
            continue
        if "expression_binary_sidecar_load_done" in line:
            match = sidecar_load_done_pattern.search(line)
            elapsed_match = elapsed_pattern.search(line)
            rss_match = rss_pattern.search(line)
            sidecar_load_points.append(
                (
                    "done",
                    1,
                    1,
                    None,
                    float(elapsed_match.group("elapsed")) if elapsed_match is not None else (
                        float(match.group("seconds")) if match is not None else None
                    ),
                    float(rss_match.group("rss")) if rss_match is not None else None,
                )
            )
            continue
        if "compose_progress" in line:
            match = progress_pattern.search(line)
            if match is None:
                continue
            elapsed_match = elapsed_pattern.search(line)
            rss_match = rss_pattern.search(line)
            points.append(
                (
                    int(match.group("done")),
                    int(match.group("total")),
                    float(elapsed_match.group("elapsed")) if elapsed_match is not None else None,
                    float(rss_match.group("rss")) if rss_match is not None else None,
                )
            )
        elif (
            "series_mul_start" in line
            or "series_mul_progress" in line
            or "series_mul_done" in line
        ):
            start_match = series_mul_start_pattern.search(line)
            if start_match is not None:
                inner_points.append(
                    (
                        "start",
                        start_match.group("label"),
                        0,
                        int(start_match.group("total")),
                        0,
                        int(start_match.group("original_total"))
                        if start_match.group("original_total")
                        else None,
                    )
                )
                continue
            match = series_mul_pattern.search(line)
            if match is not None:
                inner_points.append(
                    (
                        match.group("event"),
                        match.group("label"),
                        int(match.group("done")),
                        int(match.group("total")),
                        int(match.group("kept")),
                        int(match.group("original_total")) if match.group("original_total") else None,
                    )
                )
    if not points and not inner_points:
        if not sidecar_load_points and not evaluator_points:
            return None
    fields: list[str] = []
    suppress_expression_progress = False
    if evaluator_points:
        event, evaluator_settings, elapsed, rss = evaluator_points[-1]
        if event == "done":
            fields.append("evaluator=done")
            suppress_expression_progress = True
        else:
            fields.append("evaluator=building")
            suppress_expression_progress = True
            source = evaluator_settings.get("source")
            if source is not None:
                fields.append(f"evaluator_source={source}")
            n_cores = evaluator_settings.get("n_cores")
            if n_cores is not None:
                fields.append(f"evaluator_n_cores={n_cores}")
            verbose = evaluator_settings.get("verbose")
            if verbose is not None:
                fields.append(f"evaluator_verbose={verbose}")
            for key in (
                "iterations",
                "cpe_iterations",
                "max_horner_scheme_variables",
                "max_common_pair_cache_entries",
                "max_common_pair_distance",
            ):
                value = evaluator_settings.get(key)
                if value is not None:
                    fields.append(f"evaluator_{key}={value}")
            jit_compile = evaluator_settings.get("jit_compile")
            if jit_compile is not None:
                fields.append(f"evaluator_jit_compile={jit_compile}")
            if latest_age_seconds is not None:
                fields.append(f"evaluator_time_since_log_marker={_human_duration(latest_age_seconds)}")
        if elapsed is not None:
            fields.append(f"evaluator_elapsed_marker={elapsed:.1f}s")
        if rss is not None:
            fields.append(f"evaluator_rss_marker={rss:.1f}GiB")
    if sidecar_load_points:
        event, done, total, file_name, elapsed, rss = sidecar_load_points[-1]
        if event == "done":
            fields.append("sidecar_load=done")
        else:
            percentage = 100.0 * float(done) / float(total) if total else 0.0
            fields.append(f"sidecar_load={done}/{total} ({percentage:.1f}%)")
            if file_name is not None:
                fields.append(f"sidecar_file={file_name}")
        if elapsed is not None:
            fields.append(f"sidecar_elapsed={elapsed:.1f}s")
        if rss is not None:
            fields.append(f"sidecar_rss={rss:.1f}GiB")
        if len(sidecar_load_points) >= 2:
            prev_event, prev_done, _prev_total, _prev_file, prev_elapsed, prev_rss = sidecar_load_points[-2]
            step = max(done - prev_done, 1)
            remaining = 0 if event == "done" else max(total - done, 0)
            if elapsed is not None and prev_elapsed is not None and event != "done" and prev_event != "done":
                delta_seconds = elapsed - prev_elapsed
                if delta_seconds > 0:
                    eta_seconds = delta_seconds * remaining / step
                    eta_from_now_seconds = eta_seconds
                    if latest_age_seconds is not None:
                        eta_from_now_seconds = eta_seconds - latest_age_seconds
                    fields.append(f"sidecar_last_interval={delta_seconds:.1f}s/{step}")
                    if latest_age_seconds is not None:
                        next_marker_seconds = delta_seconds - latest_age_seconds
                        if next_marker_seconds >= 0.0:
                            fields.append(f"sidecar_next_marker_eta={_human_duration(next_marker_seconds)}")
                        else:
                            fields.append(f"sidecar_next_marker_eta={_human_overdue(next_marker_seconds)}")
                            fields.append("sidecar_eta_reliability=stale_marker_overdue")
                    fields.append(f"sidecar_eta_by_last_interval={eta_seconds / 3600.0:.2f}h")
                    if eta_from_now_seconds >= 0.0:
                        fields.append(f"sidecar_eta_from_now_by_last_interval={eta_from_now_seconds / 3600.0:.2f}h")
                        fields.append(f"sidecar_eta_utc_by_last_interval={_utc_after(eta_from_now_seconds)}")
                    else:
                        fields.append(
                            "sidecar_eta_from_now_by_last_interval="
                            f"{_human_overdue(eta_from_now_seconds)}"
                        )
            if rss is not None and prev_rss is not None:
                fields.append(f"sidecar_last_rss_delta={rss - prev_rss:.1f}GiB")
    if points and not suppress_expression_progress:
        done, total, elapsed, rss = points[-1]
        fields.append(f"progress={done}/{total}")
        if elapsed is not None:
            fields.append(f"elapsed={elapsed:.1f}s")
        if rss is not None:
            fields.append(f"worker_rss={rss:.1f}GiB")
        if len(points) >= 2:
            prev_done, _prev_total, prev_elapsed, prev_rss = points[-2]
            step = max(done - prev_done, 1)
            remaining = max(total - done, 0)
            if elapsed is not None and prev_elapsed is not None:
                delta_seconds = elapsed - prev_elapsed
                if delta_seconds > 0:
                    eta_seconds = delta_seconds * remaining / step
                    eta_from_now_seconds = eta_seconds
                    if latest_age_seconds is not None:
                        eta_from_now_seconds = eta_seconds - latest_age_seconds
                    fields.append(f"last_interval={delta_seconds:.1f}s/{step}")
                    if latest_age_seconds is not None:
                        next_marker_seconds = delta_seconds - latest_age_seconds
                        if next_marker_seconds >= 0.0:
                            fields.append(f"next_marker_eta={_human_duration(next_marker_seconds)}")
                        else:
                            fields.append(f"next_marker_eta={_human_overdue(next_marker_seconds)}")
                            fields.append("eta_reliability=stale_marker_overdue")
                    fields.append(f"eta_by_last_interval={eta_seconds / 3600.0:.2f}h")
                    if eta_from_now_seconds >= 0.0:
                        fields.append(
                            f"eta_from_now_by_last_interval={eta_from_now_seconds / 3600.0:.2f}h"
                        )
                        fields.append(f"eta_utc_by_last_interval={_utc_after(eta_from_now_seconds)}")
                    else:
                        fields.append(
                            "eta_from_now_by_last_interval="
                            f"{_human_overdue(eta_from_now_seconds)}"
                        )
            if rss is not None and prev_rss is not None:
                delta_rss = rss - prev_rss
                fields.append(f"last_rss_delta={delta_rss:.1f}GiB")
    if inner_points and not suppress_expression_progress:
        event, label, done, total, kept, original_total = inner_points[-1]
        inferred = _chain_rule_inferred_expression_progress(
            alpha=_chain_product_alpha_from_label(label),
            total=points[-1][1] if points else None,
        )
        if inferred is not None:
            fields.append(inferred)
        percentage = 100.0 * float(done) / float(total) if total else 0.0
        inner_text = f"inner_mul={label} event={event} pairs={done}/{total} ({percentage:.1f}%)"
        if original_total is not None:
            inner_text = f"{inner_text} original_pairs={original_total}"
        fields.append(f"{inner_text} kept_terms={kept}")
    return " ".join(fields)


def _du_bytes(path: Path) -> int | None:
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], text=True)
        return int(out.split()[0])
    except Exception:
        return None


def _human_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    amount = float(value)
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    return f"{value} bytes ({amount:.3f} {unit})"


def _phase_report_counts(report_root: Path) -> str | None:
    if not report_root.is_dir():
        return None
    counts: list[str] = []
    for phase_dir in sorted(path for path in report_root.iterdir() if path.is_dir()):
        count = sum(1 for _ in phase_dir.glob("*.json"))
        counts.append(f"{phase_dir.name}={count}")
    return " ".join(counts) if counts else None


def _human_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    remaining = max(int(seconds), 0)
    days, remaining = divmod(remaining, 24 * 3600)
    hours, remaining = divmod(remaining, 3600)
    minutes, seconds_int = divmod(remaining, 60)
    if days:
        return f"{days}d{hours:02d}h"
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds_int:02d}s"
    return f"{seconds_int}s"


def _human_overdue(seconds: float) -> str:
    return f"overdue_by={_human_duration(abs(float(seconds)))}"


def _watchdog_rss_trend(lines: list[str], limit_gib: float) -> str | None:
    pattern = re.compile(
        r"elapsed=(?P<elapsed>[0-9.]+)s\s+rss=(?P<rss>[0-9.]+)\s+GiB"
    )
    points: list[tuple[float, float]] = []
    for line in lines:
        match = pattern.search(line)
        if match is None:
            continue
        points.append((float(match.group("elapsed")), float(match.group("rss"))))
    reset_after_drop = False
    drop_threshold = max(20.0, 0.05 * float(limit_gib))
    start_index = 0
    for index in range(1, len(points)):
        if points[index][1] - points[index - 1][1] <= -drop_threshold:
            start_index = index
            reset_after_drop = True
    if start_index:
        points = points[start_index:]
    if len(points) < 2:
        return None
    start_elapsed, start_rss = points[0]
    end_elapsed, end_rss = points[-1]
    delta_seconds = end_elapsed - start_elapsed
    if delta_seconds <= 0:
        return None
    delta_rss = end_rss - start_rss
    rate_gib_per_hour = delta_rss * 3600.0 / delta_seconds
    fields = [
        f"window={_human_duration(delta_seconds)}",
        f"delta={delta_rss:.1f}GiB",
        f"rate={rate_gib_per_hour:.1f}GiB_per_hour",
    ]
    if reset_after_drop:
        fields.append("reset_after_rss_drop=true")
    if rate_gib_per_hour > 0:
        seconds_to_limit = (float(limit_gib) - end_rss) * 3600.0 / rate_gib_per_hour
        if seconds_to_limit >= 0:
            fields.append(f"time_to_limit={_human_duration(seconds_to_limit)}")
            fields.append(f"eta_limit_utc={_utc_after(seconds_to_limit)}")
    return " ".join(fields)


def _path_age_seconds(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def _ps_for_pids(pids: list[int], limit: int) -> str:
    if not pids:
        return ""
    pid_arg = ",".join(str(pid) for pid in pids)
    try:
        out = subprocess.check_output(
            ["ps", "-o", "pid,stat,etime,time,pcpu,rss,cmd", "-p", pid_arg],
            text=True,
        )
    except subprocess.CalledProcessError:
        return ""
    lines = out.splitlines()
    if len(lines) <= 1:
        return out.strip()
    header, rows = lines[0], lines[1:]
    rows = [
        row for row in rows
        if len(row.split(None, 6)) < 2 or not row.split(None, 6)[1].startswith("Z")
    ]
    if not rows:
        return header

    def rss(row: str) -> int:
        fields = row.split(None, 6)
        if len(fields) < 6:
            return 0
        try:
            return int(fields[5])
        except ValueError:
            return 0

    rows.sort(key=rss, reverse=True)
    return "\n".join([header, *rows[: max(int(limit), 1)]])


def _cache_log_path_from_report(report_path: Path) -> Path:
    parts = list(report_path.parts)
    try:
        report_index = parts.index("reports")
    except ValueError:
        return report_path.with_suffix(".log")
    parts[report_index] = "logs"
    return Path(*parts).with_suffix(".log")


def _standalone_cache_workers(known_pids: set[int]) -> list[dict[str, Any]]:
    """Return live direct FSD.py cache workers not owned by run_cache_shards.py."""

    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,stat=,comm=,args="],
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    rows: list[dict[str, Any]] = []
    for line in out.splitlines():
        fields = line.split(None, 3)
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        stat, comm, command = fields[1], fields[2], fields[3]
        if pid in known_pids or stat.startswith("Z") or "python" not in comm:
            continue
        if "FSD.py cache" not in command or "--cache-loop-counts 3" not in command:
            continue
        if "run_with_memory_watch.py" in command or "scripts/run_cache_shards.py" in command:
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        report_path: Path | None = None
        case = "unknown"
        try:
            report_path = Path(tokens[tokens.index("--cache-report-path") + 1])
        except (ValueError, IndexError):
            pass
        try:
            case = tokens[tokens.index("--cache-cases") + 1]
        except (ValueError, IndexError):
            pass
        if report_path is not None:
            log_path = _cache_log_path_from_report(report_path)
            task_id = f"standalone:{case}:{report_path.stem}"
        else:
            log_path = Path("")
            task_id = f"standalone:{case}:unknown"
        rows.append({"pid": pid, "task_id": task_id, "log_path": str(log_path)})
    return rows


def _worker_cpu_summary(pids: list[int]) -> str | None:
    if not pids:
        return (
            "worker_count=0 sum_pcpu=0.0 active_core_equiv=0.00 "
            "workers_ge_50pct=0 workers_ge_5pct=0 workers_ge_1pct=0 "
            "worker_rss_sum_gib=0.0"
        )
    pid_arg = ",".join(str(pid) for pid in pids)
    try:
        out = subprocess.check_output(
            ["ps", "-o", "pid=,stat=,pcpu=,rss=", "-p", pid_arg],
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    rows: list[tuple[int, str, float, int]] = []
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        if fields[1].startswith("Z"):
            continue
        try:
            rows.append((int(fields[0]), fields[1], float(fields[2]), int(fields[3])))
        except ValueError:
            continue
    if not rows:
        return (
            "worker_count=0 sum_pcpu=0.0 active_core_equiv=0.00 "
            "workers_ge_50pct=0 workers_ge_5pct=0 workers_ge_1pct=0 "
            "worker_rss_sum_gib=0.0"
        )
    sum_pcpu = sum(row[2] for row in rows)
    rss_gib = sum(row[3] for row in rows) / 1024.0 / 1024.0
    return (
        f"worker_count={len(rows)} sum_pcpu={sum_pcpu:.1f} "
        f"active_core_equiv={sum_pcpu / 100.0:.2f} "
        f"workers_ge_50pct={sum(1 for row in rows if row[2] >= 50.0)} "
        f"workers_ge_5pct={sum(1 for row in rows if row[2] >= 5.0)} "
        f"workers_ge_1pct={sum(1 for row in rows if row[2] >= 1.0)} "
        f"worker_rss_sum_gib={rss_gib:.1f}"
    )


def _worker_thread_cpu_summary(pids: list[int]) -> str | None:
    if not pids:
        return (
            "thread_count=0 sum_thread_pcpu=0.0 active_thread_core_equiv=0.00 "
            "threads_ge_50pct=0 threads_ge_5pct=0 threads_ge_1pct=0"
        )
    pid_arg = ",".join(str(pid) for pid in pids)
    try:
        out = subprocess.check_output(
            ["ps", "-L", "-o", "pid=,tid=,stat=,pcpu=", "-p", pid_arg],
            text=True,
        )
    except subprocess.CalledProcessError:
        return None
    rows: list[tuple[int, int, str, float]] = []
    for line in out.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        if fields[2].startswith("Z"):
            continue
        try:
            rows.append((int(fields[0]), int(fields[1]), fields[2], float(fields[3])))
        except ValueError:
            continue
    if not rows:
        return (
            "thread_count=0 sum_thread_pcpu=0.0 active_thread_core_equiv=0.00 "
            "threads_ge_50pct=0 threads_ge_5pct=0 threads_ge_1pct=0"
        )
    sum_pcpu = sum(row[3] for row in rows)
    top_pid, top_tid, _top_state, top_pcpu = max(rows, key=lambda row: row[3])
    return (
        f"thread_count={len(rows)} sum_thread_pcpu={sum_pcpu:.1f} "
        f"active_thread_core_equiv={sum_pcpu / 100.0:.2f} "
        f"threads_ge_50pct={sum(1 for row in rows if row[3] >= 50.0)} "
        f"threads_ge_5pct={sum(1 for row in rows if row[3] >= 5.0)} "
        f"threads_ge_1pct={sum(1 for row in rows if row[3] >= 1.0)} "
        f"top_thread_pid={top_pid} top_thread_tid={top_tid} top_thread_pcpu={top_pcpu:.1f}"
    )


def _pid_rss_kib(pid: int) -> int:
    status = Path("/proc") / str(pid) / "status"
    try:
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) if len(parts) >= 2 else 0
    except OSError:
        return 0
    return 0


def _pid_state(pid: int) -> str:
    status = Path("/proc") / str(pid) / "status"
    try:
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("State:"):
                parts = line.split()
                return parts[1] if len(parts) >= 2 else ""
    except OSError:
        return ""
    return ""


def _pid_is_live_worker(pid: int) -> bool:
    state = _pid_state(pid)
    return bool(state) and state != "Z"


def _pid_rss_gib(pid: int) -> float:
    return float(_pid_rss_kib(pid)) / 1024.0 / 1024.0


def _pid_cpu_seconds(pid: int) -> float | None:
    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        text = stat_path.read_text(encoding="utf-8", errors="replace")
        fields = text.rsplit(") ", 1)[1].split()
        ticks = int(fields[11]) + int(fields[12])
        ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    except (IndexError, KeyError, OSError, ValueError):
        return None
    return float(ticks) / float(ticks_per_second)


def _pid_symbolica_mappings(pid: int) -> str | None:
    maps_path = Path("/proc") / str(pid) / "maps"
    try:
        lines = maps_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    paths: dict[str, bool] = {}
    for line in lines:
        lower_line = line.lower()
        if "core.abi3.so" not in lower_line or "ymbolica" not in lower_line:
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        mapped_path = parts[5]
        deleted = mapped_path.endswith(" (deleted)")
        if deleted:
            mapped_path = mapped_path[: -len(" (deleted)")]
        paths[mapped_path] = paths.get(mapped_path, False) or deleted
    if not paths:
        return None
    details = ",".join(
        f"{path}{':deleted' if deleted else ':live'}"
        for path, deleted in sorted(paths.items())
    )
    deleted_count = sum(1 for deleted in paths.values() if deleted)
    return f"pid={pid} mapped_symbolica={len(paths)} deleted_mappings={deleted_count} details={details}"


def _symbolica_mapping_summary(pids: list[int]) -> str | None:
    rows = [
        row
        for pid in pids
        if (row := _pid_symbolica_mappings(pid)) is not None
    ]
    return "\n".join(rows) if rows else None


def _held_cache_lock_data(pids: list[int]) -> list[tuple[int, list[Path]]]:
    rows: list[tuple[int, list[Path]]] = []
    for pid in pids:
        fd_dir = Path("/proc") / str(pid) / "fd"
        if not fd_dir.is_dir():
            continue
        locks: list[Path] = []
        try:
            fd_paths = sorted(fd_dir.iterdir(), key=lambda path: path.name)
        except (FileNotFoundError, NotADirectoryError, PermissionError, ProcessLookupError):
            continue
        for fd_path in fd_paths:
            try:
                target = fd_path.resolve(strict=True)
            except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
                continue
            target_text = str(target)
            if "/cache/" not in target_text or not target_text.endswith(".lock"):
                continue
            locks.append(target)
        if locks:
            rows.append((pid, locks))
    return rows


def _held_cache_locks(pids: list[int], limit: int) -> list[str]:
    rows: list[str] = []
    for pid, locks in _held_cache_lock_data(pids):
        lock_text = ",".join(str(lock) for lock in locks[: max(int(limit), 1)])
        rows.append(f"pid={pid} locks={lock_text}")
    return rows


def _formula_lock_kind_and_digest(lock_path: Path) -> tuple[str, str] | None:
    match = re.match(
        r"^(?P<kind>chain_rule|regular_taylor)_(?P<digest>[0-9a-f]+)\.json\.lock$",
        lock_path.name,
    )
    if match is None:
        return None
    return match.group("kind"), match.group("digest")


def _formula_artifact_summary(formula_cache_dir: Path, kind: str, digest: str) -> str:
    """Return a compact summary of cache artifacts for one active formula."""

    metadata_path = formula_cache_dir / f"{kind}_{digest}.json"
    fields = [
        f"metadata_present={metadata_path.is_file()}",
    ]
    if kind != "chain_rule":
        return " ".join(fields)

    manifest_path = formula_cache_dir / f"{kind}_{digest}.expr_manifest.json"
    fields.append(f"expr_manifest_present={manifest_path.is_file()}")
    if not manifest_path.is_file():
        return " ".join(fields)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        fields.append("expr_manifest_readable=false")
        return " ".join(fields)
    names = [str(name) for name in data.get("expression_cache_files", [])]
    sizes: list[tuple[str, int]] = []
    present = 0
    for name in names:
        path = manifest_path.parent / name
        try:
            size = path.stat().st_size
        except OSError:
            continue
        present += 1
        sizes.append((name, int(size)))
    fields.append(f"expr_files_present={present}/{len(names)}")
    if sizes:
        total_bytes = sum(size for _name, size in sizes)
        fields.append(f"expr_files_total_gib={total_bytes / 1024.0 ** 3:.3f}")
        # The live 3L chain-rule sidecars are ordered by output coefficient.
        # The final bucket can be dramatically larger than earlier buckets,
        # so expose it when diagnosing an apparently stale load marker.
        tail_sizes = sizes[-64:]
        tail_bytes = sum(size for _name, size in tail_sizes)
        largest_name, largest_size = max(tail_sizes, key=lambda item: item[1])
        fields.append(f"expr_tail64_gib={tail_bytes / 1024.0 ** 3:.3f}")
        fields.append(f"expr_tail64_largest={largest_name}:{largest_size / 1024.0 ** 3:.3f}GiB")
    return " ".join(fields)


def _lock_owner_pids(lock_path: Path) -> list[int]:
    try:
        stat_result = lock_path.stat()
        lock_id = f"{os.major(stat_result.st_dev):02x}:{os.minor(stat_result.st_dev):02x}:{stat_result.st_ino}"
        text = Path("/proc/locks").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    owners: list[int] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[5] != lock_id:
            continue
        try:
            owners.append(int(fields[4]))
        except ValueError:
            continue
    return sorted(set(owners))


def _matching_log_lines(log_path: Path, marker: str, digest: str) -> list[str]:
    text = _tail_text(log_path)
    return [
        line
        for line in text.splitlines()
        if marker in line and digest in line
    ]


def _active_formula_progress(
    running_tasks: list[dict[str, Any]],
    pids: list[int],
    lock_limit: int,
    formula_cache_dir: Path,
) -> list[str]:
    task_by_pid = {
        int(task["pid"]): task
        for task in running_tasks
        if task.get("pid")
    }
    rows: list[str] = []
    seen_locks: set[str] = set()
    for pid, locks in _held_cache_lock_data(pids):
        for lock_path in locks[: max(int(lock_limit), 1)]:
            lock_key = str(lock_path)
            if lock_key in seen_locks:
                continue
            seen_locks.add(lock_key)
            owner_pids = _lock_owner_pids(lock_path)
            owner_pid = owner_pids[0] if owner_pids else pid
            task = task_by_pid.get(owner_pid) or task_by_pid.get(pid, {})
            task_id = task.get("task_id", "unknown")
            log_path = Path(str(task.get("log_path", "")))
            parsed = _formula_lock_kind_and_digest(lock_path)
            if parsed is None:
                rows.append(
                    f"pid={pid} owner_pid={owner_pid} task={task_id} lock={lock_path.name} "
                    "formula=unknown"
                )
                continue
            kind, digest = parsed
            marker = "[fsd-chain-rule]" if kind == "chain_rule" else "[fsd-subtraction]"
            lines = _matching_log_lines(log_path, marker, digest)
            artifact_summary = _formula_artifact_summary(formula_cache_dir, kind, digest)
            row = (
                f"pid={pid} owner_pid={owner_pid} task={task_id} formula={kind} digest={digest} "
                f"lock={lock_path.name} log={log_path.name if log_path.name else 'unknown'} "
                f"owner_rss_gib={_pid_rss_gib(owner_pid):.3f} "
                f"owner_cpu_time={_human_duration(_pid_cpu_seconds(owner_pid))} "
                f"log_age={_human_duration(_path_age_seconds(log_path))} "
                f"artifacts={artifact_summary}"
            )
            if lines:
                row = f"{row} latest={lines[-1]}"
                if kind == "chain_rule":
                    trend = _chain_rule_trend(
                        "\n".join(lines),
                        latest_age_seconds=_path_age_seconds(log_path),
                    )
                    if trend is not None:
                        row = f"{row} trend={trend}"
            else:
                row = f"{row} latest=none"
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-path", type=Path, default=Path("docs/cluster_cache_3l_shards_status.json"))
    parser.add_argument("--watchdog-log", type=Path, default=Path("docs/cluster_cache_3l_watchdog.log"))
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("docs/cache_shards/logs/triple-box-direct"),
    )
    parser.add_argument("--report-dir", type=Path, default=Path("docs/cache_shards/reports"))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--drain-file", type=Path, default=Path("docs/cluster_cache_3l_drain.order"))
    parser.add_argument("--stop-file", type=Path, default=Path("stop.order"))
    parser.add_argument(
        "--formula-stop-watcher-log",
        type=Path,
        default=Path("docs/cluster_cache_3l_formula_stop_watcher.log"),
    )
    parser.add_argument(
        "--checkpoint-guard-pid-file",
        type=Path,
        default=Path("docs/cluster_cache_3l_checkpoint_guard.pid"),
    )
    parser.add_argument(
        "--checkpoint-guard-log",
        type=Path,
        default=Path("docs/cluster_cache_3l_checkpoint_guard.log"),
    )
    parser.add_argument(
        "--drain-release-pid-file",
        type=Path,
        default=Path("docs/cluster_cache_3l_drain_release.pid"),
    )
    parser.add_argument(
        "--drain-release-log",
        type=Path,
        default=Path("docs/cluster_cache_3l_drain_release.log"),
    )
    parser.add_argument(
        "--memory-guard-pid-file",
        type=Path,
        default=Path("docs/cluster_cache_3l_memory_guard.pid"),
    )
    parser.add_argument(
        "--memory-guard-log",
        type=Path,
        default=Path("docs/cluster_cache_3l_memory_guard.log"),
    )
    parser.add_argument(
        "--status-monitor-pid-file",
        type=Path,
        default=Path("docs/cluster_cache_3l_status_monitor.pid"),
    )
    parser.add_argument(
        "--status-monitor-log",
        type=Path,
        default=Path("docs/cluster_cache_3l_status_monitor.log"),
    )
    parser.add_argument(
        "--exit-relaunch-pid-file",
        type=Path,
        default=Path("docs/cluster_cache_3l_exit_relaunch.pid"),
    )
    parser.add_argument(
        "--exit-relaunch-log",
        type=Path,
        default=Path("docs/cluster_cache_3l_exit_relaunch.log"),
    )
    parser.add_argument(
        "--post-formula-resume-pid-file",
        type=Path,
        default=Path("docs/cluster_cache_3l_post_formula_resume.pid"),
    )
    parser.add_argument(
        "--post-formula-resume-log",
        type=Path,
        default=Path("docs/cluster_cache_3l_post_formula_resume.log"),
    )
    parser.add_argument(
        "--checkpoint-restart-pid-file",
        type=Path,
        default=Path("docs/cluster_cache_3l_checkpoint_restart_2f5ed.pid"),
    )
    parser.add_argument(
        "--checkpoint-restart-log",
        type=Path,
        default=Path("docs/cluster_cache_3l_checkpoint_restart_2f5ed.log"),
    )
    parser.add_argument("--top-processes", type=int, default=10)
    parser.add_argument("--locks-per-process", type=int, default=4)
    parser.add_argument("--memory-limit-gib", type=float, default=800.0)
    args = parser.parse_args()

    status = _read_json(args.status_path)
    print("cache_run_status")
    if status:
        print(
            "shards "
            f"phase={status.get('phase')} completed={status.get('completed')} "
            f"running={status.get('running')} pending={status.get('pending')} "
            f"deferred={status.get('deferred_pending', 0)} "
            f"skipped={status.get('skipped')} failed={status.get('failed')} "
            f"jobs={status.get('jobs')} task_count={status.get('task_count')} "
            f"time_utc={status.get('time_utc')}"
        )
        failures = status.get("failures") or []
        if failures:
            print(f"failures {json.dumps(failures, sort_keys=True)}")
    else:
        print(f"shards unavailable path={args.status_path}")

    watchdog_tail = _tail_text(args.watchdog_log, max_bytes=16_000).splitlines()
    if watchdog_tail:
        latest_watchdog = watchdog_tail[-1]
        print(f"watchdog_latest {latest_watchdog}")
        limit_gib = float(args.memory_limit_gib)
        match = re.search(r"\brss=(?P<rss>[0-9.]+)\s+GiB\b", latest_watchdog)
        if match is not None:
            rss_gib = float(match.group("rss"))
            headroom_gib = limit_gib - rss_gib
            print(
                "watchdog_headroom "
                f"limit={limit_gib:.1f}GiB used={rss_gib:.1f}GiB "
                f"headroom={headroom_gib:.1f}GiB used_fraction={rss_gib / limit_gib:.3f}"
            )
        trend = _watchdog_rss_trend(watchdog_tail, limit_gib)
        if trend is not None:
            print(f"watchdog_rss_trend {trend}")
    else:
        print(f"watchdog_latest unavailable path={args.watchdog_log}")

    print(f"drain_file path={args.drain_file} present={args.drain_file.exists()}")
    print(f"stop_file path={args.stop_file} present={args.stop_file.exists()}")
    stop_watcher_count = _process_count_matching("stop_cache_run_after_chain_rule_digest.py")
    print(f"formula_stop_watcher_processes count={stop_watcher_count}")
    formula_stop_line = _latest_nonempty_line(args.formula_stop_watcher_log)
    if formula_stop_line:
        print(
            "formula_stop_watcher_latest "
            f"path={args.formula_stop_watcher_log} line={formula_stop_line}"
        )
    else:
        print(f"formula_stop_watcher_latest unavailable path={args.formula_stop_watcher_log}")
    checkpoint_guard_pid = _read_pid(args.checkpoint_guard_pid_file)
    print(
        "checkpoint_guard "
        f"pid={checkpoint_guard_pid if checkpoint_guard_pid is not None else 'unknown'} "
        f"alive={_pid_alive(checkpoint_guard_pid)} "
        f"pid_file={args.checkpoint_guard_pid_file}"
    )
    checkpoint_guard_line = _latest_nonempty_line(args.checkpoint_guard_log)
    if checkpoint_guard_line:
        print(
            "checkpoint_guard_latest "
            f"path={args.checkpoint_guard_log} line={checkpoint_guard_line}"
        )
    else:
        print(f"checkpoint_guard_latest unavailable path={args.checkpoint_guard_log}")
    drain_release_pid = _read_pid(args.drain_release_pid_file)
    print(
        "drain_release_watcher "
        f"pid={drain_release_pid if drain_release_pid is not None else 'unknown'} "
        f"alive={_pid_alive(drain_release_pid)} "
        f"pid_file={args.drain_release_pid_file}"
    )
    drain_release_line = _latest_nonempty_line(args.drain_release_log)
    if drain_release_line:
        print(
            "drain_release_latest "
            f"path={args.drain_release_log} line={drain_release_line}"
        )
    else:
        print(f"drain_release_latest unavailable path={args.drain_release_log}")
    memory_guard_pid = _read_pid(args.memory_guard_pid_file)
    print(
        "memory_guard "
        f"pid={memory_guard_pid if memory_guard_pid is not None else 'unknown'} "
        f"alive={_pid_alive(memory_guard_pid)} "
        f"pid_file={args.memory_guard_pid_file}"
    )
    memory_guard_line = _latest_nonempty_line(args.memory_guard_log)
    if memory_guard_line:
        print(
            "memory_guard_latest "
            f"path={args.memory_guard_log} line={memory_guard_line}"
        )
    else:
        print(f"memory_guard_latest unavailable path={args.memory_guard_log}")
    status_monitor_pid = _read_pid(args.status_monitor_pid_file)
    print(
        "status_monitor "
        f"pid={status_monitor_pid if status_monitor_pid is not None else 'unknown'} "
        f"alive={_pid_alive(status_monitor_pid)} "
        f"pid_file={args.status_monitor_pid_file}"
    )
    status_monitor_line = _latest_nonempty_line(args.status_monitor_log)
    if status_monitor_line:
        print(
            "status_monitor_latest "
            f"path={args.status_monitor_log} line={status_monitor_line}"
        )
    else:
        print(f"status_monitor_latest unavailable path={args.status_monitor_log}")
    exit_relaunch_pid = _read_pid(args.exit_relaunch_pid_file)
    print(
        "exit_relaunch_watcher "
        f"pid={exit_relaunch_pid if exit_relaunch_pid is not None else 'unknown'} "
        f"alive={_pid_alive(exit_relaunch_pid)} "
        f"pid_file={args.exit_relaunch_pid_file}"
    )
    exit_relaunch_line = _latest_nonempty_line(args.exit_relaunch_log)
    if exit_relaunch_line:
        print(
            "exit_relaunch_latest "
            f"path={args.exit_relaunch_log} line={exit_relaunch_line}"
        )
    else:
        print(f"exit_relaunch_latest unavailable path={args.exit_relaunch_log}")
    post_formula_resume_pid = _read_pid(args.post_formula_resume_pid_file)
    post_formula_env = _pid_environ(post_formula_resume_pid)
    post_formula_log = Path(
        post_formula_env.get(
            "FSD_POST_FORMULA_RESUME_LOG",
            str(args.post_formula_resume_log),
        )
    )
    post_formula_fields = [
        "post_formula_resume_watcher",
        f"pid={post_formula_resume_pid if post_formula_resume_pid is not None else 'unknown'}",
        f"alive={_pid_alive(post_formula_resume_pid)}",
        f"pid_file={args.post_formula_resume_pid_file}",
    ]
    if digest := post_formula_env.get("FSD_CHAIN_RULE_PROTECTED_DIGEST"):
        post_formula_fields.append(f"digest={digest}")
    if interval := post_formula_env.get("FSD_POST_FORMULA_RESUME_INTERVAL_SECONDS"):
        post_formula_fields.append(f"interval_seconds={interval}")
    for env_name, field_name in (
        ("FSD_SYMBOLICA_EVALUATOR_ITERATIONS", "iterations"),
        ("FSD_SYMBOLICA_EVALUATOR_CPE_ITERATIONS", "cpe_iterations"),
        ("FSD_SYMBOLICA_MAX_HORNER_SCHEME_VARIABLES", "max_horner_vars"),
        ("FSD_SYMBOLICA_MAX_COMMON_PAIR_CACHE_ENTRIES", "cpe_cache_entries"),
        ("FSD_SYMBOLICA_MAX_COMMON_PAIR_DISTANCE", "cpe_distance"),
        ("FSD_CACHE_WATCHDOG_LIMIT_GB", "watchdog_gib"),
    ):
        if value := post_formula_env.get(env_name):
            post_formula_fields.append(f"{field_name}={value}")
    print(" ".join(post_formula_fields))
    post_formula_resume_line = _latest_nonempty_line(post_formula_log)
    if post_formula_resume_line:
        print(
            "post_formula_resume_latest "
            f"path={post_formula_log} line={post_formula_resume_line}"
        )
    else:
        print(f"post_formula_resume_latest unavailable path={post_formula_log}")
    if post_formula_log != args.post_formula_resume_log:
        fallback_line = _latest_nonempty_line(args.post_formula_resume_log)
        if fallback_line:
            print(
                "post_formula_resume_fallback_latest "
                f"path={args.post_formula_resume_log} line={fallback_line}"
            )
        else:
            print(
                "post_formula_resume_fallback_latest "
                f"unavailable path={args.post_formula_resume_log}"
            )
    _print_pid_log_status(
        "checkpoint_restart_watcher",
        args.checkpoint_restart_pid_file,
        args.checkpoint_restart_log,
    )
    checkpoint_restart_pid = _read_pid(args.checkpoint_restart_pid_file)
    checkpoint_command = _pid_cmdline(checkpoint_restart_pid)
    checkpoint_fields = []
    for option, field_name in (
        ("--digest", "digest"),
        ("--watchdog-limit-gb", "watchdog_gib"),
        ("--evaluator-cores", "evaluator_cores"),
        ("--evaluator-iterations", "iterations"),
        ("--evaluator-cpe-iterations", "cpe_iterations"),
        ("--max-horner-scheme-variables", "max_horner_vars"),
        ("--max-common-pair-cache-entries", "cpe_cache_entries"),
        ("--max-common-pair-distance", "cpe_distance"),
    ):
        if value := _command_arg(checkpoint_command, option):
            checkpoint_fields.append(f"{field_name}={value}")
    if checkpoint_fields:
        print("checkpoint_restart_tuning " + " ".join(checkpoint_fields))
    _print_active_guard_processes()
    print(f"cache_size {_human_bytes(_du_bytes(args.cache_dir))}")
    report_counts = _phase_report_counts(args.report_dir)
    if report_counts is not None:
        print(f"phase_report_files {report_counts}")

    running_tasks = status.get("running_tasks") if status else []
    if not isinstance(running_tasks, list):
        running_tasks = []
    scheduler_raw_pids = [int(task["pid"]) for task in running_tasks if task.get("pid")]
    standalone_tasks = _standalone_cache_workers(set(scheduler_raw_pids))
    all_running_tasks = [*running_tasks, *standalone_tasks]
    raw_pids = [int(task["pid"]) for task in all_running_tasks if task.get("pid")]
    pids = [pid for pid in raw_pids if _pid_is_live_worker(pid)]
    top_lock_pids = sorted(pids, key=_pid_rss_kib, reverse=True)[: max(int(args.top_processes), 1)]
    progress_rows = _active_formula_progress(
        all_running_tasks,
        top_lock_pids,
        args.locks_per_process,
        args.cache_dir / "subtraction_formulae",
    )
    active_chain_rule_rows = [row for row in progress_rows if " formula=chain_rule " in row]

    if active_chain_rule_rows:
        print("latest_chain_rule_log source=active_lock")
        print(active_chain_rule_rows[0])
    else:
        path, line = _latest_matching_line(
            args.log_dir,
            "[fsd-chain-rule]",
            include=(
                "build_start",
                "symbols_done",
                "compose_start",
                "compose_progress",
                "series_mul_start",
                "series_mul_progress",
                "series_mul_done",
                "expressions_done",
                "expression_binary_sidecar_start",
                "expression_binary_sidecar_progress",
                "expression_binary_sidecar_done",
                "expression_binary_sidecar_load_start",
                "expression_binary_sidecar_load_progress",
                "expression_binary_sidecar_load_done",
                "evaluator_start",
                "evaluator_done",
                "metadata_write_start",
                "metadata_write_done",
            ),
            exclude=("global_cold_lock_deferred", "formula_lock_deferred"),
        )
        if line:
            print(f"latest_chain_rule_log file={path.name if path else 'unknown'}")
            print(line)
            if path is not None:
                trend = _chain_rule_trend(
                    _tail_text(path),
                    latest_age_seconds=_path_age_seconds(path),
                )
                if trend is not None:
                    print(f"chain_rule_trend {trend}")
        else:
            print("latest_chain_rule_log none")

    path, line = _latest_matching_line(
        args.log_dir,
        ("symbolica::evaluate::tree", "FSD evaluator phase", "FSD evaluator progress"),
    )
    if line:
        age = _path_age_seconds(path) if path is not None else None
        age_text = f" age={_human_duration(age)}" if age is not None else ""
        print(f"latest_symbolica_evaluator_log file={path.name if path else 'unknown'}{age_text}")
        print(line)
    else:
        print("latest_symbolica_evaluator_log none")

    path, line = _latest_matching_line(args.log_dir, "[fsd-subtraction]")
    if line:
        print(f"latest_subtraction_log file={path.name if path else 'unknown'}")
        print(line)
    else:
        print("latest_subtraction_log none")

    cpu_summary = _worker_cpu_summary(pids)
    if cpu_summary:
        print(f"worker_cpu {cpu_summary}")
        thread_cpu_summary = _worker_thread_cpu_summary(pids)
        if thread_cpu_summary:
            print(f"worker_thread_cpu {thread_cpu_summary}")
        symbolica_mapping_summary = _symbolica_mapping_summary(pids)
        if symbolica_mapping_summary:
            print("worker_symbolica_mappings")
            print(symbolica_mapping_summary)
        if standalone_tasks:
            print(f"standalone_cache_workers count={len(standalone_tasks)}")
            for task in standalone_tasks:
                print(
                    f"pid={task['pid']} task={task.get('task_id', 'unknown')} "
                    f"log={Path(str(task.get('log_path', ''))).name or 'unknown'}"
                )
        if len(pids) != len(raw_pids):
            print(f"worker_live_filter raw_worker_count={len(raw_pids)} live_worker_count={len(pids)}")
        if status:
            scheduler_running = _safe_int(status.get("running"), len(raw_pids))
            pending = _safe_int(status.get("pending"), 0)
            deferred = _safe_int(status.get("deferred_pending"), 0)
            skipped = _safe_int(status.get("skipped"), 0)
            scheduler_live = sum(1 for pid in scheduler_raw_pids if pid in set(pids))
            stale_running = max(scheduler_running - scheduler_live, 0)
            standalone_running = len([pid for pid in pids if pid not in set(scheduler_raw_pids)])
            print(
                "effective_work "
                f"live_running_tasks={len(pids)} "
                f"stale_running_tasks={stale_running} "
                f"standalone_running_tasks={standalone_running} "
                f"pending_not_started={pending} "
                f"deferred_pending={deferred} "
                f"runnable_pending={max(pending - deferred, 0)} "
                f"not_currently_worked={max(pending + stale_running - standalone_running, 0)} "
                f"skipped_resume_hits={skipped}"
            )
    ps_text = _ps_for_pids(pids, args.top_processes)
    if ps_text:
        print("top_running_processes_by_rss")
        print(ps_text)
    top_lock_pids = sorted(pids, key=_pid_rss_kib, reverse=True)[: max(int(args.top_processes), 1)]
    lock_rows = _held_cache_locks(top_lock_pids, args.locks_per_process)
    if lock_rows:
        print("held_cache_locks")
        print("\n".join(lock_rows))
    print(f"active_formula_lock_count={len(progress_rows)}")
    if progress_rows:
        print("active_formula_locks")
        print("\n".join(progress_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
