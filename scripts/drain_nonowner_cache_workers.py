#!/usr/bin/env python3
"""Drain shard launches and stop non-owner cache workers.

This is intended for memory protection during very large cold formula builds:
keep the CPU-bound formula owner processes alive, but stop workers that are
only waiting on those formula locks.  The shard runner will leave stopped
tasks pending while the drain marker exists, and the normal resume path will
pick them back up later.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAIN_FILE = ROOT / "docs" / "cluster_cache_3l_drain.order"
DEFAULT_CHILD_PID_FILE = ROOT / "docs" / "cluster_cache_3l_watchdog_child.pid"
DEFAULT_LOG_FILE = ROOT / "docs" / "cluster_cache_3l_drain_waiters.log"


def _log(path: Path, message: str) -> None:
    line = f"{datetime.now(timezone.utc).isoformat()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _status_text() -> str:
    return subprocess.check_output(
        [sys.executable, str(ROOT / "scripts" / "report_cache_run_status.py"), "--top-processes", "300"],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )


def _active_owner_pids(status: str) -> set[int]:
    owners: set[int] = set()
    for match in re.finditer(r"\bowner_pid=(\d+)\b", status):
        owners.add(int(match.group(1)))
    return owners


def _process_rows() -> list[dict[str, object]]:
    output = subprocess.check_output(
        ["ps", "-eo", "pid=,ppid=,stat=,pcpu=,rss=,args="],
        text=True,
    )
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        parts = line.split(None, 5)
        if len(parts) != 6:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            pcpu = float(parts[3])
            rss_kib = int(parts[4])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "stat": parts[2],
                "pcpu": pcpu,
                "rss_kib": rss_kib,
                "args": parts[5],
            }
        )
    return rows


def _descendant_pids(rows: list[dict[str, object]], root_pid: int) -> set[int]:
    children: dict[int, list[int]] = {}
    for row in rows:
        children.setdefault(int(row["ppid"]), []).append(int(row["pid"]))
    descendants: set[int] = set()
    stack = list(children.get(root_pid, []))
    while stack:
        pid = stack.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        stack.extend(children.get(pid, []))
    return descendants


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_targets(targets: list[int], sig: signal.Signals) -> None:
    for pid in targets:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drain-file", type=Path, default=DEFAULT_DRAIN_FILE)
    parser.add_argument("--child-pid-file", type=Path, default=DEFAULT_CHILD_PID_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument(
        "--protect-pid",
        type=int,
        action="append",
        default=[],
        help=(
            "PID to protect.  When supplied at least once, these PIDs are the "
            "only protected workers; otherwise active formula owner PIDs from "
            "the status probe are protected."
        ),
    )
    parser.add_argument("--busy-cpu-threshold", type=float, default=10.0)
    parser.add_argument("--interrupt-grace-seconds", type=float, default=20.0)
    parser.add_argument("--terminate-grace-seconds", type=float, default=10.0)
    parser.add_argument("--kill-grace-seconds", type=float, default=5.0)
    parser.add_argument(
        "--protected-digest",
        default="",
        help="Optional formula digest to record in the drain marker for later safe release.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    log_file = args.log_file if args.log_file.is_absolute() else ROOT / args.log_file
    drain_file = args.drain_file if args.drain_file.is_absolute() else ROOT / args.drain_file
    child_pid_file = (
        args.child_pid_file if args.child_pid_file.is_absolute() else ROOT / args.child_pid_file
    )
    runner_pid = _read_pid(child_pid_file)
    if runner_pid is None or not _alive(runner_pid):
        _log(log_file, f"runner_not_alive pid={runner_pid}")
        return 1

    status = _status_text()
    owners = set(int(pid) for pid in args.protect_pid) or _active_owner_pids(status)
    if not owners:
        _log(log_file, "no_active_formula_owners_found; refusing to stop workers")
        return 1

    rows = _process_rows()
    descendants = _descendant_pids(rows, runner_pid)
    candidates: list[dict[str, object]] = []
    skipped_busy: list[dict[str, object]] = []
    for row in rows:
        pid = int(row["pid"])
        args_text = str(row["args"])
        if pid not in descendants or pid in owners or "FSD.py cache" not in args_text:
            continue
        if float(row["pcpu"]) >= float(args.busy_cpu_threshold):
            skipped_busy.append(row)
            continue
        candidates.append(row)

    total_rss_gib = sum(int(row["rss_kib"]) for row in candidates) / (1024.0 * 1024.0)
    owner_text = ",".join(str(pid) for pid in sorted(owners))
    target_text = ",".join(str(int(row["pid"])) for row in candidates)
    _log(
        log_file,
        (
            f"planned_drain runner_pid={runner_pid} owners={owner_text} "
            f"targets={len(candidates)} target_rss_gib={total_rss_gib:.3f} "
            f"skipped_busy={len(skipped_busy)}"
        ),
    )
    if skipped_busy:
        skipped_text = ",".join(
            f"{int(row['pid'])}:{float(row['pcpu']):.1f}" for row in skipped_busy
        )
        _log(log_file, f"skipped_busy_workers {skipped_text}")
    if not candidates:
        _log(log_file, "no_nonowner_workers_to_stop")
        return 0

    if args.dry_run:
        _log(log_file, f"dry_run targets={target_text}")
        return 0

    drain_file.parent.mkdir(parents=True, exist_ok=True)
    protected_digest = str(args.protected_digest).strip()
    drain_text = (
        f"drain non-owner cache workers requested at {datetime.now(timezone.utc).isoformat()}\n"
    )
    if protected_digest:
        drain_text += "reason=memory_guard_nonowner_workers\n"
        drain_text += f"protected_digest={protected_digest}\n"
    drain_file.write_text(drain_text, encoding="utf-8")
    _log(log_file, f"drain_file_written path={drain_file}")

    target_pids = [int(row["pid"]) for row in candidates]
    _log(log_file, f"sending_sigint targets={target_text}")
    _signal_targets(target_pids, signal.SIGINT)
    deadline = time.monotonic() + max(float(args.interrupt_grace_seconds), 0.0)
    while time.monotonic() < deadline and any(_alive(pid) for pid in target_pids):
        time.sleep(0.5)

    remaining = [pid for pid in target_pids if _alive(pid)]
    if remaining:
        _log(log_file, "sending_sigterm targets=" + ",".join(str(pid) for pid in remaining))
        _signal_targets(remaining, signal.SIGTERM)
        deadline = time.monotonic() + max(float(args.terminate_grace_seconds), 0.0)
        while time.monotonic() < deadline and any(_alive(pid) for pid in remaining):
            time.sleep(0.5)

    remaining = [pid for pid in target_pids if _alive(pid)]
    if remaining:
        _log(log_file, "sending_sigkill targets=" + ",".join(str(pid) for pid in remaining))
        _signal_targets(remaining, signal.SIGKILL)
        deadline = time.monotonic() + max(float(args.kill_grace_seconds), 0.0)
        while time.monotonic() < deadline and any(_alive(pid) for pid in remaining):
            time.sleep(0.2)

    remaining = [pid for pid in target_pids if _alive(pid)]
    if remaining:
        _log(log_file, "remaining_after_sigkill targets=" + ",".join(str(pid) for pid in remaining))
        return 2
    _log(log_file, f"stopped_nonowner_workers count={len(target_pids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
