#!/usr/bin/env python3
"""Run a command with process-tree RSS and wall-time watchdogs.

macOS in this workspace rejects POSIX address-space limits for the Python
process used by pySecDec, so this wrapper enforces a practical cap by polling
the resident set size of the launched process tree.  It can also enforce a
wall-time timeout and watch for a local ``stop.order`` file.  If any stop
condition is met, the whole process group is terminated from the wrapper
process that owns the child, avoiding a separate external ``kill`` command.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    psutil = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-gb", type=float, required=True)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help="Optional wall-time limit for the launched command.",
    )
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=Path("stop.order"),
        help=(
            "File watched for manual termination.  Default: ./stop.order in "
            "the wrapper current working directory."
        ),
    )
    parser.add_argument(
        "--interrupt-grace-seconds",
        type=float,
        default=60.0,
        help="Seconds to wait after SIGINT before escalating to SIGTERM.",
    )
    parser.add_argument(
        "--kill-grace-seconds",
        type=float,
        default=5.0,
        help="Seconds to wait after SIGTERM before sending SIGKILL.",
    )
    parser.add_argument("--pid-file", type=Path)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command after --")
    return args


def _terminate_process_group(
    proc: subprocess.Popen[bytes],
    log_handle,
    reason: str,
    interrupt_grace_seconds: float,
    kill_grace_seconds: float,
) -> None:
    """Interrupt, then terminate, the child tree and all child process groups."""
    target_pids = _target_process_tree_pids(proc.pid)
    _log(log_handle, f"{reason}; interrupting process tree pids={target_pids}")
    _signal_process_tree(target_pids, signal.SIGINT, log_handle)
    interrupt_deadline = time.monotonic() + max(float(interrupt_grace_seconds), 0.0)
    while (
        (proc.poll() is None or _live_pids(target_pids))
        and time.monotonic() < interrupt_deadline
    ):
        time.sleep(0.5)
    if proc.poll() is not None and not _live_pids(target_pids):
        _log(log_handle, "process tree exited after SIGINT")
        return

    live_after_interrupt = _live_pids(target_pids)
    _log(log_handle, f"process tree still alive; sending SIGTERM pids={live_after_interrupt}")
    _signal_process_tree(live_after_interrupt or target_pids, signal.SIGTERM, log_handle)
    time.sleep(max(float(kill_grace_seconds), 0.0))
    live_after_term = _live_pids(target_pids)
    if proc.poll() is None or live_after_term:
        _log(log_handle, f"process tree still alive; sending SIGKILL pids={live_after_term}")
        _signal_process_tree(live_after_term or target_pids, signal.SIGKILL, log_handle)


def _target_process_tree_pids(root_pid: int) -> list[int]:
    """Return known live process ids in the watched tree."""
    _rss_kb, pids = _tree_rss_kb(root_pid)
    return sorted(set([root_pid, *pids]))


def _pid_is_live(pid: int) -> bool:
    status_path = Path("/proc") / str(pid) / "status"
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("State:"):
                parts = line.split()
                return len(parts) < 2 or parts[1] != "Z"
    except OSError:
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _live_pids(pids: list[int]) -> list[int]:
    return [pid for pid in sorted(set(pids)) if _pid_is_live(pid)]


def _signal_process_tree(pids: list[int], sig: signal.Signals, log_handle) -> None:
    pgids: set[int] = set()
    live = _live_pids(pids)
    for pid in live:
        try:
            pgids.add(os.getpgid(pid))
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            _log(log_handle, f"could not read process group for pid={pid}: {exc}")
    for pgid in sorted(pgids):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            _log(log_handle, f"could not send {sig.name} to process group {pgid}: {exc}")
    for pid in live:
        try:
            os.kill(pid, sig)
        except ProcessLookupError:
            continue
        except PermissionError as exc:
            _log(log_handle, f"could not send {sig.name} to pid={pid}: {exc}")


def _tree_rss_kb_psutil(root_pid: int) -> tuple[int | None, list[int]] | None:
    """Return process-tree RSS through psutil when available."""
    if psutil is None:
        return None
    try:
        root = psutil.Process(root_pid)
        processes = [root, *root.children(recursive=True)]
    except (psutil.Error, OSError):
        return 0, []
    total = 0
    pids: list[int] = []
    for proc in processes:
        try:
            total += int(proc.memory_info().rss)
            pids.append(int(proc.pid))
        except (psutil.Error, OSError):
            continue
    return total // 1024, sorted(set(pids))


def _process_table_from_ps() -> dict[int, tuple[int, int]] | None:
    """Return pid -> (ppid, rss_kb) from ps, if available in the sandbox."""
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,rss="],
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    table: dict[int, tuple[int, int]] = {}
    for line in output.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            pid, ppid, rss = (int(part) for part in parts)
        except ValueError:
            continue
        table[pid] = (ppid, rss)
    return table


def _tree_rss_kb(root_pid: int) -> tuple[int | None, list[int]]:
    """Return RSS in KiB for root_pid and descendants."""
    psutil_result = _tree_rss_kb_psutil(root_pid)
    if psutil_result is not None:
        return psutil_result

    table = _process_table_from_ps()
    if table is None:
        return None, [root_pid]
    children: dict[int, list[int]] = {}
    for pid, (ppid, _rss) in table.items():
        children.setdefault(ppid, []).append(pid)

    stack = [root_pid]
    seen: set[int] = set()
    total = 0
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        ppid_rss = table.get(pid)
        if ppid_rss is not None:
            total += ppid_rss[1]
        stack.extend(children.get(pid, []))
    return total, sorted(seen)


def _log(handle, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    if handle is not None:
        print(line, file=handle, flush=True)


def main() -> int:
    args = _parse_args()
    limit_kb = int(args.limit_gb * 1024.0 * 1024.0)
    poll = max(float(args.poll_seconds), 0.5)
    log_handle = args.log_file.open("a") if args.log_file else None
    stop_file = args.stop_file.expanduser()
    if not stop_file.is_absolute():
        stop_file = Path.cwd() / stop_file
    if stop_file.exists():
        stop_file.unlink()
        _log(log_handle, f"removed stale stop file {stop_file}")
    proc = subprocess.Popen(args.command, start_new_session=True)
    start_time = time.monotonic()
    if args.pid_file:
        args.pid_file.write_text(f"{proc.pid}\n")
    timeout_text = (
        f" timeout={args.timeout_seconds:g}s"
        if args.timeout_seconds is not None
        else ""
    )
    _log(log_handle, f"started pid={proc.pid} limit={args.limit_gb:g} GiB{timeout_text}")
    _log(log_handle, f"watching stop file {stop_file}")

    try:
        while True:
            rc = proc.poll()
            elapsed = time.monotonic() - start_time
            rss_kb, pids = _tree_rss_kb(proc.pid)
            if rss_kb is None:
                rss_text = "rss=unavailable"
            else:
                rss_gb = rss_kb / (1024.0 * 1024.0)
                rss_text = f"rss={rss_gb:.3f} GiB"
            _log(log_handle, f"elapsed={elapsed:.1f}s {rss_text} processes={len(pids)}")
            if rss_kb is not None and rss_kb > limit_kb:
                _terminate_process_group(
                    proc,
                    log_handle,
                    "RSS limit exceeded",
                    args.interrupt_grace_seconds,
                    args.kill_grace_seconds,
                )
                return 137
            if args.timeout_seconds is not None and elapsed > float(args.timeout_seconds):
                _terminate_process_group(
                    proc,
                    log_handle,
                    "timeout exceeded",
                    args.interrupt_grace_seconds,
                    args.kill_grace_seconds,
                )
                return 124
            if stop_file.exists():
                try:
                    stop_file.unlink()
                except FileNotFoundError:
                    pass
                _terminate_process_group(
                    proc,
                    log_handle,
                    f"stop file observed at {stop_file}",
                    args.interrupt_grace_seconds,
                    args.kill_grace_seconds,
                )
                return 130
            if rc is not None:
                _log(log_handle, f"finished returncode={rc}")
                return int(rc)
            time.sleep(poll)
    finally:
        if log_handle is not None:
            log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
