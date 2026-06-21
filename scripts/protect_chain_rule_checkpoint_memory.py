#!/usr/bin/env python3
"""Stop non-protected cache workers if RSS threatens a chain-rule cache build."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "cache" / "subtraction_formulae"
DEFAULT_WATCHDOG_LOG = ROOT / "docs" / "cluster_cache_3l_watchdog.log"
DEFAULT_DRAIN_FILE = ROOT / "docs" / "cluster_cache_3l_drain.order"
DEFAULT_LOG_FILE = ROOT / "docs" / "cluster_cache_3l_memory_guard.log"
DEFAULT_DRAIN_LOG = ROOT / "docs" / "cluster_cache_3l_drain_waiters.log"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rooted(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _log(path: Path, message: str) -> None:
    line = f"{_utc_now()} {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _tail(path: Path, *, max_bytes: int = 32_000) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _latest_watchdog_rss_gib(path: Path) -> tuple[float | None, str | None]:
    latest: str | None = None
    for line in _tail(path).splitlines():
        if " rss=" in line and " GiB" in line:
            latest = line
    if latest is None:
        return None, None
    match = re.search(r"\brss=(?P<rss>[0-9.]+)\s+GiB\b", latest)
    if match is None:
        return None, latest
    return float(match.group("rss")), latest


def _json_ready(path: Path) -> bool:
    try:
        if path.stat().st_size <= 0:
            return False
        json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return True


def _manifest_ready(path: Path) -> tuple[bool, int, int]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False, 0, 0
    names = [str(name) for name in data.get("expression_cache_files", [])]
    expected = int(data.get("output_expression_count", len(names)))
    present = 0
    for name in names:
        sidecar_path = path.parent / name
        try:
            if sidecar_path.stat().st_size > 0:
                present += 1
        except OSError:
            pass
    return bool(names) and present == len(names) and expected == len(names), present, expected


def _artifact_state(target_json: Path, manifest: Path) -> tuple[bool, bool, int, int]:
    if _json_ready(target_json):
        return True, False, 0, 0
    manifest_ok, present, expected = _manifest_ready(manifest)
    return False, manifest_ok, present, expected


def _status_text() -> str:
    return subprocess.check_output(
        [
            sys.executable,
            str(ROOT / "scripts" / "report_cache_run_status.py"),
            "--top-processes",
            "300",
        ],
        cwd=ROOT,
        text=True,
        stderr=subprocess.STDOUT,
    )


def _chain_rule_owner_pid(status: str, digest: str) -> int | None:
    for line in status.splitlines():
        if " formula=chain_rule " not in line or f" digest={digest} " not in line:
            continue
        match = re.search(r"\bowner_pid=(\d+)\b", line)
        if match is not None:
            return int(match.group(1))
    return None


def _run_emergency_drain(
    *,
    digest: str,
    protect_pid: int,
    drain_file: Path,
    drain_log: Path,
    busy_cpu_threshold: float,
    log_file: Path,
) -> int:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "drain_nonowner_cache_workers.py"),
        "--protect-pid",
        str(protect_pid),
        "--busy-cpu-threshold",
        str(busy_cpu_threshold),
        "--drain-file",
        str(drain_file),
        "--log-file",
        str(drain_log),
        "--protected-digest",
        digest,
    ]
    _log(
        log_file,
        (
            f"emergency_drain_start protect_pid={protect_pid} "
            f"busy_cpu_threshold={busy_cpu_threshold:g}"
        ),
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    for line in completed.stdout.strip().splitlines()[-8:]:
        _log(log_file, f"emergency_drain_output {line}")
    _log(log_file, f"emergency_drain_done returncode={completed.returncode}")
    return int(completed.returncode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--watchdog-log", type=Path, default=DEFAULT_WATCHDOG_LOG)
    parser.add_argument("--drain-file", type=Path, default=DEFAULT_DRAIN_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--drain-log", type=Path, default=DEFAULT_DRAIN_LOG)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--trigger-gib", type=float, default=650.0)
    parser.add_argument(
        "--busy-cpu-threshold",
        type=float,
        default=101.0,
        help="Passed to drain_nonowner_cache_workers.py so active non-protected workers can be stopped.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    digest = str(args.digest).strip()
    cache_dir = _rooted(args.cache_dir)
    watchdog_log = _rooted(args.watchdog_log)
    drain_file = _rooted(args.drain_file)
    log_file = _rooted(args.log_file)
    drain_log = _rooted(args.drain_log)
    poll_seconds = max(float(args.poll_seconds), 5.0)
    trigger_gib = float(args.trigger_gib)

    target_json = cache_dir / f"chain_rule_{digest}.json"
    manifest = cache_dir / f"chain_rule_{digest}.expr_manifest.json"
    _log(
        log_file,
        (
            f"started digest={digest} trigger_gib={trigger_gib:g} "
            f"target={target_json} manifest={manifest}"
        ),
    )
    while True:
        json_ready, manifest_ready, present, expected = _artifact_state(target_json, manifest)
        if json_ready:
            _log(
                log_file,
                f"formula_json_ready manifest_files={present}/{expected}; memory_guard_exit=true",
            )
            return 0

        rss_gib, latest = _latest_watchdog_rss_gib(watchdog_log)
        if rss_gib is None:
            _log(log_file, f"waiting_checkpoint rss_unavailable latest={latest!r}")
            time.sleep(poll_seconds)
            continue
        if rss_gib < trigger_gib:
            _log(
                log_file,
                (
                    f"waiting_final_json checkpoint_ready={str(manifest_ready).lower()} "
                    f"rss_gib={rss_gib:.3f} "
                    f"headroom_to_trigger_gib={trigger_gib - rss_gib:.3f} "
                    f"manifest_files={present}/{expected}"
                ),
            )
            time.sleep(poll_seconds)
            continue

        try:
            status = _status_text()
        except subprocess.CalledProcessError as exc:
            _log(log_file, f"status_failed returncode={exc.returncode}; retrying")
            time.sleep(poll_seconds)
            continue
        protect_pid = _chain_rule_owner_pid(status, digest)
        if protect_pid is None:
            _log(log_file, f"protected_owner_missing digest={digest}; retrying")
            time.sleep(poll_seconds)
            continue
        _log(
            log_file,
            (
                f"trigger_reached rss_gib={rss_gib:.3f} trigger_gib={trigger_gib:.3f} "
                f"protect_pid={protect_pid}"
            ),
        )
        return _run_emergency_drain(
            digest=digest,
            protect_pid=protect_pid,
            drain_file=drain_file,
            drain_log=drain_log,
            busy_cpu_threshold=float(args.busy_cpu_threshold),
            log_file=log_file,
        )


if __name__ == "__main__":
    raise SystemExit(main())
