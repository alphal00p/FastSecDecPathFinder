#!/usr/bin/env python3
"""Guard a long chain-rule expression build until its binary checkpoint exists.

The vulnerable interval is between cold expression generation start and the
native Symbolica expression sidecar write.  During that interval, a watchdog
kill loses all expression-generation progress.  This guard does not stop the
run or kill workers; it only writes the shard-runner drain marker when total
watchdog RSS crosses a configurable soft threshold before the expression
checkpoint exists.
"""

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
DEFAULT_LOG_FILE = ROOT / "docs" / "cluster_cache_3l_checkpoint_guard.log"


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
    present = sum(1 for name in names if (path.parent / name).is_file())
    return bool(names) and present == len(names) and expected == len(names), present, expected


def _write_drain(path: Path, *, digest: str, rss_gib: float, threshold_gib: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f"drain_requested_utc={_utc_now()}\n"
            f"reason=pre_checkpoint_rss_above_soft_threshold\n"
            f"digest={digest}\n"
            f"rss_gib={rss_gib:.3f}\n"
            f"soft_threshold_gib={threshold_gib:.3f}\n"
        ),
        encoding="utf-8",
    )


def _run_waiter_drain(log_file: Path, *, digest: str, rss_gib: float) -> int:
    """Stop idle non-owner cache workers using the shared helper."""

    command = [
        sys.executable,
        str(ROOT / "scripts" / "drain_nonowner_cache_workers.py"),
        "--log-file",
        str(ROOT / "docs" / "cluster_cache_3l_drain_waiters.log"),
    ]
    _log(log_file, f"waiter_drain_start digest={digest} rss_gib={rss_gib:.3f}")
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
    except OSError as exc:
        _log(log_file, f"waiter_drain_failed reason={type(exc).__name__}")
        return 127
    output_tail = completed.stdout.strip().splitlines()[-3:]
    for line in output_tail:
        _log(log_file, f"waiter_drain_output {line}")
    _log(log_file, f"waiter_drain_done returncode={completed.returncode}")
    return int(completed.returncode)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--watchdog-log", type=Path, default=DEFAULT_WATCHDOG_LOG)
    parser.add_argument("--drain-file", type=Path, default=DEFAULT_DRAIN_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--soft-drain-gib", type=float, default=600.0)
    parser.add_argument(
        "--waiter-drain-gib",
        type=float,
        default=0.0,
        help=(
            "Optional higher RSS threshold that also invokes "
            "drain_nonowner_cache_workers.py.  Set to 0 to disable."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    digest = str(args.digest).strip()
    cache_dir = _rooted(args.cache_dir)
    watchdog_log = _rooted(args.watchdog_log)
    drain_file = _rooted(args.drain_file)
    log_file = _rooted(args.log_file)
    poll_seconds = max(float(args.poll_seconds), 5.0)
    soft_drain_gib = float(args.soft_drain_gib)
    waiter_drain_gib = float(args.waiter_drain_gib)
    waiter_drain_done = False

    target_json = cache_dir / f"chain_rule_{digest}.json"
    manifest = cache_dir / f"chain_rule_{digest}.expr_manifest.json"
    _log(
        log_file,
        (
            f"started digest={digest} target={target_json} manifest={manifest} "
            f"soft_drain_gib={soft_drain_gib:g} "
            f"waiter_drain_gib={waiter_drain_gib:g} "
            f"poll_seconds={poll_seconds:g}"
        ),
    )
    while True:
        if _json_ready(target_json):
            _log(log_file, f"formula_json_ready target={target_json}; guard_exit=true")
            return 0

        manifest_ok, present, expected = _manifest_ready(manifest)
        if manifest_ok:
            _log(
                log_file,
                (
                    f"expression_checkpoint_ready manifest={manifest} "
                    f"files_present={present}/{expected}; guard_exit=true"
                ),
            )
            return 0

        rss_gib, latest = _latest_watchdog_rss_gib(watchdog_log)
        if rss_gib is None:
            _log(log_file, f"waiting_checkpoint rss_unavailable latest={latest!r}")
            time.sleep(poll_seconds)
            continue

        if rss_gib >= soft_drain_gib:
            if not drain_file.exists():
                _write_drain(
                    drain_file,
                    digest=digest,
                    rss_gib=rss_gib,
                    threshold_gib=soft_drain_gib,
                )
                _log(
                    log_file,
                    (
                        f"drain_written path={drain_file} rss_gib={rss_gib:.3f} "
                        f"soft_drain_gib={soft_drain_gib:.3f} "
                        f"manifest_files={present}/{expected}"
                    ),
                )
            else:
                _log(
                    log_file,
                    (
                        f"drain_already_present path={drain_file} rss_gib={rss_gib:.3f} "
                        f"manifest_files={present}/{expected}"
                    ),
                )
            if waiter_drain_gib > 0.0 and rss_gib >= waiter_drain_gib and not waiter_drain_done:
                waiter_drain_done = True
                _run_waiter_drain(log_file, digest=digest, rss_gib=rss_gib)
        else:
            _log(
                log_file,
                (
                    f"waiting_checkpoint rss_gib={rss_gib:.3f} "
                    f"headroom_to_soft_gib={soft_drain_gib - rss_gib:.3f} "
                    f"manifest_files={present}/{expected}"
                ),
            )
        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
