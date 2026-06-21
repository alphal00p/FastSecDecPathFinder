#!/usr/bin/env python3
"""Request cache-run shutdown after a specific chain-rule JSON is written."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest", required=True, help="64-character chain-rule digest to watch.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache/subtraction_formulae"),
        help="Formula cache directory. Default: cache/subtraction_formulae",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        default=Path("stop.order"),
        help="Stop file consumed by run_with_memory_watch.py.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=Path("docs/cluster_cache_3l_formula_stop_watcher.log"),
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--require-json-valid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the JSON to parse before writing the stop file.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        print(f"{_utc_now()} {message}", file=handle, flush=True)


def _json_ready(path: Path, require_valid: bool) -> bool:
    try:
        if path.stat().st_size <= 0:
            return False
        if require_valid:
            json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return True


def main() -> int:
    args = _parse_args()
    digest = str(args.digest).strip()
    target = args.cache_dir / f"chain_rule_{digest}.json"
    poll = max(float(args.poll_seconds), 1.0)

    _log(args.log_file, f"started digest={digest} target={target} poll_seconds={poll:g}")
    while True:
        if _json_ready(target, bool(args.require_json_valid)):
            args.stop_file.write_text(
                f"chain_rule_{digest} cache file observed at {_utc_now()}\n",
                encoding="utf-8",
            )
            _log(args.log_file, f"stop_requested stop_file={args.stop_file} target={target}")
            return 0
        _log(args.log_file, f"waiting target={target}")
        time.sleep(poll)


if __name__ == "__main__":
    raise SystemExit(main())
