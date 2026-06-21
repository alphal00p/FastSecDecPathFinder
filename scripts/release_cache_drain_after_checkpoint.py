#!/usr/bin/env python3
"""Release the shard-runner drain marker once a protected artifact exists."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / "cache" / "subtraction_formulae"
DEFAULT_DRAIN_FILE = ROOT / "docs" / "cluster_cache_3l_drain.order"
DEFAULT_LOG_FILE = ROOT / "docs" / "cluster_cache_3l_drain_release.log"


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


def _drain_matches_digest(path: Path, digest: str) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return (
        f"protected_digest={digest}" in text
        or f"digest={digest}" in text
        or f"protected_digest = {digest}" in text
        or f"digest = {digest}" in text
    )


def _archive_drain(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_name(f"{path.name}.released.{stamp}")
    counter = 1
    while archive.exists():
        archive = path.with_name(f"{path.name}.released.{stamp}.{counter}")
        counter += 1
    path.replace(archive)
    return archive


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--drain-file", type=Path, default=DEFAULT_DRAIN_FILE)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument(
        "--release-after",
        choices=("expression-checkpoint", "formula-json"),
        default="expression-checkpoint",
        help=(
            "Artifact that must exist before the drain is released.  "
            "Use formula-json to keep other shards drained through evaluator construction."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    digest = str(args.digest).strip()
    cache_dir = _rooted(args.cache_dir)
    drain_file = _rooted(args.drain_file)
    log_file = _rooted(args.log_file)
    poll_seconds = max(float(args.poll_seconds), 5.0)

    target_json = cache_dir / f"chain_rule_{digest}.json"
    manifest = cache_dir / f"chain_rule_{digest}.expr_manifest.json"
    _log(
        log_file,
        (
            f"started digest={digest} target={target_json} manifest={manifest} "
            f"drain_file={drain_file} poll_seconds={poll_seconds:g} "
            f"release_after={args.release_after}"
        ),
    )
    while True:
        if _json_ready(target_json):
            ready_reason = "formula_json_ready"
            manifest_present = manifest.is_file()
            present = 0
            expected = 0
        else:
            if args.release_after == "formula-json":
                manifest_ok, present, expected = _manifest_ready(manifest)
                _log(
                    log_file,
                    (
                        f"waiting_formula_json manifest_files={present}/{expected} "
                        f"manifest_ready={str(manifest_ok).lower()} "
                        f"drain_present={drain_file.exists()}"
                    ),
                )
                time.sleep(poll_seconds)
                continue
            manifest_ok, present, expected = _manifest_ready(manifest)
            if not manifest_ok:
                _log(
                    log_file,
                    (
                        f"waiting_checkpoint manifest_files={present}/{expected} "
                        f"drain_present={drain_file.exists()}"
                    ),
                )
                time.sleep(poll_seconds)
                continue
            ready_reason = "expression_checkpoint_ready"
            manifest_present = True

        if not drain_file.exists():
            _log(
                log_file,
                (
                    f"{ready_reason} drain_absent=true manifest_present={manifest_present} "
                    f"manifest_files={present}/{expected}; release_not_needed=true"
                ),
            )
            return 0
        if not _drain_matches_digest(drain_file, digest):
            _log(
                log_file,
                (
                    f"{ready_reason} drain_digest_mismatch path={drain_file}; "
                    f"refusing_release=true"
                ),
            )
            return 2
        archive = _archive_drain(drain_file)
        _log(
            log_file,
            (
                f"{ready_reason} drain_released path={drain_file} archive={archive} "
                f"manifest_present={manifest_present} manifest_files={present}/{expected}"
            ),
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
