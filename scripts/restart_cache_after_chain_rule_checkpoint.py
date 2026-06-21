#!/usr/bin/env python3
"""Restart the 3L cache run after a chain-rule expression checkpoint exists.

This is for the case where the expensive expression generation was launched
without the desired evaluator-tuning environment.  The script waits until the
binary expression sidecar manifest is complete, then stops the old cache tree
and starts the protected tuned resume for the same shard.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


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
        try:
            if (path.parent / name).stat().st_size > 0:
                present += 1
        except OSError:
            pass
    return bool(names) and expected == len(names) and present == len(names), present, expected


def _pid_alive(pid: int) -> bool:
    status = Path("/proc") / str(pid) / "status"
    try:
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
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


def _read_pid(path: Path) -> int | None:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _signal_process_group(pid: int, sig: signal.Signals, log_file: Path, reason: str) -> None:
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return
    except PermissionError as exc:
        _log(log_file, f"{reason} could_not_read_pgid pid={pid} error={exc}")
        return
    try:
        os.killpg(pgid, sig)
        _log(log_file, f"{reason} sent_{sig.name.lower()} pgid={pgid}")
    except ProcessLookupError:
        return
    except PermissionError as exc:
        _log(log_file, f"{reason} could_not_signal pgid={pgid} error={exc}")


def _cache_worker_pids() -> list[int]:
    try:
        output = subprocess.check_output(
            ["ps", "-eo", "pid=,stat=,args="],
            cwd=ROOT,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    pids: list[int] = []
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) < 3 or parts[1].startswith("Z"):
            continue
        if "FSD.py cache" not in parts[2] or "--cache-loop-counts 3" not in parts[2]:
            continue
        try:
            pids.append(int(parts[0]))
        except ValueError:
            pass
    return pids


def _terminate_workers(log_file: Path) -> None:
    pids = _cache_worker_pids()
    if not pids:
        return
    _log(log_file, f"terminating_remaining_cache_workers pids={','.join(map(str, pids))}")
    for sig, delay in ((signal.SIGINT, 60.0), (signal.SIGTERM, 15.0), (signal.SIGKILL, 0.0)):
        live = [pid for pid in pids if _pid_alive(pid)]
        if not live:
            return
        for pid in live:
            _signal_process_group(pid, sig, log_file, "worker_shutdown")
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                pass
        if delay:
            deadline = time.monotonic() + delay
            while time.monotonic() < deadline and any(_pid_alive(pid) for pid in live):
                time.sleep(1.0)


def _stop_post_formula_watcher(pid_file: Path, log_file: Path) -> None:
    pid = _read_pid(pid_file)
    if pid is None or not _pid_alive(pid):
        return
    _log(log_file, f"stopping_previous_post_formula_watcher pid={pid}")
    _signal_process_group(pid, signal.SIGTERM, log_file, "post_formula_watcher_shutdown")
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        pass


def _start_post_formula_watcher(args: argparse.Namespace, log_file: Path) -> None:
    env = _tuned_env(args)
    env.update(
        {
            "FSD_CHAIN_RULE_PROTECTED_DIGEST": args.digest,
            "FSD_POST_FORMULA_RESUME_INTERVAL_SECONDS": str(args.post_formula_poll_seconds),
            "FSD_POST_FORMULA_RESUME_LOG": str(args.post_formula_log),
            "FSD_CACHE_LAUNCHER_PID_FILE": str(args.launcher_pid_file),
        }
    )
    out_path = ROOT / "docs" / "cluster_cache_3l_post_formula_resume.nohup"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("ab") as out:
        proc = subprocess.Popen(
            ["bash", "scripts/resume_full_cache_after_formula_json.sh"],
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    args.post_formula_pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    _log(log_file, f"post_formula_watcher_started pid={proc.pid} digest={args.digest}")


def _wait_for_launcher_exit(args: argparse.Namespace, log_file: Path) -> None:
    args.stop_file.write_text(
        f"checkpoint restart requested for {args.digest} at {_utc_now()}\n",
        encoding="utf-8",
    )
    _log(log_file, f"stop_file_written path={args.stop_file}")
    _log(log_file, "preempting_cache_workers_after_checkpoint=true")
    _terminate_workers(log_file)
    pid = _read_pid(args.launcher_pid_file)
    deadline = time.monotonic() + float(args.old_exit_timeout_seconds)
    while pid is not None and _pid_alive(pid) and time.monotonic() < deadline:
        time.sleep(5.0)
    if pid is not None and _pid_alive(pid):
        _log(log_file, f"old_launcher_alive_after_timeout pid={pid}; escalating")
        _signal_process_group(pid, signal.SIGTERM, log_file, "old_launcher_shutdown")
        time.sleep(30.0)
    if pid is not None and _pid_alive(pid):
        _signal_process_group(pid, signal.SIGKILL, log_file, "old_launcher_shutdown")
        time.sleep(5.0)
    _terminate_workers(log_file)


def _tuned_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "FSD_CACHE_WATCHDOG_LIMIT_GB": str(args.watchdog_limit_gb),
            "FSD_SYMBOLICA_EVALUATOR_VERBOSE": "true",
            "FSD_SYMBOLICA_EVALUATOR_CORES": str(args.evaluator_cores),
            "FSD_SYMBOLICA_EVALUATOR_ITERATIONS": str(args.evaluator_iterations),
            "FSD_SYMBOLICA_EVALUATOR_CPE_ITERATIONS": str(args.evaluator_cpe_iterations),
            "FSD_SYMBOLICA_MAX_HORNER_SCHEME_VARIABLES": str(args.max_horner_scheme_variables),
            "FSD_SYMBOLICA_MAX_COMMON_PAIR_CACHE_ENTRIES": str(args.max_common_pair_cache_entries),
            "FSD_SYMBOLICA_MAX_COMMON_PAIR_DISTANCE": str(args.max_common_pair_distance),
        }
    )
    return env


def _normalize_tuning_args(args: argparse.Namespace, log_file: Path) -> None:
    requested = {
        "evaluator_cores": args.evaluator_cores,
        "evaluator_iterations": args.evaluator_iterations,
        "evaluator_cpe_iterations": args.evaluator_cpe_iterations,
        "max_horner_scheme_variables": args.max_horner_scheme_variables,
        "max_common_pair_cache_entries": args.max_common_pair_cache_entries,
        "max_common_pair_distance": args.max_common_pair_distance,
    }
    args.evaluator_cores = max(int(args.evaluator_cores), 1)
    args.evaluator_iterations = max(int(args.evaluator_iterations), 1)
    args.evaluator_cpe_iterations = max(int(args.evaluator_cpe_iterations), 50)
    args.max_horner_scheme_variables = max(int(args.max_horner_scheme_variables), 1)
    args.max_common_pair_cache_entries = max(int(args.max_common_pair_cache_entries), 1)
    args.max_common_pair_distance = max(int(args.max_common_pair_distance), 1)
    normalized = {
        "evaluator_cores": args.evaluator_cores,
        "evaluator_iterations": args.evaluator_iterations,
        "evaluator_cpe_iterations": args.evaluator_cpe_iterations,
        "max_horner_scheme_variables": args.max_horner_scheme_variables,
        "max_common_pair_cache_entries": args.max_common_pair_cache_entries,
        "max_common_pair_distance": args.max_common_pair_distance,
    }
    if normalized != requested:
        _log(
            log_file,
            "normalized_evaluator_tuning "
            + " ".join(
                f"{key}={requested[key]}->{normalized[key]}" for key in sorted(normalized)
            ),
        )


def _start_tuned_resume(args: argparse.Namespace, log_file: Path) -> None:
    env = _tuned_env(args)
    env.update(
        {
            "FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST": args.digest,
            "FSD_PROTECTED_RESUME_CASE": args.case,
            "FSD_PROTECTED_RESUME_SHARD_LABEL": args.shard_label,
            "FSD_PROTECTED_RESUME_SECTORS": " ".join(str(sector) for sector in args.sectors),
            "FSD_PROTECTED_RESUME_REPORT_PATH": str(args.report_path),
            "FSD_PROTECTED_RESUME_WORKDIR": str(args.workdir),
            "FSD_PROTECTED_RESUME_LOG_PATH": str(args.shard_log_path),
            "FSD_CACHE_SHARD_DRAIN_FILE": str(args.drain_file),
            "FSD_CACHE_STOP_FILE": str(args.stop_file),
            "FSD_CACHE_SHARD_MAX_ATTEMPTS": "1",
        }
    )
    out_path = ROOT / "docs" / "cluster_cache_3l_launcher.out"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("ab") as out:
        proc = subprocess.Popen(
            ["bash", "scripts/launch_fsd_cache_3l_tuned_resume.sh"],
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    args.launcher_pid_file.write_text(f"{proc.pid}\n", encoding="utf-8")
    _log(
        log_file,
        (
            f"tuned_protected_resume_started pid={proc.pid} digest={args.digest} "
            f"shard={args.shard_label} sectors={' '.join(map(str, args.sectors))} "
            f"iterations={args.evaluator_iterations} "
            f"cpe_iterations={args.evaluator_cpe_iterations} "
            f"max_horner_vars={args.max_horner_scheme_variables} "
            f"cpe_cache_entries={args.max_common_pair_cache_entries} "
            f"cpe_distance={args.max_common_pair_distance}"
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digest", required=True)
    parser.add_argument("--case", default="triple_box")
    parser.add_argument("--shard-label", required=True)
    parser.add_argument("--sectors", nargs="+", type=int, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/subtraction_formulae"))
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--shard-log-path", type=Path, required=True)
    parser.add_argument("--launcher-pid-file", type=Path, default=Path("docs/cluster_cache_3l_launcher.pid"))
    parser.add_argument("--post-formula-pid-file", type=Path, default=Path("docs/cluster_cache_3l_post_formula_resume.pid"))
    parser.add_argument("--post-formula-log", type=Path, default=Path("docs/cluster_cache_3l_post_formula_resume.log"))
    parser.add_argument("--stop-file", type=Path, default=Path("stop.order"))
    parser.add_argument("--drain-file", type=Path, default=Path("docs/cluster_cache_3l_drain.order"))
    parser.add_argument("--log-file", type=Path, default=Path("docs/cluster_cache_3l_checkpoint_restart.log"))
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--post-formula-poll-seconds", type=float, default=60.0)
    parser.add_argument("--old-exit-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--watchdog-limit-gb", type=float, default=950.0)
    parser.add_argument("--evaluator-cores", type=int, default=8)
    parser.add_argument("--evaluator-iterations", type=int, default=1)
    parser.add_argument("--evaluator-cpe-iterations", type=int, default=50)
    parser.add_argument("--max-horner-scheme-variables", type=int, default=6)
    parser.add_argument("--max-common-pair-cache-entries", type=int, default=20_000)
    parser.add_argument("--max-common-pair-distance", type=int, default=6)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.cache_dir = _rooted(args.cache_dir)
    args.report_path = _rooted(args.report_path)
    args.workdir = _rooted(args.workdir)
    args.shard_log_path = _rooted(args.shard_log_path)
    args.launcher_pid_file = _rooted(args.launcher_pid_file)
    args.post_formula_pid_file = _rooted(args.post_formula_pid_file)
    args.post_formula_log = _rooted(args.post_formula_log)
    args.stop_file = _rooted(args.stop_file)
    args.drain_file = _rooted(args.drain_file)
    args.log_file = _rooted(args.log_file)

    log_file = args.log_file
    _normalize_tuning_args(args, log_file)
    digest = str(args.digest).strip()
    target_json = args.cache_dir / f"chain_rule_{digest}.json"
    manifest = args.cache_dir / f"chain_rule_{digest}.expr_manifest.json"
    poll = max(float(args.poll_seconds), 1.0)

    _log(
        log_file,
        (
            f"started digest={digest} manifest={manifest} target={target_json} "
            f"shard={args.shard_label} poll_seconds={poll:g}"
        ),
    )
    while True:
        if _json_ready(target_json):
            _log(log_file, f"formula_json_already_ready target={target_json}; restart_not_needed=true")
            return 0
        manifest_ok, present, expected = _manifest_ready(manifest)
        if not manifest_ok:
            _log(log_file, f"waiting_expression_checkpoint manifest_files={present}/{expected}")
            time.sleep(poll)
            continue

        _log(log_file, f"expression_checkpoint_ready manifest_files={present}/{expected}")
        _stop_post_formula_watcher(args.post_formula_pid_file, log_file)
        _start_post_formula_watcher(args, log_file)
        _wait_for_launcher_exit(args, log_file)
        _start_tuned_resume(args, log_file)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
