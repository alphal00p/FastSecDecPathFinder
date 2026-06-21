#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p docs

digest="${FSD_CHAIN_RULE_PROTECTED_DIGEST:-8b6db81d271bc742205632d6cd44a7d488a32c01a9064d3bae4e403b36e76698}"
interval="${FSD_EXIT_RELAUNCH_INTERVAL_SECONDS:-120}"
log_path="${FSD_EXIT_RELAUNCH_LOG:-docs/cluster_cache_3l_exit_relaunch.log}"
launcher_pid_file="${FSD_CACHE_LAUNCHER_PID_FILE:-docs/cluster_cache_3l_launcher.pid}"
status_monitor_pid_file="${FSD_CACHE_STATUS_MONITOR_PID_FILE:-docs/cluster_cache_3l_status_monitor.pid}"
precheckpoint_limit_gib="${FSD_EXIT_RELAUNCH_PRECHECKPOINT_LIMIT_GB:-950}"
cache_dir="${FSD_SUBTRACTION_FORMULA_CACHE_DIR:-cache/subtraction_formulae}"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log_path"
}

pid_alive() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  if [[ -r "/proc/$pid/status" ]]; then
    ! awk '$1 == "State:" && $2 == "Z" { found = 1 } END { exit found ? 0 : 1 }' "/proc/$pid/status"
  fi
  return 0
}

launcher_pid() {
  [[ -f "$launcher_pid_file" ]] || return 1
  tr -d '[:space:]' < "$launcher_pid_file"
}

launcher_alive() {
  local pid
  pid="$(launcher_pid)" || return 1
  pid_alive "$pid"
}

cache_worker_count() {
  ps -eo stat=,comm=,args= \
    | awk '$1 !~ /^Z/ && $2 ~ /python/ && index($0, "FSD.py cache") && index($0, "--cache-loop-counts 3") { count += 1 } END { print count + 0 }'
}

restart_status_monitor_if_needed() {
  if [[ -f "$status_monitor_pid_file" ]] && pid_alive "$(<"$status_monitor_pid_file")"; then
    return 0
  fi
  setsid nohup scripts/monitor_cache_status.sh >> docs/cluster_cache_3l_status_monitor.nohup 2>&1 &
  echo $! > "$status_monitor_pid_file"
  log "status_monitor_restarted pid=$(<"$status_monitor_pid_file")"
}

json_ready() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  .venv/bin/python - "$path" <<'PY'
import json
import sys
from pathlib import Path

try:
    json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0)
PY
}

manifest_ready() {
  local path="$1"
  .venv/bin/python - "$path" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
try:
    data = json.loads(manifest.read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
names = [str(name) for name in data.get("expression_cache_files", [])]
expected = int(data.get("output_expression_count", len(names)))
if not names or expected != len(names):
    raise SystemExit(1)
for name in names:
    try:
        if (manifest.parent / name).stat().st_size <= 0:
            raise SystemExit(1)
    except OSError:
        raise SystemExit(1)
raise SystemExit(0)
PY
}

start_launcher() {
  local mode="$1"
  if [[ "$mode" == "checkpoint-resume" ]]; then
    setsid nohup bash scripts/launch_fsd_cache_3l_tuned_resume.sh >> docs/cluster_cache_3l_launcher.out 2>&1 &
  else
    setsid nohup env FSD_CACHE_WATCHDOG_LIMIT_GB="$precheckpoint_limit_gib" bash scripts/launch_fsd_cache_3l.sh >> docs/cluster_cache_3l_launcher.out 2>&1 &
  fi
  echo $! > "$launcher_pid_file"
  log "launcher_started mode=$mode pid=$(<"$launcher_pid_file")"
  restart_status_monitor_if_needed
}

target_json="${cache_dir}/chain_rule_${digest}.json"
manifest_json="${cache_dir}/chain_rule_${digest}.expr_manifest.json"

log "exit_relaunch_watcher_started digest=$digest interval_seconds=$interval target=$target_json manifest=$manifest_json"

while true; do
  if launcher_alive; then
    sleep "$interval"
    continue
  fi

  if json_ready "$target_json"; then
    log "launcher_not_alive target_json_ready=true; relaunch_not_needed=true"
    restart_status_monitor_if_needed
    exit 0
  fi

  workers="$(cache_worker_count)"
  if [[ "$workers" != "0" ]]; then
    log "launcher_not_alive_but_cache_workers_alive count=$workers; waiting"
    sleep "$interval"
    continue
  fi

  if manifest_ready "$manifest_json"; then
    log "launcher_not_alive expression_checkpoint_ready=true; starting_tuned_resume=true"
    start_launcher "checkpoint-resume"
  else
    log "launcher_not_alive expression_checkpoint_ready=false; starting_precheckpoint_resume=true limit_gib=$precheckpoint_limit_gib"
    start_launcher "precheckpoint-resume"
  fi
  sleep "$interval"
done
