#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p docs

digest="${FSD_CHAIN_RULE_PROTECTED_DIGEST:-8b6db81d271bc742205632d6cd44a7d488a32c01a9064d3bae4e403b36e76698}"
interval="${FSD_POST_FORMULA_RESUME_INTERVAL_SECONDS:-120}"
log_path="${FSD_POST_FORMULA_RESUME_LOG:-docs/cluster_cache_3l_post_formula_resume.log}"
launcher_pid_file="${FSD_CACHE_LAUNCHER_PID_FILE:-docs/cluster_cache_3l_launcher.pid}"
cache_dir="${FSD_SUBTRACTION_FORMULA_CACHE_DIR:-cache/subtraction_formulae}"
target_json="${cache_dir}/chain_rule_${digest}.json"
status_path="${FSD_CACHE_SHARD_STATUS_PATH:-docs/cluster_cache_3l_shards_status.json}"

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

all_phases_done() {
  local path="$1"
  [[ -s "$path" ]] || return 1
  .venv/bin/python - "$path" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
except Exception:
    raise SystemExit(1)
if data.get("phase") != "ibp-3l":
    raise SystemExit(1)
if int(data.get("pending", 1)) != 0 or int(data.get("running", 1)) != 0:
    raise SystemExit(1)
if int(data.get("failed", 1)) != 0:
    raise SystemExit(1)
task_count = int(data.get("task_count", 0))
done = int(data.get("completed", 0)) + int(data.get("skipped", 0))
raise SystemExit(0 if task_count > 0 and done >= task_count else 1)
PY
}

start_normal_launcher() {
  setsid nohup env \
    FSD_CACHE_WATCHDOG_LIMIT_GB="${FSD_CACHE_WATCHDOG_LIMIT_GB:-950}" \
    FSD_CACHE_SHARD_MAX_ATTEMPTS="${FSD_CACHE_SHARD_MAX_ATTEMPTS:-2}" \
    FSD_CACHE_PRESERVE_DRAIN_FILE=false \
    FSD_CACHE_INITIAL_DRAIN_UNTIL_FORMULA_JSON=false \
    FSD_SYMBOLICA_EVALUATOR_VERBOSE="${FSD_SYMBOLICA_EVALUATOR_VERBOSE:-true}" \
    FSD_SYMBOLICA_EVALUATOR_CORES="${FSD_SYMBOLICA_EVALUATOR_CORES:-8}" \
    FSD_SYMBOLICA_EVALUATOR_ITERATIONS="${FSD_SYMBOLICA_EVALUATOR_ITERATIONS:-1}" \
    FSD_SYMBOLICA_EVALUATOR_CPE_ITERATIONS="${FSD_SYMBOLICA_EVALUATOR_CPE_ITERATIONS:-50}" \
    FSD_SYMBOLICA_MAX_HORNER_SCHEME_VARIABLES="${FSD_SYMBOLICA_MAX_HORNER_SCHEME_VARIABLES:-6}" \
    FSD_SYMBOLICA_MAX_COMMON_PAIR_CACHE_ENTRIES="${FSD_SYMBOLICA_MAX_COMMON_PAIR_CACHE_ENTRIES:-20000}" \
    FSD_SYMBOLICA_MAX_COMMON_PAIR_DISTANCE="${FSD_SYMBOLICA_MAX_COMMON_PAIR_DISTANCE:-6}" \
    bash scripts/launch_fsd_cache_3l.sh >> docs/cluster_cache_3l_launcher.out 2>&1 &
  echo $! > "$launcher_pid_file"
  log "normal_launcher_started pid=$(<"$launcher_pid_file")"
}

log "post_formula_resume_watcher_started digest=$digest interval_seconds=$interval target=$target_json"

while true; do
  if all_phases_done "$status_path"; then
    log "all_phases_done=true; exiting"
    exit 0
  fi

  if ! json_ready "$target_json"; then
    log "waiting_target_json target=$target_json"
    sleep "$interval"
    continue
  fi

  if launcher_alive; then
    log "target_ready=true launcher_alive=true; waiting"
    sleep "$interval"
    continue
  fi

  workers="$(cache_worker_count)"
  if [[ "$workers" != "0" ]]; then
    log "target_ready=true launcher_dead=true cache_workers_alive=$workers; waiting"
    sleep "$interval"
    continue
  fi

  log "target_ready=true launcher_dead=true workers=0; starting_normal_launcher=true"
  start_normal_launcher
  sleep "$interval"
done
