#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p docs

interval="${FSD_SAFE_RELAUNCH_INTERVAL_SECONDS:-120}"
log_path="${FSD_SAFE_RELAUNCH_LOG:-docs/cluster_cache_3l_safe_relaunch.log}"
launcher_pid_file="${FSD_CACHE_LAUNCHER_PID_FILE:-docs/cluster_cache_3l_launcher.pid}"
monitor_pid_file="${FSD_CACHE_STATUS_MONITOR_PID_FILE:-docs/cluster_cache_3l_status_monitor.pid}"
stop_file="${FSD_CACHE_STOP_FILE:-stop.order}"
formula_stop_log="${FSD_FORMULA_STOP_WATCHER_LOG:-docs/cluster_cache_3l_formula_stop_watcher.log}"
status_tmp="docs/cluster_cache_3l_safe_relaunch_status.tmp"

log() {
  printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$log_path"
}

cache_worker_pids() {
  ps -eo pid=,stat=,args= \
    | awk '$2 !~ /^Z/ && index($0, "FSD.py cache") && index($0, "--cache-loop-counts 3") { print $1 }'
}

cache_worker_count() {
  cache_worker_pids | wc -l | tr -d '[:space:]'
}

formula_stop_requested() {
  [[ -f "$formula_stop_log" ]] && tail -n 20 "$formula_stop_log" | grep -q ' stop_requested '
}

terminate_existing_cache_workers() {
  local pids
  mapfile -t pids < <(cache_worker_pids)
  if (( ${#pids[@]} == 0 )); then
    log "no_existing_cache_workers_before_launch"
    return 0
  fi

  log "terminating_existing_cache_workers_before_launch pids=${pids[*]}"
  local pid
  for pid in "${pids[@]}"; do
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
  done
  local deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )) && [[ "$(cache_worker_count)" != "0" ]]; do
    sleep 2
  done
  if [[ "$(cache_worker_count)" == "0" ]]; then
    log "existing_cache_workers_exited_after_sigint"
    return 0
  fi

  mapfile -t pids < <(cache_worker_pids)
  log "existing_cache_workers_still_alive_after_sigint pids=${pids[*]}; sending_sigterm"
  for pid in "${pids[@]}"; do
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  done
  deadline=$((SECONDS + 15))
  while (( SECONDS < deadline )) && [[ "$(cache_worker_count)" != "0" ]]; do
    sleep 1
  done
  if [[ "$(cache_worker_count)" == "0" ]]; then
    log "existing_cache_workers_exited_after_sigterm"
    return 0
  fi

  mapfile -t pids < <(cache_worker_pids)
  log "existing_cache_workers_still_alive_after_sigterm pids=${pids[*]}; sending_sigkill"
  for pid in "${pids[@]}"; do
    kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
  done
}

launcher_pid() {
  [[ -f "$launcher_pid_file" ]] || return 1
  tr -d '[:space:]' < "$launcher_pid_file"
}

launcher_alive() {
  local pid
  pid="$(launcher_pid)" || return 1
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

stop_old_launcher_or_escalate() {
  local old_pid="$1"
  local reason="$2"

  printf 'safe relaunch requested (%s) at %s\n' "$reason" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$stop_file"

  local deadline=$((SECONDS + 900))
  while kill -0 "$old_pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 10
  done
  if ! kill -0 "$old_pid" 2>/dev/null; then
    log "old_launcher_exited reason=$reason"
    return 0
  fi

  log "old_launcher_still_alive_after_timeout pid=$old_pid reason=$reason; sending_sigterm"
  kill -TERM -- "-$old_pid" 2>/dev/null || kill -TERM "$old_pid" 2>/dev/null || true
  deadline=$((SECONDS + 60))
  while kill -0 "$old_pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 2
  done
  if ! kill -0 "$old_pid" 2>/dev/null; then
    log "old_launcher_exited_after_sigterm reason=$reason"
    return 0
  fi

  log "old_launcher_still_alive_after_sigterm pid=$old_pid reason=$reason; sending_sigkill"
  kill -KILL -- "-$old_pid" 2>/dev/null || kill -KILL "$old_pid" 2>/dev/null || true
  deadline=$((SECONDS + 30))
  while kill -0 "$old_pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if kill -0 "$old_pid" 2>/dev/null; then
    log "old_launcher_still_alive_after_sigkill pid=$old_pid reason=$reason; not_relaunching"
    return 1
  fi
  log "old_launcher_exited_after_sigkill reason=$reason"
}

restart_status_monitor_if_needed() {
  if [[ -f "$monitor_pid_file" ]] && kill -0 "$(<"$monitor_pid_file")" 2>/dev/null; then
    return 0
  fi
  setsid nohup scripts/monitor_cache_status.sh >> docs/cluster_cache_3l_status_monitor.nohup 2>&1 &
  echo $! > "$monitor_pid_file"
  log "status_monitor_restarted pid=$(<"$monitor_pid_file")"
}

status_probe_is_complete() {
  grep -q '^cache_run_status$' "$status_tmp" \
    && grep -q '^shards phase=' "$status_tmp" \
    && grep -q '^watchdog_latest ' "$status_tmp" \
    && grep -q '^worker_cpu worker_count=' "$status_tmp" \
    && grep -q '^active_formula_lock_count=' "$status_tmp"
}

start_patched_launcher() {
  terminate_existing_cache_workers
  setsid nohup bash scripts/launch_fsd_cache_3l.sh >> docs/cluster_cache_3l_launcher.out 2>&1 &
  echo $! > "$launcher_pid_file"
  log "launcher_started pid=$(<"$launcher_pid_file")"
  restart_status_monitor_if_needed
}

log "safe_relaunch_watcher_started interval_seconds=$interval"

while true; do
  if ! launcher_alive; then
    status_rc=0
    .venv/bin/python scripts/report_cache_run_status.py \
      --top-processes 200 \
      --memory-limit-gib 800 > "$status_tmp" 2>&1 || status_rc=$?
    cat "$status_tmp" >> "$log_path"
    if (( status_rc == 0 )) && status_probe_is_complete; then
      lock_count="$(sed -n 's/^active_formula_lock_count=//p' "$status_tmp" | tail -n 1)"
      if [[ "$lock_count" =~ ^[1-9][0-9]*$ ]]; then
        log "launcher_not_alive_but_active_formula_locks_present count=$lock_count; waiting"
        sleep "$interval"
        continue
      fi
    elif [[ "$(cache_worker_count)" != "0" ]] && ! formula_stop_requested; then
      log "launcher_not_alive_status_probe_incomplete rc=$status_rc existing_workers=$(cache_worker_count); waiting"
      sleep "$interval"
      continue
    fi

    log "launcher_not_alive; starting patched launcher"
    start_patched_launcher
    exit 0
  fi

  status_rc=0
  .venv/bin/python scripts/report_cache_run_status.py \
    --top-processes 200 \
    --memory-limit-gib 800 > "$status_tmp" 2>&1 || status_rc=$?
  cat "$status_tmp" >> "$log_path"

  if (( status_rc != 0 )) || ! status_probe_is_complete; then
    log "status_probe_incomplete rc=$status_rc; keeping current run alive"
    sleep "$interval"
    continue
  fi

  lock_count="$(sed -n 's/^active_formula_lock_count=//p' "$status_tmp" | tail -n 1)"
  if [[ "$lock_count" =~ ^[1-9][0-9]*$ ]]; then
    log "active_formula_locks_present count=$lock_count; keeping current run alive"
    sleep "$interval"
    continue
  fi
  if [[ "$lock_count" != "0" ]]; then
    log "active_formula_lock_count_invalid value=${lock_count:-missing}; keeping current run alive"
    sleep "$interval"
    continue
  fi

  old_pid="$(launcher_pid)"
  log "no_active_formula_locks; requesting watchdog stop for launcher pid=$old_pid"
  stop_old_launcher_or_escalate "$old_pid" "no_active_formula_locks" || exit 1
  log "starting_patched_launcher_after_no_active_formula_locks"
  start_patched_launcher
  exit 0
done
