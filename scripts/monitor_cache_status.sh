#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p docs

interval="${FSD_CACHE_STATUS_INTERVAL_SECONDS:-300}"
log_path="${FSD_CACHE_STATUS_MONITOR_LOG:-docs/cluster_cache_3l_status_monitor.log}"
launcher_pid_file="${FSD_CACHE_LAUNCHER_PID_FILE:-docs/cluster_cache_3l_launcher.pid}"
top_processes="${FSD_CACHE_STATUS_TOP_PROCESSES:-8}"
memory_limit_gib="${FSD_CACHE_MEMORY_LIMIT_GIB:-800}"

while true; do
  {
    printf '\n=== cache status %s ===\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    .venv/bin/python scripts/report_cache_run_status.py \
      --top-processes "$top_processes" \
      --memory-limit-gib "$memory_limit_gib"
  } >> "$log_path" 2>&1 || true

  if [[ ! -f "$launcher_pid_file" ]]; then
    printf 'launcher_pid_file_missing path=%s; waiting_for_relaunch=true\n' "$launcher_pid_file" >> "$log_path"
    sleep "$interval"
    continue
  fi

  launcher_pid="$(<"$launcher_pid_file")"
  if ! kill -0 "$launcher_pid" 2>/dev/null; then
    printf 'launcher_exited pid=%s time_utc=%s waiting_for_relaunch=true\n' "$launcher_pid" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$log_path"
    sleep "$interval"
    continue
  fi

  sleep "$interval"
done
