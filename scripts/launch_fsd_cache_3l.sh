#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p docs

exec nix shell \
  nixpkgs#gfortran \
  nixpkgs#gcc \
  nixpkgs#gmp \
  nixpkgs#libffi \
  nixpkgs#zlib \
  --command bash -lc '
set -euo pipefail

source .deps/fsd_cache_env.sh

GCC_LIB=$(nix eval --raw nixpkgs#gcc.cc.lib.outPath)
GFORTRAN_LIB=$(nix eval --raw nixpkgs#gfortran.cc.lib.outPath)
GMP_LIB=$(nix eval --raw nixpkgs#gmp.outPath)
ZLIB_LIB=$(nix eval --raw nixpkgs#zlib.outPath)
export LD_LIBRARY_PATH="$GCC_LIB/lib:$GFORTRAN_LIB/lib:$GMP_LIB/lib:$ZLIB_LIB/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export FSD_CHAIN_RULE_MONITOR="${FSD_CHAIN_RULE_MONITOR:-true}"
export FSD_CHAIN_RULE_COMPOSE_MODE="${FSD_CHAIN_RULE_COMPOSE_MODE:-inplace}"
export FSD_CHAIN_RULE_COMPOSE_PROGRESS_EVERY="${FSD_CHAIN_RULE_COMPOSE_PROGRESS_EVERY:-10}"
export FSD_CHAIN_RULE_TERM_PROGRESS_EVERY="${FSD_CHAIN_RULE_TERM_PROGRESS_EVERY:-10000}"
export FSD_CHAIN_RULE_MUL_PROGRESS_EVERY="${FSD_CHAIN_RULE_MUL_PROGRESS_EVERY:-250000}"
export FSD_CHAIN_RULE_EXPRESSION_SIDECAR_REQUIRED="${FSD_CHAIN_RULE_EXPRESSION_SIDECAR_REQUIRED:-true}"
export FSD_CHAIN_RULE_EXPRESSION_PROGRESS_EVERY="${FSD_CHAIN_RULE_EXPRESSION_PROGRESS_EVERY:-64}"
export FSD_CHAIN_RULE_EXPRESSION_COMPRESSION_LEVEL="${FSD_CHAIN_RULE_EXPRESSION_COMPRESSION_LEVEL:-9}"
export FSD_CHAIN_RULE_GLOBAL_COLD_LOCK="${FSD_CHAIN_RULE_GLOBAL_COLD_LOCK:-true}"
export FSD_CHAIN_RULE_DEFER_WHEN_GLOBAL_LOCKED="${FSD_CHAIN_RULE_DEFER_WHEN_GLOBAL_LOCKED:-true}"
export FSD_CHAIN_RULE_DEFER_WHEN_CACHE_LOCKED="${FSD_CHAIN_RULE_DEFER_WHEN_CACHE_LOCKED:-true}"
export FSD_SUBTRACTION_FORMULA_MONITOR="${FSD_SUBTRACTION_FORMULA_MONITOR:-true}"
export FSD_SUBTRACTION_FORMULA_PROGRESS_SECONDS="${FSD_SUBTRACTION_FORMULA_PROGRESS_SECONDS:-30}"
export FSD_SYMBOLICA_EVALUATOR_VERBOSE="${FSD_SYMBOLICA_EVALUATOR_VERBOSE:-true}"
export FSD_SYMBOLICA_EVALUATOR_CORES="${FSD_SYMBOLICA_EVALUATOR_CORES:-8}"
export FSD_SYMBOLICA_EVALUATOR_ITERATIONS="${FSD_SYMBOLICA_EVALUATOR_ITERATIONS:-1}"
export FSD_SYMBOLICA_EVALUATOR_CPE_ITERATIONS="${FSD_SYMBOLICA_EVALUATOR_CPE_ITERATIONS:-50}"
export FSD_SYMBOLICA_MAX_HORNER_SCHEME_VARIABLES="${FSD_SYMBOLICA_MAX_HORNER_SCHEME_VARIABLES:-6}"
export FSD_SYMBOLICA_MAX_COMMON_PAIR_CACHE_ENTRIES="${FSD_SYMBOLICA_MAX_COMMON_PAIR_CACHE_ENTRIES:-20000}"
export FSD_SYMBOLICA_MAX_COMMON_PAIR_DISTANCE="${FSD_SYMBOLICA_MAX_COMMON_PAIR_DISTANCE:-6}"
export FSD_CACHE_WATCHDOG_LIMIT_GB="${FSD_CACHE_WATCHDOG_LIMIT_GB:-950}"
export FSD_CACHE_SHARD_JOBS="${FSD_CACHE_SHARD_JOBS:-100}"
export FSD_CACHE_SHARD_MAX_ATTEMPTS="${FSD_CACHE_SHARD_MAX_ATTEMPTS:-2}"
export FSD_CACHE_SHARD_DRAIN_FILE="${FSD_CACHE_SHARD_DRAIN_FILE:-docs/cluster_cache_3l_drain.order}"
export FSD_CACHE_PRESERVE_DRAIN_FILE="${FSD_CACHE_PRESERVE_DRAIN_FILE:-false}"
export FSD_CACHE_INITIAL_DRAIN_UNTIL_FORMULA_JSON="${FSD_CACHE_INITIAL_DRAIN_UNTIL_FORMULA_JSON:-false}"
export FSD_CACHE_STOP_FILE="${FSD_CACHE_STOP_FILE:-stop.order}"
export FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST="${FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST:-8b6db81d271bc742205632d6cd44a7d488a32c01a9064d3bae4e403b36e76698}"
export FSD_CHAIN_RULE_CHECKPOINT_GUARD_SOFT_DRAIN_GIB="${FSD_CHAIN_RULE_CHECKPOINT_GUARD_SOFT_DRAIN_GIB:-600}"
export FSD_CHAIN_RULE_CHECKPOINT_GUARD_WAITER_DRAIN_GIB="${FSD_CHAIN_RULE_CHECKPOINT_GUARD_WAITER_DRAIN_GIB:-0}"
export FSD_CHAIN_RULE_CHECKPOINT_GUARD_POLL_SECONDS="${FSD_CHAIN_RULE_CHECKPOINT_GUARD_POLL_SECONDS:-60}"
export FSD_CHAIN_RULE_CHECKPOINT_GUARD_PID_FILE="${FSD_CHAIN_RULE_CHECKPOINT_GUARD_PID_FILE:-docs/cluster_cache_3l_checkpoint_guard.pid}"
export FSD_CHAIN_RULE_CHECKPOINT_GUARD_LOG_FILE="${FSD_CHAIN_RULE_CHECKPOINT_GUARD_LOG_FILE:-docs/cluster_cache_3l_checkpoint_guard.log}"
export FSD_CHAIN_RULE_DRAIN_RELEASE_PID_FILE="${FSD_CHAIN_RULE_DRAIN_RELEASE_PID_FILE:-docs/cluster_cache_3l_drain_release.pid}"
export FSD_CHAIN_RULE_DRAIN_RELEASE_LOG_FILE="${FSD_CHAIN_RULE_DRAIN_RELEASE_LOG_FILE:-docs/cluster_cache_3l_drain_release.log}"
export FSD_CHAIN_RULE_DRAIN_RELEASE_AFTER="${FSD_CHAIN_RULE_DRAIN_RELEASE_AFTER:-formula-json}"
export FSD_CHAIN_RULE_MEMORY_GUARD_PID_FILE="${FSD_CHAIN_RULE_MEMORY_GUARD_PID_FILE:-docs/cluster_cache_3l_memory_guard.pid}"
export FSD_CHAIN_RULE_MEMORY_GUARD_LOG_FILE="${FSD_CHAIN_RULE_MEMORY_GUARD_LOG_FILE:-docs/cluster_cache_3l_memory_guard.log}"
export FSD_CHAIN_RULE_MEMORY_GUARD_TRIGGER_GIB="${FSD_CHAIN_RULE_MEMORY_GUARD_TRIGGER_GIB:-650}"
export FSD_CHAIN_RULE_MEMORY_GUARD_BUSY_CPU_THRESHOLD="${FSD_CHAIN_RULE_MEMORY_GUARD_BUSY_CPU_THRESHOLD:-101}"

if [[ "$FSD_CACHE_INITIAL_DRAIN_UNTIL_FORMULA_JSON" == "true" && -n "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST" ]]; then
  if [[ ! -e "$FSD_CACHE_SHARD_DRAIN_FILE" ]]; then
    cat > "$FSD_CACHE_SHARD_DRAIN_FILE" <<EOF
drain_requested_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
reason=initial_drain_until_formula_json
protected_digest=$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST
EOF
    echo "created initial drain file $FSD_CACHE_SHARD_DRAIN_FILE"
  fi
fi

if [[ -e "$FSD_CACHE_SHARD_DRAIN_FILE" && "$FSD_CACHE_PRESERVE_DRAIN_FILE" != "true" && "$FSD_CACHE_INITIAL_DRAIN_UNTIL_FORMULA_JSON" != "true" ]]; then
  rm -f -- "$FSD_CACHE_SHARD_DRAIN_FILE"
  echo "removed stale drain file $FSD_CACHE_SHARD_DRAIN_FILE"
elif [[ -e "$FSD_CACHE_SHARD_DRAIN_FILE" ]]; then
  echo "preserving drain file $FSD_CACHE_SHARD_DRAIN_FILE"
fi
if [[ -e "$FSD_CACHE_STOP_FILE" ]]; then
  rm -f -- "$FSD_CACHE_STOP_FILE"
  echo "removed stale stop file $FSD_CACHE_STOP_FILE"
fi

exec > >(tee -a docs/cluster_cache_3l_driver.log) 2>&1

if [[ -n "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST" ]]; then
  guard_pid=""
  if [[ -f "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_PID_FILE" ]]; then
    guard_pid="$(<"$FSD_CHAIN_RULE_CHECKPOINT_GUARD_PID_FILE")"
  fi
  if [[ -n "$guard_pid" ]] && kill -0 "$guard_pid" 2>/dev/null; then
    echo "checkpoint guard already running pid=$guard_pid"
  else
    setsid nohup .venv/bin/python scripts/guard_chain_rule_expression_checkpoint.py \
      --digest "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST" \
      --poll-seconds "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_POLL_SECONDS" \
      --soft-drain-gib "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_SOFT_DRAIN_GIB" \
      --waiter-drain-gib "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_WAITER_DRAIN_GIB" \
      --drain-file "$FSD_CACHE_SHARD_DRAIN_FILE" \
      --log-file "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_LOG_FILE" \
      > docs/cluster_cache_3l_checkpoint_guard.nohup 2>&1 &
    echo $! > "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_PID_FILE"
    echo "checkpoint guard started pid=$(<"$FSD_CHAIN_RULE_CHECKPOINT_GUARD_PID_FILE") digest=$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST"
  fi
  release_pid=""
  if [[ -f "$FSD_CHAIN_RULE_DRAIN_RELEASE_PID_FILE" ]]; then
    release_pid="$(<"$FSD_CHAIN_RULE_DRAIN_RELEASE_PID_FILE")"
  fi
  if [[ -n "$release_pid" ]] && kill -0 "$release_pid" 2>/dev/null; then
    echo "drain release watcher already running pid=$release_pid"
  else
    setsid nohup .venv/bin/python scripts/release_cache_drain_after_checkpoint.py \
      --digest "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST" \
      --poll-seconds "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_POLL_SECONDS" \
      --drain-file "$FSD_CACHE_SHARD_DRAIN_FILE" \
      --log-file "$FSD_CHAIN_RULE_DRAIN_RELEASE_LOG_FILE" \
      --release-after "$FSD_CHAIN_RULE_DRAIN_RELEASE_AFTER" \
      > docs/cluster_cache_3l_drain_release.nohup 2>&1 &
    echo $! > "$FSD_CHAIN_RULE_DRAIN_RELEASE_PID_FILE"
    echo "drain release watcher started pid=$(<"$FSD_CHAIN_RULE_DRAIN_RELEASE_PID_FILE") digest=$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST"
  fi
  memory_guard_pid=""
  if [[ -f "$FSD_CHAIN_RULE_MEMORY_GUARD_PID_FILE" ]]; then
    memory_guard_pid="$(<"$FSD_CHAIN_RULE_MEMORY_GUARD_PID_FILE")"
  fi
  if [[ -n "$memory_guard_pid" ]] && kill -0 "$memory_guard_pid" 2>/dev/null; then
    echo "memory guard already running pid=$memory_guard_pid"
  else
    setsid nohup .venv/bin/python scripts/protect_chain_rule_checkpoint_memory.py \
      --digest "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST" \
      --poll-seconds "$FSD_CHAIN_RULE_CHECKPOINT_GUARD_POLL_SECONDS" \
      --trigger-gib "$FSD_CHAIN_RULE_MEMORY_GUARD_TRIGGER_GIB" \
      --busy-cpu-threshold "$FSD_CHAIN_RULE_MEMORY_GUARD_BUSY_CPU_THRESHOLD" \
      --drain-file "$FSD_CACHE_SHARD_DRAIN_FILE" \
      --log-file "$FSD_CHAIN_RULE_MEMORY_GUARD_LOG_FILE" \
      > docs/cluster_cache_3l_memory_guard.nohup 2>&1 &
    echo $! > "$FSD_CHAIN_RULE_MEMORY_GUARD_PID_FILE"
    echo "memory guard started pid=$(<"$FSD_CHAIN_RULE_MEMORY_GUARD_PID_FILE") digest=$FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST"
  fi
fi

exec .venv/bin/python run_with_memory_watch.py \
  --limit-gb "$FSD_CACHE_WATCHDOG_LIMIT_GB" \
  --poll-seconds 30 \
  --pid-file docs/cluster_cache_3l_watchdog_child.pid \
  --log-file docs/cluster_cache_3l_watchdog.log \
  --stop-file "$FSD_CACHE_STOP_FILE" \
  -- \
  .venv/bin/python scripts/run_cache_shards.py \
    --variant all \
    --jobs "$FSD_CACHE_SHARD_JOBS" \
    --max-task-attempts "$FSD_CACHE_SHARD_MAX_ATTEMPTS" \
    --deferred-retry-seconds "${FSD_CACHE_DEFERRED_RETRY_SECONDS:-300}" \
    --drain-file "$FSD_CACHE_SHARD_DRAIN_FILE" \
    --shards-per-case 100 \
    --cache-verify-samples-per-sector 1 \
    --progress-seconds 30 \
    --resume
'
