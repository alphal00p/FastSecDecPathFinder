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
export FSD_CACHE_STOP_FILE="${FSD_CACHE_STOP_FILE:-stop.order}"
export FSD_CACHE_SHARD_DRAIN_FILE="${FSD_CACHE_SHARD_DRAIN_FILE:-docs/cluster_cache_3l_drain.order}"

protected_digest="${FSD_CHAIN_RULE_CHECKPOINT_GUARD_DIGEST:-8b6db81d271bc742205632d6cd44a7d488a32c01a9064d3bae4e403b36e76698}"
if [[ ! -e "$FSD_CACHE_SHARD_DRAIN_FILE" ]]; then
  cat > "$FSD_CACHE_SHARD_DRAIN_FILE" <<EOF
drain_requested_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
reason=protected_chain_rule_resume
protected_digest=$protected_digest
EOF
  echo "created protected resume drain file $FSD_CACHE_SHARD_DRAIN_FILE"
else
  echo "preserving drain file $FSD_CACHE_SHARD_DRAIN_FILE"
fi
if [[ -e "$FSD_CACHE_STOP_FILE" ]]; then
  rm -f -- "$FSD_CACHE_STOP_FILE"
  echo "removed stale stop file $FSD_CACHE_STOP_FILE"
fi

protected_case="${FSD_PROTECTED_RESUME_CASE:-triple_box}"
protected_shard_label="${FSD_PROTECTED_RESUME_SHARD_LABEL:-triple_box_shard_0013_of_0099}"
protected_sectors="${FSD_PROTECTED_RESUME_SECTORS:-240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259}"
read -r -a protected_sector_args <<< "$protected_sectors"
if (( ${#protected_sector_args[@]} == 0 )); then
  echo "FSD_PROTECTED_RESUME_SECTORS must contain at least one sector id" >&2
  exit 2
fi

report_path="${FSD_PROTECTED_RESUME_REPORT_PATH:-docs/cache_shards/reports/triple-box-direct/${protected_shard_label}.json}"
workdir="${FSD_PROTECTED_RESUME_WORKDIR:-.cache_warm_cluster_shards/triple-box-direct/${protected_shard_label}}"
log_path="${FSD_PROTECTED_RESUME_LOG_PATH:-docs/cache_shards/logs/triple-box-direct/${protected_shard_label}.log}"
mkdir -p "$(dirname "$report_path")" "$(dirname "$log_path")" "$workdir"
rm -f -- "$report_path" "$report_path.tmp"

{
  printf "[cache-shards] protected_chain_rule_resume_start time_utc=%s digest=%s case=%s shard=%s sectors=%s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$protected_digest" "$protected_case" "$protected_shard_label" "$protected_sectors"
  printf "[cache-shards] protected_chain_rule_resume_env iterations=%s cpe_iterations=%s max_horner_vars=%s cpe_cache_entries=%s cpe_distance=%s watchdog_gib=%s\n" \
    "$FSD_SYMBOLICA_EVALUATOR_ITERATIONS" \
    "$FSD_SYMBOLICA_EVALUATOR_CPE_ITERATIONS" \
    "$FSD_SYMBOLICA_MAX_HORNER_SCHEME_VARIABLES" \
    "$FSD_SYMBOLICA_MAX_COMMON_PAIR_CACHE_ENTRIES" \
    "$FSD_SYMBOLICA_MAX_COMMON_PAIR_DISTANCE" \
    "$FSD_CACHE_WATCHDOG_LIMIT_GB"
} >> "$log_path"

export FSD_PROTECTED_RESUME_LOG_PATH="$log_path"

exec > >(tee -a docs/cluster_cache_3l_driver.log) 2>&1

exec .venv/bin/python run_with_memory_watch.py \
  --limit-gb "$FSD_CACHE_WATCHDOG_LIMIT_GB" \
  --poll-seconds 30 \
  --pid-file docs/cluster_cache_3l_watchdog_child.pid \
  --log-file docs/cluster_cache_3l_watchdog.log \
  --stop-file "$FSD_CACHE_STOP_FILE" \
  -- \
  bash -lc '"'"'exec "$@" >> "$FSD_PROTECTED_RESUME_LOG_PATH" 2>&1'"'"' _ \
  .venv/bin/python FSD.py \
    cache \
    --cache-loop-counts 3 \
    --cache-verify-samples-per-sector 1 \
    --max-eps-order 0 \
    --force-regular-taylor-formulas \
    --chain-rule-formula-signature-limit 1000000 \
    --chain-rule-formula-output-length-limit 0 \
    --no-cache-estimate-3l \
    --no-progress \
    --json \
    --cache-cases "$protected_case" \
    --sectors \
    "${protected_sector_args[@]}" \
    --cache-report-path "$report_path" \
    --cache-workdir "$workdir"
'
