#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${LSGEMU_CONFIG_FILE:-$PROJECT_ROOT/configs/lsgemu_config.yaml}"

OUT="${1:-/tmp/lsgemu_elfmultifuzz_full_4h_10m_profiles_20260603_serial}"
JOBS="${JOBS:-1}"
FIRMWARE_MINUTES="${FIRMWARE_MINUTES:-240}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-600}"
CASE_TIMEOUT_GRACE_SECONDS="${CASE_TIMEOUT_GRACE_SECONDS:-1200}"
# Unicorn and angr reserve large virtual address ranges; RLIMIT_AS can fail
# translator-buffer allocation even when physical RAM is sufficient.
CASE_MEMORY_LIMIT_GB="${CASE_MEMORY_LIMIT_GB:-0}"
RUNNER_LOG_LEVEL="${RUNNER_LOG_LEVEL:-INFO}"
RESOURCE_MONITOR_SECONDS="${RESOURCE_MONITOR_SECONDS:-600}"
NICE_LEVEL="${NICE_LEVEL:-10}"
IONICE_CLASS="${IONICE_CLASS:-2}"
IONICE_LEVEL="${IONICE_LEVEL:-7}"

cd "$PROJECT_ROOT"

export LSGEMU_CONFIG_FILE="$CONFIG_FILE"
eval "$(
python3 - <<'PY'
from lsgemu.deployment_config import load_and_apply_deployment_config
cfg = load_and_apply_deployment_config()
for key in ("PYTHONPATH", "LIBUNICORN_PATH", "LD_LIBRARY_PATH", "LSGEMU_SOURCE_ROOT", "LSGEMU_PROJECT_ROOT"):
    import os
    value = os.environ.get(key)
    if value:
        print(f"export {key}={value!r}")
PY
)"
export PYTHONUNBUFFERED=1

mkdir -p "$OUT"

{
    echo "[start] $(date -Is)"
    echo "[config] output_root=$OUT"
    echo "[config] jobs=$JOBS firmware_minutes=$FIRMWARE_MINUTES progress_interval_seconds=$PROGRESS_INTERVAL_SECONDS"
    echo "[config] case_timeout_grace_seconds=$CASE_TIMEOUT_GRACE_SECONDS case_memory_limit_gb=$CASE_MEMORY_LIMIT_GB"
    echo "[config] nice=$NICE_LEVEL ionice_class=$IONICE_CLASS ionice_level=$IONICE_LEVEL"
} | tee "$OUT/driver_console.log"

monitor_resources() {
    while true; do
        sleep "$RESOURCE_MONITOR_SECONDS" || break
        {
            echo "[resource] $(date -Is)"
            free -h
            ps -eo pid,ppid,pcpu,pmem,rss,vsz,etime,cmd --sort=-rss | head -30
        } >> "$OUT/resource_monitor.log" 2>&1
    done
}

monitor_resources &
monitor_pid=$!
cleanup_monitor() {
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
}
trap cleanup_monitor EXIT

runner_cmd=(
python3 -m lsgemu.run_elfmultifuzz_interleaved_strict_parallel_campaign \
    --output-root "$OUT" \
    --firmware-minutes "$FIRMWARE_MINUTES" \
    --progress-interval-seconds "$PROGRESS_INTERVAL_SECONDS" \
    --jobs "$JOBS" \
    --case-timeout-grace-seconds "$CASE_TIMEOUT_GRACE_SECONDS" \
    --runner-log-level "$RUNNER_LOG_LEVEL" \
    --case-memory-limit-gb "$CASE_MEMORY_LIMIT_GB" \
    --config "$CONFIG_FILE" \
    --disable-low-coverage-retry
)

if command -v ionice >/dev/null 2>&1; then
    nice -n "$NICE_LEVEL" ionice -c "$IONICE_CLASS" -n "$IONICE_LEVEL" "${runner_cmd[@]}" \
        2>&1 | tee -a "$OUT/driver_console.log"
else
    nice -n "$NICE_LEVEL" "${runner_cmd[@]}" \
        2>&1 | tee -a "$OUT/driver_console.log"
fi

status=${PIPESTATUS[0]}
cleanup_monitor
echo "[end] $(date -Is) status=$status" | tee -a "$OUT/driver_console.log"
exit "$status"
