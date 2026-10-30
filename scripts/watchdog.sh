#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Watchdog for the queue sweep (tslib.benchmark.scheduler). Two jobs, one loop
# (every INTERVAL sec):
#   1. FORENSIC LOG: append a resource snapshot to resource_monitor.csv (host
#      RAM/swap, per-GPU util+mem, sweep process-tree RSS, running job count)
#      so a silent death can be diagnosed after the fact.
#   2. AUTO-RESUME: if the scheduler is dead, relaunch it via run_queue.sh
#      (which self-guards double-launch and skips finished jobs).
#
# FLAP-RESILIENT: the shared storage may be an NFS mount that flaps — the v003
# sweep died twice from it. So this runs from a LOCAL cwd, touches the repo only
# by ABSOLUTE path, and never uses `set -e`: an NFS blip fails one iteration's
# file ops but the loop survives and resumes the sweep once the mount returns.
#
# Usage:  bash <this> [INTERVAL_SEC] [GPUS...]     # defaults 60, "0 1 2"
# Stop:   kill $(cat <BASE>/runs/bench/watchdog.pid)
# ---------------------------------------------------------------------------
INTERVAL="${1:-60}"; shift 2>/dev/null
GPUS="${*:-0 1 2}"
# Resolve the repo root BEFORE leaving it (BASH_SOURCE may be a relative path).
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${TMPDIR:-/tmp}" 2>/dev/null || cd /   # stable local cwd — survive NFS flaps

PIDF="$BASE/runs/bench/queue_sweep.pid"
LOG="$BASE/runs/bench/queue_sweep.log"
WD_LOG="$BASE/runs/bench/watchdog.log"
WD_PIDF="$BASE/runs/bench/watchdog.pid"
RES_CSV="$BASE/runs/bench/resource_monitor.csv"
RUN="$BASE/scripts/run_queue.sh"

# --- guard: never run two watchdogs (but don't match myself) ---
if [ -f "$WD_PIDF" ]; then
  oldpid=$(cat "$WD_PIDF" 2>/dev/null)
  if [ -n "$oldpid" ] && [ "$oldpid" != "$$" ] && [ -e "/proc/$oldpid" ] \
     && grep -qa watchdog.sh "/proc/$oldpid/cmdline" 2>/dev/null; then
    echo "watchdog already running (pid $oldpid) — not starting a second"
    exit 0
  fi
fi
echo $$ > "$WD_PIDF"

if [ ! -f "$RES_CSV" ]; then
  echo "ts_utc,sched_alive,done,total,mem_total_mb,mem_used_mb,mem_avail_mb,swap_used_mb,gpu_util,gpu_mem_mb,tree_rss_mb,n_procs,n_jobs" > "$RES_CSV"
fi

sched_alive() {
  local p; p=$(cat "$PIDF" 2>/dev/null)
  [ -n "$p" ] && [ -e "/proc/$p" ] && grep -qa "benchmark.scheduler" "/proc/$p/cmdline" 2>/dev/null
}

tree_rss_and_count() {
  local root; root=$(cat "$PIDF" 2>/dev/null)
  if [ -z "$root" ] || [ ! -e "/proc/$root" ]; then echo "0 0"; return; fi
  local pids="$root" frontier="$root" next p kids
  while [ -n "${frontier// /}" ]; do
    next=""
    for p in $frontier; do kids=$(pgrep -P "$p" 2>/dev/null | tr '\n' ' '); next="$next $kids"; done
    frontier="$next"; pids="$pids $next"
  done
  local total=0 n=0 r
  for p in $pids; do
    [ -z "$p" ] && continue
    r=$(awk '/^VmRSS:/{print $2}' "/proc/$p/status" 2>/dev/null)
    if [ -n "$r" ]; then total=$((total + r)); n=$((n + 1)); fi
  done
  echo "$((total / 1024)) $n"
}

echo "=== watchdog start $(date -u) (interval ${INTERVAL}s, gpus $GPUS, cwd=$(pwd)) ===" >> "$WD_LOG"

while true; do
  ts=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
  read -r mt mu ma <<<"$(free -m | awk '/^Mem:/{print $2, $3, $7}')"
  su=$(free -m | awk '/^Swap:/{print $3}')
  gpu_list=$(echo "$GPUS" | tr ' ' ',')
  gu=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_list" 2>/dev/null | tr -d ' ' | paste -sd'|')
  gm=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_list" 2>/dev/null | tr -d ' ' | paste -sd'|')
  read -r rss nproc <<<"$(tree_rss_and_count)"
  njobs=$(pgrep -fc "[m]ain.py --data_path" 2>/dev/null); njobs=${njobs:-0}
  if sched_alive; then alive=1; else alive=0; fi
  prog=$(grep -oE '\[[0-9]+/[0-9]+\] gpu' "$LOG" 2>/dev/null | tail -1 | tr -d '[]' | sed 's/ gpu//')
  done_n=${prog%%/*}; total_n=${prog##*/}
  echo "$ts,$alive,${done_n:-NA},${total_n:-NA},${mt:-NA},${mu:-NA},${ma:-NA},${su:-NA},${gu:-NA},${gm:-NA},${rss:-NA},${nproc:-NA},$njobs" >> "$RES_CSV"

  # A FINISHED sweep is not a dead one. Without this check the watchdog keeps
  # relaunching a scheduler that immediately skips all 870 jobs and exits, once
  # a minute, forever — the v003 watchdog ran that way for weeks and the v004
  # one wrote 2.9M skip lines (283 MB) over the three days after completion.
  # The result rows are the truth: when every job has one, retire.
  rows=$(find "$BASE/runs/bench" -path '*/score/rows/*.json' 2>/dev/null | wc -l)
  if [ -n "$total_n" ] && [ "$total_n" != "NA" ] && [ "$rows" -ge "$total_n" ] 2>/dev/null; then
    echo "$(date -u) SWEEP COMPLETE ($rows/$total_n result rows) — watchdog retiring" >> "$WD_LOG"
    rm -f "$WD_PIDF"
    exit 0
  fi

  if [ "$alive" -eq 0 ]; then
    echo "$(date -u) SCHEDULER DOWN — resuming (progress=${prog:-NA} tree_rss=${rss}MB mem_avail=${ma}MB)" >> "$WD_LOG"
    bash "$RUN" $GPUS >> "$WD_LOG" 2>&1
  fi
  sleep "$INTERVAL"
done
