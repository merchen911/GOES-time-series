#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# v004 multi-GPU queue launcher / resumer. Grid = the driver's SEQ_LENS x
# PRED_LENS (36/72/144/288) x folds 0-4, direct strategy; the scheduler's own
# flags (--pred-lens, --folds, --runs-root, ...) select a subset when run directly.
#
#   * One job = ONE MODEL. The scheduler keeps every GPU fed from a shared
#     queue, so a slow model delays only itself, never a whole GPU.
#   * SAFE TO RE-RUN: a job whose result row exists is skipped, so this command
#     doubles as the RESUME after any crash, kill, or NFS flap.
#   * Detached (setsid) — it survives the shell that started it.
#
# Usage:  bash scripts/run_queue.sh              # GPUs 0 1 2
#         bash scripts/run_queue.sh 0 1          # a subset
# Watch:  tail -f runs/bench/queue_sweep.log
#         column -t -s, runs/bench/resource_monitor.csv | tail
# Stop:   kill $(cat runs/bench/queue_sweep.pid)      # SIGTERM stops cleanly
# ---------------------------------------------------------------------------
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE" || exit 1

GPUS="${*:-0 1 2}"
LOG=runs/bench/queue_sweep.log
PIDF=runs/bench/queue_sweep.pid

# The parquet pair lives outside the repo (never committed); the driver reads
# this and fails loudly rather than silently pointing at a missing file.
export SW_DATA_DIR="${SW_DATA_DIR:-$BASE/data/goes_data/processed}"

if pgrep -f "[t]slib.benchmark.scheduler" >/dev/null; then
  echo "ALREADY RUNNING: pids $(pgrep -f '[t]slib.benchmark.scheduler' | tr '\n' ' ')"
  exit 0
fi

mkdir -p runs/bench
echo "=== launch/resume $(date -u) (gpus $GPUS) ===" >> "$LOG"
setsid bash -c 'echo $$ > "'"$PIDF"'"; exec python3.12 -m tslib.benchmark.scheduler --gpus '"$GPUS"' >> "'"$LOG"'" 2>&1' < /dev/null > /dev/null 2>&1 &
disown
sleep 1
echo "launched (gpus $GPUS). pid file: $PIDF  log: $LOG"
