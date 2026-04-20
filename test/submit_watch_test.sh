#!/bin/bash
# Submit long-running SLURM jobs that watch run directories and evaluate
# new programs on the test set every --interval seconds.
#
# Usage (single run):
#   bash submit_watch_test.sh --log-dir  /share/j_sun/as2637/logs/clevr_count/aSSL_0_1_metric_priorlora2_greedy/k10/seed26/aira/1111 --max-rounds 5 
# Usage (all runs under a setting — submits one job per run dir):
#   bash submit_watch_test.sh --setting aSSL_ami_prior
#   bash submit_watch_test.sh --setting aSSL_backbone_1804 --interval 1800 --max-rounds 10
#
# Other flags:
#   --interval   N    seconds between checks (default: 1800 = 30 min)
#   --max-rounds 5    stop after N rounds (default: 48 = 24 h at 30 min interval)
#   --dry-run         print without submitting

set -euo pipefail

LOG_BASE="/share/j_sun/as2637/logs/clevr_count"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=""
FILTER_SETTING=""
INTERVAL=1800
MAX_ROUNDS=48
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)    LOG_DIR="$2";          shift 2 ;;
        --setting)    FILTER_SETTING="$2";   shift 2 ;;
        --interval)   INTERVAL="$2";         shift 2 ;;
        --max-rounds) MAX_ROUNDS="$2";       shift 2 ;;
        --dry-run)    DRY_RUN=1;             shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$LOG_DIR" && -z "$FILTER_SETTING" ]]; then
    echo "ERROR: provide --log-dir <run-dir> or --setting <setting-name>"
    exit 1
fi

# Time limit: MAX_ROUNDS × INTERVAL + 10 min buffer, capped at 48 h
TIME_SECS=$(( MAX_ROUNDS * INTERVAL + 600 ))
TIME_SECS=$(( TIME_SECS < 172800 ? TIME_SECS : 172800 ))
TIME_LIMIT=$(printf "%d:%02d:%02d" $((TIME_SECS/3600)) $(( (TIME_SECS%3600)/60 )) $((TIME_SECS%60)))

submit_one() {
    local run_dir="$1"
    local job_name="watchtest_$(basename "$(dirname "$(dirname "$run_dir")")")_$(basename "$run_dir")"

    echo "Submitting watch job:"
    echo "  log-dir:    $run_dir"
    echo "  interval:   ${INTERVAL}s  |  max-rounds: $MAX_ROUNDS  |  time-limit: $TIME_LIMIT"
    echo "  job-name:   $job_name"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  (dry-run, not submitting)"
        return
    fi

    sbatch \
        --job-name="$job_name" \
        --output="$run_dir/watchtest_slurm_%j.out" \
        --error="$run_dir/watchtest_slurm_%j.err" \
        --partition=jjs533,gpu \
        --gres=gpu:1 \
        --constraint='gpu-high' \
        --exclude=abdelfattah-compute-02 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --time="$TIME_LIMIT" \
        --requeue \
        --wrap="#!/bin/bash
eval \"\$(/home/as2637/miniconda/bin/conda shell.bash hook)\"
conda activate aira-dojo
bash '$TEST_DIR/watch_and_test.sh' \
    --log-dir '$run_dir' \
    --interval '$INTERVAL' \
    --max-rounds '$MAX_ROUNDS'
"
    echo "  Submitted. Logs: $run_dir/watchtest_slurm_<jobid>.out"
    echo ""
}

# ── Single run ────────────────────────────────────────────────────────────────
if [[ -n "$LOG_DIR" ]]; then
    submit_one "$(realpath "$LOG_DIR")"
    exit 0
fi

# ── All runs under a setting ─────────────────────────────────────────────────
mapfile -t RUN_DIRS < <(
    find "$LOG_BASE" -type d -name "json" \
        | grep "/aira/" \
        | xargs -I{} dirname {} \
        | grep "/$FILTER_SETTING/" \
        | sort
)

echo "Found ${#RUN_DIRS[@]} run dirs for setting '$FILTER_SETTING'"
echo ""

for RUN_DIR in "${RUN_DIRS[@]}"; do
    submit_one "$RUN_DIR"
done
