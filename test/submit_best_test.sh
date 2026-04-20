#!/bin/bash
# Submit a SLURM job to evaluate the best program at every N-th step for a run.
#
# Usage (single run):
#   bash submit_best_test.sh --log-dir /share/j_sun/as2637/logs/clevr_count/aSSL_noisy_val/k10/seed26/aira/1000
#   bash submit_best_test.sh --log-dir /share/j_sun/as2637/logs/clevr_count/aSSL_noisy_val/k10/seed26/aira/1000 --every 10
#
# Usage (all runs under a setting):
#   bash submit_best_test.sh --setting aSSL_noisy_val
#   bash submit_best_test.sh --setting aSSL_noisy_val --every 10
#
# Other flags:
#   --every  N    evaluate best program at every N-th step (default: 5)
#   --dry-run     print without submitting

set -euo pipefail

LOG_BASE="/share/j_sun/as2637/logs/clevr_count"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=""
FILTER_SETTING=""
EVERY=5
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir) LOG_DIR="$2";          shift 2 ;;
        --setting) FILTER_SETTING="$2";   shift 2 ;;
        --every)   EVERY="$2";            shift 2 ;;
        --dry-run) DRY_RUN=1;             shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$LOG_DIR" && -z "$FILTER_SETTING" ]]; then
    echo "ERROR: provide --log-dir <run-dir> or --setting <setting-name>"
    exit 1
fi

submit_one() {
    local run_dir="$1"
    local job_name="besttest_$(basename "$(dirname "$(dirname "$run_dir")")")_$(basename "$run_dir")"

    echo "Submitting best_prog_eval job:"
    echo "  log-dir:  $run_dir"
    echo "  every:    ${EVERY} steps"
    echo "  job-name: $job_name"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  (dry-run, not submitting)"
        return
    fi

    sbatch \
        --job-name="$job_name" \
        --output="$run_dir/besttest_slurm_%j.out" \
        --error="$run_dir/besttest_slurm_%j.err" \
        --partition=jjs533,gpu \
        --gres=gpu:1 \
        --constraint='gpu-high' \
        --exclude=abdelfattah-compute-02 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --mem=128G \
        --time=24:00:00 \
        --requeue \
        --wrap="#!/bin/bash
eval \"\$(/home/as2637/miniconda/bin/conda shell.bash hook)\"
conda activate aira-dojo
cd '$TEST_DIR'
python validate_run_best.py --log-dir '$run_dir' --every '$EVERY'
"
    echo "  Submitted. Logs: $run_dir/besttest_slurm_<jobid>.out"
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
