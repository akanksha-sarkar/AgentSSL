#!/bin/bash
# Submit SLURM jobs to evaluate every non-buggy program on the test set for all (or selected) resisc45 runs.
#
# Usage:
#   bash submit_all_test.sh                    # all runs
#   bash submit_all_test.sh --setting aSSL_0_1_metric_prior      # only aSSL
#   bash submit_all_test.sh --setting aSSL_all_metric_prior      # only aSL
#   bash submit_all_test.sh --setting aSSL_ami_prior      # only aSL
#   bash submit_all_test.sh --setting aSSL_backbone_1804    # only aSL
#   bash submit_all_test.sh --setting aSSL_0_1_metric_priorlora2_mcts     # only aSL
#   bash submit_all_test.sh --dry-run          # print without submitting

set -euo pipefail

LOG_BASE="/share/j_sun/as2637/logs/clevr_count"
TEST_DIR="$(cd "$(dirname "$0")" && pwd)"
FILTER_SETTING=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --setting) FILTER_SETTING="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1;           shift   ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# Collect all run dirs: .../clevr_count/{setting}/{k}/{seed}/aira/{run_id}
mapfile -t RUN_DIRS < <(
    find "$LOG_BASE" -type d -name "json" \
        | grep "/aira/" \
        | xargs -I{} dirname {} \
        | sort
)

echo "Found ${#RUN_DIRS[@]} run dirs"

for RUN_DIR in "${RUN_DIRS[@]}"; do
    if [[ -n "$FILTER_SETTING" ]]; then
        if [[ "$RUN_DIR" != *"/$FILTER_SETTING/"* ]]; then
            continue
        fi
    fi

    JOB_NAME="alltest_$(basename "$(dirname "$(dirname "$RUN_DIR")")")_$(basename "$RUN_DIR")"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "Would submit: $RUN_DIR"
        continue
    fi

    sbatch \
        --job-name="$JOB_NAME" \
        --output="$RUN_DIR/alltest_slurm_%j.out" \
        --error="$RUN_DIR/alltest_slurm_%j.err" \
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
python validate_run_all_back.py --log-dir '$RUN_DIR'
"
    echo "Submitted: $RUN_DIR"
done
