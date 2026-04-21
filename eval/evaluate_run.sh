#!/bin/bash
#SBATCH --job-name=all_test_eval
#SBATCH --partition=jjs533,gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint='gpu-high'
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=15:00:00
#SBATCH --requeue
#SBATCH --output=/share/j_sun/as2637/logs/slurm-%x-%j.out
#SBATCH --error=/share/j_sun/as2637/logs/slurm-%x-%j.err


#--------------------------------
# Example usage:
#    sbatch evaluate_run.sh --log-dir /share/j_sun/as2637/logs/dtd/aSSL_backbone_unsupervised/k6/seed26/aira/2 --eval-type test
#--------------------------------

set -euo pipefail

# -------------------
# parse flags
# -------------------
LOG_DIR=""
EVAL_TYPE="test"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --eval-type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$LOG_DIR" ]]; then
    echo "Error: --log-dir is required"
    exit 1
fi

# -------------------
# env setup
# -------------------
source ~/miniconda/etc/profile.d/conda.sh
conda activate aira-dojo

#cd /home/as2637/agentSSL/eval

# -------------------
# run
# -------------------
python evaluate_run.py \
    --log-dir "$LOG_DIR" \
    --eval-type "$EVAL_TYPE"