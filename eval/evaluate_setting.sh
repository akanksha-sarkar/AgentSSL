#!/bin/bash
# Submit one evaluation job per run directory under a setting folder.
#
# This is intentionally styled like `submit_all_test.sh`:
# - discover run dirs by finding `json/` folders under `.../aira/<run_id>`
# - optionally filter
# - dry-run option
# - write stdout/err inside each run dir
#
# Dry-run:
#   bash evaluate_setting.sh --setting-dir /share/j_sun/as2637/logs/clevr_count/aSSL_backbone_unsupervised --eval-type test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$SCRIPT_DIR"
EVALUATE_RUN_PY="$EVAL_DIR/evaluate_run.py"

SETTING_DIR=""
FILTER_K=""
FILTER_SEED=""
EVAL_TYPE="test"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setting-dir) SETTING_DIR="$2"; shift 2 ;;
    --k)           FILTER_K="$2"; shift 2 ;;
    --seed)        FILTER_SEED="$2"; shift 2 ;;
    --eval-type)   EVAL_TYPE="$2"; shift 2 ;;
    --dry-run)     DRY_RUN=1; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$SETTING_DIR" ]]; then
  echo "ERROR: --setting-dir is required" >&2
  exit 1
fi
if [[ "$EVAL_TYPE" != "test" && "$EVAL_TYPE" != "val" ]]; then
  echo "ERROR: --eval-type must be 'test' or 'val' (got: $EVAL_TYPE)" >&2
  exit 1
fi
if [[ ! -f "$EVALUATE_RUN_PY" ]]; then
  echo "ERROR: evaluate_run.py not found at: $EVALUATE_RUN_PY" >&2
  exit 1
fi

SETTING_DIR="$(python3 - "$SETTING_DIR" <<'PY'
import os, sys
print(os.path.abspath(sys.argv[1]))
PY
)"

echo "[evaluate_setting] setting_dir=$SETTING_DIR"
echo "[evaluate_setting] k_filter=${FILTER_K:-<none>}"
echo "[evaluate_setting] seed_filter=${FILTER_SEED:-<none>}"
echo "[evaluate_setting] eval_type=$EVAL_TYPE"
echo "[evaluate_setting] dry_run=$DRY_RUN"
echo "[evaluate_setting] eval_dir=$EVAL_DIR"
echo "[evaluate_setting] evaluator=$EVALUATE_RUN_PY"

# Discover run dirs under:
#   <setting-dir>/<k>/<seed>/aira/<run_id>
mapfile -t RUN_DIRS < <(
  find "$SETTING_DIR" -type d -name "json" \
    | grep "/aira/" \
    | xargs -I{} dirname {} \
    | sort
)

echo "[evaluate_setting] found_runs=${#RUN_DIRS[@]}"
if [[ ${#RUN_DIRS[@]} -eq 0 ]]; then
  echo "[evaluate_setting] nothing to submit"
  exit 0
fi

submitted=0
skipped=0

for RUN_DIR in "${RUN_DIRS[@]}"; do
  if [[ -n "$FILTER_K" && "$RUN_DIR" != *"/$FILTER_K/"* ]]; then
    continue
  fi
  if [[ -n "$FILTER_SEED" && "$RUN_DIR" != *"/$FILTER_SEED/"* ]]; then
    continue
  fi

  # Make it easy to grep logs
  echo "[evaluate_setting] run=$RUN_DIR"

  JOB_NAME="eval_$(basename "$(dirname "$(dirname "$RUN_DIR")")")_$(basename "$RUN_DIR")"
  OUT_PATH="$RUN_DIR/evalrun_slurm_%j.out"
  ERR_PATH="$RUN_DIR/evalrun_slurm_%j.err"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[evaluate_setting]   would_submit: $RUN_DIR"
    skipped=$((skipped + 1))
    continue
  fi

  sbatch \
    --job-name="$JOB_NAME" \
    --output="$OUT_PATH" \
    --error="$ERR_PATH" \
    --partition=jjs533,gpu \
    --gres=gpu:1 \
    --constraint='gpu-high' \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --time=24:00:00 \
    --requeue \
    --wrap="#!/bin/bash
eval \"\$(/home/as2637/miniconda/bin/conda shell.bash hook)\"
conda activate aira-dojo
cd '$EVAL_DIR'
python '$EVALUATE_RUN_PY' --log-dir '$RUN_DIR' --eval-type '$EVAL_TYPE'
"

  submitted=$((submitted + 1))
done

echo "[evaluate_setting] done submitted=$submitted dryrun_skipped=$skipped total=${#RUN_DIRS[@]}"