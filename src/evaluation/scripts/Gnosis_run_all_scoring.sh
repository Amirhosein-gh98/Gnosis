#!/usr/bin/env bash
set -euo pipefail

# # === Paths ===
MODEL="Path_to_trained_gnosis_backbone"


# Datasets
MMLUPRO_DIR="Path_to_backbone_reposnes_for_MMLUPRO"
MATH10_DIR="Path_to_backbone_reposnes_for_Math"
TRIVIA_DIR="Path_to_backbone_reposnes_for_TriviaQA_eval"


# Where to write results
MODEL_PARENT="$(basename "$(dirname "$MODEL")")"
OUT_BASE="outputs/scored_runs/Gnosis"

# Optional
PATTERN="shard-*.parquet"
THRESHOLDS=""   # e.g., "0.2,0.5,0.8"

# Scorer script name you gave
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(dirname "$SCRIPT_DIR")"                     # .../project (drop /scripts)
SCORER="${SCRIPT_DIR}/score_completions_Gnosis_outputscores_script_version.py"


run_job () {
  local input_dir="$1"
  local task="$2"

  if [[ ! -d "$input_dir" ]]; then
    echo "!! Skipping (missing dir): $input_dir" >&2
    return
  fi

  echo "==== Running task='${task}' on dir='${input_dir}' ===="
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python "$SCORER" \
    --model "$MODEL" \
    --input_dir "$input_dir" \
    --output_dir "$OUT_BASE" \
    --task "$task" \
    --pattern "$PATTERN" \
    ${THRESHOLDS:+--thresholds "$THRESHOLDS"}
  echo
}

# Run sequentially (no string splitting needed)

run_job "$MATH10_DIR"  math
# run_job "$TRIVIA_DIR"  trivia
# run_job "$MMLUPRO_DIR" gpqa
run_job "$MATH24_DIR"  math

echo "All runs finished. Outputs under: $OUT_BASE/scored/<model_name>/"
