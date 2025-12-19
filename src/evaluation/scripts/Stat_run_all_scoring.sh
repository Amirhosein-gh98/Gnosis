#!/usr/bin/env bash
set -euo pipefail

# # === Paths ===
MODEL="Qwen/Qwen3-1.7B" #backnone

# Datasets
MMLUPRO_DIR="Path_to_backbone_reposnes_for_MMLUPRO"
MATH10_DIR="Path_to_backbone_reposnes_for_Math"
TRIVIA_DIR="Path_to_backbone_reposnes_for_TriviaQA_eval"



# Where to write results (STAT saves under: /.../STAT/<model>_<dataname>/scored/<model>/stats_full/)
MODEL_NAME="$(basename "$MODEL")"
OUT_ROOT="outputs/scored_runs/STAT"

# Optional
PATTERN="shard-*.parquet"
THRESHOLDS=""          # e.g., "0.2,0.5,0.8,0.9"
OUT_SUBDIR="scored"
STATS_SUBDIR="stats_full"

# Scorer script (statistical outputs)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(dirname "$SCRIPT_DIR")"                     # .../project (drop /scripts)
SCORER="${SCRIPT_DIR}/score_completions_statistic_outputscores_script_version.py"


run_job () {
  local input_dir="$1"
  local task="$2"

  if [[ ! -d "$input_dir" ]]; then
    echo "!! Skipping (missing dir): $input_dir" >&2
    return
  fi

  local data_name
  data_name="$(basename "$input_dir")"
  local out_base="${OUT_ROOT}/${MODEL_NAME}_${data_name}"

  echo "==== STAT run: task='${task}' on dir='${input_dir}' → out='${out_base}' ===="
  CUDA_VISIBLE_DEVICES=1 python "$SCORER" \
    --model "$MODEL" \
    --input_dir "$input_dir" \
    --output_dir "$out_base" \
    --task "$task" \
    --pattern "$PATTERN" \
    --out_subdir "$OUT_SUBDIR" \
    --stats_subdir "$STATS_SUBDIR" \
    ${THRESHOLDS:+--thresholds "$THRESHOLDS"}
  echo
}

# Run sequentially (no string splitting needed)

run_job "$MATH10_DIR"  math
run_job "$TRIVIA_DIR"  trivia
run_job "$MMLUPRO_DIR" gpqa

echo "All runs finished. Outputs under: ${OUT_ROOT}/${MODEL_NAME}_<dataname>/${OUT_SUBDIR}/${MODEL_NAME}/${STATS_SUBDIR}/"
