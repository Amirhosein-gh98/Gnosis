#!/usr/bin/env bash 
set -euo pipefail

# ========= Config =========
# Reward model
# RM_MODEL="Skywork/Skywork-Reward-V2-Llama-3.1-8B"
RM_MODEL="Skywork/Skywork-Reward-V2-Qwen3-8B"
RM_NAME="$(basename "$RM_MODEL")"                 # e.g., Skywork-Reward-V2-Llama-3.1-8B


# Datasets
MMLUPRO_DIR="Path_to_backbone_reposnes_for_MMLUPRO"
MATH10_DIR="Path_to_backbone_reposnes_for_Math"
TRIVIA_DIR="Path_to_backbone_reposnes_for_TriviaQA_eval"



# Output root; scorer will still add: /scored/<rm_model_name>/ underneath
OUT_BASE="outputs/scored_runs/RM"

# Optional knobs
PATTERN="shard-*.parquet"   # or *.jsonl
THRESHOLDS=""               # e.g., "0.1,0.2,0.5"
STRIDE=1
LIMIT_SHARDS=0
BATCH_SIZE=1
MAX_LEN=32000

# Path to the scorer script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_DIR="$(dirname "$SCRIPT_DIR")"                     # .../project (drop /scripts)
SCORER="${SCRIPT_DIR}/score_completions_ORM_outputscores_script_version.py"


run_job () {
  local input_dir="$1"
  local task="$2"

  if [[ ! -d "$input_dir" ]]; then
    echo "!! Skipping (missing dir): $input_dir" >&2
    return
  fi

  # combined folder: ORM/<rm_name>_<data_name>/
  local data_name
  data_name="$(basename "$input_dir")"
  local combo="${RM_NAME}_${data_name}"
  local out_dir="${OUT_BASE}/${combo}"

  echo "==== RM eval: task='${task}' dir='${input_dir}' out='${out_dir}' ===="
  CUDA_VISIBLE_DEVICES=1 python "$SCORER" \
    --rm_model "$RM_MODEL" \
    --input_dir "$input_dir" \
    --output_dir "$out_dir" \
    --task "$task" \
    --pattern "$PATTERN" \
    --stride "$STRIDE" \
    --limit_shards "$LIMIT_SHARDS" \
    --batch_size "$BATCH_SIZE" \
    --max_len "$MAX_LEN" \
    ${THRESHOLDS:+--thresholds "$THRESHOLDS"}
  echo
}

# Run all

run_job "$MATH10_DIR"  math
# run_job "$TRIVIA_DIR"  trivia
# run_job "$MMLUPRO_DIR" gpqa
run_job "$MATH24_DIR"  math
# run_job "$MMLUPRO_DIR_" gpqa
# run_job "$MATH10_DIR_"  math
# run_job "$MATH24_DIR_"  math
# run_job "$TRIVIA_DIR_"  trivia

echo "All RM evals finished."
echo "Outputs under: ${OUT_BASE}/<rm_name>_<data_name>/ ..."