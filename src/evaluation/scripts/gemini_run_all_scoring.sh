#!/usr/bin/env bash 
set -euo pipefail

# ========= Config =========
# Judge Model
JUDGE_MODEL="gemini-2.5-pro" 

# Datasets
MMLUPRO_DIR="Path_to_backbone_reposnes_for_MMLUPRO"
MATH10_DIR="Path_to_backbone_reposnes_for_Math"
TRIVIA_DIR="Path_to_backbone_reposnes_for_TriviaQA_eval"

# Output root
OUT_BASE="outputs/scored_runs/Gemini_Judge"

# Optional knobs
PATTERN="shard-*.parquet"   # or *.jsonl
THRESHOLDS=""               # e.g., "0.1,0.2,0.5"
STRIDE=1
LIMIT_SHARDS=0
BATCH_SIZE=64               # Increased for API concurrency (Thread pool)



# Path to the scorer script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../project/scripts
SCRIPT_DIR="$(dirname "$SCRIPT_DIR")"                     # .../project (drop /scripts)
SCORER="${PROJECT_ROOT}/score_completions_gemini_outputscores_script_version.py"


# Ensure API Key is available (optional check, useful for debugging)
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "Info: GEMINI_API_KEY env var not set. Relying on hardcoded key in python script."
fi

run_job () {
  local input_dir="$1"
  local task="$2"

  if [[ ! -d "$input_dir" ]]; then
    echo "!! Skipping (missing dir): $input_dir" >&2
    return
  fi

  # combined folder: <judge_model>_<data_name>/
  local judge_clean_name
  judge_clean_name="${JUDGE_MODEL//:/-}" # replace : with - if present
  
  local data_name
  data_name="$(basename "$input_dir")"
  
  local combo="${judge_clean_name}_${data_name}"
  local out_dir="${OUT_BASE}/${combo}"

  echo "==== Gemini Judge eval: task='${task}' dir='${input_dir}' out='${out_dir}' ===="
  

  python "$SCORER" \
    --model "$JUDGE_MODEL" \
    --input_dir "$input_dir" \
    --output_dir "$out_dir" \
    --task "$task" \
    --pattern "$PATTERN" \
    --stride "$STRIDE" \
    --limit_shards "$LIMIT_SHARDS" \
    --batch_size "$BATCH_SIZE" \
    ${THRESHOLDS:+--thresholds "$THRESHOLDS"}
  echo
}

# Run all (Modify as needed)
run_job "$MATH10_DIR"  math
run_job "$TRIVIA_DIR"  trivia
run_job "$MMLUPRO_DIR" gpqa


echo "All Gemini Judge evals finished."
echo "Outputs under: ${OUT_BASE}/ ..."