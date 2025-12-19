# Gnosis Data Pipeline
**Training + Benchmarks (Test/Eval)**

This repo uses one unified pipeline to:
1) **generate model completions** (two-step, chat-template, vLLM),
2) **verify/label those completions** (task-specific evaluators),
3) **(training only) merge + rebalance tasks** into a single SFT dataset.

> ✅ Same generation codepath for **training** and **benchmarks** — only the **data source** + **system prompt** change.

---

## Directory conventions

- **Training outputs**
  - `data/train/<MODEL_TAG>_<DATASET_NAME>/shard-*.{parquet|jsonl}`
  - Verified: `data/train/<...>/verified/shard-*.verified.{parquet|jsonl}`
  - Final merged SFT: `data/train/Final/<...>/merged_balanced.parquet`

- **Benchmark outputs**
  - `data/test/<MODEL_TAG>_<BENCH_NAME>/shard-*.{parquet|jsonl}`
  - Verified (optional): `data/test/<...>/verified/shard-*.verified.{parquet|jsonl}`

---

## Scripts at a glance

### Core
- **Generate completions (training + benchmarks)**  
  `src/data_preprocess/data_generation.py`

- **Verify / label completions (training + benchmarks)**  
  `src/data_preprocess/Label_for_SFT.py`

### Training-only
- **Merge + rebalance verified datasets (e.g., trivia + math)**  
  `src/data_preprocess/merge_sft_data.py`

### Benchmark preprocessing
- **Merge math benchmarks → standardized CSV**  
  `src/data_preprocess/test_data_prepreprocess/merge_eval_math_data.py`
- **Convert MMLU-Pro → minimal CSV**  
  `src/data_preprocess/test_data_prepreprocess/mmlupro.py`

---

## Pipeline overview

### A) Training data (SFT) ✅

#### A1) Generate raw training completions (two-step vLLM)
**Script:** `src/data_preprocess/data_generation.py`  
**Output:** shards with `question`, optional gold (`answer/solution`), and `completion`.

**Examples**

**Math (DAPO)**
```bash
python src/data_preprocess/data_generation.py \
  --data_mode hf \
  --dataset_id open-r1/DAPO-Math-17k-Processed \
  --dataset_config en \
  --dataset_split train \
  --system_prompt "Please reason step by step, and put your final answer within \\boxed{}." \
  --model_id Qwen/Qwen3-8B \
  --save_dir data/train/Qwen3_8B_DAPO_Math_9k3k_2gen
````

**TriviaQA (train)**

```bash
python src/data_preprocess/data_generation.py \
  --data_mode hf \
  --dataset_id mandarjoshi/trivia_qa \
  --dataset_config rc \
  --dataset_split train \
  --system_prompt "This is a trivia question. Put your final answer within \\boxed{}." \
  --model_id Qwen/Qwen3-8B \
  --save_dir data/train/Qwen3_8B_trivia_qa40k-6k
```

---

#### A2) Verify + label training completions

**Script:** `src/data_preprocess/Label_for_SFT.py`
**Output:** mirrored shards under `verified/` with:

* `correctness_label` ∈ {0, 1}
* `pred_parsed` (bool)

```bash
# Trivia
python src/data_preprocess/Label_for_SFT.py \
  --save_dir data/train/Qwen3_8B_trivia_qa40k-6k \
  --task trivia

# Math
python src/data_preprocess/Label_for_SFT.py \
  --save_dir data/train/Qwen3_8B_DAPO_Math_9k3k_2gen \
  --task math
```

---

#### A3) Merge + rebalance (training only)

**Script:** `src/data_preprocess/merge_sft_data.py`
**Does:** unify schema + rebalance per-question completion buckets:

* **all-correct**, **all-wrong**, **mixed**
* optional downsample/upsample per question
* optional per-task row caps

```bash
python src/data_preprocess/merge_sft_data.py
```

**Final output**

* `data/train/Final/.../merged_balanced.parquet`

---

## B) Benchmarks (Test/Eval) ✅

### B0) Prepare benchmark CSVs (one-time)

#### B0.1 Math benchmark CSV (merged)

**Script:** `src/data_preprocess/test_data_prepreprocess/merge_eval_math_data.py`
**Output columns:** `question`, `solution`, `original_source`

```bash
python src/data_preprocess/test_data_prepreprocess/merge_eval_math_data.py \
  --out_csv data/test/merged_math.csv \
  --out_hf_dir data/test/merged_math_hf
```

#### B0.2 MMLU-Pro CSV

**Script:** `src/data_preprocess/test_data_prepreprocess/mmlupro.py`
**Output columns:** `question`, `answer` (letter)

```bash
python src/data_preprocess/test_data_prepreprocess/mmlupro.py \
  --out_dir data/test/mmlu_pro_csv
```

> TriviaQA can be loaded directly from HF (or exported to CSV if you want consistent IO).

---

### B1) Generate benchmark completions (same generator as training)

**Script:** `src/data_preprocess/data_generation.py`

#### Math (from merged CSV)

```bash
python src/data_preprocess/data_generation.py \
  --data_mode csv \
  --data_path data/test/merged_math.csv \
  --system_prompt "Please reason step by step, and put your final answer within \\boxed{}." \
  --model_id Qwen/Qwen3-8B \
  --save_dir data/test/Qwen3_8B_MergedMath
```

#### MMLU-Pro (from CSV)

```bash
python src/data_preprocess/data_generation.py \
  --data_mode csv \
  --data_path data/test/mmlu_pro_csv/test.csv \
  --system_prompt "Please reason step by step, and put your final answer with only the choice letter within \\boxed{}." \
  --model_id Qwen/Qwen3-8B \
  --mcq_append_options auto \
  --save_dir data/test/Qwen3_8B_MMLUPro
```

#### TriviaQA (HF)

```bash
python src/data_preprocess/data_generation.py \
  --data_mode hf \
  --dataset_id mandarjoshi/trivia_qa \
  --dataset_config rc \
  --dataset_split validation \
  --system_prompt "This is a trivia question. Put your final answer within \\boxed{}." \
  --model_id Qwen/Qwen3-8B \
  --save_dir data/test/Qwen3_8B_TriviaQA_val
```

---

### B2) (Optional) Verify / score benchmark completions

**Script:** `src/data_preprocess/Label_for_SFT.py`
**Output:** `verified/` shards with `correctness_label` + `pred_parsed`

```bash
# Math
python src/data_preprocess/Label_for_SFT.py \
  --save_dir data/test/Qwen3_8B_MergedMath \
  --task math

# Trivia
python src/data_preprocess/Label_for_SFT.py \
  --save_dir data/test/Qwen3_8B_TriviaQA_val \
  --task trivia

# MMLU-Pro (letter evaluation)
python src/data_preprocess/Label_for_SFT.py \
  --save_dir data/test/Qwen3_8B_MMLUPro \
  --task gpqa
```

---

## Prompts (important)

Use task-matched prompts during generation:

* **Math / reasoning**

  * final answer in `\boxed{}`

* **Trivia**

  * short factoid; final in `\boxed{}`

* **MMLU-Pro**

  * final is **only the choice letter** in `\boxed{}`

---

## TL;DR

* **Generate** with: `data_generation.py`
* **Verify/label** with: `Label_for_SFT.py`
* **Merge (train only)** with: `merge_sft_data.py`
* **Bench prep**: `merge_eval_math_data.py` + `mmlupro.py`


