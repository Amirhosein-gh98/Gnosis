"""
Two-step data generation with Qwen + vLLM for Gnosis.

This script loads QA-style data (HF / disk / CSV / Parquet), wraps each example
with the model’s chat template, and runs two-stage generation:
(1) a chain-of-thought “thinking” segment, (2) a short final answer.
Outputs are sharded Parquet/JSONL files under `data/train/<MODEL_TAG>_*`
(relative to the Gnosis repo root).

Backbones used in the paper (MODEL_ID):
- Qwen/Qwen3-1.7B
- Qwen/Qwen3-4B-Thinking-2507
- Qwen/Qwen3-4B-Instruct-2507
- Qwen/Qwen3-8B  (default)

Example data setups (see commented blocks below):
- DAPO Math (HF train)       → data/train/<MODEL_TAG>_DAPO_Math
- TriviaQA RC (train / val)  → data/train/<MODEL_TAG>_TriviaQA_*
- Merged math CSV            → data/train/<MODEL_TAG>_MergedMath
- MMLU-Pro CSV (active)      → data/train/<MODEL_TAG>_MMLUPro

Two-stage sampling (token budgets):
- NUM_GENERATIONS = 2:
    Number of first-stage (thinking) samples per question.
- Stage budgets (defaults used in the paper):
    • Stage 1 (thinking):  SAMPLING_KW["max_tokens"] = 9000
    • Stage 2 (answer):    STAGE2_TOKENS = 3000
  So a question can use up to ~12k generated tokens if both stages fire.
- stage_2_ = True:
    If the first segment does not contain `</think>`, we append `</think>`
    and run a second stage (up to STAGE2_TOKENS) to force a concise answer
    instead of spending the entire budget on reasoning alone.

Usage:
- From the Gnosis repo root, uncomment the desired DATA_*/SAVE_DIR config block 
  at the top of this file and run the script.
"""
"""
Two-step data generation with Qwen + HuggingFace Transformers (no vLLM).

Same behavior as vLLM version:
(1) Stage-1 "thinking" generation with N samples per question
(2) Optional Stage-2 short answer if </think> missing (forced close)

Outputs are sharded Parquet/JSONL files under `data/train/<MODEL_TAG>_*`
(relative to the Gnosis repo root).
"""

import os
from glob import glob
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, load_from_disk as hf_load_from_disk
from datasets import load_dataset as hf_load_dataset
from rich.progress import (
    Progress, BarColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn, TextColumn
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ======================== CONFIG (defaults; override via CLI) =========================
# MODEL_ID = "Qwen/Qwen3-1.7B"
# MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507"
# MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
# MODEL_ID = "Qwen/Qwen3-8B"
MODEL_ID = "EleutherAI/pythia-410m"
MODEL_TAG = MODEL_ID.split("/")[-1].replace("-", "_").replace(".", "_")

# Two-step budget
STAGE2_TOKENS = 3000
# Sampling config
NUM_GENERATIONS = 2
SAMPLING_KW = {
    "temperature": 0.6,
    "top_p": 0.95,
    "max_tokens": 1000,   # total max gen budget (stage1+stage2 reserve)
}
stage_2_ = False  # keep your original switch

# Batching / IO
CHUNK_SIZE = 1
SHARD_SIZE = 4000
SHARD_FMT = "parquet"  # 'parquet' or 'jsonl'
PUSH_AT_END = False
HF_REPO_ID = ""
HF_PRIVATE = False

# NEW: optional cap on the number of questions to process
MAX_QUESTIONS = 40000
SEED = 1337
# ======================================================================================

# --------- Active data config example ----------
SYSTEM_PROMPT = "This is a Trivia question, put your final answer within \\boxed{}"
THINKING_MODE = True
DATA_MODE = "hf"
DATASET_ID = "mandarjoshi/trivia_qa"
DATASET_CONFIG = "rc"
DATASET_SPLIT = "validation"
DATA_PATH = ""
SAVE_DIR = f"data/test/{MODEL_TAG}_TriviaQA_val"

# Column candidates
PROMPT_CANDIDATES = ["prompt", "question", "query", "input", "instruction"]
ANSWER_CANDIDATES = ["answer", "final_answer", "target", "label", "answers"]
SOLUTION_CANDIDATES = ["solution", "rationale", "steps", "explanation", "cot"]
ORIGINAL_SOURCE_CANDIDATES = ["original_source", "source"]


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield i, iterable[i:i + n]

def _next_shard_index(save_dir: Path) -> int:
    save_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(glob(str(save_dir / "shard-*.parquet"))) + sorted(glob(str(save_dir / "shard-*.jsonl")))
    if not existing:
        return 1
    last = Path(existing[-1]).stem
    try:
        return int(last.split("-")[-1]) + 1
    except Exception:
        return 1

def _save_shard(rows: List[Dict[str, Any]], save_dir: Path, shard_idx: int, fmt: str = "parquet") -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    shard_name = f"shard-{shard_idx:06d}.{ 'parquet' if fmt == 'parquet' else 'jsonl' }"
    shard_path = save_dir / shard_name
    df = pd.DataFrame(rows)
    if fmt == "parquet":
        df.to_parquet(shard_path, index=False)
    else:
        df.to_json(shard_path, lines=True, orient="records", force_ascii=False)
    return shard_path

def _push_all_shards_to_hub(save_dir: Path, repo_id: str, private: bool):
    parquet_files = sorted(glob(str(save_dir / "shard-*.parquet")))
    jsonl_files   = sorted(glob(str(save_dir / "shard-*.jsonl")))
    if not parquet_files and not jsonl_files:
        print("No shards found to push.")
        return

    if parquet_files:
        ds = hf_load_dataset("parquet", data_files={"train": parquet_files})["train"]
    else:
        ds = hf_load_dataset("json", data_files={"train": jsonl_files})["train"]

    dd = DatasetDict({"train": ds})
    dd.push_to_hub(repo_id=repo_id, private=private, commit_message="Add shards")
    print(f"✅ Pushed shards to Hub: {repo_id} (private={private})")

def _get_end_think_id(tokenizer: AutoTokenizer) -> Optional[int]:
    # Keep your original behavior (single token id if possible)
    try:
        tid = tokenizer.convert_tokens_to_ids("</think>")
        if isinstance(tid, int) and tid >= 0:
            return tid
    except Exception:
        pass
    try:
        ids = tokenizer.encode("</think>", add_special_tokens=False)
        if ids:
            return int(ids[-1])
    except Exception:
        pass
    print("[warn] '</think>' token not found; two-step forcing is disabled.")
    return None

def _pick_split(ds: Union[Dataset, DatasetDict], desired: str = "train") -> Tuple[str, Dataset]:
    if isinstance(ds, DatasetDict):
        if desired in ds:
            return desired, ds[desired]
        for s in ["train", "validation", "dev", "val", "test"]:
            if s in ds:
                return s, ds[s]
        k = next(iter(ds.keys()))
        return k, ds[k]
    else:
        return desired, ds

def _load_source_dataset(
    mode: str,
    dataset_id: str,
    split: str,
    path: str,
    dataset_config: Optional[str] = None,
) -> Dataset:
    if mode == "hf":
        if dataset_config:
            ds = load_dataset(dataset_id, dataset_config, split=split)
        else:
            ds = load_dataset(dataset_id, split=split)
        return ds
    elif mode == "disk":
        if not path:
            raise ValueError("DATA_MODE='disk' requires --data_path pointing to save_to_disk directory.")
        disk_ds = hf_load_from_disk(path)
        split_name, split_ds = _pick_split(disk_ds, desired=split)
        print(f"[data] Loaded from disk: {path} (split='{split_name}', rows={len(split_ds)})")
        return split_ds
    elif mode == "csv":
        if not path:
            raise ValueError("DATA_MODE='csv' requires --data_path pointing to a CSV file.")
        ds = hf_load_dataset("csv", data_files=path)["train"]
        print(f"[data] Loaded CSV: {path} (rows={len(ds)})")
        return ds
    elif mode == "parquet":
        if not path:
            raise ValueError("DATA_MODE='parquet' requires --data_path pointing to a Parquet file or glob.")
        ds = hf_load_dataset("parquet", data_files=path)["train"]
        print(f"[data] Loaded Parquet: {path} (rows={len(ds)})")
        return ds
    else:
        raise ValueError(f"Unknown DATA_MODE '{mode}'. Use one of ['hf','disk','csv','parquet'].")

def _resolve_column(ds: Dataset, candidates: List[str], required: bool) -> Optional[str]:
    for name in candidates:
        if name in ds.column_names:
            return name
    if required:
        raise ValueError(f"Required column not found. Tried {candidates}. Available: {ds.column_names}")
    return None

def _extract_cell(val: Any) -> Any:
    if isinstance(val, (list, tuple)) and val:
        return val[0]
    if isinstance(val, dict):
        for k in ["text", "answer", "label"]:
            if k in val:
                return val[k]
    return val

def _get_opt_col_value(ds: Dataset, name: Optional[str], i: int) -> Any:
    if name is None or name not in ds.column_names:
        return None
    return _extract_cell(ds[name][i])

def _question_with_options(question: str, ex: Dict[str, Any], mode: str = "auto") -> str:
    has_opts = all(k in ex for k in ["option_a", "option_b", "option_c", "option_d"])
    if not has_opts or mode == "none":
        return question

    def already_inline(q: str) -> bool:
        ql = q.lower()
        return (" a. " in ql and " b. " in ql and " c. " in ql and " d. " in ql) or \
               ("\na." in ql and "\nb." in ql and "\nc." in ql and "\nd." in ql)

    if mode == "auto" and already_inline(question):
        return question

    a = str(ex["option_a"]); b = str(ex["option_b"]); c = str(ex["option_c"]); d = str(ex["option_d"])
    if mode == "newlines":
        return f"{question}\n\nA. {a}\nB. {b}\nC. {c}\nD. {d}"
    return f"{question}  A. {a}  B. {b}  C. {c}  D. {d}"

def _format_with_model_template(
    tokenizer: AutoTokenizer,
    question: str,
    system_prompt: Optional[str] = None,
    thinking: bool = True
) -> str:
    # Robust fallback for non-chat models (e.g., pythia)
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking=thinking,
        )
    # Plain text fallback
    if system_prompt:
        return f"{system_prompt}\n\nQ: {question}\nA:"
    return f"Q: {question}\nA:"

def _infer_context_len(tokenizer: AutoTokenizer, model: AutoModelForCausalLM) -> int:
    ctx = getattr(model.config, "max_position_embeddings", None)
    if ctx is None:
        ctx = getattr(tokenizer, "model_max_length", 2048)
    # Some tokenizers use huge sentinel values; clamp to a sane default if needed
    if ctx is None or ctx > 1_000_000:
        ctx = 2048
    return int(ctx)

def _as_set(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, set)):
        return set(x)
    return {int(x)}

def _trim_gen_ids(gen_ids: List[int], eos_ids_set: Optional[set], pad_id: Optional[int]) -> Tuple[List[int], str]:
    """
    Returns (trimmed_ids, finish_reason)
      - trims trailing pad tokens
      - trims at first EOS if present (EOS excluded from returned ids)
    """
    ids = list(gen_ids)

    # strip trailing pad
    if pad_id is not None:
        while ids and ids[-1] == pad_id:
            ids.pop()

    finish = "length"
    if eos_ids_set is not None:
        for k, t in enumerate(ids):
            if t in eos_ids_set:
                ids = ids[:k]  # exclude EOS itself
                finish = "eos"
                break
    return ids, finish

@torch.inference_mode()
def _hf_generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_return_sequences: int,
):
    gen_kw = dict(
        max_new_tokens=int(max_new_tokens),
        num_return_sequences=int(num_return_sequences),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=False,
    )
    do_sample = float(temperature) > 0.0
    gen_kw["do_sample"] = do_sample
    if do_sample:
        gen_kw["temperature"] = float(temperature)
        gen_kw["top_p"] = float(top_p)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kw
    )
    return out.sequences  # shape: (batch*num_return_sequences, seq_len_total)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Two-step generation with HF Transformers using model chat template.")
    # Data source
    parser.add_argument("--data_mode", type=str, default=DATA_MODE, choices=["hf", "disk", "csv", "parquet"])
    parser.add_argument("--dataset_id", type=str, default=DATASET_ID)
    parser.add_argument("--dataset_config", type=str, default=DATASET_CONFIG, help="HF dataset config (e.g., 'en', 'rc')")
    parser.add_argument("--dataset_split", type=str, default=DATASET_SPLIT)
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Path for disk/csv/parquet modes")
    # Model / prompts
    parser.add_argument("--model_id", type=str, default=MODEL_ID)
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--thinking_mode", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=THINKING_MODE)
    parser.add_argument("--mcq_append_options", type=str, default="auto", choices=["auto","inline","newlines","none"])
    # Generation control
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS)
    parser.add_argument("--stage2_tokens", type=int, default=STAGE2_TOKENS)
    parser.add_argument("--max_tokens", type=int, default=SAMPLING_KW["max_tokens"])
    # NEW: limit dataset size
    parser.add_argument("--max_questions", type=int, default=MAX_QUESTIONS, help="If set, process at most this many rows.")
    parser.add_argument("--seed", type=int, default=SEED, help="Shuffle seed for subsampling.")
    # IO
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR)
    parser.add_argument("--shard_fmt", type=str, default=SHARD_FMT, choices=["parquet","jsonl"])
    parser.add_argument("--shard_size", type=int, default=SHARD_SIZE)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--push_at_end", action="store_true", default=PUSH_AT_END)
    parser.add_argument("--hf_repo_id", type=str, default=HF_REPO_ID)
    parser.add_argument("--hf_private", action="store_true", default=HF_PRIVATE)
    args = parser.parse_args()

    # -------- Load source data --------
    src = _load_source_dataset(
        mode=args.data_mode,
        dataset_id=args.dataset_id,
        split=args.dataset_split,
        path=args.data_path,
        dataset_config=args.dataset_config,
    )

    # -------- Optional subsample (random, reproducible) --------
    if args.max_questions is not None and args.max_questions < len(src):
        src = src.shuffle(seed=args.seed).select(range(args.max_questions))
        print(f"[data] Using random subset: {len(src)} rows (seed={args.seed})")
    else:
        print(f"[data] Using full dataset: {len(src)} rows")

    # Resolve columns
    prompt_col = _resolve_column(src, PROMPT_CANDIDATES, required=True)
    answer_col = _resolve_column(src, ANSWER_CANDIDATES, required=False)
    solution_col = _resolve_column(src, SOLUTION_CANDIDATES, required=False)
    orig_src_col = _resolve_column(src, ORIGINAL_SOURCE_CANDIDATES, required=False)

    # -------- Init tokenizer + HF model --------
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.padding_side = "left"

    # Ensure pad_token exists for decoder-only generate batching
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()
    # If we added pad token, resize embeddings
    if getattr(model, "resize_token_embeddings", None) and (len(tokenizer) > model.get_input_embeddings().num_embeddings):
        model.resize_token_embeddings(len(tokenizer))

    end_thinking_id = _get_end_think_id(tokenizer)
    eos_ids_set = _as_set(tokenizer.eos_token_id)
    pad_id = tokenizer.pad_token_id

    ctx_len = _infer_context_len(tokenizer, model)
    # We need room for total generation + small margin (+1 for forced </think>)
    margin = 8
    max_input_len = max(16, ctx_len - int(args.max_tokens) - margin)

    # -------- Build formatted prompts (MCQ-safe) --------
    questions_raw: List[str] = []
    for i in range(len(src)):
        q = str(_extract_cell(src[prompt_col][i]))
        ex = {k: src[k][i] for k in src.column_names}
        q2 = _question_with_options(q, ex, mode=args.mcq_append_options)
        questions_raw.append(q2)

    prompts_fmt: List[str] = [
        _format_with_model_template(tokenizer, q, system_prompt=args.system_prompt, thinking=args.thinking_mode)
        for q in questions_raw
    ]

    # Stage-1 & Stage-2 budgets
    stage1_max = max(1, int(args.max_tokens) - int(args.stage2_tokens))

    save_dir = Path(args.save_dir)
    shard_idx = _next_shard_index(save_dir)
    buffer_rows: List[Dict[str, Any]] = []
    total_saved = 0

    # -------- Generate in batches with a progress bar --------
    with Progress(
        TextColumn("[bold]Generating (two-step, chat-template, HF)[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("HF", total=len(prompts_fmt))

        for start_idx, chunk in batched(prompts_fmt, args.chunk_size):
            # Tokenize chunk
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_len,
                add_special_tokens=False,
            )
            dev = next(model.parameters()).device
            inputs = {k: v.to(dev) for k, v in inputs.items()}
            input_max_len = inputs["input_ids"].shape[1]

            # Keep per-example prompt_ids (unpadded) for stage2 seeding
            prompt_ids_chunk: List[List[int]] = [
                tokenizer.encode(p, add_special_tokens=False) for p in chunk
            ]

            # Stage 1 generate
            stage1_seqs = _hf_generate_batch(
                model=model,
                tokenizer=tokenizer,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=stage1_max,
                temperature=SAMPLING_KW["temperature"],
                top_p=SAMPLING_KW["top_p"],
                num_return_sequences=int(args.num_generations),
            )

            pending = []
            stage2_seed_ids = []
            stage2_map = []

            # HF ordering: for each prompt i, its num_return_sequences are contiguous
            B = len(chunk)
            N = int(args.num_generations)

            for local_i in range(B):
                global_i = start_idx + local_i
                base = {
                    "question": _extract_cell(src[prompt_col][global_i]),
                    "solution": _get_opt_col_value(src, solution_col, global_i),
                    "answer": _get_opt_col_value(src, answer_col, global_i),
                    "original_source": _get_opt_col_value(src, orig_src_col, global_i),
                }

                for j in range(N):
                    seq = stage1_seqs[local_i * N + j].tolist()

                    # new tokens start after padded input length
                    gen_ids_raw = seq[input_max_len:]
                    gen_ids, finish_reason = _trim_gen_ids(gen_ids_raw, eos_ids_set, pad_id)
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

                    need_stage2 = False
                    if end_thinking_id is not None:
                        need_stage2 = (end_thinking_id not in gen_ids)
                        need_stage2 = need_stage2 and stage_2_

                    pending.append({
                        "base": base,
                        "prompt_ids": prompt_ids_chunk[local_i],
                        "stage1_ids": gen_ids,
                        "stage1_text": gen_text,
                        "finish_reason": finish_reason,
                        "need_stage2": need_stage2,
                    })

            # Stage 2 (only for those missing </think>)
            stage2_outputs = None
            end_tok_str = tokenizer.decode([end_thinking_id], skip_special_tokens=False) if end_thinking_id is not None else ""

            if end_thinking_id is not None:
                for idx, row in enumerate(pending):
                    if row["need_stage2"]:
                        seed = row["prompt_ids"] + row["stage1_ids"] + [end_thinking_id]
                        stage2_seed_ids.append(seed)
                        stage2_map.append(idx)

            if stage2_seed_ids:
                max_seed_len = max(len(x) for x in stage2_seed_ids)
                max_seed_len = min(max_seed_len, ctx_len - int(args.stage2_tokens) - 1)

                # Left-pad seeds
                input_ids2 = torch.full((len(stage2_seed_ids), max_seed_len), pad_id, dtype=torch.long, device=dev)
                attn2 = torch.zeros((len(stage2_seed_ids), max_seed_len), dtype=torch.long, device=dev)

                for r, ids in enumerate(stage2_seed_ids):
                    ids = ids[-max_seed_len:]  # safety truncate from left if needed
                    input_ids2[r, -len(ids):] = torch.tensor(ids, dtype=torch.long, device=dev)
                    attn2[r, -len(ids):] = 1

                stage2_seqs = _hf_generate_batch(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=input_ids2,
                    attention_mask=attn2,
                    max_new_tokens=int(args.stage2_tokens),
                    temperature=SAMPLING_KW["temperature"],
                    top_p=SAMPLING_KW["top_p"],
                    num_return_sequences=1,
                )
                stage2_outputs = stage2_seqs  # (num_stage2, seed_len + new)

            s2_cursor = 0
            for idx, row in enumerate(pending):
                final_text = row["stage1_text"]
                final_num_tokens = len(row["stage1_ids"])

                if row["need_stage2"] and stage2_outputs is not None:
                    seq2 = stage2_outputs[s2_cursor].tolist()
                    s2_cursor += 1

                    # new tokens start after padded seed length (=input_ids2.shape[1])
                    seed_padded_len = input_ids2.shape[1]
                    gen2_ids_raw = seq2[seed_padded_len:]
                    gen2_ids, _ = _trim_gen_ids(gen2_ids_raw, eos_ids_set, pad_id)
                    s2_text = tokenizer.decode(gen2_ids, skip_special_tokens=False)

                    final_text = row["stage1_text"] + end_tok_str + s2_text
                    final_num_tokens = len(row["stage1_ids"]) + 1 + len(gen2_ids)

                buffer_rows.append({
                    **row["base"],
                    "completion": final_text,
                    "finish_reason": row["finish_reason"],
                    "num_output_tokens": final_num_tokens,
                    "two_step_applied": bool(row["need_stage2"]),
                })
                print(final_text)
                if len(buffer_rows) >= args.shard_size:
                    shard_path = _save_shard(buffer_rows[:args.shard_size], save_dir, shard_idx, fmt=args.shard_fmt)
                    total_saved += args.shard_size
                    print(f"💾 Saved shard {shard_idx:06d} -> {shard_path} | total rows saved: {total_saved}")
                    buffer_rows = buffer_rows[args.shard_size:]
                    shard_idx += 1

            progress.advance(task, len(chunk))

    if buffer_rows:
        shard_path = _save_shard(buffer_rows, save_dir, shard_idx, fmt=args.shard_fmt)
        total_saved += len(buffer_rows)
        print(f"💾 Saved final shard {shard_idx:06d} -> {shard_path} | total rows saved: {total_saved}")

    print(f"✅ All shards written to: {save_dir}")

    if args.push_at_end:
        _push_all_shards_to_hub(save_dir, args.hf_repo_id, args.hf_private)

if __name__ == "__main__":
    main()
