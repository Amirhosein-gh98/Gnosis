#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_sft_datasets.py

Merge and rebalance verified SFT shards (trivia / math) into a single
Gnosis-ready training set.

What it does
------------
- Loads per-task verified shards:
    TRIVIA_DIR / MATH_DIR
  Each directory is expected to contain: `verified/shard-*.verified.parquet` (or .jsonl).

- Unifies schema:
    * question      ← from question / prompt / query / input / instruction
    * ground_truth  ← from answer / solution
    * task          ∈ {trivia, science, math}
    * correctness_label ∈ {0, 1}
    * pred_parsed   (boolean; defaults to True if missing)

- Computes per-(task, question) stats and buckets questions into:
    * all_correct (AC)   all completions correct
    * all_wrong   (AW)   all completions wrong
    * mixed         mixture of correct and wrong completions

- Selects AC/AW questions using:
    * AC_FRAC, AC_QUESTIONS, AW_QUESTIONS
  and trims per-question completions via:
    * AC_KEEP_PER_Q, AW_KEEP_PER_Q

- For mixed questions, balances correct vs. wrong completions per question using:
    * MIXED_STRATEGY ∈ {"upsample", "downsample", "none"}

- Optionally caps rows per task:
    * MAX_ROWS_PER_TASK (absolute caps)
    * MAX_FINAL_TASK_RATIOS (fractional caps)

- Reorders data so all rows for a given (task, question) are contiguous,
  with optional block-level shuffling (SHUFFLE_BLOCKS_BEFORE_SAVE).

Outputs (written under OUT_DIR)
-------------------------------
- merged_balanced.parquet       (main merged dataset)


Usage
-----
1. Point TRIVIA_DIR / SCIENCE_DIR / MATH_DIR to the verified shard roots
   you want to merge (see examples below).
2. Optionally tune AC_FRAC / AC_QUESTIONS / AW_QUESTIONS / *_KEEP_PER_Q /
   MAX_ROWS_PER_TASK / MAX_FINAL_TASK_RATIOS.
3. Run:

    python merge_sft_datasets.py

4. To push to Hugging Face Hub, set:
    PUSH_TO_HUB = True, HF_REPO_ID, and HF_TOKEN / HUGGINGFACE_TOKEN env var.
"""


from __future__ import annotations

import os
import json
import math
import random
from pathlib import Path
from glob import glob
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Hugging Face Hub / Datasets
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo

# =========================
# USER CONFIG (edit here)
# =========================


# Input roots (each should contain verified/shard-*.verified.parquet or .jsonl)
# ---- Active config: Qwen3 8B ----
TRIVIA_DIR  = "data/train/Qwen3_8B_trivia_qa40k-6k"
MATH_DIR    = "data/train/Qwen3_8B_DAPO_Math_9k3k_2gen"
SCIENCE_DIR = None

# Verified shard pattern inside each root
PATTERN = "verified/shard-*.verified.parquet"  # supports .parquet or .jsonl

# Output directory (merged dataset + optional artifacts)
OUT_DIR = "data/train/Final/Qwen3_8B_Merged_sft_data" 


# Selection & strategy controls
ONLY_PARSED  = True     # consider rows with pred_parsed==True for selection/analysis
SEED         = 1337

# AC/AW bucket sizing
AC_FRAC= None   # share of ALL-CORRECT among AC+AW; set None to disable ratio logic
AC_QUESTIONS  = None              # explicit AC question count (int or None). Overrides ratio when set
AW_QUESTIONS  = None              # explicit AW question count (int or None). Overrides ratio when set

# Per-question completion limits for AC/AW buckets (PER TASK)
# Accepts:
#   - int: global cap for all tasks
#   - dict: {"trivia": 1, "science": 1, "math": 3, "*": 1}; "*" is wildcard default
#   - -1 means keep all for that task
AC_KEEP_PER_Q: Union[int, Dict[str, int], None] = {"trivia": 1, "science": 1, "math": 2}
AW_KEEP_PER_Q: Union[int, Dict[str, int], None] = {"trivia": 1, "science": 1, "math": 2}

# Mixed handling: "upsample" (duplicate minority), "downsample" (drop majority), or "none"
MIXED_STRATEGY = "downsample"

# Per-task cap on total ROWS in the final dataset (None disables).
# Can be an int (applies to all tasks) or dict like {"trivia": 40000, "science": 30000, "math": 45000}.
# MAX_ROWS_PER_TASK: Optional[Union[int, Dict[str, int]]] = {"trivia": 40000, "math": 45000}
MAX_ROWS_PER_TASK = None

# Optional cap on final per-task ROW fractions (downsample any task that exceeds its fraction of total)
# Example: {"trivia": 0.4, "science": 0.4, "math": 0.4}; disable by setting to None or {}
MAX_FINAL_TASK_RATIOS: Optional[Dict[str, float]] = None

# Block shuffle (keeps rows for same (task, question) together)
SHUFFLE_BLOCKS_BEFORE_SAVE = True

# -------------------------
# Hugging Face Hub pushing
# -------------------------
PUSH_TO_HUB: bool = False
HF_REPO_ID: Optional[str] = "ID"  # change if you want
HF_PRIVATE: bool = True
HF_COMMIT_MSG_DATASET: str = "Add merged dataset (train split)"
HF_COMMIT_MSG_ARTIFACTS: str = "Upload merge artifacts"
HF_ARTIFACTS_SUBFOLDER: str = "artifacts"  # where to put CSV/Parquet/JSON under the repo
HF_TOKEN: Optional[str] = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# =========================
# IO helpers
# =========================

def _find_verified_files(root: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in sorted(glob(str(root / pattern)))]

def _load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, orient="records")
    raise ValueError(f"Unsupported file type: {path}")

def _serialize_cell(v):
    """Make a cell parquet- and arrow-safe. Dict/list/tuple/set → JSON string; bytes → utf-8 str."""
    try:
        if isinstance(v, (dict, list, tuple, set)):
            if isinstance(v, set):
                v = list(v)
            return json.dumps(v, ensure_ascii=False, default=str)
        if isinstance(v, bytes):
            return v.decode("utf-8", errors="ignore")
        return v
    except Exception:
        return str(v)

def _coerce_for_storage(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize booleans/ints; JSON-encode object cols containing containers/bytes."""
    out = df.copy()
    if "pred_parsed" in out.columns:
        out["pred_parsed"] = out["pred_parsed"].astype("boolean").fillna(False).astype(bool)
    if "correctness_label" in out.columns:
        try:
            out["correctness_label"] = out["correctness_label"].astype("Int8")
        except Exception:
            out["correctness_label"] = out["correctness_label"].fillna(-1).astype("int8")
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]):
            if out[c].map(lambda x: isinstance(x, (dict, list, tuple, set, bytes))).any():
                out[c] = out[c].map(_serialize_cell)
    return out

def _save_parquet(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_safe = _coerce_for_storage(df)
    df_safe.to_parquet(out_path, index=False)

def _save_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

def _save_json(obj: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# =========================
# Schema helpers
# =========================

QUESTION_CANDIDATES = ("question", "prompt", "query", "input", "instruction")

def unify_question_column(df: pd.DataFrame, target_col: str = "question") -> pd.DataFrame:
    if target_col in df.columns:
        return df
    for c in QUESTION_CANDIDATES:
        if c in df.columns:
            df = df.copy()
            df[target_col] = df[c]
            return df
    raise ValueError("No suitable question column found (tried: question, prompt, query, input, instruction).")

def unify_ground_truth_column(df: pd.DataFrame, target_col: str = "ground_truth") -> pd.DataFrame:
    """
    Create a single `ground_truth` column from possible sources:
      - prefer `answer` (trivia/science), else `solution` (math), else keep existing `ground_truth` if present.
    Drops `answer`/`solution` after unification to ensure only `ground_truth` remains.
    """
    if target_col in df.columns:
        out = df.copy()
        for col in ("answer", "solution"):
            if col in out.columns and col != target_col:
                out = out.drop(columns=[col])
        return out

    out = df.copy()
    if "answer" in out.columns and "solution" in out.columns:
        base = out["answer"].combine_first(out["solution"])
    elif "answer" in out.columns:
        base = out["answer"]
    elif "solution" in out.columns:
        base = out["solution"]
    else:
        base = pd.Series([None] * len(out), index=out.index, dtype="object")
    out[target_col] = base

    for col in ("answer", "solution"):
        if col in out.columns and col != target_col:
            out = out.drop(columns=[col])
    return out

def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "correctness_label" not in df.columns:
        raise ValueError("Verified data must include `correctness_label`.")
    if "pred_parsed" not in df.columns:
        df = df.copy()
        df["pred_parsed"] = True
    return df

# =========================
# Analysis helpers
# =========================

def per_question_stats(df: pd.DataFrame, only_parsed: bool) -> pd.DataFrame:
    if only_parsed:
        df = df[df["pred_parsed"].astype(bool)]
    gb = df.groupby(["task", "question"], dropna=False)
    out = gb.agg(
        n_total=("correctness_label", "size"),
        n_correct=("correctness_label", lambda s: int(np.nansum((s == 1).astype(int)))),
        n_wrong=("correctness_label",  lambda s: int(np.nansum((s == 0).astype(int)))),
    ).reset_index()
    out["any_correct"] = out["n_correct"] > 0
    out["all_correct"] = (out["n_total"] > 0) & (out["n_correct"] == out["n_total"])
    out["all_wrong"]   = (out["n_total"] > 0) & (out["n_wrong"]   == out["n_total"])
    out["mixed"]       = (out["n_correct"] > 0) & (out["n_wrong"] > 0)
    return out

def category_counts(per_q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    total = len(per_q)
    rows.append(dict(category="ALL", count=total))
    for name, val in [("all_correct", per_q["all_correct"].sum()),
                      ("all_wrong",   per_q["all_wrong"].sum()),
                      ("mixed",       per_q["mixed"].sum())]:
        rows.append(dict(category=name, count=int(val)))
    return pd.DataFrame(rows)

# =========================
# Balancers
# =========================

def _upsample_minority_to(df: pd.DataFrame, label_col: str, target_label: int, target_count: int, rng: random.Random) -> Tuple[pd.DataFrame, int]:
    cur = df[df[label_col] == target_label]
    n = len(cur)
    if n == 0:
        return cur.iloc[0:0], 0
    if n == target_count:
        return cur.copy(), 0
    reps = target_count // n
    rem  = target_count % n
    blocks = [cur] * reps
    if rem > 0:
        idx = list(cur.index)
        rng.shuffle(idx)
        blocks.append(cur.loc[idx[:rem]])
    out = pd.concat(blocks, ignore_index=False).copy()
    return out, (target_count - n)

def _downsample_majority_to(df: pd.DataFrame, label_col: str, target_label: int, target_count: int, rng: random.Random) -> Tuple[pd.DataFrame, int]:
    cur = df[df[label_col] == target_label]
    n = len(cur)
    if n <= target_count:
        return cur.copy(), 0
    keep = cur.sample(n=target_count, random_state=rng.randint(0, 2**32 - 1))
    removed = n - target_count
    return keep.sort_index(kind="stable"), removed

def balance_mixed_question(group: pd.DataFrame, rng: random.Random, strategy: str = "upsample") -> Tuple[pd.DataFrame, int]:
    n_c = int((group["correctness_label"] == 1).sum())
    n_w = int((group["correctness_label"] == 0).sum())
    if n_c == 0 or n_w == 0 or n_c == n_w or strategy == "none":
        return group.copy(), 0

    if strategy == "upsample":
        maj = max(n_c, n_w)
        if n_c > n_w:
            up_w, added = _upsample_minority_to(group, "correctness_label", 0, maj, rng)
            keep_c = group[group["correctness_label"] == 1]
            out = pd.concat([keep_c, up_w], ignore_index=False)
            return out.sort_index(kind="stable"), added
        else:
            up_c, added = _upsample_minority_to(group, "correctness_label", 1, maj, rng)
            keep_w = group[group["correctness_label"] == 0]
            out = pd.concat([keep_w, up_c], ignore_index=False)
            return out.sort_index(kind="stable"), added

    if strategy == "downsample":
        minn = min(n_c, n_w)
        if n_c > n_w:
            keep_c, removed = _downsample_majority_to(group, "correctness_label", 1, minn, rng)
            keep_w = group[group["correctness_label"] == 0]
            out = pd.concat([keep_w, keep_c], ignore_index=False)
            return out.sort_index(kind="stable"), removed
        else:
            keep_w, removed = _downsample_majority_to(group, "correctness_label", 0, minn, rng)
            keep_c = group[group["correctness_label"] == 1]
            out = pd.concat([keep_c, keep_w], ignore_index=False)
            return out.sort_index(kind="stable"), removed

    return group.copy(), 0

# =========================
# Selection helpers
# =========================

def _key_mask(df: pd.DataFrame, keys: List[Tuple[str, str]]) -> pd.Series:
    if not keys:
        return pd.Series([False] * len(df), index=df.index)
    tuples = list(zip(df["task"].astype(str), df["question"].astype(str)))
    keyset = set(keys)
    return pd.Series([t in keyset for t in tuples], index=df.index)

def _trim_completions_per_question_per_task(
    df: pd.DataFrame,
    keep_cfg: Union[int, Dict[str, int], None],
    rng: random.Random
) -> pd.DataFrame:
    """
    Trim completions per (task, question) with per-task limits.
    keep_cfg can be:
      - int: same cap for all tasks (use -1 to keep all)
      - dict: {"trivia": 1, "science": 2, "math": -1, "*": 1} ("*" is default fallback)
      - None: keep all
    """
    if keep_cfg is None:
        return df

    def k_for(task: str) -> int:
        if isinstance(keep_cfg, dict):
            if task in keep_cfg:
                return int(keep_cfg[task])
            if "*" in keep_cfg:
                return int(keep_cfg["*"])
            return -1
        return int(keep_cfg)

    pieces = []
    for (t, q), g in df.groupby(["task", "question"], dropna=False):
        k = k_for(str(t))
        if k is None or k < 0 or len(g) <= k:
            pieces.append(g)
        else:
            pieces.append(g.sample(n=k, random_state=rng.randint(0, 2**32 - 1)))
    return pd.concat(pieces, ignore_index=False) if pieces else df.iloc[0:0]

# =========================
# Task cap helpers (rows)
# =========================

def _normalize_task_row_caps(val, tasks: List[str]) -> Dict[str, int]:
    if not val:
        return {}
    if isinstance(val, int):
        return {t: int(val) for t in tasks}
    return {str(t): int(v) for t, v in dict(val).items()}

def _apply_task_row_count_caps(
    df: pd.DataFrame,
    caps_by_task: Dict[str, int],
    rng: random.Random
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if not caps_by_task or df.empty:
        return df, {}
    infos = []
    kept_parts = []
    for t, g in df.groupby("task", dropna=False):
        n = len(g)
        allowed = max(0, int(caps_by_task.get(t, n)))
        if n <= allowed:
            kept_parts.append(g)
            infos.append(dict(task=t, before=n, allowed=allowed, kept=n, removed=0))
        else:
            keep = g.sample(n=allowed, random_state=rng.randint(0, 2**32 - 1)).sort_index(kind="stable")
            kept_parts.append(keep)
            infos.append(dict(task=t, before=n, allowed=allowed, kept=allowed, removed=n - allowed))
    df_out = pd.concat(kept_parts, ignore_index=True) if kept_parts else df.iloc[0:0]
    info_by_task = {r["task"]: r for r in infos}
    return df_out, info_by_task

def _apply_task_ratio_caps(
    df: pd.DataFrame,
    max_ratios: Dict[str, float],
    rng: random.Random
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    if not max_ratios:
        return df, {}
    max_ratios = {t: max(0.0, min(1.0, float(f))) for t, f in max_ratios.items()}
    total = len(df)
    if total == 0:
        return df, {}
    allowed_by_task = {t: int(math.floor(max_ratios[t] * total)) for t in max_ratios}
    infos = []
    kept_blocks = []
    for t, g in df.groupby("task", dropna=False):
        allowed = allowed_by_task.get(t, len(g))
        n = len(g)
        if n <= allowed:
            kept_blocks.append(g)
            infos.append(dict(task=t, before=n, allowed=allowed, kept=n, removed=0))
        else:
            keep = g.sample(n=allowed, random_state=rng.randint(0, 2**32 - 1)).sort_index(kind="stable")
            kept_blocks.append(keep)
            infos.append(dict(task=t, before=n, allowed=allowed, kept=allowed, removed=n - allowed))
    df_out = pd.concat(kept_blocks, ignore_index=True) if kept_blocks else df.iloc[0:0]
    cap_info = {r["task"]: {k: r[k] for k in ("before", "allowed", "kept", "removed")} for r in infos}
    return df_out, cap_info

# =========================
# Block ordering helpers
# =========================

def reorder_as_question_blocks(
    df: pd.DataFrame,
    rng: random.Random,
    shuffle_blocks: bool = False,
    group_keys: Tuple[str, str] = ("task", "question"),
) -> pd.DataFrame:
    """
    Reorder the dataframe so all rows for the same (task, question) are contiguous.
    If shuffle_blocks=True, the *order of question blocks* is randomized, but rows
    within each block remain together (stable).
    """
    if df.empty:
        return df
    groups = list(df.groupby(list(group_keys), sort=False, dropna=False))
    if shuffle_blocks:
        rng.shuffle(groups)
    blocks = [g.sort_index(kind="stable") for _, g in groups]
    return pd.concat(blocks, ignore_index=True)

# =========================
# Core
# =========================

def load_one_dataset(root: Optional[Path], task: str, pattern: str) -> pd.DataFrame:
    if root is None:
        return pd.DataFrame()
    files = _find_verified_files(root, pattern)
    if not files:
        print(f"[warn] No verified files under: {root}/{pattern}")
        return pd.DataFrame()
    dfs = []
    for p in files:
        df = _load_df(p)
        df = ensure_required_columns(df)
        df = unify_question_column(df, "question")
        df = unify_ground_truth_column(df, "ground_truth")  # unify answer/solution
        df = df.copy()
        df["task"] = task
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def compute_ac_aw_take_counts(
    n_ac: int,
    n_aw: int,
    ac_questions: Optional[int],
    aw_questions: Optional[int],
    ac_frac: Optional[float],
) -> Tuple[int, int]:
    if ac_frac is None:
        ac_take = n_ac if ac_questions is None else min(n_ac, int(ac_questions))
        aw_take = n_aw if aw_questions is None else min(n_aw, int(aw_questions))
        return ac_take, aw_take

    ac_frac = max(0.0, min(1.0, float(ac_frac)))

    if ac_questions is not None and aw_questions is not None:
        return min(n_ac, ac_questions), min(n_aw, aw_questions)

    if ac_questions is not None and aw_questions is None:
        if ac_frac == 0.0:
            return 0, 0
        aw_target = int(round(ac_questions * (1.0 - ac_frac) / ac_frac))
        return min(n_ac, ac_questions), min(n_aw, aw_target)

    if aw_questions is not None and ac_questions is None:
        if ac_frac == 1.0:
            return 0, 0
        ac_target = int(round(aw_questions * ac_frac / (1.0 - ac_frac)))
        return min(n_ac, ac_target), min(n_aw, aw_questions)

    if ac_frac == 0.0:
        return 0, n_aw
    if ac_frac == 1.0:
        return n_ac, 0
    total_take = min(int(n_ac / ac_frac), int(n_aw / (1.0 - ac_frac)))
    ac_take = int(math.floor(total_take * ac_frac))
    aw_take = int(math.floor(total_take * (1.0 - ac_frac)))
    if ac_take == 0 and n_ac > 0 and ac_frac > 0:
        ac_take = 1
    if aw_take == 0 and n_aw > 0 and (1.0 - ac_frac) > 0:
        aw_take = 1
    return min(n_ac, ac_take), min(n_aw, aw_take)

# =========================
# Hub helpers
# =========================

def build_readme_md(stats: dict) -> str:
    pre = stats.get("pre", {})
    post = stats.get("post", {})
    controls = stats.get("controls", {})
    lines = [
        "# Merged SFT dataset",
        "",
        "This repository contains a merged and balanced dataset across tasks (trivia/science/math),",
        "with per-question multi-completion handling and optional AC/AW balancing.",
        "",
        "## Summary",
        f"- Seed: `{stats.get('seed')}`",
        f"- Only parsed rows considered (input): `{stats.get('only_parsed')}`",
        "",
        "### Pre (input)",
        f"- Rows (all): `{pre.get('rows_total')}`",
        f"- Rows (considered): `{pre.get('rows_considered')}`",
        f"- Questions: `{pre.get('questions_total')}`",
        f"- Categories: `{pre.get('categories')}`",
        f"- Rows per task: `{pre.get('per_task_rows')}`",
        "",
        "### Post (final)",
        f"- Rows (final): `{post.get('rows_total')}`",
        f"- Questions (final): `{post.get('questions_total')}`",
        f"- Categories: `{post.get('categories')}`",
        f"- Rows per task: `{post.get('per_task_rows')}`",
        f"- Task fractions: `{post.get('per_task_fractions')}`",
        "",
        "## Controls",
        f"```json\n{json.dumps(controls, indent=2, ensure_ascii=False)}\n```",
        "",
        "## Artifacts",
        f"- Parquet/CSVs/JSON stats under `{HF_ARTIFACTS_SUBFOLDER}/`.",
        "",
        "## Schema",
        "- `question`: unified question field",
        "- `ground_truth`: unified target (from `answer` or `solution`)",
        "- `task`: one of {trivia, science, math}",
        "- `correctness_label`: 1 (correct) / 0 (wrong)",
    ]
    return "\n".join(lines)

def push_to_hub(
    df_final: pd.DataFrame,
    stats: dict,
    out_dir: Path,
    repo_id: str,
    private: bool = True,
    token: Optional[str] = None,
    commit_msg_dataset: str = "Add merged dataset",
    commit_msg_artifacts: str = "Add artifacts",
    artifacts_subfolder: str = "artifacts",
):
    print("\n[Hub] Creating/updating repo and pushing dataset...")
    api = HfApi()
    create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True, token=token)

    # Build arrow dataset (train split)
    df_arrow = _coerce_for_storage(df_final)
    ds = Dataset.from_pandas(df_arrow, preserve_index=False)
    dsd = DatasetDict({"train": ds})
    dsd.push_to_hub(repo_id, private=private, token=token, commit_message=commit_msg_dataset)

    # Upload artifacts folder
    if out_dir.exists():
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(out_dir),
            path_in_repo=artifacts_subfolder.rstrip("/"),
            token=token,
            commit_message=commit_msg_artifacts,
        )

    # Upload README.md
    readme = build_readme_md(stats)
    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    api.upload_file(
        repo_id=repo_id,
        repo_type="dataset",
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        token=token,
        commit_message="Add README.md",
    )
    print(f"[Hub] Done. Pushed to: https://huggingface.co/datasets/{repo_id}")

# =========================
# Main
# =========================

def main():
    rng = random.Random(SEED)

    trivia_dir  = Path(TRIVIA_DIR)  if TRIVIA_DIR  else None
    science_dir = Path(SCIENCE_DIR) if SCIENCE_DIR else None
    math_dir    = Path(MATH_DIR)    if MATH_DIR    else None
    out_dir     = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print("\n[1/7] Loading verified datasets...")
    df_trivia  = load_one_dataset(trivia_dir,  "trivia",  PATTERN)
    df_science = load_one_dataset(science_dir, "science", PATTERN)
    df_math    = load_one_dataset(math_dir,    "math",    PATTERN)

    frames = [df for df in (df_trivia, df_science, df_math) if not df.empty]
    if not frames:
        print("No data loaded. Exiting.")
        return
    df_all = pd.concat(frames, ignore_index=True)

    # Selection mask
    select_mask = df_all["pred_parsed"].astype(bool) if ONLY_PARSED else pd.Series([True] * len(df_all), index=df_all.index)

    # Pre-analysis
    print("[2/7] Computing per-question stats (BEFORE)...")
    per_q_before = per_question_stats(df_all.copy(), ONLY_PARSED)
    # _save_csv(per_q_before, out_dir / "per_question_before.csv")
    cat_before = category_counts(per_q_before)
    # _save_csv(cat_before, out_dir / "category_counts_before.csv")

    # Keys by category
    print("[3/7] Selecting categories...")
    per_q_idxed = per_q_before.set_index(["task", "question"])
    all_correct_keys = per_q_idxed.index[per_q_idxed["all_correct"].values].tolist()
    all_wrong_keys   = per_q_idxed.index[per_q_idxed["all_wrong"].values].tolist()
    mixed_keys       = per_q_idxed.index[per_q_idxed["mixed"].values].tolist()

    # Decide AC/AW sample sizes
    n_ac_avail, n_aw_avail = len(all_correct_keys), len(all_wrong_keys)
    ac_take, aw_take = compute_ac_aw_take_counts(
        n_ac=n_ac_avail,
        n_aw=n_aw_avail,
        ac_questions=AC_QUESTIONS,
        aw_questions=AW_QUESTIONS,
        ac_frac=AC_FRAC,
    )

    # Sample question keys deterministically
    def _sample_keys(keys: List[Tuple[str, str]], k: int) -> List[Tuple[str, str]]:
        if k <= 0 or not keys:
            return []
        if k >= len(keys):
            return list(keys)
        return list(rng.sample(keys, k))

    ac_sel = _sample_keys(all_correct_keys, ac_take)
    aw_sel = _sample_keys(all_wrong_keys,   aw_take)

    # AC/AW selection
    mask_ac = _key_mask(df_all, ac_sel)
    mask_aw = _key_mask(df_all, aw_sel)

    df_ac = df_all[mask_ac & select_mask].copy()
    df_aw = df_all[mask_aw & select_mask].copy()

    # Per-task trim: max completions per (task, question)
    df_ac = _trim_completions_per_question_per_task(df_ac, AC_KEEP_PER_Q, rng)
    df_aw = _trim_completions_per_question_per_task(df_aw, AW_KEEP_PER_Q, rng)

    # Mixed handling
    print(f"[4/7] Handling mixed questions (strategy: {MIXED_STRATEGY})...")
    mask_mixed = _key_mask(df_all, mixed_keys)
    df_mixed = df_all[mask_mixed & select_mask].copy()

    change_total = 0
    balanced_groups = []
    if not df_mixed.empty and MIXED_STRATEGY in ("upsample", "downsample"):
        for (t, q), g in df_mixed.groupby(["task", "question"], dropna=False):
            bg, change = balance_mixed_question(g, rng, strategy=MIXED_STRATEGY)
            change_total += change
            balanced_groups.append(bg)
        df_mixed_bal = pd.concat(balanced_groups, ignore_index=False) if balanced_groups else df_mixed
    else:
        df_mixed_bal = df_mixed

    # Merge final dataset (before optional caps)
    print("[5/7] Concatenating final dataset...]")
    df_final = pd.concat([df_ac, df_aw, df_mixed_bal], ignore_index=True)

    # --- Per-task max ROWS cap ---
    row_count_cap_info = {}
    if MAX_ROWS_PER_TASK:
        tasks_present = sorted(df_final["task"].dropna().astype(str).unique().tolist())
        row_caps = _normalize_task_row_caps(MAX_ROWS_PER_TASK, tasks_present)
        print("[6a/7] Enforcing per-task max ROWS...")
        df_final, row_count_cap_info = _apply_task_row_count_caps(df_final, row_caps, rng)
    else:
        print("[6a/7] Skipping per-task row caps (disabled).")

    # Optional per-task ROW fraction cap
    cap_info = {}
    if MAX_FINAL_TASK_RATIOS:
        print("[6b/7] Enforcing max final task ratios...")
        df_final, cap_info = _apply_task_ratio_caps(df_final, MAX_FINAL_TASK_RATIOS, rng)
    else:
        print("[6b/7] Skipping task ratio caps (disabled).")

    # Block-ordering (keeps rows for same (task, question) together)
    df_final = reorder_as_question_blocks(
        df_final,
        rng=rng,
        shuffle_blocks=bool(SHUFFLE_BLOCKS_BEFORE_SAVE),
        group_keys=("task", "question"),
    )

    # Post-analysis & save
    print("[7/7] Computing per-question stats (AFTER) & saving outputs...")
    per_q_after = per_question_stats(df_final.copy(), only_parsed=False)
    cat_after = category_counts(per_q_after)

    # Label counts
    row_counts_before = df_all[select_mask]["correctness_label"].value_counts().to_dict()
    row_counts_after  = df_final["correctness_label"].value_counts().to_dict()

    # Per-task rows
    per_task_rows_before = df_all[select_mask].groupby("task").size().to_dict()
    per_task_rows_after  = df_final.groupby("task").size().to_dict()

    # Save outputs
    _save_parquet(df_final, out_dir / "merged_balanced.parquet")
    # _save_csv(per_q_after, out_dir / "per_question_after.csv")
    # _save_csv(cat_after, out_dir / "category_counts_after.csv")

    # Distributions of completions per question
    def _dist(series: pd.Series) -> Dict[str, int]:
        vc = series.value_counts().sort_index()
        return {str(int(k)): int(v) for k, v in vc.items()}

    comp_dist_before = _dist(per_q_before.loc[per_q_before["n_total"] > 0, "n_total"]) if len(per_q_before) else {}
    comp_dist_after  = _dist(per_q_after.loc[per_q_after["n_total"] > 0,  "n_total"]) if len(per_q_after)  else {}

    # Post-task fractions
    total_final = max(1, int(len(df_final)))
    post_task_fractions = {t: round(c / total_final, 6) for t, c in (per_task_rows_after or {}).items()}

    stats = dict(
        seed=SEED,
        only_parsed=ONLY_PARSED,
        inputs=dict(
            trivia_dir=str(trivia_dir) if trivia_dir else None,
            science_dir=str(science_dir) if science_dir else None,
            math_dir=str(math_dir) if math_dir else None,
            pattern=PATTERN,
        ),
        controls=dict(
            ac_frac=(None if AC_FRAC is None else float(AC_FRAC)),
            ac_questions=AC_QUESTIONS,
            aw_questions=AW_QUESTIONS,
            ac_keep_per_q=AC_KEEP_PER_Q,
            aw_keep_per_q=AW_KEEP_PER_Q,
            mixed_strategy=MIXED_STRATEGY,
            max_rows_per_task=MAX_ROWS_PER_TASK,
            max_final_task_ratios=MAX_FINAL_TASK_RATIOS,
            shuffle_blocks_before_save=SHUFFLE_BLOCKS_BEFORE_SAVE,
        ),
        availability=dict(
            all_correct_available=len(all_correct_keys),
            all_wrong_available=len(all_wrong_keys),
            mixed_available=len(mixed_keys),
        ),
        selection=dict(
            ac_selected=len(ac_sel),
            aw_selected=len(aw_sel),
            mixed_change_total=int(change_total),
        ),
        row_count_caps=dict(
            applied=bool(MAX_ROWS_PER_TASK),
            cap_info=row_count_cap_info,
        ),
        task_caps=dict(
            applied=bool(MAX_FINAL_TASK_RATIOS),
            cap_info=cap_info,
        ),
        pre=dict(
            rows_total=int(len(df_all)),
            rows_considered=int(int(select_mask.sum())),
            questions_total=int(len(per_q_before)),
            categories=cat_before.set_index("category")["count"].to_dict(),
            row_label_counts=row_counts_before,
            per_task_rows=per_task_rows_before,
            completions_per_question_distribution=comp_dist_before,
        ),
        post=dict(
            rows_total=int(len(df_final)),
            questions_total=int(len(per_q_after)),
            categories=cat_after.set_index("category")["count"].to_dict(),
            row_label_counts=row_counts_after,
            per_task_rows=per_task_rows_after,
            per_task_fractions=post_task_fractions,
            completions_per_question_distribution=comp_dist_after,
        ),
    )
    # _save_json(stats, out_dir / "stats.json")

    # Human-readable printout
    print("\n===== SUMMARY =====")
    print(f"Seed                       : {SEED}")
    print(f"Only parsed rows (input)   : {ONLY_PARSED}")
    print("Inputs:")
    print(f"  - trivia : {trivia_dir}")
    print(f"  - science: {science_dir}")
    print(f"  - math   : {math_dir}")
    print(f"  - pattern: {PATTERN}")

    print("\n[Controls]")
    print(f"  AC share (AC_FRAC)       : {'disabled' if AC_FRAC is None else AC_FRAC}")
    print(f"  AC/AW requested counts   : AC={AC_QUESTIONS}  AW={AW_QUESTIONS}")
    print(f"  AC keep per question     : {AC_KEEP_PER_Q}")
    print(f"  AW keep per question     : {AW_KEEP_PER_Q}")
    print(f"  Mixed strategy           : {MIXED_STRATEGY}")
    print(f"  Max rows per task        : {MAX_ROWS_PER_TASK if MAX_ROWS_PER_TASK else 'disabled'}")
    print(f"  Max task ratios (rows)   : {MAX_FINAL_TASK_RATIOS if MAX_FINAL_TASK_RATIOS else 'disabled'}")
    print(f"  Shuffle blocks before save: {SHUFFLE_BLOCKS_BEFORE_SAVE}")

    if MAX_ROWS_PER_TASK and row_count_cap_info:
        print("\n[Per-Task Row Caps Applied]")
        for t, info in row_count_cap_info.items():
            print(f"  - {t:<8} before={info['before']:,} "
                  f"allowed={info['allowed']:,} kept={info['kept']:,} "
                  f"removed={info['removed']:,}")

    if MAX_FINAL_TASK_RATIOS and cap_info:
        print("\n[Task Row-Fraction Caps Applied]")
        for t, info in cap_info.items():
            print(f"  - {t:<8} before={info['before']:,} allowed={info['allowed']:,} "
                  f"kept={info['kept']:,} removed={info['removed']:,}")

    print("\n[Before]")
    print(f"  Rows (all)               : {len(df_all):,}")
    print(f"  Rows (considered)        : {int(select_mask.sum()):,}")
    print(f"  Questions                : {len(per_q_before):,}")
    print(f"  Categories               : {stats['pre']['categories']}")
    print(f"  Row labels (0/1)         : {row_counts_before}")
    print(f"  Rows per task            : {per_task_rows_before}")

    print("\n[After]")
    print(f"  Rows (final)             : {len(df_final):,}")
    print(f"  Questions (final)        : {len(per_q_after):,}")
    print(f"  Categories               : {stats['post']['categories']}")
    print(f"  Row labels (0/1)         : {row_counts_after}")
    print(f"  Rows per task            : {per_task_rows_after}")
    print(f"  Task fractions           : {post_task_fractions}")
    print("  Saved:")
    print(f"    - {out_dir / 'merged_balanced.parquet'}")
    print(f"    - {out_dir / 'per_question_before.csv'}")
    print(f"    - {out_dir / 'per_question_after.csv'}")
    print(f"    - {out_dir / 'category_counts_before.csv'}")
    print(f"    - {out_dir / 'category_counts_after.csv'}")
    print(f"    - {out_dir / 'stats.json'}")

    # Push to Hugging Face Hub (dataset + artifacts)
    if PUSH_TO_HUB and HF_REPO_ID:
        try:
            push_to_hub(
                df_final=df_final,
                stats=stats,
                out_dir=out_dir,
                repo_id=HF_REPO_ID,
                private=HF_PRIVATE,
                token=HF_TOKEN,
                commit_msg_dataset=HF_COMMIT_MSG_DATASET,
                commit_msg_artifacts=HF_COMMIT_MSG_ARTIFACTS,
                artifacts_subfolder=HF_ARTIFACTS_SUBFOLDER,
            )
        except Exception as e:
            print(f"[Hub][ERROR] Failed to push to hub: {e}")
            print("Tip: run `huggingface-cli login` or set HF_TOKEN/HUGGINGFACE_TOKEN.")

if __name__ == "__main__":
    main()
