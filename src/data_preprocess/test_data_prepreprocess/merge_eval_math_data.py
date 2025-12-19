#!/usr/bin/env python3
"""
Merge and standardize math datasets for Gnosis training.

This script is tailored to merge a small set of Hugging Face math benchmarks
into a single, standardized dataset with columns:
    ["question", "solution", "original_source"].

By default it:
- Loads the following datasets (see `--datasets`):
    - "math-ai/aime24"
    - "math-ai/aime25"
    - "MathArena/hmmt_feb_2025"
    - "AI-MO/aimo-validation-amc"


With default arguments it writes:
- CSV to: Gnosis/data/test/merged_math.csv
- HF `save_to_disk` dataset to: "merged_math_hf"
- Prints a 5-row preview (`--print_rows`).


"""

import argparse, sys, re
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi

# ------------ heuristics -------------
QUESTION_CANDIDATES = [
    "question", "prompt", "problem", "input", "instruction", "query", "text", "title"
]
SOLUTION_CANDIDATES = [
    "solution", "final_solution", "final_answer", "answer", "output", "response"
]
SOURCE_CANDIDATES = [
    "source", "dataset", "origin", "original_source"
]
PREFERRED_SPLITS = ["train", "validation", "dev", "val", "test"]  # pick in order if present

# ------------ helpers -------------
def _pick_split(ds: DatasetDict) -> Tuple[str, Dataset]:
    for s in PREFERRED_SPLITS:
        if s in ds:
            return s, ds[s]
    first_key = next(iter(ds.keys()))
    return first_key, ds[first_key]

def _detect_col(example: Dict[str, Any], candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in example and example[c] is not None:
            v = example[c]
            if isinstance(v, str) and v.strip(): return c
            if isinstance(v, (int, float)):      return c
            if isinstance(v, list) and v:        return c
    return None

def _coerce_question(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, list): return " ".join(str(x) for x in v if x is not None)
    return str(v)

def _coerce_solution(v: Any) -> str:
    if v is None: return ""
    if isinstance(v, list): return "\n".join(str(x) for x in v if x is not None)
    return str(v)

# ---- AIME boxed cleanup ----
_BOXED_RE = re.compile(r"\\boxed\s*\{(.*?)\}", re.DOTALL)

def _extract_boxed(s: str) -> Optional[str]:
    if not s: return None
    m = _BOXED_RE.search(s)
    if not m: return None
    inside = m.group(1)
    inside = inside.replace(r"\,", " ").replace(r"\!", " ").replace("$", " ")
    inside = re.sub(r"\\text\s*\{.*?\}", " ", inside)
    inside = re.sub(r"[()\[\]]", " ", inside)
    inside = re.sub(r"\s+", " ", inside).strip().strip(".")
    num = re.search(r"-?\d+(?:\.\d+)?", inside)
    return (num.group(0) if num else inside).strip()

def _aime24_cleanup(solution: str) -> str:
    boxed = _extract_boxed(solution)
    return boxed if boxed else solution.strip()

def _row_to_standard(
    ex: Dict[str, Any],
    ds_id: str,
    split_name: str
) -> Dict[str, Any]:
    q_key = _detect_col(ex, QUESTION_CANDIDATES)
    solution_key = _detect_col(ex, SOLUTION_CANDIDATES)
    source_key = _detect_col(ex, SOURCE_CANDIDATES)

    if not q_key:
        for k in ex.keys():
            if "problem" in k.lower() or "question" in k.lower():
                q_key = k; break
    if not solution_key:
        for k in ex.keys():
            kl = k.lower()
            if "solution" in kl or "final" in kl or kl == "ans" or kl.endswith("_ans"):
                solution_key = k; break
        if not solution_key and "answer" in ex:
            solution_key = "answer"

    question = _coerce_question(ex.get(q_key, "")) if q_key else ""
    solution = _coerce_solution(ex.get(solution_key, "")) if solution_key else ""

    # dataset-specific post-processing
    if ds_id == "math-ai/aime24":
        solution = _aime24_cleanup(solution)

    native_src = str(ex.get(source_key)) if source_key else None
    original_source = native_src if (native_src and native_src.strip()) else f"{ds_id}::{split_name}"

    return {
        "question": question.strip(),
        "solution": solution.strip(),
        "original_source": original_source.strip(),
    }

def load_one_dataset(ds_id: str) -> pd.DataFrame:
    try:
        ds = load_dataset(ds_id)  # default config
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {ds_id}. If it needs a config, pass it via --datasets "
            f"(e.g., org/name:config). Original error: {e}"
        ) from e

    if isinstance(ds, DatasetDict):
        split_name, split = _pick_split(ds)
    else:
        split_name, split = "train", ds  # type: ignore

    rows = [_row_to_standard(ex, ds_id, split_name) for ex in split]
    df = pd.DataFrame(rows, columns=["question", "solution", "original_source"])
    # drop empties
    df = df[(df["question"].astype(str).str.strip() != "") & (df["solution"].astype(str).str.strip() != "")]
    return df

def main():
    parser = argparse.ArgumentParser(description="Merge HF math datasets into CSV and HF Dataset")
    parser.add_argument("--out_csv", type=str, default="data/test/merged_math_hf", help="CSV path")
    parser.add_argument("--out_hf_dir", type=str, default="merged_math_hf", help="save_to_disk directory")
    parser.add_argument("--dedupe", action="store_true", help="dedupe by identical question")
    parser.add_argument("--keep_all", action="store_true", help="keep rows with empty solution")
    parser.add_argument("--print_rows", type=int, default=5, help="how many rows to print")
    parser.add_argument(
        "--datasets", nargs="+", default=[
            "math-ai/aime24",
            "math-ai/aime25",
            "MathArena/hmmt_feb_2025",
            "AI-MO/aimo-validation-amc",
        ],
        help="Dataset IDs (optionally with :config) to merge",
    )
    parser.add_argument("--push_to_hub", type=str, default="", help="Repo ID to push (e.g., username/merged-math)")
    parser.add_argument("--private", action="store_true", help="create private repo when pushing")
    args = parser.parse_args()

    # ---- load & merge ----
    frames = []
    for ds_id in args.datasets:
        print(f"[load] {ds_id}", file=sys.stderr)
        frames.append(load_one_dataset(ds_id))
    merged = pd.concat(frames, ignore_index=True)

    if not args.keep_all:
        merged = merged[(merged["question"].astype(str).str.strip() != "") &
                        (merged["solution"].astype(str).str.strip() != "")]

    if args.dedupe:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["question"]).reset_index(drop=True)
        print(f"[dedupe] {before} -> {len(merged)}", file=sys.stderr)

    # ---- save CSV ----
    merged.to_csv(args.out_csv, index=False)
    print(f"[saved] CSV -> {args.out_csv}  (rows={len(merged)})")

    # ---- make HF Dataset & save_to_disk ----
    hf_ds = Dataset.from_pandas(merged, preserve_index=False)
    hf_ds.save_to_disk(args.out_hf_dir)
    print(f"[saved] HF Dataset (Arrow) -> {args.out_hf_dir}  (rows={hf_ds.num_rows})")

    # ---- optional push to hub ----
    if args.push_to_hub:
        # requires `huggingface-cli login` beforehand
        repo_id = args.push_to_hub
        print(f"[push] Creating/updating repo: {repo_id}", file=sys.stderr)
        api = HfApi()
        try:
            api.create_repo(repo_id=repo_id, private=args.private, exist_ok=True)
        except Exception as e:
            print(f"[warn] create_repo: {e}", file=sys.stderr)
        hf_ds.push_to_hub(repo_id)
        print(f"[pushed] {repo_id}")

    # ---- print preview ----
    print("\n=== Dataset preview ===")
    print(hf_ds)  # schema + size
    print(hf_ds.select(range(min(args.print_rows, hf_ds.num_rows))).to_pandas())

if __name__ == "__main__":
    main()
