"""
verify_completions.py

Post-process vLLM-generated shards into verified datasets for Gnosis.

For each shard in `SAVE_DIR` matching `PATTERN`, this script:
- Loads parquet/jsonl completions (requires `question` and `completion` columns).
- Selects the appropriate evaluator based on `TASK_` (math, trivia, gpqa, science),
  using task-specific answer columns (e.g., `solution` for math, `answer` for trivia).
- Runs batched evaluation via `evaluator.py` (e.g., `evaluate_trivia_batch`),
  producing correctness scores and parse flags.
- Writes out a mirrored shard under `SAVE_DIR / OUT_SUBDIR` with:
    - `correctness_label` ∈ {0, 1}
    - `pred_parsed` (boolean)

Defaults are set up for TriviaQA-style training data under
`data/train/Qwen3_8B_trivia_qa40k-6k`. To use it, point `SAVE_DIR` to the directory
containing your shards (or override via `--save_dir`) and set `--task` appropriately:

    python verify_completions.py --save_dir data/train/Qwen3_8B_trivia_qa40k-6k --task trivia
"""

from glob import glob
from pathlib import Path
from typing import List, Tuple, Optional, Any

import argparse
import pandas as pd

import evaluator as ev  # <- ensure your GPQA evaluator has the JSON `"answer": "C"` fallback

# ---------- Defaults (override via CLI) ----------
# Example: DAPO Math train shards (Qwen3-8B)
# SAVE_DIR      = Path("data/train/Qwen3_8B_DAPO_Math_9k3k_2gen")
# PATTERN       = "shard-*.parquet"   # or *.jsonl
# OUT_SUBDIR    = "verified"
# BATCH_SIZE    = 512
# DROP_UNPARSED = False               # False => keep all rows (None->0); True => drop unparsed/unverifiable
# TASK_         = "math"

# Active default: TriviaQA train (Qwen3-8B)
SAVE_DIR      = Path("data/train/Qwen3_8B_trivia_qa40k-6k")
PATTERN       = "shard-*.parquet"   # or *.jsonl
OUT_SUBDIR    = "verified"
BATCH_SIZE    = 512
DROP_UNPARSED = False               # False => keep all rows (None->0); True => drop unparsed/unverifiable
TASK_         = "trivia"
# -------------------------------------------------


def find_shards(save_dir: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in sorted(glob(str(save_dir / pattern)))]

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, orient="records")
    raise ValueError(f"Unsupported shard format: {path.suffix}")

def save_df(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix == ".jsonl":
        df.to_json(out_path, lines=True, orient="records", force_ascii=False)
    else:
        out_path = out_path.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)

def map_task_to_eval_and_answer_col(task: str, override_col: Optional[str]) -> Tuple[str, str]:
    """
    Return (eval_task, answer_col_name).
    - math   => evaluator: math,   answer col: 'solution'
    - trivia => evaluator: trivia, answer col: 'answer' (dict-like gold)
    - gpqa   => evaluator: gpqa,   answer col: 'answer' (letter/text)
    - science=> evaluator: gpqa,   answer col: 'answer' (letter/text)
    If override_col is provided, it is used as the answer column name.
    """
    if task == "math":
        default_col = "solution"
        return "math", (override_col or default_col)
    if task in ("gpqa", "science"):
        default_col = "answer"
        return "gpqa", (override_col or default_col)
    if task == "trivia":
        default_col = "answer"
        return "trivia", (override_col or default_col)
    raise ValueError(f"Unsupported task: {task}")

def eval_batch(eval_task: str, comps: list, gold: list):
    if eval_task == "trivia":
        return ev.evaluate_trivia_batch(comps, gold)
    elif eval_task == "gpqa":
        return ev.evaluate_gpqa_batch(comps, gold)
    elif eval_task == "math":
        return ev.evaluate_math_batch(comps, gold)
    else:
        raise ValueError(f"Unknown eval task: {eval_task}")

def to_label_keep_all(x: Optional[float]) -> int:
    # Keep-all mode: None (unverifiable) -> 0
    if x is None:
        return 0
    return 1 if x >= 0.5 else 0

def verify_shard(
    shard_path: Path,
    out_dir: Path,
    eval_task: str,
    answer_col: str,
    batch_size: int,
    drop_unparsed: bool,
) -> Tuple[int, int, int, int]:
    """
    Verify one shard. Returns (n_in, n_kept, n_parsed, n_correct).
    """
    df = load_df(shard_path)
    drop_unparsed = False
    # require these from your generation script
    if "completion" not in df.columns or "question" not in df.columns:
        print(f"Skipping {shard_path.name}: missing 'question' and/or 'completion'")
        return (0, 0, 0, 0)

    # keep prompt explicitly for SFT
    if "prompt" not in df.columns:
        df["prompt"] = df["question"].fillna("").astype(str)

    # prepare gold/completions
    if answer_col in df.columns:
        gold = df[answer_col].tolist()
    else:
        # No ground truth available -> unverifiable (Math will need math_verify installed).
        gold = [None] * len(df)

    comps = df["completion"].astype(str).tolist()

    n = len(df)
    labels: List[int] = [0] * n
    parsed_mask: List[bool] = [False] * n
    keep_mask: Optional[List[bool]] = [False] * n if drop_unparsed else None

    # batch evaluation with per-item fallback
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        comps_b = comps[i:j]
        gold_b  = gold[i:j]
        try:
            gt, _pp, _gp, parsed = eval_batch(eval_task, comps_b, gold_b)
            for k in range(len(comps_b)):
                pk = bool(parsed[k])
                parsed_mask[i + k] = pk
                if drop_unparsed:
                    if pk and gt[k] is not None:
                        keep_mask[i + k] = True
                        labels[i + k] = 1 if gt[k] >= 0.5 else 0
                    # else:
                    #     print(comps_b[k])
                else:
                    labels[i + k] = to_label_keep_all(gt[k])
        except Exception:
            # per-item fallback
            for k in range(len(comps_b)):
                try:
                    gt1, _pp1, _gp1, parsed1 = eval_batch(eval_task, [comps_b[k]], [gold_b[k]])
                    pk = bool(parsed1[0])
                    parsed_mask[i + k] = pk
                    if drop_unparsed:
                        if pk and gt1[0] is not None:
                            keep_mask[i + k] = True
                            labels[i + k] = 1 if gt1[0] >= 0.5 else 0
                        # else:
                        #     print(comps_b[k])
                    else:
                        labels[i + k] = to_label_keep_all(gt1[0])
                except Exception:
                    # keep-all mode keeps the row with label=0; drop mode leaves it dropped
                    pass

    # construct output
    if drop_unparsed:
        kept_idx = [idx for idx, v in enumerate(keep_mask) if v]
        dropped = n - len(kept_idx)
        if kept_idx:
            df_out = df.iloc[kept_idx].copy()
            df_out["correctness_label"] = pd.Series([labels[i] for i in kept_idx], index=df_out.index, dtype="int8")
            df_out["pred_parsed"] = True
        else:
            df_out = df.iloc[0:0].copy()
            df_out["correctness_label"] = pd.Series(dtype="int8")
            df_out["pred_parsed"] = pd.Series(dtype="boolean")
    else:
        df_out = df.copy()
        df_out["correctness_label"] = pd.Series(labels, dtype="int8")
        df_out["pred_parsed"] = pd.Series(parsed_mask, dtype="boolean")
        dropped = 0

    out_path = out_dir / (shard_path.stem + ".verified" + shard_path.suffix)
    save_df(df_out, out_path)

    # stats
    n_kept = len(df_out)
    n_parsed = int(df_out["pred_parsed"].sum()) if n_kept else 0
    n_correct = int(df_out["correctness_label"].sum()) if n_kept else 0
    acc = (n_correct / n_parsed) if n_parsed else 0.0
    kept_msg = f"kept={n_kept}" if drop_unparsed else f"kept(all)={n_kept}"
    print(f"✅ {shard_path.name} → {out_path.name} | in={n} {kept_msg} dropped={dropped} "
          f"parsed={n_parsed} correct={n_correct} acc@parsed={acc:.4f}")

    return (n, n_kept, n_parsed, n_correct)

def main():
    ap = argparse.ArgumentParser(description="Verify completions with a FIXED task (no inference).")
    ap.add_argument("--save_dir", type=str, default=str(SAVE_DIR))
    ap.add_argument("--pattern", type=str, default=PATTERN)
    ap.add_argument("--out_subdir", type=str, default=OUT_SUBDIR)
    ap.add_argument("--task", type=str, default=TASK_,
                    choices=["math","trivia","gpqa","science"],
                    help="Task type. 'science' maps to GPQA-style letter evaluation.")
    ap.add_argument("--answer_col", type=str, default=None,
                    help="Optional: override the answer column name (default maps by task).")
    ap.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    ap.add_argument("--drop_unparsed", action="store_true", default=DROP_UNPARSED,
                    help="Drop rows that are unparsed/unverifiable. Default keeps all (None->0).")
    args = ap.parse_args()

    save_dir = Path(args.save_dir)
    out_dir = save_dir / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    shards = find_shards(save_dir, args.pattern)
    if not shards:
        print(f"No shards matched {save_dir}/{args.pattern}")
        return

    eval_task, ans_col = map_task_to_eval_and_answer_col(args.task, args.answer_col)
    print(f"Verifying {len(shards)} shard(s) in {save_dir} with task='{args.task}' "
          f"(eval='{eval_task}', answer_col='{ans_col}') drop_unparsed={args.drop_unparsed}")

    total_in = total_kept = total_parsed = total_correct = 0

    for idx, shard_path in enumerate(shards, 1):
        print(f"\n--> [{idx}/{len(shards)}] {shard_path.name}")
        n_in, n_kept, n_parsed, n_correct = verify_shard(
            shard_path=shard_path,
            out_dir=out_dir,
            eval_task=eval_task,
            answer_col=ans_col,
            batch_size=args.batch_size,
            drop_unparsed=args.drop_unparsed,
        )
        total_in      += n_in
        total_kept    += n_kept
        total_parsed  += n_parsed
        total_correct += n_correct

    overall_acc = (total_correct / total_parsed) if total_parsed else 0.0
    dropped = total_in - total_kept
    kept_msg = f"kept={total_kept}" if args.drop_unparsed else f"kept(all)={total_kept}"
    print("\n=== Overall ===")
    print(f"in={total_in} {kept_msg} dropped={dropped} parsed={total_parsed} "
          f"correct={total_correct} acc@parsed={overall_acc:.4f}")

if __name__ == "__main__":
    main()
