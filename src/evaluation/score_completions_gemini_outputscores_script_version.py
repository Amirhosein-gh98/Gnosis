import os
import sys
import re
import argparse
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np

# --- New Google GenAI SDK ---
from google import genai
from google.genai import types

# --- plotting (non-GUI backend) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.progress import (
    Progress, BarColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn, TextColumn
)

# Import your existing evaluator if available
try:
    import evaluator as ev
except ImportError:
    print("Error: 'evaluator.py' not found. Please ensure it is in the same directory.")
    sys.exit(1)

# ======== CONFIGURATION =========
# ---------------------------------------------------------
# PASTE YOUR API KEY HERE (inside the quotes)
MY_GEMINI_KEY = "API" 
# ---------------------------------------------------------

# CHANGED: Updated to Gemini 2.5 Pro
DEFAULT_MODEL_ID = "gemini-2.5-pro" 
SHARD_GLOB_DEFAULT  = "shard-*.parquet"
OUT_SUBDIR          = "scored_gemini_25_pro"
THRESHOLDS_DEFAULT  = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]


# ---------- IO ----------
def find_shards(save_dir: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in sorted(glob(str(save_dir / pattern)))]

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, orient="records")
    raise ValueError(f"Unsupported shard format: {path.suffix}")

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".jsonl":
        df.to_json(path, lines=True, orient="records", force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format: {path.suffix}")

# ---------- tiny helpers ----------
_LETTER_ONLY = re.compile(r"^[A-Fa-f]$")

def _looks_like_gpqa_answers(ans_list: List[Any], sample_k: int = 50, ratio: float = 0.6) -> bool:
    if not ans_list:
        return False
    n = 0
    hits = 0
    for a in ans_list[:sample_k]:
        if a is None:
            continue
        s = str(a).strip()
        if not s:
            continue
        n += 1
        if _LETTER_ONLY.fullmatch(s) is not None:
            hits += 1
    return (n > 0) and (hits / n >= ratio)

def _infer_task_and_ans_col(df: pd.DataFrame, requested: str) -> Tuple[str, Optional[str]]:
    if requested != "auto":
        if requested == "math":
            return "math", "solution"
        if requested == "gpqa":
            return "gpqa", "answer"
        return "trivia", "answer"

    if "solution" in df.columns:
        return "math", "solution"
    if "answer" in df.columns:
        if _looks_like_gpqa_answers(df["answer"].tolist()):
            return "gpqa", "answer"
        else:
            return "trivia", "answer"
    return "math", "solution"

# ---------- GEMINI JUDGE LOGIC ----------

SYSTEM_PROMPT = """You are an objective, expert evaluator. 
Your task is to review a Question and a proposed Answer.
Determine if the Answer is correct, relevant, and accurate based on the Question.
Assign a correctness score between 0.0 (completely wrong) and 1.0 (perfectly correct).

IMPORTANT:
Output the final score inside <score> tags. Example: <score>0.95</score> or <score>0.0</score>.
"""

def _call_gemini_single(client: genai.Client, model_id: str, q: str, c: str) -> float:
    """
    Calls Gemini 2.5 Pro using the new SDK.
    """
    prompt = f"Question: {q}\n\nProposed Answer: {c}\n\nAssess the correctness and output the score in <score> tags."
    
    try:
        # NEW SDK CALL
        response = client.models.generate_content(
            model=model_id,
            contents=[SYSTEM_PROMPT, prompt],
            config=types.GenerateContentConfig(
                temperature=0.0,
                # NOTE: We removed 'thinking_config' because standard Pro models 
                # usually do not support the explicit thinking parameter.
            ),
        )
        
        text = response.text.strip() if response.text else ""
        
        # Robust extraction: look for <score>TAG</score>
        match = re.search(r"<score>\s*(\d+(\.\d+)?)\s*</score>", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
            
        # Fallback: Look for the *last* number in the text
        matches = list(re.finditer(r"(\d+(\.\d+)?)", text))
        if matches:
            score = float(matches[-1].group(1))
            if 0.0 <= score <= 1.0:
                return score

        return 0.0
    except Exception as e:
        # print(f"API Error: {e}") 
        return 0.0

def gemini_scores_batch(
    client: genai.Client, model_id: str, qs: List[str], cs: List[str], batch_size: int
) -> Tuple[List[float], List[float]]:
    """
    Parallelizes calls using ThreadPoolExecutor.
    """
    results = [0.0] * len(qs)
    
    # We pass the shared 'client' instance which is thread-safe in the new SDK
    with ThreadPoolExecutor(max_workers=min(batch_size, 20)) as executor:
        future_to_idx = {
            executor.submit(_call_gemini_single, client, model_id, qs[i], cs[i]): i 
            for i in range(len(qs))
        }
        
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                score = future.result()
                results[idx] = score
            except Exception:
                results[idx] = 0.0

    return results, results

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_ID,
                        help=f"Gemini model ID (default: {DEFAULT_MODEL_ID})")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing shards (parquet/jsonl).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to write scored shards + reports.")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "math", "trivia", "gpqa"],
                        help="auto: infer from columns")
    parser.add_argument("--pattern", type=str, default=SHARD_GLOB_DEFAULT,
                        help=f"Glob to match shards (default: {SHARD_GLOB_DEFAULT}).")
    parser.add_argument("--thresholds", type=str, default="",
                        help="Comma-separated thresholds.")
    parser.add_argument("--limit_shards", type=int, default=0,
                        help="If >0, cap the number of shards processed.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Row stride for downsampling within each shard.")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Concurrency/Threads for API calls (Pro is slower than Flash, lowered default to 5).")
    args = parser.parse_args()

    thresholds = THRESHOLDS_DEFAULT if not args.thresholds.strip() else [
        float(x.strip()) for x in args.thresholds.split(",") if x.strip()
    ]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_id = args.model
    
    # --- Initialize Client with Hardcoded Key or Env Var ---
    # Prioritize MY_GEMINI_KEY if set, otherwise fallback to os.environ
    active_key = MY_GEMINI_KEY if MY_GEMINI_KEY.strip() else os.environ.get("GEMINI_API_KEY")
    
    if not active_key:
        print("CRITICAL ERROR: No API Key found. Please set MY_GEMINI_KEY in the script or GEMINI_API_KEY env var.")
        return

    client = genai.Client(api_key=active_key)
    # -------------------------------------------------------

    # Sanitize model name for folder path
    model_safe_name = model_id.replace("models/", "").replace(":", "-")

    # Shards
    in_shards = find_shards(input_dir, args.pattern)
    if args.limit_shards and args.limit_shards > 0:
        in_shards = in_shards[: args.limit_shards]
    if not in_shards:
        print(f"No shards matched {input_dir}/{args.pattern}")
        return

    # Output directory
    out_dir_model = output_dir / OUT_SUBDIR / model_safe_name
    out_dir_model.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(in_shards)} shard(s) -> {out_dir_model}")
    print(f"Using Judge: {model_id} (Pro Mode)")

    # Global accumulators
    y_true_all: List[Optional[int]] = []    
    y_prob_all: List[Optional[float]] = []  
    pred_parsed_all: List[bool] = []        

    with Progress(
        TextColumn("[bold]Scoring (Gemini 2.5 Pro)[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        shard_task_id = progress.add_task("shards", total=len(in_shards))

        for shard_path in in_shards:
            df = load_df(shard_path)
            # Determine task + answer column
            task, ans_col = _infer_task_and_ans_col(df, args.task)

            needed = [c for c in ("question", "completion") if c not in df.columns]
            if ans_col and ans_col not in df.columns:
                needed.append(ans_col)
            if needed:
                print(f"Skipping {shard_path.name}: missing columns {needed}")
                progress.advance(shard_task_id, 1)
                continue

            questions   = df["question"].tolist()
            completions = df["completion"].tolist()
            gold_any    = df[ans_col].tolist() if ans_col in df.columns else [None] * len(df)

            row_task_id = progress.add_task(f"{shard_path.name}", total=len(questions))

            probs_all: List[float] = []
            rm_scores_all: List[float] = []

            parsed_flags_shard: List[bool] = []
            is_correct_all: List[Optional[float]] = []
            pred_prev_all:  List[Optional[str]]  = []
            gold_prev_all:  List[Optional[str]]  = []

            # Batching Loop
            B = max(1, int(args.batch_size))
            for i in range(0, len(questions), B):
                batch_idx = list(range(i, min(i + B, len(questions))))
                qs = [questions[k] for k in batch_idx]
                cs = [completions[k] if completions[k] is not None else "" for k in batch_idx]

                # --- CALL GEMINI 2.5 PRO ---
                raw_scores, probs = gemini_scores_batch(client, model_id, qs, cs, B)
                
                rm_scores_all.extend(raw_scores)
                probs_all.extend(probs)

                # Ground-truth evaluation
                batch_comps = [completions[k] for k in batch_idx]
                batch_gold  = [gold_any[k]    for k in batch_idx]
                
                if task == "trivia":
                    gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_trivia_batch(batch_comps, batch_gold)
                elif task == "gpqa":
                    gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_gpqa_batch(batch_comps, batch_gold)
                else:
                    gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_math_batch(batch_comps, batch_gold)

                is_correct_all.extend(gt)
                pred_prev_all.extend(pred_prev)
                gold_prev_all.extend(gold_prev)
                parsed_flags_shard.extend([bool(x) for x in parsed_flags])

                # Console Logging
                for rel, k in enumerate(batch_idx):
                    p   = float(probs[rel])
                    ok  = gt[rel]   
                    
                    pred_parsed_all.append(bool(parsed_flags[rel]))
                    if pred_prev[rel]:
                        print(f"  - idx={k:>6}  score={p:>7.4f}  correct={ok}  pred≈{p}")
                        y_prob_all.append(p)
                        y_true_all.append(None if ok is None else (1 if ok >= 0.5 else 0))

                progress.advance(row_task_id, len(batch_idx))

            # --- attach columns for this shard ---
            df["correctness_prob"] = probs_all
            df["reward_score"]     = rm_scores_all
            df["is_correct"]       = is_correct_all
            df["pred_ans_preview"] = pred_prev_all
            df["gold_ans_preview"] = gold_prev_all
            df["pred_parsed"]      = parsed_flags_shard

            # per-shard summaries
            parsable_df = df.loc[df["pred_parsed"].astype(bool)]
            if parsable_df["correctness_prob"].notna().any():
                t = np.array([x for x in parsable_df["correctness_prob"] if x is not None])
                print(f"Summary [{shard_path.name}] (parsed only) gemini_score:"
                      f" mean={t.mean():.4f} median={np.median(t):.4f} ")

            # Write scored shard
            out_name = shard_path.stem + ".scored" + shard_path.suffix
            out_path = out_dir_model / out_name
            save_df(df, out_path)

            progress.remove_task(row_task_id)
            progress.advance(shard_task_id, 1)
            print(f"✅ {shard_path.name} -> {out_path}")

    # ---------- FINAL OVERALL METRICS (parsed-only) ----------
    metric_idx = [
        i for i in range(len(y_true_all))
        if (y_true_all[i] is not None and y_prob_all[i] is not None and pred_parsed_all[i])
    ]
    print("\n===== Overall metrics (parsed predictions ONLY) =====")
    print(f"Total rows: {len(y_true_all)} | Parsed rows with GT & prob: {len(metric_idx)}")

    metrics_lines: List[str] = []
    metrics_lines.append(f"Judge Model: {model_id} (Pro)")
    metrics_lines.append(f"Task: {args.task}")
    metrics_lines.append(f"Date: {datetime.now().isoformat()}")
    metrics_lines.append(f"Total rows: {len(y_true_all)}")
    metrics_lines.append(f"Parsed rows with GT & prob: {len(metric_idx)}")
    metrics_lines.append("")

    if metric_idx:
        y_true_m = [y_true_all[i] for i in metric_idx]
        y_prob_m = [y_prob_all[i] for i in metric_idx]

        # ------- Thresholded (parsed-only) with F1 -------
        for TH in thresholds:
            y_pred = [1 if p >= TH else 0 for p in y_prob_m]
            TP, FP, TN, FN = ev.confusion(y_true_m, y_pred)
            acc, prec, rec = ev.metrics(TP, FP, TN, FN)
            f1 = ev.f1_from_counts(TP, FP, TN, FN)

            header = f"--- Threshold = {TH:.3f} ---"
            print(f"\n{header}")
            print(f"TP={TP} FP={FP} TN={TN} FN={FN}")
            print(f"Accuracy (↑ better): {acc:.4f}" if not np.isnan(acc) else "Accuracy: NA")
            print(f"Precision(↑ better): {prec:.4f}" if not np.isnan(prec) else "Precision: NA")
            print(f"Recall   (↑ better): {rec:.4f}"  if not np.isnan(rec)  else "Recall: NA")
            print(f"F1       (↑ better): {f1:.4f}"   if not np.isnan(f1)   else "F1: NA")

            metrics_lines.append(f"{header}")
            metrics_lines.append(f"TP={TP} FP={FP} TN={TN} FN={FN}")
            metrics_lines.append(f"Acc: {acc:.4f} Prec: {prec:.4f} Rec: {rec:.4f} F1: {f1:.4f}")

        # ------- Threshold-free metrics + Plot (parsed only) -------
        y_true_arr = np.array(y_true_m, dtype=np.int32)
        y_prob_arr = np.array(y_prob_m, dtype=np.float64)

        # Discrimination
        _auroc = ev.auroc(y_true_arr, y_prob_arr)
        _fpr95 = ev.fpr_at_tpr(y_true_arr, y_prob_arr, target_tpr=0.95)
        _aupr_correct, _aupr_error = ev.aupr_both_classes(y_true_arr, y_prob_arr)

        # Probabilistic & calibration
        _nll = ev.nll_binary(y_true_arr, y_prob_arr)
        _ece_adapt = ev.ece_equal_mass(y_true_arr, y_prob_arr, n_bins=15)
        _bss, _brier_val, _brier_base = ev.brier_skill_score(y_true_arr, y_prob_arr)
        
        # Originals / Fixed Width
        _ece_fixed = ev.ece_fixed(y_true_arr, y_prob_arr, n_bins=15)
        _smece     = ev.sm_ece(y_true_arr, y_prob_arr, bin_counts=(5,10,15,20))

        print("\n===== Threshold-free metrics (parsed predictions ONLY) =====")
        print(f"AUROC              (↑ better): {_auroc:.4f}" if _auroc is not None else "AUROC: NA")
        print(f"AUPR correct=pos   (↑ better): {_aupr_correct:.4f}" if _aupr_correct is not None else "AUPR (correct): NA")
        print(f"AUPR error=pos     (↑ better): {_aupr_error:.4f}"   if _aupr_error   is not None else "AUPR (error): NA")
        print(f"FPR@95%TPR         (↓ better): {_fpr95:.4f}" if _fpr95 is not None else "FPR@95: NA")
        print(f"NLL                (↓ better): {_nll:.6f}")
        print(f"ECE (adaptive)     (↓ better): {_ece_adapt:.4f}" if _ece_adapt is not None else "ECE (adapt): NA")
        print(f"Brier              (↓ better): {_brier_val:.6f}")
        print(f"Brier (baseline)   (reference): {_brier_base:.6f}")
        print(f"Brier Skill Score  (↑ better): {_bss:.4f}" if np.isfinite(_bss) else "BSS: NA")
        print(f"ECE (fixed)        (↓ better): {_ece_fixed:.4f}" if _ece_fixed is not None else "ECE (fixed): NA")
        print(f"smECE              (↓ better): {_smece:.4f}" if _smece is not None else "smECE: NA")

        metrics_lines.append("")
        metrics_lines.append("===== Threshold-free metrics =====")
        metrics_lines.append(f"AUROC: {_auroc}")
        metrics_lines.append(f"AUPR (correct): {_aupr_correct}")
        metrics_lines.append(f"AUPR (error): {_aupr_error}")
        metrics_lines.append(f"FPR@95: {_fpr95}")
        metrics_lines.append(f"NLL: {_nll}")
        metrics_lines.append(f"ECE (adaptive): {_ece_adapt}")
        metrics_lines.append(f"Brier: {_brier_val}")
        metrics_lines.append(f"BSS: {_bss}")
        metrics_lines.append(f"smECE: {_smece}")

        # Plot distribution
        plt.figure(figsize=(8, 5))
        bins = 20
        # Weights for normalized histograms
        correct_probs = y_prob_arr[y_true_arr == 1]
        wrong_probs   = y_prob_arr[y_true_arr == 0]
        
        if len(correct_probs) > 0:
            w = np.ones_like(correct_probs) / len(correct_probs)
            plt.hist(correct_probs, bins=bins, range=(0,1), alpha=0.6, label="Correct", weights=w)
        if len(wrong_probs) > 0:
            w_r = np.ones_like(wrong_probs) / len(wrong_probs)
            plt.hist(wrong_probs, bins=bins, range=(0,1), alpha=0.6, label="Wrong", weights=w_r)
            
        plt.xlabel("Gemini Score")
        plt.ylabel("Proportion")
        plt.legend()
        plt.title(f"Score Distribution: {model_safe_name} ({args.task})")
        plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        
        plot_path = out_dir_model / f"dist_{args.task}_{model_safe_name}.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"📊 Plot saved: {plot_path}")
        metrics_lines.append(f"Plot saved to: {plot_path}")

    else:
        msg = "\n(No parsed rows had both verifiable GT and a predicted probability—skipping metrics.)"
        print(msg)
        metrics_lines.append(msg.strip())

    # Write report
    metrics_txt_path = out_dir_model / f"metrics_{args.task}_{model_safe_name}.txt"
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_lines) + "\n")
    print(f"📝 Saved metrics report to: {metrics_txt_path}")
    print("\nAll shards processed.")

if __name__ == "__main__":
    main()