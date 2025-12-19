from glob import glob
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import torch
import pandas as pd
import numpy as np

# --- plotting (non-GUI backend) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rich.progress import (
    Progress, BarColumn, MofNCompleteColumn,
    TimeElapsedColumn, TimeRemainingColumn, TextColumn
)
from transformers import AutoTokenizer, AutoModelForCausalLM

import evaluator as ev  # all shared utilities live here
import torch
import time

class MeasureBlock:
    def __init__(self, name="Block"):
        self.name = name
        self.start_event = None
        self.end_event = None

    def __enter__(self):
        if torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()  # WAITS for GPU to finish
            elapsed_time_ms = self.start_event.elapsed_time(self.end_event)
        else:
            elapsed_time_ms = (time.time() - self.start_time) * 1000
            
        print(f"{self.name} execution time: {elapsed_time_ms:.3f} ms")

        
# ======== CONFIG (constants; can expose as flags later if needed) =========
SHARD_GLOB     = "shard-*.parquet"  # or *.jsonl
OUT_SUBDIR     = "scored"
BATCH_SIZE     = 1
MAX_LEN        = 32000
USE_ATTENTIONS = True    # auto-disabled if maps look huge

# Thresholds for reporting (same as ORM)
THRESHOLDS = [0.1, 0.2, 0.3, 0.5, 0.7]

SYS_INSTRUCTION_MATH   = "Please reason step by step, and put your final answer within \\boxed{}"
SYS_INSTRUCTION_TRIVIA = "Put your final answer within \\boxed{}."
# For GPQA we reuse the trivia-style chat prefix when computing the head.

# ---------- small helpers ----------
def has_correctness_head(model) -> bool:
    return (
        hasattr(model, "_should_stop")
        and hasattr(model, "stop_head")
        and hasattr(model, "hid_extractor")
        and hasattr(model, "conf_extractor")
    )

def find_shards(save_dir: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in sorted(glob(str(save_dir / pattern)))]

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, orient="records")
    raise ValueError(f"Unsupported shard format: {path.suffix}")

def infer_model_name(model_id_or_path: str) -> str:
    """
    Prefer parent directory name if this is a '.../checkpoint-XXXX' path;
    otherwise use the last path component or HF repo name.
    """
    s = str(model_id_or_path).rstrip("/")
    if "checkpoint-" in s:
        try:
            return Path(s).parent.name  # directory before 'checkpoint-XXXX'
        except Exception:
            pass
    # For HF IDs like "org/name", take last segment; for paths, take name
    return s.split("/")[-1] if "/" in s else Path(s).name

# ---------- correctness head (batch) ----------
@torch.no_grad()
def score_batch(
    model,
    tokenizer: AutoTokenizer,
    texts: List[str],
    device: torch.device,
) -> Optional[List[float]]:
    """Returns a list of correctness probs if the head exists, else None."""
    if not has_correctness_head(model):
        return None

    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
    input_ids      = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    out = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=USE_ATTENTIONS,
        output_hidden_states=False,
    )
    hidden_states = out.last_hidden_state

    # token-prob features for the head (model uses them internally)
    logits        = model.lm_head(hidden_states)
    token_probs   = ev.build_token_probs(input_ids, logits)

    attn_stack = out.attentions
    
    with MeasureBlock("Stop Mechanism"):
        probs = model._should_stop(
            last_hidden=hidden_states,
            attn_stack=attn_stack,
            token_probs=token_probs,
            mask=attention_mask.float(),
            input_ids=input_ids,
        )  # (B,1)

    return probs.squeeze(-1).float().clamp(1e-6, 1 - 1e-6).tolist()

# ---------- main ----------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HF model ID or local checkpoint path (e.g., .../checkpoint-XXXX)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing shards to score (the old SAVE_DIR)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory to write scored shards and reports."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="auto",
        choices=["auto", "math", "trivia", "gpqa"],
        help="Task type. 'auto' infers from columns: 'solution' -> math, 'answer' -> gpqa (else trivia)."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=SHARD_GLOB,
        help=f"Glob to match shards (default: {SHARD_GLOB})."
    )
    parser.add_argument("--thresholds", type=str, default="", help="Comma-separated thresholds to override default.")
    args = parser.parse_args()

    thresholds = THRESHOLDS if not args.thresholds.strip() else [
        float(x.strip()) for x in args.thresholds.split(",") if x.strip()
    ]

    model_ckpt_or_id = args.model
    model_name = infer_model_name(model_ckpt_or_id)
    print(model_name)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt_or_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_ckpt_or_id,
        torch_dtype=dtype,
        trust_remote_code=True,
        use_cache=False,
    ).to(device)
    model.eval()

    in_shards = find_shards(input_dir, args.pattern)
    if not in_shards:
        print(f"No shards matched {input_dir}/{args.pattern}")
        return

    # >>> per-model output directory under the user-provided output_dir
    out_dir_model = output_dir / OUT_SUBDIR / model_name
    out_dir_model.mkdir(parents=True, exist_ok=True)
    print(f"Scoring {len(in_shards)} shard(s) -> {out_dir_model}")
    if not has_correctness_head(model):
        print("⚠️  Model has no correctness head (`_should_stop`). Will compute GT correctness but skip head scoring.")

    # Collect for final metrics (across all shards)
    y_true_all: List[Optional[int]] = []     # 0/1/None
    y_prob_all: List[Optional[float]] = []   # float/None
    pred_parsed_all: List[bool] = []         # did we parse a candidate/answer?

    with Progress(
        TextColumn("[bold]Scoring shards[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        if args.task == "math":
            sub_ = 2
        else:
            sub_ = 4
        shard_task_id = progress.add_task("shards", total=len(in_shards))
        for shard_path in in_shards:
            df = load_df(shard_path)
            df = df.iloc[::sub_].reset_index(drop=True)

            task_choice = args.task
            ans_col = None
            if task_choice == "auto":
                if "solution" in df.columns:
                    task_choice, ans_col = "math", "solution"
                elif "answer" in df.columns:
                    task_choice, ans_col = "gpqa", "answer"  # prefer GPQA if answers present
                else:
                    task_choice, ans_col = "math", "solution"  # fallback
            else:
                ans_col = "answer" if task_choice in ("trivia", "gpqa") else "solution"

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

            # Build chat prefixes (use trivia-style for GPQA when computing head)
            eff_task_for_prefix = "trivia" if task_choice == "gpqa" else task_choice
            prefixes = [
                ev.chat_prefix(
                    tokenizer,
                    q,
                    task=eff_task_for_prefix,
                    sys_math=SYS_INSTRUCTION_MATH,
                    sys_trivia=SYS_INSTRUCTION_TRIVIA,
                )
                for q in questions
            ]
            texts = [p + (c or "") for p, c in zip(prefixes, completions)]

            row_task_id = progress.add_task(f"{shard_path.name}", total=len(texts))

            # --- correctness head probs (if available) + GT eval in batches ---
            probs_all: Optional[List[float]] = None
            parsed_flags_shard: List[bool] = []
            is_correct_all: List[Optional[float]] = []
            pred_prev_all:  List[Optional[str]]  = []
            gold_prev_all:  List[Optional[str]]  = []

            if has_correctness_head(model):
                probs_all = []
                for i in range(0, len(texts), BATCH_SIZE):
                    batch_idx    = list(range(i, min(i + BATCH_SIZE, len(texts))))
                    batch_texts  = [texts[k] for k in batch_idx]

                    probs = score_batch(model, tokenizer, batch_texts, device)
                    if probs is None:
                        probs_all = None
                        break

                    # GT for same rows (math / trivia / gpqa)
                    batch_comps = [completions[k] for k in batch_idx]
                    batch_gold  = [gold_any[k]    for k in batch_idx]
                    if task_choice == "trivia":
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_trivia_batch(batch_comps, batch_gold)
                    elif task_choice == "gpqa":
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_gpqa_batch(batch_comps, batch_gold)
                    else:
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_math_batch(batch_comps, batch_gold)

                    is_correct_all.extend(gt)
                    pred_prev_all.extend(pred_prev)
                    gold_prev_all.extend(gold_prev)
                    parsed_flags_shard.extend([bool(x) for x in parsed_flags])

                    for rel, k in enumerate(batch_idx):
                        p   = probs[rel]
                        ok  = gt[rel]
                        ppv = ev.preview_text(pred_prev[rel])
                        gpv = ev.preview_text(gold_prev[rel])

                        pred_parsed_all.append(bool(parsed_flags[rel]))
                        print(f"  - idx={k:>6}  prob={p:>7.4f}  correct={ok}  pred≈{ppv}  gt≈{gpv}")
                        if pred_prev[rel]:
                            y_prob_all.append(p)
                            y_true_all.append(None if ok is None else (1 if ok >= 0.5 else 0))

                    probs_all.extend(probs)
                    progress.advance(row_task_id, len(batch_idx))

            # even if head missing, still do GT evaluation; store None probs
            if probs_all is None:
                for i in range(0, len(completions), BATCH_SIZE):
                    batch_idx = list(range(i, min(i + BATCH_SIZE, len(completions))))
                    batch_comps = [completions[k] for k in batch_idx]
                    batch_gold  = [gold_any[k]    for k in batch_idx]
                    if task_choice == "trivia":
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_trivia_batch(batch_comps, batch_gold)
                    elif task_choice == "gpqa":
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_gpqa_batch(batch_comps, batch_gold)
                    else:
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_math_batch(batch_comps, batch_gold)

                    is_correct_all.extend(gt)
                    pred_prev_all.extend(pred_prev)
                    gold_prev_all.extend(gold_prev)
                    parsed_flags_shard.extend([bool(x) for x in parsed_flags])

                    for rel, k in enumerate(batch_idx):
                        ok = gt[rel]
                        pred_parsed_all.append(bool(parsed_flags[rel]))
                        y_prob_all.append(None)  # no head → no prediction prob
                        y_true_all.append(None if ok is None else (1 if ok >= 0.5 else 0))
                    progress.advance(row_task_id, len(batch_idx))

            # --- attach columns for this shard ---
            if probs_all is not None:
                assert len(probs_all) == len(df), "length mismatch after scoring"
                df["correctness_prob"] = probs_all
            else:
                df["correctness_prob"] = None  # entire shard lacks predictions

            df["is_correct"]       = is_correct_all
            df["pred_ans_preview"] = pred_prev_all
            df["gold_ans_preview"] = gold_prev_all
            df["pred_parsed"]      = parsed_flags_shard

            # Per-shard summaries use ONLY parsed predictions
            parsed_mask = df["pred_parsed"].astype(bool)
            parsable_df = df.loc[parsed_mask]

            if parsable_df["correctness_prob"].notna().any():
                t = torch.tensor(
                    [x for x in parsable_df["correctness_prob"] if x is not None],
                    dtype=torch.float32
                )
                print(f"Summary [{shard_path.name}] (parsed only) stop_prob:"
                      f" mean={t.mean():.4f} median={t.median():.4f} "
                      f"min={t.min():.4f} max={t.max():.4f}")

            valid = [x for x in parsable_df["is_correct"] if x is not None]
            if valid:
                acc = sum(valid) / len(valid)
                print(f"Summary [{shard_path.name}] GT accuracy (parsed only): {acc:.4f} "
                      f"({len(valid)}/{len(parsable_df)})")

            # write scored shard
            out_name = shard_path.stem + ".scored" + shard_path.suffix
            out_path = out_dir_model / out_name
            if shard_path.suffix == ".parquet":
                df.to_parquet(out_path, index=False)
            else:
                df.to_json(out_path, lines=True, orient="records", force_ascii=False)

            progress.remove_task(row_task_id)
            progress.advance(shard_task_id, 1)
            print(f"✅ {shard_path.name} -> {out_path} "
                  f"(added correctness_prob, is_correct, previews, pred_parsed)")

    # ---------- FINAL OVERALL METRICS (parsed-only; matches ORM) ----------
    metric_idx = [
        i for i in range(len(y_true_all))
        if (y_true_all[i] is not None and y_prob_all[i] is not None and pred_parsed_all[i])
    ]
    print("\n===== Overall metrics (parsed predictions ONLY) =====")
    print(f"Total rows: {len(y_true_all)} | Parsed rows with GT & prob: {len(metric_idx)}")

    # accumulate a metrics report to save as text
    metrics_lines: List[str] = []
    metrics_lines.append(f"Model: {model_name}")
    metrics_lines.append(f"Checkpoint: {model_ckpt_or_id}")
    metrics_lines.append(f"Task (arg): {args.task}")
    metrics_lines.append(f"Evaluated at: {datetime.now().isoformat(timespec='seconds')}")
    metrics_lines.append(f"Shards dir: {input_dir}")
    metrics_lines.append(f"Per-model output dir: {out_dir_model}")
    metrics_lines.append("")
    metrics_lines.append("===== Overall metrics (parsed predictions ONLY) =====")
    metrics_lines.append(f"Total rows: {len(y_true_all)}")
    metrics_lines.append(f"Parsed rows with GT & prob: {len(metric_idx)}")

    if metric_idx:
        y_true_m = [y_true_all[i] for i in metric_idx]
        y_prob_m = [y_prob_all[i] for i in metric_idx]

        # ------- Thresholded (parsed-only) with F1 -------
        for TH in thresholds:
            y_pred = [1 if p >= TH else 0 for p in y_prob_m]
            TP, FP, TN, FN = ev.confusion(y_true_m, y_pred)
            acc, prec, rec = ev.metrics(TP, FP, TN, FN)
            f1 = ev.f1_from_counts(TP, FP, TN, FN)

            header = f"--- Threshold = {TH:.3f} (parsed only) ---"
            print(f"\n{header}")
            print(f"TP={TP} FP={FP} TN={TN} FN={FN}")
            print(f"Accuracy : {acc:.4f}" if not np.isnan(acc) else "Accuracy : NA")
            print(f"Precision: {prec:.4f}" if not np.isnan(prec) else "Precision: NA")
            print(f"Recall   : {rec:.4f}"  if not np.isnan(rec)  else "Recall   : NA")
            print(f"F1       : {f1:.4f}"   if not np.isnan(f1)   else "F1       : NA")

            metrics_lines.append("")
            metrics_lines.append(header)
            metrics_lines.append(f"TP={TP} FP={FP} TN={TN} FN={FN}")
            metrics_lines.append(f"Accuracy : {acc:.4f}" if not np.isnan(acc) else "Accuracy : NA")
            metrics_lines.append(f"Precision: {prec:.4f}" if not np.isnan(prec) else "Precision: NA")
            metrics_lines.append(f"Recall   : {rec:.4f}"  if not np.isnan(rec)  else "Recall   : NA")
            metrics_lines.append(f"F1       : {f1:.4f}"   if not np.isnan(f1)   else "F1       : NA")
    else:
        skip_msg = ("\n(No parsed rows had both verifiable GT and a predicted probability—"
                    "skipping thresholded metrics.)")
        print(skip_msg)
        metrics_lines.append(skip_msg.strip())

    # ---------- Threshold-free metrics + Plot (parsed-only; matches ORM) ----------
    if metric_idx:
        y_true_arr = np.array([y_true_all[i] for i in metric_idx], dtype=np.int32)
        y_prob_arr = np.array([y_prob_all[i] for i in metric_idx], dtype=np.float64)

        # Discrimination
        _auroc = ev.auroc(y_true_arr, y_prob_arr)
        _fpr95 = ev.fpr_at_tpr(y_true_arr, y_prob_arr, target_tpr=0.95)
        _aupr_correct, _aupr_error = ev.aupr_both_classes(y_true_arr, y_prob_arr)

        # Probabilistic & calibration (core paper-5)
        _nll = ev.nll_binary(y_true_arr, y_prob_arr)
        _ece_adapt = ev.ece_equal_mass(y_true_arr, y_prob_arr, n_bins=15)
        _bss, _brier_val, _brier_base = ev.brier_skill_score(y_true_arr, y_prob_arr)

        # Originals you already had
        _ece_fixed = ev.ece_fixed(y_true_arr, y_prob_arr, n_bins=15)
        _smece     = ev.sm_ece(y_true_arr, y_prob_arr, bin_counts=(5,10,15,20))

        print("\n===== Threshold-free metrics (parsed predictions ONLY) =====")
        print(f"AUROC              (↑ better): {_auroc:.4f}" if _auroc is not None else "AUROC              (↑ better): NA")
        print(f"AUPR correct=pos   (↑ better): {_aupr_correct:.4f}" if _aupr_correct is not None else "AUPR correct=pos   (↑ better): NA")
        print(f"AUPR error=pos     (↑ better): {_aupr_error:.4f}"  if _aupr_error   is not None else "AUPR error=pos     (↑ better): NA")
        print(f"FPR@95%TPR         (↓ better): {_fpr95:.4f}" if _fpr95 is not None else "FPR@95%TPR         (↓ better): NA")
        print(f"NLL                (↓ better): {_nll:.6f}")
        print(f"ECE (adaptive)     (↓ better): {_ece_adapt:.4f}" if _ece_adapt is not None else "ECE (adaptive)     (↓ better): NA")
        print(f"Brier              (↓ better): {_brier_val:.6f}")
        print(f"Brier (baseline)   (reference): {_brier_base:.6f}")
        print(f"Brier Skill Score  (↑ better): {_bss:.4f}" if np.isfinite(_bss) else "Brier Skill Score  (↑ better): NA")
        print(f"ECE (fixed)        (↓ better): {_ece_fixed:.4f}" if _ece_fixed is not None else "ECE (fixed)        (↓ better): NA")
        print(f"smECE              (↓ better): {_smece:.4f}" if _smece is not None else "smECE              (↓ better): NA")

        metrics_lines.append("")
        metrics_lines.append("===== Threshold-free metrics (parsed predictions ONLY) =====")
        metrics_lines.append(f"AUROC              (↑ better): {_auroc:.4f}" if _auroc is not None else "AUROC              (↑ better): NA")
        metrics_lines.append(f"AUPR correct=pos   (↑ better): {_aupr_correct:.4f}" if _aupr_correct is not None else "AUPR correct=pos   (↑ better): NA")
        metrics_lines.append(f"AUPR error=pos     (↑ better): {_aupr_error:.4f}"  if _aupr_error   is not None else "AUPR error=pos     (↑ better): NA")
        metrics_lines.append(f"FPR@95%TPR         (↓ better): {_fpr95:.4f}" if _fpr95 is not None else "FPR@95%TPR         (↓ better): NA")
        metrics_lines.append(f"NLL                (↓ better): {_nll:.6f}")
        metrics_lines.append(f"ECE (adaptive)     (↓ better): {_ece_adapt:.4f}" if _ece_adapt is not None else "ECE (adaptive)     (↓ better): NA")
        metrics_lines.append(f"Brier              (↓ better): {_brier_val:.6f}")
        metrics_lines.append(f"Brier (baseline)   (reference): {_brier_base:.6f}")
        metrics_lines.append(f"Brier Skill Score  (↑ better): {_bss:.4f}" if np.isfinite(_bss) else "Brier Skill Score  (↑ better): NA")
        metrics_lines.append(f"ECE (fixed)        (↓ better): {_ece_fixed:.4f}" if _ece_fixed is not None else "ECE (fixed)        (↓ better): NA")
        metrics_lines.append(f"smECE              (↓ better): {_smece:.4f}" if _smece is not None else "smECE              (↓ better): NA")

        # Define colors
        COLOR_WRONG_DARK    = "#2b8cbe"
        COLOR_CORRECT_DARK  = "#cb181d"
        # --- TASK MAPPING LOGIC ---
        # Maps the raw input string (key) to the Desired Title (value)
        # We use .lower() on the input key to ensure it matches regardless of case
        task_map = {
            "math":   "Math-Reasoning",
            "trivia": "TriviaQA",
            "gpqa":   "GPQA",
            "mmlu":   "MMLUPro"
        }
        
        # Get the clean display name. 
        # Defaults to the original TASK string (Title Cased) if not found in the map.
        # Assuming your variable is named 'task' or 'TASK'
        raw_task = str(args.task).lower()
        task_display = task_map.get(raw_task, raw_task.title())

        # Plot distribution (normalized proportions)
        wrong_probs   = y_prob_arr[y_true_arr == 0]
        correct_probs = y_prob_arr[y_true_arr == 1]

        plt.figure(figsize=(8, 5))
        bins = 20
        
        # Plot Correct
        if len(correct_probs) > 0:
            w = np.ones_like(correct_probs) / len(correct_probs)
            plt.hist(correct_probs, bins=bins, range=(0, 1), alpha=0.6,
                     label="Correct", weights=w, color=COLOR_CORRECT_DARK)
        
        # Plot Wrong
        if len(wrong_probs) > 0:
            w_r = np.ones_like(wrong_probs) / len(wrong_probs)
            plt.hist(wrong_probs, bins=bins, range=(0, 1), alpha=0.6,
                     label="Wrong", weights=w_r, color=COLOR_WRONG_DARK)

        # --- UPDATED LABELS & TITLE ---
        
        plt.xlabel("Predicted Correctness Score", fontsize=16)
        plt.ylabel("Proportion", fontsize=16)
        plt.title(f"Predicted Correctness Scores (Correctness Head) — {task_display}", fontsize=16)
        
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize="x-large")
        plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        # --- UPDATED FILENAME ---
        # Uses task_display in the filename (e.g. ..._Math-Reasoning_...)
        plot_path = out_dir_model / f"correctness_prob_distribution_head_{task_display}_{model_name}.png"
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        #*******************************************************************************************
        
        print(f"📊 Saved score distribution plot to: {plot_path}")
        metrics_lines.append(f"Plot saved to: {plot_path}")
    else:
        msg = ("\n(No parsed rows had both verifiable GT and a predicted probability—"
               "skipping AUROC/AUPR/FPR@95, Brier/ECE/smECE and the plot.)")
        print(msg)
        metrics_lines.append(msg.strip())

    # write metrics report
    metrics_txt_path = out_dir_model / f"metrics_{args.task}_{model_name}.txt"
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_lines) + "\n")
    print(f"📝 Saved metrics report to: {metrics_txt_path}")

    print("\nAll shards processed.")

if __name__ == "__main__":
    main()
