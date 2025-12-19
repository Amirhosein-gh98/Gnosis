from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Any

import re
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import evaluator as ev  # shared evaluator utilities
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

# ======== DEFAULTS (overridable via CLI) =========
DEFAULT_RM_MODEL_ID = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
SHARD_GLOB_DEFAULT  = "shard-*.parquet"
OUT_SUBDIR          = "scored"
THRESHOLDS_DEFAULT  = [0.05, 0.1, 0.2, 0.3, 0.5]

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
    """Heuristic: treat as GPQA if ≥ratio of sampled non-empty answers are single letters A–F."""
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
    """
    Returns (task, ans_col) with task in {"math","trivia","gpqa"}.
    'auto' infers from columns/content.
    """
    if requested != "auto":
        if requested == "math":
            return "math", "solution"
        if requested == "gpqa":
            return "gpqa", "answer"
        return "trivia", "answer"

    # auto
    if "solution" in df.columns:
        return "math", "solution"
    if "answer" in df.columns:
        if _looks_like_gpqa_answers(df["answer"].tolist()):
            return "gpqa", "answer"
        else:
            return "trivia", "answer"
    # fallback
    return "math", "solution"

# ---------- reward-model scoring ----------
@torch.no_grad()
def rm_scores_batch(
    rm, rm_tokenizer: AutoTokenizer, qs: List[str], cs: List[str],
    device: torch.device, max_len: int
) -> Tuple[List[float], List[float]]:
    """
    Returns:
        raw_scores: list of raw reward logits (float, unbounded)
        probs     : normalized score in [0,1] (clamped+scaled raw)
    """
    assert len(qs) == len(cs)
    texts = []
    bos = rm_tokenizer.bos_token
    for q, c in zip(qs, cs):
        conv = [{"role": "user", "content": q or ""}, {"role": "assistant", "content": c or ""}]
        txt = rm_tokenizer.apply_chat_template(conv, tokenize=False)
        if bos is not None and isinstance(txt, str) and txt.startswith(bos):
            txt = txt[len(bos):]
        texts.append(txt)

    with MeasureBlock("Stop Mechanism"):
        enc = rm_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = rm(**enc).logits.squeeze(-1)  # (B,)
        raw = out.detach().float()

    # map raw → [0,1] with clamp+linear scale (customizable via CONSTS if needed)
    raw_clamped = torch.clamp(raw, min=0.0, max=30.0)
    probs = (raw_clamped / 30.0).tolist()
    return raw.tolist(), probs

# ---------- main ----------
def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--rm_model", type=str, default=DEFAULT_RM_MODEL_ID,
                        help="HF reward model ID (default: Skywork/Skywork-Reward-V2-Llama-3.1-8B)")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing shards (parquet/jsonl).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base directory to write scored shards + reports.")
    parser.add_argument("--task", type=str, default="auto",
                        choices=["auto", "math", "trivia", "gpqa"],
                        help="auto: infer from columns ('solution'→math; 'answer'→gpqa if mostly letters A–F, else trivia)")
    parser.add_argument("--pattern", type=str, default=SHARD_GLOB_DEFAULT,
                        help=f"Glob to match shards (default: {SHARD_GLOB_DEFAULT}).")
    parser.add_argument("--thresholds", type=str, default="",
                        help="Comma-separated thresholds (default: 0.05,0.1,0.2,0.3,0.5).")
    parser.add_argument("--limit_shards", type=int, default=0,
                        help="If >0, cap the number of shards processed.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Row stride for downsampling within each shard (e.g., 4 keeps every 4th row).")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for RM scoring.")
    parser.add_argument("--max_len", type=int, default=32000,
                        help="Max sequence length for RM tokenizer.")
    args = parser.parse_args()

    thresholds = THRESHOLDS_DEFAULT if not args.thresholds.strip() else [
        float(x.strip()) for x in args.thresholds.split(",") if x.strip()
    ]

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    rm_model_id = args.rm_model
    rm_model_name = rm_model_id.split("/")[-1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rm_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    # Load reward model
    rm = AutoModelForSequenceClassification.from_pretrained(
        rm_model_id,
        torch_dtype=rm_dtype,
        num_labels=1,
    ).to(device)
    rm.eval()
    rm_tokenizer = AutoTokenizer.from_pretrained(rm_model_id)

    # Shards
    in_shards = find_shards(input_dir, args.pattern)
    if args.limit_shards and args.limit_shards > 0:
        in_shards = in_shards[: args.limit_shards]
    if not in_shards:
        print(f"No shards matched {input_dir}/{args.pattern}")
        return

    # Per-RM output directory (under user-provided output_dir)
    out_dir_model = output_dir / OUT_SUBDIR / rm_model_name
    out_dir_model.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(in_shards)} shard(s) -> {out_dir_model}")
    print(f"Using reward model: {rm_model_id}")

    # Global accumulators
    y_true_all: List[Optional[int]] = []    # 0/1/None
    y_prob_all: List[Optional[float]] = []  # float/None
    pred_parsed_all: List[bool] = []        # parsed flag (math/trivia/gpqa)

    with Progress(
        TextColumn("[bold]Scoring shards (Reward Model)[/bold]"),
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

            # RM scoring + GT evaluation in batches
            B = max(1, int(args.batch_size))
            for i in range(0, len(questions), B):
                batch_idx = list(range(i, min(i + B, len(questions))))
                qs = [questions[k] for k in batch_idx]
                cs = [completions[k] if completions[k] is not None else "" for k in batch_idx]

                raw_scores, probs = rm_scores_batch(rm, rm_tokenizer, qs, cs, device, args.max_len)
                rm_scores_all.extend(raw_scores)
                probs_all.extend(probs)

                # Ground-truth correctness (math / trivia / gpqa)
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

                for rel, k in enumerate(batch_idx):
                    p   = float(probs[rel])
                    ok  = gt[rel]  # None / 0.0 / 1.0
                    ppv = ev.preview_text(pred_prev[rel])
                    gpv = ev.preview_text(gold_prev[rel])

                    pred_parsed_all.append(bool(parsed_flags[rel]))
                    if pred_prev[rel]:
                        print(f"  - idx={k:>6}  rm_prob={p:>7.4f}  correct={ok}  pred≈{ppv}  gt≈{gpv}")

                        y_prob_all.append(p)
                        y_true_all.append(None if ok is None else (1 if ok >= 0.5 else 0))

                progress.advance(row_task_id, len(batch_idx))

            # --- attach columns for this shard ---
            assert len(probs_all) == len(df), "length mismatch after scoring"
            df["correctness_prob"] = probs_all          # normalized RM score in [0,1]
            df["reward_score"]     = rm_scores_all      # raw RM logit/score
            df["is_correct"]       = is_correct_all
            df["pred_ans_preview"] = pred_prev_all
            df["gold_ans_preview"] = gold_prev_all
            df["pred_parsed"]      = parsed_flags_shard

            # per-shard summaries use ONLY parsed predictions
            parsable_df = df.loc[df["pred_parsed"].astype(bool)]

            if parsable_df["correctness_prob"].notna().any():
                t = torch.tensor(
                    [x for x in parsable_df["correctness_prob"] if x is not None],
                    dtype=torch.float32
                )
                print(f"Summary [{shard_path.name}] (parsed only) rm_prob:"
                      f" mean={t.mean():.4f} median={t.median():.4f} "
                      f"min={t.min():.4f} max={t.max():.4f}")

            valid = [x for x in parsable_df["is_correct"] if x is not None]
            if valid:
                acc = sum(valid) / len(valid)
                print(f"Summary [{shard_path.name}] GT accuracy (parsed only): {acc:.4f} "
                      f"({len(valid)}/{len(parsable_df)})")

            # Write scored shard (under per-RM folder in output_dir)
            out_name = shard_path.stem + ".scored" + shard_path.suffix
            out_path = out_dir_model / out_name
            save_df(df, out_path)

            progress.remove_task(row_task_id)
            progress.advance(shard_task_id, 1)
            print(f"✅ {shard_path.name} -> {out_path} "
                  f"(added correctness_prob=norm(reward), reward_score, is_correct, previews, pred_parsed)")

    # ---------- FINAL OVERALL METRICS (parsed-only) ----------
    metric_idx = [
        i for i in range(len(y_true_all))
        if (y_true_all[i] is not None and y_prob_all[i] is not None and pred_parsed_all[i])
    ]
    print("\n===== Overall metrics (parsed predictions ONLY) =====")
    print(f"Total rows: {len(y_true_all)} | Parsed rows with GT & prob: {len(metric_idx)}")

    # Prepare a text report
    metrics_lines: List[str] = []
    metrics_lines.append(f"Reward model: {rm_model_name}")
    metrics_lines.append(f"RM Model ID: {rm_model_id}")
    metrics_lines.append(f"Task (arg): {args.task}")
    metrics_lines.append(f"Evaluated at: {datetime.now().isoformat(timespec='seconds')}")
    metrics_lines.append(f"Shards dir: {input_dir}")
    metrics_lines.append(f"Per-RM output dir: {out_dir_model}")
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
            print(f"Accuracy (↑ better): {acc:.4f}" if not np.isnan(acc) else "Accuracy (↑ better): NA")
            print(f"Precision(↑ better): {prec:.4f}" if not np.isnan(prec) else "Precision(↑ better): NA")
            print(f"Recall   (↑ better): {rec:.4f}"  if not np.isnan(rec)  else "Recall   (↑ better): NA")
            print(f"F1       (↑ better): {f1:.4f}"   if not np.isnan(f1)   else "F1       (↑ better): NA")

            metrics_lines.append("")
            metrics_lines.append(header)
            metrics_lines.append(f"TP={TP} FP={FP} TN={TN} FN={FN}")
            metrics_lines.append(f"Accuracy (↑ better): {acc:.4f}" if not np.isnan(acc) else "Accuracy (↑ better): NA")
            metrics_lines.append(f"Precision(↑ better): {prec:.4f}" if not np.isnan(prec) else "Precision(↑ better): NA")
            metrics_lines.append(f"Recall   (↑ better): {rec:.4f}"  if not np.isnan(rec)  else "Recall   (↑ better): NA")
            metrics_lines.append(f"F1       (↑ better): {f1:.4f}"   if not np.isnan(f1)   else "F1       (↑ better): NA")
    else:
        skip_msg = ("\n(No parsed rows had both verifiable GT and a predicted probability—"
                    "skipping thresholded metrics.)")
        print(skip_msg)
        metrics_lines.append(skip_msg.strip())

    # ---------- Threshold-free metrics + Plot (parsed only) ----------
    if metric_idx:
        y_true_arr = np.array([y_true_all[i] for i in metric_idx], dtype=np.int32)
        y_prob_arr = np.array([y_prob_all[i] for i in metric_idx], dtype=np.float64)

        # Discrimination
        _auroc = ev.auroc(y_true_arr, y_prob_arr)
        _fpr95 = ev.fpr_at_tpr(y_true_arr, y_prob_arr, target_tpr=0.95)
        _aupr_correct, _aupr_error = ev.aupr_both_classes(y_true_arr, y_prob_arr)

        # Probabilistic & calibration
        _nll = ev.nll_binary(y_true_arr, y_prob_arr)
        _ece_adapt = ev.ece_equal_mass(y_true_arr, y_prob_arr, n_bins=15)
        _bss, _brier_val, _brier_base = ev.brier_skill_score(y_true_arr, y_prob_arr)

        # Originals
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
        COLOR_CORRECT_DARK = "#2b8cbe"
        COLOR_WRONG_DARK = "#df0808"
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
        plt.ylabel("Density", fontsize=16)
        plt.title(f"{task_display}", fontsize=16)
        
        plt.tick_params(axis='both', which='major', labelsize=14)
        plt.legend(fontsize="x-large")
        plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        # --- UPDATED FILENAME ---
        # Uses task_display in the filename (e.g. ..._Math-Reasoning_...)
        plot_path = out_dir_model / f"correctness_prob_distribution_head_{task_display}_{rm_model_name}.png"
        
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

    # Write metrics report
    metrics_txt_path = out_dir_model / f"metrics_{args.task}_{rm_model_name}.txt"
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(metrics_lines) + "\n")
    print(f"📝 Saved metrics report to: {metrics_txt_path}")

    print("\nAll shards processed.")

if __name__ == "__main__":
    main()
