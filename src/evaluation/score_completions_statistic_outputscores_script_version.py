#!/usr/bin/env python3
# Statistical methods ONLY (no correctness head).
# Single forward on a tail window; saves per-metric reports + plots.

import os, math, json, re
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from transformers import AutoTokenizer, AutoModelForCausalLM

# plotting (non-GUI)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import evaluator as ev  # your evaluation utilities

# ======== DEFAULTS (overridable by CLI) ========
DEFAULT_MODEL = "Qwen/Qwen3-1.7B"
DEFAULT_PATTERN = "shard-*.parquet"
DEFAULT_OUT_SUBDIR = "scored"
DEFAULT_STATS_SUBDIR = "stats"
DEFAULT_BATCH = 4
DEFAULT_MAXLEN = 32000
DEFAULT_TASK = "auto"  # {auto, math, trivia, gpqa}
DEFAULT_COE_WIN = 256
DEFAULT_THRESH = [0.2, 0.5, 0.8, 0.9]

SYS_INSTRUCTION_MATH   = "Please reason step by step, and put your final answer within \\boxed{}"
SYS_INSTRUCTION_TRIVIA = "This is a Trivia question, put your final answer within \\boxed{Final_answer}"

METRIC_FLAGS_DEFAULT = {
    "maxprob": True,
    "ppl": True,
    "entropy": True,
    "coe": True,
    "logit": True,
    "hidden": False,
    "attns": True,
}
USE_TOKLENS_FOR_EXTRAS = True

# ---------- IO ----------
def find_shards(save_dir: Path, pattern: str) -> List[Path]:
    return [Path(p) for p in sorted(glob(str(save_dir / pattern)))]

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True, orient="records")
    raise ValueError(f"Unsupported shard format: {path.suffix}")

# ---------- GPQA detection helper ----------
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
        return "trivia", "answer"
    return "math", "solution"

def infer_model_name(m: str) -> str:
    parts = m.rstrip("/").split("/")
    if len(parts) >= 2 and parts[-1].startswith("checkpoint-"):
        return parts[-2]
    return parts[-1]

# ---------- Helpers ----------
def _is_finite(x: Optional[float]) -> bool:
    return (x is not None) and (not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))))

def _rank_to_prob_oriented(raw: List[Optional[float]], valid_idx: List[int], higher_is_better: bool) -> np.ndarray:
    vals = [float(raw[j]) for j in valid_idx]
    if not vals:
        return np.array([], dtype=np.float64)
    arr = np.array(vals, dtype=np.float64)
    if not higher_is_better:
        arr = -arr
    if np.all(arr == arr[0]):
        return np.full_like(arr, 0.5, dtype=np.float64)
    order = np.argsort(arr)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(arr), dtype=np.float64)
    ranks /= max(len(arr) - 1, 1.0)
    return ranks

def _spearman_auto_orientation(y_true_known: np.ndarray, raw_known: np.ndarray) -> Optional[bool]:
    try:
        r1 = pd.Series(y_true_known).rank().to_numpy()
        r2 = pd.Series(raw_known).rank().to_numpy()
        num = ((r1 - r1.mean()) * (r2 - r2.mean())).sum()
        den = np.sqrt(((r1 - r1.mean())**2).sum()) * np.sqrt(((r2 - r2.mean())**2).sum())
        if den == 0:
            return None
        rho = num / den
        if rho > 0: return True
        if rho < 0: return False
        return None
    except Exception:
        return None

def _print_and_save_threshold_free_block(name: str,
                                         y_true_all: List[Optional[int]],
                                         raw_scores: List[Optional[float]],
                                         known_idx: List[int],
                                         higher_is_better: bool,
                                         out_dir: Path,
                                         model_id: str,
                                         task_label: str,
                                         note: str = ""):
    valid_idx = [i for i in known_idx if _is_finite(raw_scores[i])]
    print(f"\n===== Threshold-free metrics [{name}] (verifiable GT rows) =====")
    if note:
        print(note)

    report_path = out_dir / f"{name}_threshold_free_metrics.txt"
    if not valid_idx:
        print("Rows used          : 0")
        for line in [
            "AUROC : NA","AUPR (correct=pos) : NA","AUPR (error=pos) : NA","FPR@95%TPR : NA",
            "NLL : NA","ECE (adaptive) : NA","Brier : NA","Brier (baseline) : NA","Brier Skill Score : NA",
            "ECE (fixed) : NA","smECE : NA"
        ]: print(line)
        with open(report_path, "w") as f:
            f.write(f"Metric: {name}\n")
            if note: f.write(note + "\n")
            f.write("Rows used          : 0\n")
            f.write("AUROC              : NA\nAUPR (correct=pos) : NA\nAUPR (error=pos)   : NA\n")
            f.write("FPR@95%TPR         : NA\nNLL                : NA\nECE (adaptive)     : NA\n")
            f.write("Brier              : NA\nBrier (baseline)   : NA\nBrier Skill Score  : NA\n")
            f.write("ECE (fixed)        : NA\nsmECE              : NA\n")
        print(f"   ↳ saved: {report_path.name}")
        return

    y_true = np.array([y_true_all[i] for i in valid_idx], dtype=np.int32)
    probs  = _rank_to_prob_oriented(raw_scores, valid_idx, higher_is_better=higher_is_better)

    _auroc        = ev.auroc(y_true, probs)
    _fpr95        = ev.fpr_at_tpr(y_true, probs, target_tpr=0.95)
    _aupr_c, _aupr_e = ev.aupr_both_classes(y_true, probs)
    _nll          = ev.nll_binary(y_true, probs)
    _ece_adapt    = ev.ece_equal_mass(y_true, probs, n_bins=15)
    _bss, _brier_val, _brier_base = ev.brier_skill_score(y_true, probs)
    _ece_fixed    = ev.ece_fixed(y_true, probs, n_bins=15)
    _smece        = ev.sm_ece(y_true, probs, bin_counts=(5,10,15,20))

    print(f"Rows used          : {len(probs)}")
    print(f"AUROC              : {_auroc:.4f}" if _auroc is not None else "AUROC : NA")
    print(f"AUPR (correct=pos) : {_aupr_c:.4f}" if _aupr_c is not None else "AUPR (correct=pos) : NA")
    print(f"AUPR (error=pos)   : {_aupr_e:.4f}" if _aupr_e is not None else "AUPR (error=pos)   : NA")
    print(f"FPR@95%TPR         : {_fpr95:.4f}" if _fpr95 is not None else "FPR@95%TPR : NA")
    print(f"NLL                : {_nll:.6f}")
    print(f"ECE (adaptive)     : {_ece_adapt:.4f}" if _ece_adapt is not None else "ECE (adaptive) : NA")
    print(f"Brier              : {_brier_val:.6f}")
    print(f"Brier (baseline)   : {_brier_base:.6f}")
    print(f"Brier Skill Score  : {_bss:.4f}" if np.isfinite(_bss) else "Brier Skill Score : NA")
    print(f"ECE (fixed)        : {_ece_fixed:.4f}" if _ece_fixed is not None else "ECE (fixed) : NA")
    print(f"smECE              : {_smece:.4f}" if _smece is not None else "smECE : NA")

    with open(report_path, "w") as f:
        f.write(f"Metric: {name}\n")
        if note: f.write(note + "\n")
        f.write(f"Rows used          : {len(probs)}\n")
        f.write(f"AUROC              : {(_auroc if _auroc is not None else 'NA')}\n")
        f.write(f"AUPR (correct=pos) : {(_aupr_c if _aupr_c is not None else 'NA')}\n")
        f.write(f"AUPR (error=pos)   : {(_aupr_e if _aupr_e is not None else 'NA')}\n")
        f.write(f"FPR@95%TPR         : {(_fpr95 if _fpr95 is not None else 'NA')}\n")
        f.write(f"NLL                : {_nll}\n")
        f.write(f"ECE (adaptive)     : {(_ece_adapt if _ece_adapt is not None else 'NA')}\n")
        f.write(f"Brier              : {_brier_val}\n")
        f.write(f"Brier (baseline)   : {_brier_base}\n")
        f.write(f"Brier Skill Score  : {(_bss if np.isfinite(_bss) else 'NA')}\n")
        f.write(f"ECE (fixed)        : {(_ece_fixed if _ece_fixed is not None else 'NA')}\n")
        f.write(f"smECE              : {(_smece if _smece is not None else 'NA')}\n")

    # normalized histogram
    wrong_probs   = probs[y_true == 0]
    correct_probs = probs[y_true == 1]
    plt.figure(figsize=(8, 5))
    bins = 20
    if len(correct_probs) > 0:
        w = np.ones_like(correct_probs) / len(correct_probs)
        plt.hist(correct_probs, bins=bins, range=(0, 1), alpha=0.6, label="Correct", weights=w)
    if len(wrong_probs) > 0:
        w_r = np.ones_like(wrong_probs) / len(wrong_probs)
        plt.hist(wrong_probs, bins=bins, range=(0, 1), alpha=0.6, label="Wrong", weights=w_r)
    plt.xlabel("Oriented score (rank-normalized)")
    plt.ylabel("Proportion")
    plt.title(f"{name} — threshold-free oriented distribution")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    plot_path = out_dir / f"{name}_distribution_{task_label}_{model_id}.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"   ↳ saved: {report_path.name}, {plot_path.name}")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF id or local checkpoint dir")
    parser.add_argument("--input_dir", type=Path, required=True, help="Folder containing shards")
    parser.add_argument("--output_dir", type=Path, default=None, help="Root to write outputs (default: input_dir)")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, choices=["auto","math","trivia","gpqa"])
    parser.add_argument("--pattern", type=str, default=DEFAULT_PATTERN, help="Shard filename glob")
    parser.add_argument("--out_subdir", type=str, default=DEFAULT_OUT_SUBDIR, help="First-level output subdir")
    parser.add_argument("--stats_subdir", type=str, default=DEFAULT_STATS_SUBDIR, help="Where to save stats under model folder")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--max_len", type=int, default=DEFAULT_MAXLEN)
    parser.add_argument("--coe_window_tokens", type=int, default=DEFAULT_COE_WIN)
    parser.add_argument("--metrics", type=str, default="",
                        help="Comma-separated subset from {maxprob,ppl,entropy,coe,logit,hidden,attns}")
    parser.add_argument("--thresholds", type=str, default="", help="Comma-separated thresholds (normalized)")
    parser.add_argument("--save_indiv_scores", dest="save_indiv_scores", action="store_true", default=True)
    parser.add_argument("--no_save_indiv_scores", dest="save_indiv_scores", action="store_false")
    parser.add_argument("--sample_n", type=int, default=0, help="Sample up to N rows per shard (0=all)")
    parser.add_argument("--stride", type=int, default=1, help="Row stride downsample (e.g., 2 keeps every other)")
    parser.add_argument("--limit_shards", type=int, default=0, help="Only process first N shards (0=all)")
    args = parser.parse_args()

    # metric flags
    flags = METRIC_FLAGS_DEFAULT.copy()
    if args.metrics.strip():
        wanted = {x.strip().lower() for x in args.metrics.split(",") if x.strip()}
        for k in flags:
            flags[k] = (k in wanted)

    thresholds = DEFAULT_THRESH if not args.thresholds.strip() else [
        float(x.strip()) for x in args.thresholds.split(",") if x.strip()
    ]

    device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device_t.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True, use_cache=False,
    ).to(device_t)
    model.eval()

    in_shards = find_shards(args.input_dir, args.pattern)
    if args.limit_shards > 0:
        in_shards = in_shards[:args.limit_shards]

    if not in_shards:
        print(f"No shards matched {args.input_dir}/{args.pattern}")
        return

    model_name = infer_model_name(args.model)
    out_root = args.output_dir if args.output_dir is not None else args.input_dir
    out_dir = Path(out_root) / args.out_subdir / model_name / args.stats_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {len(in_shards)} shard(s) -> {out_dir}")
    print("Enabled metrics:", [k for k, v in flags.items() if v])
    print("Thresholds (normalized):", thresholds)
    print(f"Task mode requested: {args.task}")

    # Global arrays for final reports
    y_true_all: List[Optional[int]] = []
    pred_parsed_all: List[bool] = []
    scores_global: Dict[str, List[Optional[float]]] = {
        "maxprob": [], "ppl": [], "entropy": [],
        "coe_mag_mean": [], "coe_mag_var": [], "coe_ang_mean": [], "coe_ang_var": [], "coe_R": [], "coe_C": [],
        "extra_perplexity": [], "extra_window_entropy": [], "extra_logit_entropy": [],
        "hidden_svd_mean": [], "hidden_svd_last": [],
        "attn_eigprod_mean": [], "attn_eigprod_last": [],
    }

    def tokenize_len(txt: str) -> int:
        return tokenizer(txt, return_tensors="pt", padding=False, truncation=True, max_length=args.max_len).input_ids.shape[1]

    def fmt_val(v):
        try:
            if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
                return "NA"
            return f"{float(v):.4f}"
        except Exception:
            return "NA"

    with Progress(
        TextColumn("[bold]Scoring shards[/bold]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        shard_task = progress.add_task("shards", total=len(in_shards))

        for shard_path in in_shards:
            # Load this shard
            df = load_df(shard_path)

            # Optional row downsampling
            if args.stride > 1:
                df = df.iloc[::args.stride].reset_index(drop=True)

            # Optional heuristic thinning by task (kept from your code)
            sub_ = 2 if args.task == "math" else 4
            if sub_ > 1:
                df = df.iloc[::sub_].reset_index(drop=True)

            if args.sample_n > 0 and len(df) > args.sample_n:
                df = df.sample(n=args.sample_n, random_state=42).reset_index(drop=True)

            # Decide task + answer column
            task = args.task
            if task == "auto":
                task, ans_col = _infer_task_and_ans_col(df, "auto")
            else:
                ans_col = "answer" if task in ("trivia", "gpqa") else "solution"

            needed = [c for c in ("question", "completion") if c not in df.columns]
            if ans_col and ans_col not in df.columns:
                needed.append(ans_col)
            if needed:
                print(f"Skipping {shard_path.name}: missing columns {needed}")
                progress.advance(shard_task, 1)
                continue

            questions   = df["question"].tolist()
            completions = df["completion"].tolist()
            gold_any    = df[ans_col].tolist() if ans_col in df.columns else [None]*len(df)

            prefixes = [
                ev.chat_prefix(tokenizer, q, task, SYS_INSTRUCTION_MATH, SYS_INSTRUCTION_TRIVIA)
                for q in questions
            ]
            texts = [p + (c or "") for p, c in zip(prefixes, completions)]
            print(f"\n→ {shard_path.name}: task={task} (rows={len(df)})")

            row_task = progress.add_task(f"{shard_path.name}", total=len(texts))

            # Per-shard columns (collect)
            col_maxprob, col_ppl, col_entropy = [], [], []
            col_coe_mag_mean, col_coe_mag_var = [], []
            col_coe_ang_mean, col_coe_ang_var = [], []
            col_coe_R, col_coe_C = [], []
            col_extra_ppl, col_extra_win_ent, col_extra_log_ent = [], [], []
            col_hidden_mean, col_hidden_last = [], []
            col_attn_mean, col_attn_last = [], []

            indiv_scores = {"logit": {"perplexity": [], "window_entropy": [], "logit_entropy": []}, "hidden": {}, "attns": {}}
            parsed_flags_shard: List[bool] = []

            for i in range(0, len(texts), args.batch_size):
                batch_idx   = list(range(i, min(i + args.batch_size, len(texts))))
                batch_texts = [texts[k] for k in batch_idx]
                with ev.timing("tokenize"):
                    enc = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_len)
                input_ids      = enc["input_ids"].to(device_t)
                attention_mask = enc["attention_mask"].to(device_t)

                # per-sample: single forward on tail window (logits/hidden/attns together)
                for rel, k in enumerate(batch_idx):
                    full_len = int(attention_mask[rel].sum().item())
                    pref_len = tokenize_len(prefixes[k])

                    ws = max(0, full_len - args.coe_window_tokens)
                    ids_win = input_ids[rel:rel+1, ws:full_len]
                    att_win = attention_mask[rel:rel+1, ws:full_len]
                    win_len = int(att_win.sum().item())
                    pref_in_win = max(0, min(win_len, pref_len - ws))

                    need_hidden = flags["coe"] or flags["hidden"] or flags["attns"]
                    need_attn   = flags["attns"]

                    with torch.no_grad():
                        with ev.timing("forward_tail_all"):
                            out = model.model(
                                input_ids=ids_win,
                                attention_mask=att_win,
                                use_cache=False,
                                output_hidden_states=need_hidden,
                                output_attentions=need_attn,
                            )
                    hidden_last = out.last_hidden_state
                    logits_win  = model.lm_head(hidden_last)

                    # GT for this row
                    gold_here = gold_any[k]
                    comp_here = completions[k]
                    if task == "trivia":
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_trivia_batch([comp_here], [gold_here])
                    elif task == "gpqa":
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_gpqa_batch([comp_here], [gold_here])
                    else:
                        gt, pred_prev, gold_prev, parsed_flags = ev.evaluate_math_batch([comp_here], [gold_here])

                    ok = gt[0]
                    y_true_all.append(None if ok is None else (1 if ok >= 0.5 else 0))
                    parsed_here = bool(parsed_flags[0])
                    pred_parsed_all.append(parsed_here)
                    parsed_flags_shard.append(parsed_here)

                    # step logits for output-score metrics
                    start_step = max(pref_in_win - 1, 0)
                    end_step   = max(win_len - 1, 0)
                    step_logits = [logits_win[0, s, :].float().detach() for s in range(start_step, end_step)] if end_step > start_step else []
                    out_info = ev.OutputScoreInfo(step_logits)

                    maxprob = ppl = entr = None
                    if flags["maxprob"]:
                        with ev.timing("metric:maxprob"):
                            maxprob = out_info.compute_maxprob()
                    if flags["ppl"]:
                        with ev.timing("metric:ppl"):
                            ppl = out_info.compute_ppl()
                    if flags["entropy"]:
                        with ev.timing("metric:entropy"):
                            entr = out_info.compute_entropy()

                    perpx_val = went_val = lent_val = None
                    if flags["logit"]:
                        with ev.timing("metric:extra_logit"):
                            l_win_cpu = logits_win[0, :win_len, :].float().detach().cpu()
                            tmp_scores = []
                            ev.compute_scores(
                                logits=[l_win_cpu], hidden_acts=[], attns=[], scores=tmp_scores,
                                indiv_scores=indiv_scores, mt_list=["logit"],
                                tok_ins=[ids_win.detach().cpu()],
                                tok_lens=[(pref_in_win, win_len)],
                                use_toklens=USE_TOKLENS_FOR_EXTRAS,
                            )
                            perpx_val, went_val, lent_val = tmp_scores[0]

                    coe_vals = {"mag_mean": None, "mag_var": None, "ang_mean": None, "ang_var": None, "R": None, "C": None}
                    hid_mean_val = hid_last_val = None
                    attn_mean_val = attn_last_val = None

                    if need_hidden:
                        win_last_pos = win_len - 1
                        if flags["coe"] and win_last_pos >= 0 and getattr(out, "hidden_states", None) is not None:
                            with ev.timing("metric:coe"):
                                hs_tuple = out.hidden_states
                                hs_list = [h[0, win_last_pos, :].float().detach().cpu().numpy() for h in hs_tuple]
                                if len(hs_list) >= 2:
                                    coe = ev.CoEScoreInfo(hs_list)
                                    _, mag_mean, mag_var = coe.compute_CoE_Mag()
                                    _, ang_mean, ang_var = coe.compute_CoE_Ang()
                                    coe_R = coe.compute_CoE_R()
                                    coe_C = coe.compute_CoE_C()
                                    coe_vals = {"mag_mean": mag_mean, "mag_var": mag_var, "ang_mean": ang_mean, "ang_var": ang_var, "R": coe_R, "C": coe_C}

                        if flags["hidden"] and getattr(out, "hidden_states", None) is not None:
                            with ev.timing("metric:extra_hidden"):
                                hs_tensors = tuple(h[0, :win_len, :].detach().to(torch.float32).cpu() for h in out.hidden_states)
                                tmp_scores = []
                                ev.compute_scores(
                                    logits=[], hidden_acts=[hs_tensors], attns=[], scores=tmp_scores,
                                    indiv_scores=indiv_scores, mt_list=["hidden"],
                                    tok_ins=[], tok_lens=[(max(0, pref_in_win), win_len)], use_toklens=USE_TOKLENS_FOR_EXTRAS,
                                )
                                hvals, last_val = [], None
                                L = len(hs_tensors) - 1
                                for ly in range(1, L + 1):
                                    key = f"Hly{ly}"
                                    if key in indiv_scores["hidden"] and len(indiv_scores["hidden"][key]) > 0:
                                        v = indiv_scores["hidden"][key][-1]
                                        hvals.append(v)
                                        if ly == L:
                                            last_val = v
                                hid_mean_val = np.mean(hvals) if hvals else None
                                hid_last_val = last_val

                    if flags["attns"] and getattr(out, "attentions", None) is not None:
                        with ev.timing("metric:extra_attns"):
                            attns_layers = [None]
                            h0_all = out.attentions
                            for layer_tensor in h0_all:
                                h0 = layer_tensor[0, :, :win_len, :win_len].detach().to(torch.float32).cpu()
                                per_heads = [h0[h, :, :] for h in range(h0.shape[0])]
                                attns_layers.append(per_heads)
                            tmp_scores = []
                            ev.compute_scores(
                                logits=[], hidden_acts=[], attns=[attns_layers], scores=tmp_scores,
                                indiv_scores=indiv_scores, mt_list=["attns"],
                                tok_ins=[], tok_lens=[(max(0, pref_in_win), win_len)], use_toklens=USE_TOKLENS_FOR_EXTRAS,
                            )
                            avals, last_val = [], None
                            L = len(attns_layers) - 1
                            for ly in range(1, L + 1):
                                key = f"Attn{ly}"
                                if key in indiv_scores["attns"] and len(indiv_scores["attns"][key]) > 0:
                                    v = indiv_scores["attns"][key][-1]
                                    avals.append(v)
                                    if ly == L:
                                        last_val = v
                            attn_mean_val = np.mean(avals) if avals else None
                            attn_last_val = last_val

                    # save per-row columns
                    if flags["maxprob"]:  col_maxprob.append(maxprob)
                    if flags["ppl"]:      col_ppl.append(ppl)
                    if flags["entropy"]:  col_entropy.append(entr)
                    if flags["coe"]:
                        col_coe_mag_mean.append(coe_vals["mag_mean"]); col_coe_mag_var.append(coe_vals["mag_var"])
                        col_coe_ang_mean.append(coe_vals["ang_mean"]); col_coe_ang_var.append(coe_vals["ang_var"])
                        col_coe_R.append(coe_vals["R"]);               col_coe_C.append(coe_vals["C"])
                    if flags["logit"]:
                        col_extra_ppl.append(perpx_val); col_extra_win_ent.append(went_val); col_extra_log_ent.append(lent_val)
                    if flags["hidden"]:
                        col_hidden_mean.append(hid_mean_val); col_hidden_last.append(hid_last_val)
                    if flags["attns"]:
                        col_attn_mean.append(attn_mean_val); col_attn_last.append(attn_last_val)

                    # logging
                    ppv = ev.preview_text(pred_prev[0]); gpv = ev.preview_text(gold_prev[0])
                    parts = []
                    if flags["maxprob"]: parts += [f"maxprob={fmt_val(maxprob)}"]
                    if flags["ppl"]:     parts += [f"ppl={fmt_val(ppl)}"]
                    if flags["entropy"]: parts += [f"entr={fmt_val(entr)}"]
                    if flags["coe"]:
                        parts += [f"coeR={fmt_val(coe_vals['R'])}", f"coeC={fmt_val(coe_vals['C'])}",
                                 f"coeMagMu={fmt_val(coe_vals['mag_mean'])}", f"coeAngMu={fmt_val(coe_vals['ang_mean'])}"]
                    if flags["logit"]:
                        parts += [f"xPPL={fmt_val(perpx_val)}", f"winEnt={fmt_val(went_val)}", f"logEntK50={fmt_val(lent_val)}"]
                    if flags["hidden"]:
                        parts += [f"hidSVD_mean={fmt_val(hid_mean_val)}", f"hidSVD_last={fmt_val(hid_last_val)}"]
                    if flags["attns"]:
                        parts += [f"attnEig_mean={fmt_val(attn_mean_val)}", f"attnEig_last={fmt_val(attn_last_val)}"]
                    parts += [f"correct={ok}", f"pred≈{ppv}", f"gt≈{gpv}"]
                    if pred_prev[0]:
                        print(f"  - idx={k:>6}  " + "  ".join(parts))

                progress.advance(row_task, len(batch_idx))

            # attach columns & write shard
            if flags["maxprob"]:   df["score_maxprob"]    = col_maxprob
            if flags["ppl"]:       df["score_ppl"]        = col_ppl
            if flags["entropy"]:   df["score_entropy"]    = col_entropy
            if flags["coe"]:
                df["score_coe_mag_mu"] = col_coe_mag_mean
                df["score_coe_mag_var"]= col_coe_mag_var
                df["score_coe_ang_mu"] = col_coe_ang_mean
                df["score_coe_ang_var"]= col_coe_ang_var
                df["score_coe_R"]      = col_coe_R
                df["score_coe_C"]      = col_coe_C
            if flags["logit"]:
                df["score_extra_perplexity"]     = col_extra_ppl
                df["score_extra_window_entropy"] = col_extra_win_ent
                df["score_extra_logit_entropy"]  = col_extra_log_ent
            if flags["hidden"]:
                df["score_hidden_svd_mean"]     = col_hidden_mean
                df["score_hidden_svd_last"]     = col_hidden_last
            if flags["attns"]:
                df["score_attn_eigprod_mean"]   = col_attn_mean
                df["score_attn_eigprod_last"]   = col_attn_last

            # GT previews (parsed-only summaries)
            is_correct_all, pred_prev_all, gold_prev_all = [], [], []
            for i in range(0, len(completions), args.batch_size):
                batch_idx = list(range(i, min(i + args.batch_size, len(completions))))
                batch_comps = [completions[k] for k in batch_idx]
                batch_golds = [gold_any[k]    for k in batch_idx]
                if task == "trivia":
                    gt2, pprev2, gprev2, _ = ev.evaluate_trivia_batch(batch_comps, batch_golds)
                elif task == "gpqa":
                    gt2, pprev2, gprev2, _ = ev.evaluate_gpqa_batch(batch_comps, batch_golds)
                else:
                    gt2, pprev2, gprev2, _ = ev.evaluate_math_batch(batch_comps, batch_golds)
                is_correct_all.extend(gt2); pred_prev_all.extend(pprev2); gold_prev_all.extend(gprev2)
            df["is_correct"]       = is_correct_all
            df["pred_ans_preview"] = pred_prev_all
            df["gold_ans_preview"] = gold_prev_all
            df["pred_parsed"]      = parsed_flags_shard if len(parsed_flags_shard) == len(df) else [False] * len(df)

            parsable_df = df.loc[df["pred_parsed"].astype(bool)]
            valid = [x for x in parsable_df["is_correct"] if x is not None]
            if valid:
                acc = sum(valid) / len(valid)
                print(f"Summary [{shard_path.name}] GT accuracy (parsed only): {acc:.4f} ({len(valid)}/{len(parsable_df)})")

            out_name = shard_path.stem + ".scored" + shard_path.suffix
            out_path = out_dir / out_name
            if shard_path.suffix == ".parquet":
                df.to_parquet(out_path, index=False)
            else:
                df.to_json(out_path, lines=True, orient="records", force_ascii=False)

            if args.save_indiv_scores:
                indiv_path = out_dir / (shard_path.stem + ".indiv_scores.json")
                clean_indiv = {
                    "logit": indiv_scores["logit"],
                    "hidden": {k: v for k, v in indiv_scores["hidden"].items()},
                    "attns": {k: v for k, v in indiv_scores["attns"].items()},
                }
                with open(indiv_path, "w") as f:
                    json.dump(clean_indiv, f)

            progress.remove_task(row_task)
            print(f"✅ {shard_path.name} -> {out_path} (added score_* columns)")

            # add to globals
            for name, col in [
                ("maxprob", col_maxprob), ("ppl", col_ppl), ("entropy", col_entropy),
                ("coe_mag_mean", col_coe_mag_mean), ("coe_mag_var",  col_coe_mag_var),
                ("coe_ang_mean", col_coe_ang_mean), ("coe_ang_var",  col_coe_ang_var),
                ("coe_R", col_coe_R), ("coe_C", col_coe_C),
                ("extra_perplexity", col_extra_ppl), ("extra_window_entropy", col_extra_win_ent),
                ("extra_logit_entropy", col_extra_log_ent),
                ("hidden_svd_mean", col_hidden_mean), ("hidden_svd_last", col_hidden_last),
                ("attn_eigprod_mean", col_attn_mean), ("attn_eigprod_last", col_attn_last),
            ]:
                scores_global[name].extend(col if col else [None] * len(df))

        # end shards loop
        progress.advance(shard_task, 0)

    # ---------- FINAL EVALS (AUTO orientation) ----------
    known_idx = [i for i, t in enumerate(y_true_all) if t is not None]
    print("\n===== Threshold-free metrics per scoring function (verifiable GT rows) =====")
    print(f"Total rows: {len(y_true_all)} | Verifiable GT rows: {len(known_idx)}")

    orient_map: Dict[str, bool] = {}
    for name, raw in scores_global.items():
        raw_vals = [raw[i] for i in known_idx if _is_finite(raw[i])]
        y_vals   = [y_true_all[i] for i in known_idx if _is_finite(raw[i])]
        higher_is_better = True
        decided = False
        if len(raw_vals) >= 3 and not all(v == raw_vals[0] for v in raw_vals):
            auto = _spearman_auto_orientation(np.array(y_vals, dtype=int), np.array(raw_vals, dtype=float))
            if auto is not None:
                higher_is_better = bool(auto); decided = True
        orient_map[name] = higher_is_better
        print(f"  auto-orientation [{name:<22}] -> higher_is_better={higher_is_better} {'' if decided else '(fallback)'}")

    model_id_for_paths = infer_model_name(args.model)
    for name, raw in scores_global.items():
        if any(_is_finite(v) for v in raw):
            note = f"(orientation: higher_is_better={orient_map.get(name, True)}; auto)"
            _print_and_save_threshold_free_block(
                name=name,
                y_true_all=y_true_all,
                raw_scores=raw,
                known_idx=known_idx,
                higher_is_better=orient_map.get(name, True),
                out_dir=out_dir,
                model_id=model_id_for_paths,
                task_label=args.task,
                note=note,
            )

    # ---------- THRESHOLDED (normalized, both orientations) ----------
    print("\n===== Normalized thresholded final reports (per metric, both orientations) =====")
    for name, raw in scores_global.items():
        if any(_is_finite(v) for v in raw):
            ev.threshold_reports_normalized_both(
                name=name,
                raw_scores=raw,
                known_idx=known_idx,
                y_true_all=y_true_all,
                pred_parsed_all=pred_parsed_all,
                thresholds=thresholds if thresholds else [0.5],
            )

    # Short run summary
    summary_path = out_dir / f"run_summary_{args.task}_{model_id_for_paths}.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_id_for_paths}\n")
        f.write(f"Checkpoint: {args.model}\n")
        f.write(f"Task: {args.task}\n")
        f.write(f"Evaluated at: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Shards dir: {args.input_dir}\n")
        f.write(f"Per-model output dir: {out_dir}\n")
        f.write(f"Metrics computed: {', '.join([k for k,v in flags.items() if v])}\n")
        f.write(f"Thresholds: {thresholds}\n")
        f.write(f"Stride: {args.stride}, SampleN: {args.sample_n}, LimitShards: {args.limit_shards}\n")
    print(f"📝 Saved run summary to: {summary_path}")
    ev.print_timings()

if __name__ == "__main__":
    main()
