# evaluator.py
import math, re, unicodedata, time, json
from contextlib import contextmanager
from typing import List, Optional, Tuple, Any, Dict
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional math verify stack
try:
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig, parse, verify
    HAVE_MATH_VERIFY = True
except Exception:
    HAVE_MATH_VERIFY = False


# ========================= Timing =========================
_metric_times: Dict[str, float] = defaultdict(float)

@contextmanager
def timing(name: str):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _metric_times[name] += (time.perf_counter() - t0)

def print_timings():
    if not _metric_times:
        return
    print("\n===== Timing (cumulative wall time) =====")
    for k, v in sorted(_metric_times.items(), key=lambda x: -x[1]):
        print(f"{k:>32}: {v:8.3f} s")


# ========================= Helpers =========================
def preview_text(s: Optional[str], n: int = 80) -> str:
    if s is None:
        return "None"
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else (s[: n - 1] + "…")


def build_token_probs(input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """
    Return per-token probabilities aligned with input_ids (B, S).
    First token prob=1.0 by convention; others are p(target_t | prefix<=t-1).
    """
    B, S = input_ids.shape
    if logits.size(1) == S:
        logp_step = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        tgt = input_ids[:, 1:]
        log_tok_p = logp_step.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tok_p = input_ids.new_ones((B, S), dtype=logits.dtype)
        tok_p[:, 1:] = torch.clamp(log_tok_p.exp().to(logits.dtype), 1e-8, 1.0)
        return tok_p
    else:
        S_logits = logits.size(1)
        T = min(S - 1, S_logits)
        logp_step = torch.log_softmax(logits[:, :T, :].float(), dim=-1)
        tgt = input_ids[:, 1:1 + T]
        log_tok_p = logp_step.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        tok_p = input_ids.new_ones((B, S), dtype=logits.dtype)
        tok_p[:, 1:1 + T] = torch.clamp(log_tok_p.exp().to(logits.dtype), 1e-8, 1.0)
        return tok_p


# ========================= Output / CoE classes =========================
class OutputScoreInfo:
    """Compute token-wise maxprob / ppl / entropy from a list of per-step logits vectors."""
    def __init__(self, output_scores: List[torch.Tensor]):
        self.output_scores = output_scores
        self.all_token_re: List[List[float]] = []
        self.all_token_max_re: List[float] = []
        for token in range(len(self.output_scores)):
            logits = self.output_scores[token]
            vec = logits[0] if (logits.ndim == 2 and logits.shape[0] == 1) else logits
            re = F.softmax(vec.float(), dim=-1).detach().cpu().numpy().tolist()
            self.all_token_re.append(re)
            self.all_token_max_re.append(max(re) if re else float("nan"))

    def compute_maxprob(self) -> Optional[float]:
        return None if not self.all_token_max_re else float(np.mean(self.all_token_max_re))

    def compute_ppl(self) -> Optional[float]:
        if not self.all_token_max_re:
            return None
        seq_ppl_list = [math.log(max(1e-12, float(mr))) for mr in self.all_token_max_re]
        return float(-np.mean(seq_ppl_list))

    def compute_entropy(self) -> Optional[float]:
        if not self.all_token_re:
            return None
        # scipy.stats.entropy with base=2 -> manual to avoid heavy deps here
        def _H(p):
            p = np.array(p, dtype=np.float64)
            p = p[p > 0]
            return -float(np.sum(p * (np.log(p) / np.log(2))))
        return float(np.mean([_H(re) for re in self.all_token_re]))


class CoEScoreInfo:
    """CoE magnitudes/angles across layers for the LAST token in a window."""
    def __init__(self, hidden_states: List[np.ndarray]):
        self.hidden_states = hidden_states

    def _valid(self) -> bool:
        return isinstance(self.hidden_states, list) and len(self.hidden_states) >= 2

    def compute_CoE_Mag(self):
        if not self._valid():
            return None, None, None
        hs_all_layer = np.stack(self.hidden_states, axis=0)  # (L, H)
        denom = np.linalg.norm(hs_all_layer[-1] - hs_all_layer[0], ord=2)
        denom = denom if denom != 0 else 1e-12
        diffs = np.diff(hs_all_layer, axis=0)  # (L-1, H)
        norms = np.linalg.norm(diffs, axis=1, ord=2) / denom
        return norms.tolist(), float(np.mean(norms)), float(np.var(norms))

    def compute_CoE_Ang(self):
        if not self._valid():
            return None, None, None
        hs_all_layer = np.stack(self.hidden_states, axis=0)

        def _cos(a, b):
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            if na == 0 or nb == 0: return 1.0
            v = float(np.dot(a, b) / (na * nb))
            return max(min(v, 1.0), -1.0)

        denom = math.acos(_cos(hs_all_layer[-1], hs_all_layer[0]))
        denom = denom if denom != 0 else 1e-12
        al = []
        for i in range(hs_all_layer.shape[0] - 1):
            c = _cos(hs_all_layer[i+1], hs_all_layer[i])
            al.append(math.acos(c) / denom)
        arr = np.array(al, dtype=np.float64)
        return arr.tolist(), float(np.mean(arr)), float(np.var(arr))

    def compute_CoE_R(self):
        _, mag_mean, _ = self.compute_CoE_Mag()
        _, ang_mean, _ = self.compute_CoE_Ang()
        if mag_mean is None or ang_mean is None:
            return None
        return float(mag_mean - ang_mean)

    def compute_CoE_C(self):
        mag_list, _, _ = self.compute_CoE_Mag()
        ang_list, _, _ = self.compute_CoE_Ang()
        if mag_list is None or ang_list is None or len(mag_list) != len(ang_list) or len(mag_list) == 0:
            return None
        mag = np.array(mag_list, dtype=np.float64)
        ang = np.array(ang_list, dtype=np.float64)
        x_list = mag * np.cos(ang)
        y_list = mag * np.sin(ang)
        x_ave = float(np.mean(x_list))
        y_ave = float(np.mean(y_list))
        return float(math.sqrt(x_ave ** 2 + y_ave ** 2))


# ========================= Extra metrics =========================
def centered_svd_val(Z: torch.Tensor, alpha: float = 0.001) -> torch.Tensor:
    I = torch.eye(Z.shape[0], dtype=Z.dtype, device=Z.device)
    O = torch.ones(Z.shape[0], Z.shape[0], dtype=Z.dtype, device=Z.device)
    J = I - (1 / Z.shape[0]) * O
    Sigma = torch.matmul(torch.matmul(Z.t(), J), Z)
    Sigma = Sigma + alpha * torch.eye(Sigma.shape[0], dtype=Z.dtype, device=Z.device)
    svdvals = torch.linalg.svdvals(Sigma)
    eigscore = torch.log(svdvals).mean()
    return eigscore


def get_svd_eval(hidden_acts, layer_num=15, tok_lens=[], use_toklens=True):
    svd_scores = []
    for i in range(len(hidden_acts)):
        Z = hidden_acts[i][layer_num]  # (S,H)
        if use_toklens and tok_lens[i]:
            i1, i2 = tok_lens[i][0], tok_lens[i][1]
            Z = Z[i1:i2, :]
        Z = torch.transpose(Z, 0, 1)  # (H,S)
        svd_scores.append(centered_svd_val(Z).item())
    return np.stack(svd_scores)


def get_attn_eig_prod(attns, layer_num=15, tok_lens=[], use_toklens=True):
    attn_scores = []
    for i in range(len(attns)):  # samples
        eigscore = 0.0
        for attn_head_num in range(len(attns[i][layer_num])):  # heads
            Sigma = attns[i][layer_num][attn_head_num]  # (S,S)
            if use_toklens and tok_lens[i]:
                i1, i2 = tok_lens[i][0], tok_lens[i][1]
                Sigma = Sigma[i1:i2, i1:i2]
            eigscore += torch.log(torch.diagonal(Sigma, 0)).mean()
        attn_scores.append(eigscore.item())
    return np.stack(attn_scores)


def perplexity(logits, tok_ins, tok_lens, min_k=None):
    softmax = torch.nn.Softmax(dim=-1)
    ppls = []
    for i in range(len(logits)):
        i1, i2 = tok_lens[i][0], tok_lens[i][1]
        pr = torch.log(softmax(logits[i]))[torch.arange(i1, i2) - 1, tok_ins[i][0, i1:i2]]
        if min_k is not None:
            pr = torch.topk(pr, k=int(min_k * len(pr)), largest=False).values
        ppls.append(float(torch.exp(-pr.mean()).item()))
    return np.stack(ppls)


def logit_entropy(logits, tok_lens, top_k=None):
    softmax = torch.nn.Softmax(dim=-1)
    scores = []
    for i in range(len(logits)):
        i1, i2 = tok_lens[i][0], tok_lens[i][1]
        if top_k is None:
            l = softmax(torch.tensor(logits[i]))[i1:i2]
            scores.append(float(((-l) * torch.log(l)).mean().item()))
        else:
            l = logits[i][i1:i2]
            l = softmax(torch.topk(l, top_k, 1).values)
            scores.append(float(((-l) * torch.log(l)).mean().item()))
    return np.stack(scores)


def window_logit_entropy(logits, tok_lens, top_k=None, w=1):
    softmax = torch.nn.Softmax(dim=-1)
    scores = []
    for i in range(len(logits)):
        i1, i2 = tok_lens[i][0], tok_lens[i][1]
        if top_k is None:
            l = softmax(logits[i])[i1:i2]
        else:
            l = torch.tensor(logits[i])[i1:i2]
            l = softmax(torch.topk(l, top_k, 1).values)
        windows = torch.max(((-l) * torch.log(l)).mean(1).unfold(0, w, w).mean(1))
        scores.append(float(windows.item()))
    return np.stack(scores)


def compute_scores(logits, hidden_acts, attns, scores, indiv_scores, mt_list, tok_ins, tok_lens, use_toklens=True):
    sample_scores = []
    for mt in mt_list:
        mt_score = []
        if mt == "logit":
            mt_score.append(perplexity(logits, tok_ins, tok_lens)[0])
            indiv_scores[mt]["perplexity"].append(mt_score[-1])

            mt_score.append(window_logit_entropy(logits, tok_lens, w=1)[0])
            indiv_scores[mt]["window_entropy"].append(mt_score[-1])

            mt_score.append(logit_entropy(logits, tok_lens, top_k=50)[0])
            indiv_scores[mt]["logit_entropy"].append(mt_score[-1])

        elif mt == "hidden":
            for layer_num in range(1, len(hidden_acts[0])):
                mt_score.append(get_svd_eval(hidden_acts, layer_num, tok_lens, use_toklens)[0])
                key = "Hly" + str(layer_num)
                if key not in indiv_scores[mt]:
                    indiv_scores[mt][key] = []
                indiv_scores[mt][key].append(mt_score[-1])

        elif mt == "attns":
            for layer_num in range(1, len(attns[0])):
                mt_score.append(get_attn_eig_prod(attns, layer_num, tok_lens, use_toklens)[0])
                key = "Attn" + str(layer_num)
                if key not in indiv_scores[mt]:
                    indiv_scores[mt][key] = []
                indiv_scores[mt][key].append(mt_score[-1])

        else:
            raise ValueError("Invalid method type")
        sample_scores.extend(mt_score)
    scores.append(sample_scores)


# ========================= Math verify =========================
def _parse_gold(sol: str):
    try:
        return parse(sol, extraction_mode="first_match")
    except Exception:
        return []

def _parse_pred(pred: str):
    try:
        return parse(
            pred,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed="all",
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )
    except Exception:
        return []


def evaluate_math_batch(
    completions: List[str], solutions: List[str]
) -> Tuple[List[Optional[float]], List[Optional[str]], List[Optional[str]], List[bool]]:
    if not HAVE_MATH_VERIFY:
        n = len(completions)
        return [None]*n, [None]*n, [None]*n, [False]*n

    correctness, pred_prev, gold_prev, parsed_flag = [], [], [], []
    for pred, sol in zip(completions, solutions):
        gold_parsed = _parse_gold(sol)
        if len(gold_parsed) == 0:
            correctness.append(None); pred_prev.append(None); gold_prev.append(None); parsed_flag.append(False)
            continue

        ans_parsed = _parse_pred(pred)
        gold_prev.append(str(gold_parsed[0]) if len(gold_parsed) else None)
        pred_prev.append(str(ans_parsed[0]) if len(ans_parsed) else None)
        parsed_flag.append(len(ans_parsed) > 0)

        if len(ans_parsed) == 0:
            correctness.append(0.0)
            continue

        try:
            ok = bool(verify(gold_parsed, ans_parsed))
            correctness.append(1.0 if ok else 0.0)
        except Exception as e:
            print(f"[verify error] {e}")
            correctness.append(None)

    return correctness, pred_prev, gold_prev, parsed_flag


# ========================= Trivia utils =========================
def _norm(s):
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _unwrap_latex_wrappers(s: str) -> str:
    if not s: return s
    pat = re.compile(r"\\(text|mathrm|operatorname)\s*\{(.+)\}\s*$", flags=re.DOTALL)
    for _ in range(3):
        m = pat.fullmatch(s.strip())
        if not m: break
        s = m.group(2)
    return s.strip()

def _collapse_spelled_letters(s: str) -> str:
    t = s.strip()
    if re.fullmatch(r'(?:[A-Za-z]\s+){2,}[A-Za-z]', t):
        return re.sub(r'\s+', '', t)
    return s

def _clean_pred_text(p: str) -> str:
    if not p: return p
    p = _unwrap_latex_wrappers(p)
    p = p.strip().strip('\'"`“”’.,;:!-()[]')
    p = _collapse_spelled_letters(p)
    return p.strip()

def _extract_all_boxed(content: str) -> List[str]:
    results, i = [], 0
    while True:
        m = re.search(r'\\boxed\s*\{', content[i:])
        if not m: break
        start_brace = i + m.end()
        depth, j = 1, start_brace
        while j < len(content) and depth > 0:
            ch = content[j]
            if ch == '{': depth += 1
            elif ch == '}': depth -= 1
            j += 1
        if depth == 0:
            inner = content[start_brace:j-1]
            results.append(inner.strip())
            i = j
        else:
            break
    return results

def _singularize_token(tok: str) -> str:
    if len(tok) <= 3:
        return tok
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("oes") and len(tok) > 4:
        return tok[:-2]
    if tok.endswith("es") and len(tok) > 3:
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 3:
        return tok[:-1]
    return tok

def _lemmatize_phrase(norm: str) -> str:
    toks = [t for t in norm.split() if t]
    return " ".join(_singularize_token(t) for t in toks)

def _nospace(s: str) -> str:
    return s.replace(" ", "")

def _tokens(s: str) -> set:
    return set(t for t in s.split() if t)

def _soft_match_pred_to_cand(pred_clean: str, cand_norm: str) -> bool:
    pn        = _norm(pred_clean)
    pn_lem    = _lemmatize_phrase(pn)
    pn_ns     = _nospace(pn)
    pn_lem_ns = _nospace(pn_lem)

    cn        = cand_norm
    cn_lem    = _lemmatize_phrase(cn)
    cn_ns     = _nospace(cn)
    cn_lem_ns = _nospace(cn_lem)

    if pn == cn or pn_lem == cn or pn == cn_lem or pn_lem == cn_lem:
        return True
    if pn_ns == cn_ns or pn_lem_ns == cn_ns or pn_ns == cn_lem_ns or pn_lem_ns == cn_lem_ns:
        return True

    pt = _tokens(pn)
    ct = _tokens(cn)
    if pt and ct:
        small, large = (pt, ct) if len(pt) <= len(ct) else (ct, pt)
        if small.issubset(large) and (len(small) >= 2 or (len(large) - len(small) <= 1)):
            return True
    return False


def evaluate_trivia_batch(
    completions: List[Any],
    gold_answers: List[Any],
):
    correctness, pred_prev, gold_prev, parsed_flag = [], [], [], []
    for comp, sol in zip(completions, gold_answers):
        if not isinstance(sol, dict):
            correctness.append(None); pred_prev.append(None); gold_prev.append(None); parsed_flag.append(False); continue

        cands, rep = set(), None
        for k in ("value","normalized_value","matched_wiki_entity_name","normalized_matched_wiki_entity_name"):
            v = sol.get(k)
            if isinstance(v, str) and v.strip():
                if rep is None: rep = v
                cands.add(_norm(v))
        for k in ("aliases","normalized_aliases"):
            arr = sol.get(k, [])
            if isinstance(arr, list):
                for v in arr:
                    if isinstance(v, str) and v.strip():
                        if rep is None: rep = v
                        cands.add(_norm(v))

        if not cands:
            correctness.append(None); pred_prev.append(None); gold_prev.append(None); parsed_flag.append(False); continue

        boxed = _extract_all_boxed(str(comp))
        preds_clean = [_clean_pred_text(b) for b in boxed if b.strip()]
        has_pred = len(preds_clean) > 0
        pred_prev.append(preds_clean[0] if has_pred else None)
        gold_prev.append(rep)

        seen_raw, preds_dedup = set(), []
        for p in preds_clean:
            if p not in seen_raw:
                seen_raw.add(p)
                preds_dedup.append(p)

        hit = False
        for p in preds_dedup:
            for cn in cands:
                if _soft_match_pred_to_cand(p, cn):
                    hit = True
                    break
            if hit: break

        correctness.append(1.0 if (has_pred and hit) else (0.0 if has_pred else None))
        parsed_flag.append(has_pred)

    return correctness, pred_prev, gold_prev, parsed_flag



# ---------- GPQA evaluation (boxed prediction, gold in `answer`) ----------
import re
import unicodedata
from typing import List, Optional, Tuple, Any

# -------------------- Utilities --------------------

_TEXTLIKE_FUNCS = ("text", "mathrm", "operatorname", "mathbf", "mathit", "mathsf", "mathtt")

def _norm(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = unicodedata.normalize("NFKD", str(s)).casefold()
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _soft_match_pred_to_cand(pred_raw: str, gold_norm: str) -> bool:
    # very light heuristic: check containment on normalized text
    pn = _norm(pred_raw)
    return pn == gold_norm or pn.replace(" ", "") == gold_norm.replace(" ", "") or \
           (pn and gold_norm and (pn in gold_norm or gold_norm in pn))

def _extract_all_boxed(s: str) -> List[str]:
    """Extract all \boxed{...} segments with brace matching (no nested boxes inside)."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        j = s.find(r"\boxed{", i)
        if j < 0:
            break
        k = j + len(r"\boxed{") - 1  # index at '{'
        # find matching brace
        depth = 0
        start = None
        for t in range(j, n):
            if s[t] == "{":
                depth += 1
                if start is None:
                    start = t + 1
            elif s[t] == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(s[start:t])
                    i = t + 1
                    break
        else:
            # unmatched brace; stop
            break
    return out

def _clean_pred_text(s: str) -> str:
    """Trim and remove trivial surrounding $...$ or spaces."""
    if not isinstance(s, str):
        return ""
    t = s.strip()
    # strip inline math $...$
    if len(t) >= 2 and t[0] == "$" and t[-1] == "$":
        t = t[1:-1].strip()
    return t

# ---------- Leading-wrapper peeler (ONLY at the start) ----------

def _strip_leading_textlike_wrappers(s: str) -> str:
    """
    Repeatedly peel leading \\text{...}, \\mathrm{...}, etc. from the START only,
    concatenating the inside with the remaining suffix.
    """
    t = s
    for _ in range(5):  # nested a few times at most
        m = re.match(r"^\s*\\(" + "|".join(_TEXTLIKE_FUNCS) + r")\s*\{", t)
        if not m:
            break
        # find the matching closing brace for the opening at m.end()-1
        start_brace = m.end() - 1  # index of '{'
        depth = 0
        end_idx = None
        for i in range(start_brace, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx is None:
            break  # malformed; stop peeling
        inside = t[start_brace + 1:end_idx].strip()
        suffix = t[end_idx + 1:]  # keep whatever comes after the wrapper
        t = (inside + " " + suffix).strip()
    return t

# ---------- Letter extraction ----------

_LETTER_ONLY = re.compile(r"^[A-Fa-f]$")

def _as_letter_strict(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    m = _LETTER_ONLY.fullmatch(s)
    return m.group(0).upper() if m else None

# Accepts prefixes like: "A.", "A)", "A :", "A -", "A\ ", "(B)", "[C].", "{D}", "\left( E )", "\text{E. } ..."
_LEAD_LETTER_RE = re.compile(
    r"""^\s*
        (?:\\left\s*)?           # optional \left
        [\(\[\{]?\s*             # optional opening bracket
        ([A-Fa-f])               # the choice letter
        \s*(?:                   # allowable separators after the letter
            [\.\)\]:;-] |        # typical punctuation
            \\\s |               # LaTeX backslash-space (\ )
            \\[ ,;!]? |          # LaTeX spacing like \, \; \!
            $                    # or end
        )
    """,
    re.VERBOSE,
)

def _lead_letter_fuzzy(s: Optional[str]) -> Optional[str]:
    if not isinstance(s, str):
        return None
    t = _strip_leading_textlike_wrappers(s)
    m = _LEAD_LETTER_RE.match(t)
    return m.group(1).upper() if m else None

def _letter_from_any(s: Optional[str]) -> Optional[str]:
    # try strict "A" first (covers gt="A")
    L = _as_letter_strict(s)
    if L is not None:
        return L
    # then fuzzy leading pattern (covers pred like "A.\ 48.90", "\text{E. } $5,600 ...")
    return _lead_letter_fuzzy(s)

# # -------------------- Main evaluator --------------------

# def evaluate_gpqa_batch(
#     completions: List[Any],
#     answers: List[Any],
# ) -> Tuple[List[Optional[float]], List[Optional[str]], List[Optional[str]], List[bool]]:
#     """
#     Evaluate GPQA-style MC outputs where the model's final answer is the (last) \\boxed{...}
#     in `completion` and the ground-truth is in the `answer` column.

#     Returns:
#         correctness: List[Optional[float]]  # 1.0/0.0/None
#         pred_prev : List[Optional[str]]     # extracted boxed content (string)
#         gold_prev : List[Optional[str]]     # raw gold answer string
#         parsed_flag: List[bool]             # True if a boxed answer was extracted
#     """
#     correctness: List[Optional[float]] = []
#     pred_prev:  List[Optional[str]]    = []
#     gold_prev:  List[Optional[str]]    = []
#     parsed_flag: List[bool]            = []

#     for comp, gold in zip(completions, answers):
#         comp_str = "" if comp is None else str(comp)
#         gold_str = None if (gold is None or str(gold).strip() == "") else str(gold).strip()

#         boxed_all = _extract_all_boxed(comp_str)
#         pred_clean = None
#         if boxed_all:
#             last_box = boxed_all[-1]
#             pred_clean = _clean_pred_text(last_box) if isinstance(last_box, str) else None

#         parsed = bool(pred_clean)
#         parsed_flag.append(parsed)
#         pred_prev.append(pred_clean)
#         gold_prev.append(gold_str)

#         if (not parsed) or (gold_str is None):
#             correctness.append(None)
#             continue

#         # ---- Primary: compare letters (robust) ----
#         pL = _letter_from_any(pred_clean)
#         gL = _letter_from_any(gold_str)
#         if pL is not None and gL is not None:
#             correctness.append(1.0 if pL == gL else 0.0)
#             continue

#         # ---- Fallback: normalized text & soft match ----
#         pn = _norm(pred_clean)
#         gn = _norm(gold_str)
#         if pn == gn or pn.replace(" ", "") == gn.replace(" ", ""):
#             correctness.append(1.0)
#             continue

#         hit = _soft_match_pred_to_cand(pred_clean, gn)
#         correctness.append(1.0 if hit else 0.0)

#     return correctness, pred_prev, gold_prev, parsed_flag

_ANSWER_FIELD_RE = re.compile(
    r"""
    [({\[\s,"'*_`-]*              # optional leading punc/space/markdown
    (?:["'“‘])?                   # optional quote before key (unbalanced ok)
    (?:\*{1,2}|_{1,2}|`)?         # optional markdown opener (*, **, _, __, `)
    \banswer\b                    # the key
    (?:\*{1,2}|_{1,2}|`)?         # optional markdown closer
    (?:["'”’])?                   # optional quote after key (unbalanced ok)
    \s*[:=]\s*                    # separator
    (?:["'“‘])?                   # optional opening quote for value
    \s*([A-F])\s*                 # <-- capture single letter A..F
    (?:["'”’])?                   # optional closing quote for value
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _extract_choice_letter_from_answer_field(text: str) -> Optional[str]:
    """
    Finds the LAST occurrence of an 'answer' field, tolerant to markdown/quotes:
      **Answer**: C, "answer": "C", 'answer': 'B', answer: D, answer": "B", (answer": "B")
    Returns the letter (A..F) uppercased, or None if not found.
    """
    if not text:
        return None
    letter = None
    for m in _ANSWER_FIELD_RE.finditer(text):
        letter = (m.group(1) or "").upper()
    return letter

    
def evaluate_gpqa_batch(
    completions: List[Any],
    answers: List[Any],
) -> Tuple[List[Optional[float]], List[Optional[str]], List[Optional[str]], List[bool]]:
    correctness: List[Optional[float]] = []
    pred_prev:  List[Optional[str]]    = []
    gold_prev:  List[Optional[str]]    = []
    parsed_flag: List[bool]            = []

    for comp, gold in zip(completions, answers):
        comp_str = "" if comp is None else str(comp)
        gold_str = None if (gold is None or str(gold).strip() == "") else str(gold).strip()

        # ---- Primary: take the LAST \boxed{...} ----
        boxed_all = _extract_all_boxed(comp_str)
        pred_clean = None
        if boxed_all:
            last_box = boxed_all[-1]
            pred_clean = _clean_pred_text(last_box) if isinstance(last_box, str) else None

        # ---- Fallback 1: JSON-like `"answer": "C"` anywhere in text ----
        pL_json = _extract_choice_letter_from_answer_field(comp_str) if not pred_clean else None
        if pL_json is not None and (pred_clean is None or not _letter_from_any(pred_clean)):
            # If boxed not present or not a clean letter, treat extracted letter as the prediction
            pred_clean = pL_json

        parsed_flag.append(bool(pred_clean))
        pred_prev.append(pred_clean)
        gold_prev.append(gold_str)

        if (not pred_clean) or (gold_str is None):
            correctness.append(None)
            continue

        # ---- Prefer letter-vs-letter match ----
        pL = _letter_from_any(pred_clean)
        gL = _letter_from_any(gold_str)
        if pL is not None and gL is not None:
            correctness.append(1.0 if pL == gL else 0.0)
            continue

        # ---- Fallback: normalized text equivalence / containment ----
        pn = _norm(pred_clean)
        gn = _norm(gold_str)
        if pn == gn or pn.replace(" ", "") == gn.replace(" ", ""):
            correctness.append(1.0)
            continue

        hit = _soft_match_pred_to_cand(pred_clean, gn)
        correctness.append(1.0 if hit else 0.0)

    return correctness, pred_prev, gold_prev, parsed_flag


# ========================= Chat prefix =========================
def chat_prefix(tokenizer, question: str, task: str, sys_math: str, sys_trivia: str) -> str:
    sys_inst = sys_trivia if task == "trivia" else sys_math
    messages = [
        {"role": "system", "content": sys_inst},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# -*- coding: utf-8 -*-
"""
Threshold-free & thresholded metrics (imbalance-robust)
Adds:
  • AUROC
  • AUPR (correct-positive) and AUPR (error-positive)
  • NLL (negative log-likelihood)
  • ECE (adaptive / equal-mass bins)
  • Brier + Brier Skill Score (vs prevalence baseline)
  • F1 in thresholded report

Keeps your originals:
  • ECE (fixed bins), smECE, FPR@95%TPR
  • Distribution plots (raw & normalized)
"""

from typing import List, Optional, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
import time

# ---------------------------
# Minimal timing context mgr.
# ---------------------------
@contextmanager
def timing(name: str):
    t0 = time.time()
    try:
        yield
    finally:
        dt = time.time() - t0
        # Print or no-op depending on your preference:
        # print(f"[timing] {name}: {dt*1000:.1f} ms")

# ========================= Basic confusion & thresholded metrics =========================
def confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
    TP = FP = TN = FN = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1: TP += 1
        elif t == 0 and p == 1: FP += 1
        elif t == 0 and p == 0: TN += 1
        elif t == 1 and p == 0: FN += 1
    return TP, FP, TN, FN

def metrics(TP: int, FP: int, TN: int, FN: int) -> Tuple[float, float, float]:
    N = TP + FP + TN + FN
    if N == 0:
        return float("nan"), float("nan"), float("nan")
    acc = (TP + TN) / N
    prec = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
    rec = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    return acc, prec, rec

def f1_from_counts(TP: int, FP: int, TN: int, FN: int) -> float:
    denom = (2 * TP + FP + FN)
    if denom == 0:
        return float("nan")
    return float((2 * TP) / denom)

# ========================= ROC / PR curves & areas =========================
def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
    assert y_true.shape == y_score.shape
    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return None, None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    tpr = np.r_[0.0, tp / pos, 1.0]
    fpr = np.r_[0.0, fp / neg, 1.0]
    return fpr, tpr

def auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    fpr, tpr = _roc_curve(y_true, y_score)
    if fpr is None: return None
    return float(np.trapz(tpr, fpr))

def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> Optional[float]:
    fpr, tpr = _roc_curve(y_true, y_score)
    if fpr is None: return None
    idx = np.searchsorted(tpr, target_tpr, side="left")
    if idx >= len(fpr): return float(fpr[-1])
    if idx == 0: return float(fpr[0])
    t0, t1 = tpr[idx-1], tpr[idx]
    f0, f1 = fpr[idx-1], fpr[idx]
    if t1 == t0: return float(f1)
    w = (target_tpr - t0) / (t1 - t0)
    return float(f0 + w * (f1 - f0))

def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
    pos = int(np.sum(y_true == 1))
    if pos == 0: return None, None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / pos
    precision = np.r_[1.0, precision]
    recall    = np.r_[0.0, recall]
    return precision, recall

def aupr(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    precision, recall = _precision_recall_curve(y_true, y_score)
    if precision is None: return None
    return float(np.trapz(precision, recall))

def aupr_both_classes(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    """
    AUPR for both orientations:
      - AUPR_correct: treat y=1 (correct) as positive
      - AUPR_error  : treat y=0 (error)   as positive (flip labels & invert scores)
    """
    aupr_correct = aupr(y_true, y_score)
    aupr_error   = aupr(1 - y_true, -y_score)
    return aupr_correct, aupr_error

# ========================= Probabilistic scoring & calibration =========================
def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))

def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    """
    Brier Skill Score vs. prevalence baseline.
    Returns (BSS, brier_model, brier_baseline).
    """
    if y_prob.size == 0:
        return float("nan"), float("nan"), float("nan")
    y = y_true.astype(np.float64)
    p = np.clip(y_prob.astype(np.float64), 0.0, 1.0)
    b_model = float(np.mean((p - y) ** 2))
    p_base = float(np.mean(y))  # prevalence
    b_base = float(np.mean((p_base - y) ** 2))
    if b_base == 0.0:
        return float("nan"), b_model, b_base
    bss = 1.0 - (b_model / b_base)
    return float(bss), b_model, b_base

def nll_binary(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Negative Log-Likelihood (binary cross-entropy) with clipping."""
    if y_prob.size == 0:
        return float("nan")
    p = np.clip(y_prob.astype(np.float64), 1e-15, 1 - 1e-15)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

def ece_fixed(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> Optional[float]:
    if len(y_true) == 0: return None
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    inds = np.digitize(y_score, bins, right=False) - 1
    inds = np.clip(inds, 0, n_bins - 1)
    ece = 0.0
    N = len(y_true)
    for b in range(n_bins):
        mask = (inds == b)
        cnt = np.sum(mask)
        if cnt == 0: continue
        conf = float(np.mean(y_score[mask]))
        acc  = float(np.mean(y_true[mask]))
        ece += (cnt / N) * abs(acc - conf)
    return float(ece)

def ece_equal_mass(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> Optional[float]:
    N = len(y_true)
    if N == 0 or n_bins <= 1: return None
    order = np.argsort(y_score)
    ys, ps = y_true[order], y_score[order]
    ece, start = 0.0, 0
    for b in range(n_bins):
        end = int(round((b + 1) * N / n_bins))
        end = min(max(end, start + 1), N)
        cnt = end - start
        if cnt <= 0: continue
        conf = float(np.mean(ps[start:end]))
        acc  = float(np.mean(ys[start:end]))
        ece += (cnt / N) * abs(acc - conf)
        start = end
    return float(ece)

def sm_ece(y_true: np.ndarray, y_score: np.ndarray, bin_counts=(5,10,15,20)) -> Optional[float]:
    vals = []
    for k in bin_counts:
        v = ece_equal_mass(y_true, y_score, n_bins=k)
        if v is not None and not np.isnan(v):
            vals.append(v)
    if not vals: return None
    return float(np.mean(vals))

# ========================= Master evaluators =========================
def threshold_free_for_one(
    name: str,
    raw_scores: List[Optional[float]],
    prob_like: bool,
    known_idx: List[int],
    y_true_all: List[Optional[int]],
    out_dir: Path,
):
    # Align to known rows
    vals = [(i, raw_scores[i]) for i in range(len(raw_scores))
            if (i in known_idx and raw_scores[i] is not None)]
    if not vals:
        print(f"\n[{name}] no valid rows — skipping.")
        return

    idxs, s_raw = zip(*vals)
    s_raw = np.array(s_raw, dtype=np.float64)
    y_true_known = np.array([y_true_all[i] for i in known_idx], dtype=np.int32)
    y = y_true_known[[known_idx.index(i) for i in idxs]].astype(np.int32)

    finite = np.isfinite(s_raw)
    s_raw = s_raw[finite]
    y = y[finite]
    if s_raw.size == 0:
        print(f"\n[{name}] no finite values — skipping.")
        return

    # Raw distribution plot
    with timing(f"plot:{name}:raw"):
        plt.figure(figsize=(8, 5))
        bins = 30
        if prob_like:
            rmin, rmax = 0.0, 1.0
            plt.hist(s_raw[y == 1], bins=bins, range=(rmin, rmax), alpha=0.6, label="Correct")
            plt.hist(s_raw[y == 0], bins=bins, range=(rmin, rmax), alpha=0.6, label="Wrong")
        else:
            rmin, rmax = np.min(s_raw), np.max(s_raw)
            if np.isfinite(rmin) and np.isfinite(rmax) and rmin < rmax:
                plt.hist(s_raw[y == 1], bins=bins, range=(float(rmin), float(rmax)), alpha=0.6, label="Correct")
                plt.hist(s_raw[y == 0], bins=bins, range=(float(rmin), float(rmax)), alpha=0.6, label="Wrong")
            else:
                plt.hist(s_raw[y == 1], bins=bins, alpha=0.6, label="Correct")
                plt.hist(s_raw[y == 0], bins=bins, alpha=0.6, label="Wrong")
        plt.xlabel(name); plt.ylabel("Count")
        plt.title(f"Distribution of {name} by correctness (raw)")
        plt.legend(); plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        raw_path = out_dir / f"{name}_distribution_raw.png"
        plt.tight_layout(); plt.savefig(raw_path, dpi=150); plt.close()
    print(f"📊 Saved {name} raw distribution plot to: {raw_path}")

    # Both orientations
    for tag, sign in (("↑ (higher=correct)", +1.0), ("↓ (lower=correct)", -1.0)):
        s_oriented = s_raw * sign

        with timing(f"final_eval:{name}:{tag}:auroc"):
            _auroc = auroc(y, s_oriented)
        with timing(f"final_eval:{name}:{tag}:aupr_both"):
            _aupr_correct, _aupr_error = aupr_both_classes(y, s_oriented)
        with timing(f"final_eval:{name}:{tag}:fpr95"):
            _fpr95 = fpr_at_tpr(y, s_oriented, target_tpr=0.95)

        print(f"\n--- {name} | {tag} ---")
        print(f"AUROC                : {_auroc:.4f}" if _auroc is not None else "AUROC                : NA")
        print(f"AUPR (correct=pos)   : {_aupr_correct:.4f}" if _aupr_correct is not None else "AUPR (correct=pos)   : NA")
        print(f"AUPR (error=pos)     : {_aupr_error:.4f}"  if _aupr_error   is not None else "AUPR (error=pos)     : NA")
        print(f"FPR@95%TPR           : {_fpr95:.4f}" if _fpr95 is not None else "FPR@95%TPR           : NA")

        # Only for probability-like scores in the "higher=correct" orientation
        if prob_like and sign > 0:
            with timing(f"final_eval:{name}:{tag}:calibration"):
                s_clip = np.clip(s_oriented, 0.0, 1.0)

                # Core paper-5:
                _ece_adapt = ece_equal_mass(y, s_clip, n_bins=15)  # Adaptive ECE
                _nll       = nll_binary(y, s_clip)                 # NLL
                _bss, _brier_val, _brier_base = brier_skill_score(y, s_clip)  # BSS + pieces

                # Keep your existing outputs
                _ece_fixed = ece_fixed(y, s_clip, n_bins=15)
                _smece     = sm_ece(y, s_clip, bin_counts=(5,10,15,20))

            # Print the core "paper 5"
            print(f"ECE (adaptive)       : {_ece_adapt:.4f}" if _ece_adapt is not None else "ECE (adaptive)       : NA")
            print(f"NLL                  : {_nll:.6f}")
            print(f"Brier                : {_brier_val:.6f}")
            print(f"Brier (baseline)     : {_brier_base:.6f}")
            print(f"Brier Skill Score    : {_bss:.4f}" if np.isfinite(_bss) else "Brier Skill Score    : NA")

            # Original calibration reports for reference
            print(f"ECE (fixed)          : {_ece_fixed:.4f}" if _ece_fixed is not None else "ECE (fixed)          : NA")
            print(f"smECE                : {_smece:.4f}" if _smece is not None else "smECE                : NA")

        # Normalized distribution plot
        with timing(f"plot:{name}:norm:{'up' if sign>0 else 'down'}"):
            smin, smax = float(np.min(s_oriented)), float(np.max(s_oriented))
            if smax - smin <= 1e-12:
                s_norm = np.full_like(s_oriented, 0.5, dtype=np.float64)
                norm_info = "degenerate (all equal) -> set to 0.5"
            else:
                s_norm = (s_oriented - smin) / (smax - smin)
                norm_info = f"min={smin:.6f}, max={smax:.6f}"
            plt.figure(figsize=(8, 5))
            bins = 30
            plt.hist(s_norm[y == 1], bins=bins, range=(0.0, 1.0), alpha=0.6, label="Correct")
            plt.hist(s_norm[y == 0], bins=bins, range=(0.0, 1.0), alpha=0.6, label="Wrong")
            plt.xlabel(f"{name} (normalized)")
            plt.ylabel("Count")
            plt.title(f"Distribution of {name} by correctness (normalized, {tag})\n[{norm_info}]")
            plt.legend(); plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
            norm_path = out_dir / f"{name}_distribution_norm_{'up' if sign>0 else 'down'}.png"
            plt.tight_layout(); plt.savefig(norm_path, dpi=150); plt.close()
        print(f"📊 Saved {name} normalized {tag} distribution plot to: {norm_path}")

def threshold_reports_normalized_both(
    name: str,
    raw_scores: List[Optional[float]],
    known_idx: List[int],
    y_true_all: List[Optional[int]],
    pred_parsed_all: List[bool],
    thresholds: List[float],
):
    y_true_known_local = np.array([y_true_all[i] for i in known_idx], dtype=np.int32)
    parsed_known = [pred_parsed_all[i] for i in known_idx]

    valid_pairs = []
    for j, i in enumerate(known_idx):
        val = raw_scores[i] if i < len(raw_scores) else None
        if val is not None and np.isfinite(val):
            valid_pairs.append((j, float(val)))
    if not valid_pairs:
        print(f"\n[{name}] no valid rows for thresholding — skipping.")
        return

    idx_valid, s_valid_raw = zip(*valid_pairs)
    idx_valid = np.array(idx_valid, dtype=np.int32)
    s_valid_raw = np.array(s_valid_raw, dtype=np.float64)

    for tag, sign in (("↑ (higher=correct)", +1.0), ("↓ (lower=correct)", -1.0)):
        s_use = s_valid_raw * sign
        with timing(f"thresh:{name}:{tag}:normalize"):
            smin, smax = float(np.min(s_use)), float(np.max(s_use))
            if smax - smin <= 1e-12:
                s_valid_norm = np.full_like(s_use, 0.5, dtype=np.float64)
                norm_info = "degenerate (all equal) -> set to 0.5"
            else:
                s_valid_norm = (s_use - smin) / (smax - smin)
                norm_info = f"min={smin:.6f}, max={smax:.6f}"
            s_norm_all: List[Optional[float]] = [None] * len(known_idx)
            for jj, v in zip(idx_valid, s_valid_norm):
                s_norm_all[jj] = float(v)

        print(f"\n=== Thresholded report for {name} | {tag} [normalized to [0,1], {norm_info}] ===")
        for TH in thresholds:
            with timing(f"thresh:{name}:{tag}:TH={TH}"):
                y_pred_A = [1 if (v is not None and v >= TH) else 0 for v in s_norm_all]
                TP, FP, TN, FN = confusion(y_true_known_local.tolist(), y_pred_A)
                accA, precA, recA = metrics(TP, FP, TN, FN)
                f1A = f1_from_counts(TP, FP, TN, FN)

                b_idx = [j for j, (pk, v) in enumerate(zip(parsed_known, s_norm_all)) if pk and (v is not None)]
                if b_idx:
                    y_true_B = [int(y_true_known_local[j]) for j in b_idx]
                    y_pred_B = [1 if s_norm_all[j] >= TH else 0 for j in b_idx]
                    TP2, FP2, TN2, FN2 = confusion(y_true_B, y_pred_B)
                    accB, precB, recB = metrics(TP2, FP2, TN2, FN2)
                    f1B = f1_from_counts(TP2, FP2, TN2, FN2)
                else:
                    accB = precB = recB = f1B = float("nan")

                print(f"\n--- Threshold = {TH:.3f} ---")
                # Summary for Set B (parsed answers subset)
                if not np.isnan(accB):
                    print(
                        f"TP={TP2} FP={FP2} TN={TN2} FN={FN2} N={len(b_idx)} | "
                        f"Acc={accB:.4f} | Prec={precB:.4f} | Rec={recB:.4f} | F1={f1B:.4f}"
                    )
                else:
                    print("No rows with parsed answers & valid scores at this threshold.")



# # ========================= Threshold-free & thresholded metrics =========================
# def confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
#     TP = FP = TN = FN = 0
#     for t, p in zip(y_true, y_pred):
#         if t == 1 and p == 1: TP += 1
#         elif t == 0 and p == 1: FP += 1
#         elif t == 0 and p == 0: TN += 1
#         elif t == 1 and p == 0: FN += 1
#     return TP, FP, TN, FN

# def metrics(TP: int, FP: int, TN: int, FN: int) -> Tuple[float, float, float]:
#     N = TP + FP + TN + FN
#     if N == 0:
#         return float("nan"), float("nan"), float("nan")
#     acc = (TP + TN) / N
#     prec = TP / (TP + FP) if (TP + FP) > 0 else float("nan")
#     rec = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
#     return acc, prec, rec

# def _roc_curve(y_true: np.ndarray, y_score: np.ndarray):
#     assert y_true.shape == y_score.shape
#     pos = int(np.sum(y_true == 1))
#     neg = int(np.sum(y_true == 0))
#     if pos == 0 or neg == 0:
#         return None, None
#     order = np.argsort(-y_score)
#     y_sorted = y_true[order]
#     tp = np.cumsum(y_sorted == 1)
#     fp = np.cumsum(y_sorted == 0)
#     tpr = np.r_[0.0, tp / pos, 1.0]
#     fpr = np.r_[0.0, fp / neg, 1.0]
#     return fpr, tpr

# def auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
#     fpr, tpr = _roc_curve(y_true, y_score)
#     if fpr is None: return None
#     return float(np.trapz(tpr, fpr))

# def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, target_tpr: float = 0.95) -> Optional[float]:
#     fpr, tpr = _roc_curve(y_true, y_score)
#     if fpr is None: return None
#     idx = np.searchsorted(tpr, target_tpr, side="left")
#     if idx >= len(fpr): return float(fpr[-1])
#     if idx == 0: return float(fpr[0])
#     t0, t1 = tpr[idx-1], tpr[idx]
#     f0, f1 = fpr[idx-1], fpr[idx]
#     if t1 == t0: return float(f1)
#     w = (target_tpr - t0) / (t1 - t0)
#     return float(f0 + w * (f1 - f0))

# def _precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray):
#     pos = int(np.sum(y_true == 1))
#     if pos == 0: return None, None
#     order = np.argsort(-y_score)
#     y_sorted = y_true[order]
#     tp = np.cumsum(y_sorted == 1)
#     fp = np.cumsum(y_sorted == 0)
#     precision = tp / (tp + fp + 1e-12)
#     recall = tp / pos
#     precision = np.r_[1.0, precision]
#     recall    = np.r_[0.0, recall]
#     return precision, recall

# def aupr(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
#     precision, recall = _precision_recall_curve(y_true, y_score)
#     if precision is None: return None
#     return float(np.trapz(precision, recall))

# def brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
#     return float(np.mean((y_score - y_true) ** 2))

# def ece_fixed(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> Optional[float]:
#     if len(y_true) == 0: return None
#     bins = np.linspace(0.0, 1.0, n_bins + 1)
#     inds = np.digitize(y_score, bins, right=False) - 1
#     inds = np.clip(inds, 0, n_bins - 1)
#     ece = 0.0
#     N = len(y_true)
#     for b in range(n_bins):
#         mask = (inds == b)
#         cnt = np.sum(mask)
#         if cnt == 0: continue
#         conf = float(np.mean(y_score[mask]))
#         acc  = float(np.mean(y_true[mask]))
#         ece += (cnt / N) * abs(acc - conf)
#     return float(ece)

# def ece_equal_mass(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 15) -> Optional[float]:
#     N = len(y_true)
#     if N == 0 or n_bins <= 1: return None
#     order = np.argsort(y_score)
#     ys, ps = y_true[order], y_score[order]
#     ece, start = 0.0, 0
#     for b in range(n_bins):
#         end = int(round((b + 1) * N / n_bins))
#         end = min(max(end, start + 1), N)
#         cnt = end - start
#         if cnt <= 0: continue
#         conf = float(np.mean(ps[start:end]))
#         acc  = float(np.mean(ys[start:end]))
#         ece += (cnt / N) * abs(acc - conf)
#         start = end
#     return float(ece)

# def sm_ece(y_true: np.ndarray, y_score: np.ndarray, bin_counts=(5,10,15,20)) -> Optional[float]:
#     vals = []
#     for k in bin_counts:
#         v = ece_equal_mass(y_true, y_score, n_bins=k)
#         if v is not None and not np.isnan(v):
#             vals.append(v)
#     if not vals: return None
#     return float(np.mean(vals))


# def threshold_free_for_one(
#     name: str,
#     raw_scores: List[Optional[float]],
#     prob_like: bool,
#     known_idx: List[int],
#     y_true_all: List[Optional[int]],
#     out_dir: Path,
# ):
#     # Align to known rows
#     vals = [(i, raw_scores[i]) for i in range(len(raw_scores))
#             if (i in known_idx and raw_scores[i] is not None)]
#     if not vals:
#         print(f"\n[{name}] no valid rows — skipping.")
#         return

#     idxs, s_raw = zip(*vals)
#     s_raw = np.array(s_raw, dtype=np.float64)
#     y_true_known = np.array([y_true_all[i] for i in known_idx], dtype=np.int32)
#     y = y_true_known[[known_idx.index(i) for i in idxs]].astype(np.int32)

#     finite = np.isfinite(s_raw)
#     s_raw = s_raw[finite]
#     y = y[finite]
#     if s_raw.size == 0:
#         print(f"\n[{name}] no finite values — skipping.")
#         return

#     # Raw distribution plot
#     with timing(f"plot:{name}:raw"):
#         plt.figure(figsize=(8, 5))
#         bins = 30
#         if prob_like:
#             rmin, rmax = 0.0, 1.0
#             plt.hist(s_raw[y == 1], bins=bins, range=(rmin, rmax), alpha=0.6, label="Correct")
#             plt.hist(s_raw[y == 0], bins=bins, range=(rmin, rmax), alpha=0.6, label="Wrong")
#         else:
#             rmin, rmax = np.min(s_raw), np.max(s_raw)
#             if np.isfinite(rmin) and np.isfinite(rmax) and rmin < rmax:
#                 plt.hist(s_raw[y == 1], bins=bins, range=(float(rmin), float(rmax)), alpha=0.6, label="Correct")
#                 plt.hist(s_raw[y == 0], bins=bins, range=(float(rmin), float(rmax)), alpha=0.6, label="Wrong")
#             else:
#                 plt.hist(s_raw[y == 1], bins=bins, alpha=0.6, label="Correct")
#                 plt.hist(s_raw[y == 0], bins=bins, alpha=0.6, label="Wrong")
#         plt.xlabel(name); plt.ylabel("Count")
#         plt.title(f"Distribution of {name} by correctness (raw)")
#         plt.legend(); plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
#         raw_path = out_dir / f"{name}_distribution_raw.png"
#         plt.tight_layout(); plt.savefig(raw_path, dpi=150); plt.close()
#     print(f"📊 Saved {name} raw distribution plot to: {raw_path}")

#     # Both orientations
#     for tag, sign in (("↑ (higher=correct)", +1.0), ("↓ (lower=correct)", -1.0)):
#         s_oriented = s_raw * sign
#         with timing(f"final_eval:{name}:{tag}:auroc"):
#             _auroc = auroc(y, s_oriented)
#         with timing(f"final_eval:{name}:{tag}:fpr95"):
#             _fpr95 = fpr_at_tpr(y, s_oriented, target_tpr=0.95)
#         with timing(f"final_eval:{name}:{tag}:aupr"):
#             _aupr  = aupr(y, s_oriented)

#         print(f"\n--- {name} | {tag} ---")
#         print(f"AUROC       : {_auroc:.4f}" if _auroc is not None else "AUROC       : NA")
#         print(f"FPR@95%TPR  : {_fpr95:.4f}" if _fpr95 is not None else "FPR@95%TPR  : NA")
#         print(f"AUPR (pos=1): {_aupr:.4f}"  if _aupr  is not None else "AUPR (pos=1): NA")

#         if prob_like and sign > 0:
#             with timing(f"final_eval:{name}:{tag}:calibration"):
#                 s_clip = np.clip(s_oriented, 0.0, 1.0)
#                 _brier = brier(y, s_clip)
#                 _ece   = ece_fixed(y, s_clip, n_bins=15)
#                 _smece = sm_ece(y, s_clip, bin_counts=(5,10,15,20))
#             print(f"Brier       : {_brier:.4f}")
#             print(f"ECE (fixed) : {_ece:.4f}"   if _ece   is not None else "ECE (fixed) : NA")
#             print(f"smECE       : {_smece:.4f}" if _smece is not None else "smECE       : NA")

#         with timing(f"plot:{name}:norm:{'up' if sign>0 else 'down'}"):
#             smin, smax = float(np.min(s_oriented)), float(np.max(s_oriented))
#             if smax - smin <= 1e-12:
#                 s_norm = np.full_like(s_oriented, 0.5, dtype=np.float64)
#                 norm_info = "degenerate (all equal) -> set to 0.5"
#             else:
#                 s_norm = (s_oriented - smin) / (smax - smin)
#                 norm_info = f"min={smin:.6f}, max={smax:.6f}"
#             plt.figure(figsize=(8, 5))
#             bins = 30
#             plt.hist(s_norm[y == 1], bins=bins, range=(0.0, 1.0), alpha=0.6, label="Correct")
#             plt.hist(s_norm[y == 0], bins=bins, range=(0.0, 1.0), alpha=0.6, label="Wrong")
#             plt.xlabel(f"{name} (normalized)")
#             plt.ylabel("Count")
#             plt.title(f"Distribution of {name} by correctness (normalized, {tag})\n[{norm_info}]")
#             plt.legend(); plt.grid(alpha=0.3, linestyle="--", linewidth=0.5)
#             norm_path = out_dir / f"{name}_distribution_norm_{'up' if sign>0 else 'down'}.png"
#             plt.tight_layout(); plt.savefig(norm_path, dpi=150); plt.close()
#         print(f"📊 Saved {name} normalized {tag} distribution plot to: {norm_path}")


# def threshold_reports_normalized_both(
#     name: str,
#     raw_scores: List[Optional[float]],
#     known_idx: List[int],
#     y_true_all: List[Optional[int]],
#     pred_parsed_all: List[bool],
#     thresholds: List[float],
# ):
#     y_true_known_local = np.array([y_true_all[i] for i in known_idx], dtype=np.int32)
#     parsed_known = [pred_parsed_all[i] for i in known_idx]

#     valid_pairs = []
#     for j, i in enumerate(known_idx):
#         val = raw_scores[i] if i < len(raw_scores) else None
#         if val is not None and np.isfinite(val):
#             valid_pairs.append((j, float(val)))
#     if not valid_pairs:
#         print(f"\n[{name}] no valid rows for thresholding — skipping.")
#         return

#     idx_valid, s_valid_raw = zip(*valid_pairs)
#     idx_valid = np.array(idx_valid, dtype=np.int32)
#     s_valid_raw = np.array(s_valid_raw, dtype=np.float64)

#     for tag, sign in (("↑ (higher=correct)", +1.0), ("↓ (lower=correct)", -1.0)):
#         s_use = s_valid_raw * sign
#         with timing(f"thresh:{name}:{tag}:normalize"):
#             smin, smax = float(np.min(s_use)), float(np.max(s_use))
#             if smax - smin <= 1e-12:
#                 s_valid_norm = np.full_like(s_use, 0.5, dtype=np.float64)
#                 norm_info = "degenerate (all equal) -> set to 0.5"
#             else:
#                 s_valid_norm = (s_use - smin) / (smax - smin)
#                 norm_info = f"min={smin:.6f}, max={smax:.6f}"
#             s_norm_all: List[Optional[float]] = [None] * len(known_idx)
#             for jj, v in zip(idx_valid, s_valid_norm):
#                 s_norm_all[jj] = float(v)

#         print(f"\n=== Thresholded report for {name} | {tag} [normalized to [0,1], {norm_info}] ===")
#         for TH in thresholds:
#             with timing(f"thresh:{name}:{tag}:TH={TH}"):
#                 y_pred_A = [1 if (v is not None and v >= TH) else 0 for v in s_norm_all]
#                 TP, FP, TN, FN = confusion(y_true_known_local.tolist(), y_pred_A)
#                 accA, precA, recA = metrics(TP, FP, TN, FN)

#                 b_idx = [j for j, (pk, v) in enumerate(zip(parsed_known, s_norm_all)) if pk and (v is not None)]
#                 if b_idx:
#                     y_true_B = [int(y_true_known_local[j]) for j in b_idx]
#                     y_pred_B = [1 if s_norm_all[j] >= TH else 0 for j in b_idx]
#                     TP2, FP2, TN2, FN2 = confusion(y_true_B, y_pred_B)
#                     accB, precB, recB = metrics(TP2, FP2, TN2, FN2)
#                 else:
#                     accB = precB = recB = float("nan")

#                 print(f"\n--- Threshold = {TH:.3f} ---")
#                 # print("-- Set A: All verifiable rows (None => negative) --")
#                 # if not np.isnan(precA):
#                 #     print(f"TP={TP} FP={FP} TN={TN} FN={FN} N={len(y_true_known_local)} | "
#                 #           f"Acc={accA:.4f} | Prec={precA:.4f} | Rec={recA:.4f}")
#                 # else:
#                 #     print(f"TP={TP} FP={FP} TN={TN} FN={FN} N={len(y_true_known_local)} | "
#                 #           f"Acc={accA:.4f} | Prec=NA | Rec={recA:.4f}" if not np.isnan(recA) else
#                 #           f"TP={TP} FP={FP} TN={TN} FN={FN} N={len(y_true_known_local)} | Acc={accA:.4f} | Prec=NA | Rec=NA")

#                 # print("\n-- Set B: Rows with parsed answers (exclude missing) --")
#                 if not np.isnan(accB):
#                     print(f"TP={TP2} FP={FP2} TN={TN2} FN={FN2} N={len(b_idx)} | "
#                           f"Acc={accB:.4f} | Prec={precB:.4f} | Rec={recB:.4f}")
#                 else:
#                     print("No rows with parsed answers & valid scores at this threshold.")


