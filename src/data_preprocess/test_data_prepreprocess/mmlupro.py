"""
Convert MMLU-Pro to minimal CSV format for Gnosis.

Loads `TIGER-Lab/MMLU-Pro` from Hugging Face and writes 2-column CSV files
(question, answer) per split. Each question is the stem plus lettered options
("A. ...", "B. ...", ...); each answer is a single letter (A/B/C/...).

Defaults:
- out_dir (relative to Gnosis root): data/test/mmlu_pro_csv
"""


from datasets import load_dataset, DatasetDict
from typing import Dict, Any, List
import string
import argparse
import os

LETTERS = list(string.ascii_uppercase)

def to_letter(idx: int):
    return LETTERS[idx] if isinstance(idx, int) and 0 <= idx < len(LETTERS) else None

def format_example(x: Dict[str, Any]) -> Dict[str, Any]:
    q = (x.get("question") or "").strip()
    opts: List[str] = x.get("options") or []
    labels = LETTERS[:len(opts)]
    labeled = [f"{lbl}. {opt}" for lbl, opt in zip(labels, opts)]
    merged = f"{q}\n\n" + "\n".join(labeled) if labeled else q

    ans_letter = (x.get("answer") or "") or to_letter(x.get("answer_index"))
    return {"question": merged, "answer": ans_letter}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="TIGER-Lab/MMLU-Pro",
                    help="HF dataset repo id")
    ap.add_argument("--splits", nargs="+", default=["validation", "test"],
                    help="Splits to process if present")
    ap.add_argument("--out_dir", default="data/test/mmlu_pro_csv",
                    help="Directory to write CSV files")
    ap.add_argument("--num_proc", type=int, default=None,
                    help="Parallel workers for map()")
    args = ap.parse_args()

    raw = load_dataset(args.dataset)
    os.makedirs(args.out_dir, exist_ok=True)

    any_written = False
    for split in args.splits:
        if split not in raw:
            print(f"[warn] split '{split}' not found; skipping")
            continue

        ds = raw[split].map(
            format_example,
            remove_columns=raw[split].column_names,
            num_proc=args.num_proc
        )
        # Ensure only two columns and in this order
        ds = ds.select_columns(["question", "answer"])

        csv_path = os.path.join(args.out_dir, f"{split}.csv")
        # datasets has a native CSV export:
        ds.to_csv(csv_path)
        print(f"[ok] wrote {csv_path}")
        any_written = True

    if not any_written:
        raise SystemExit("No splits processed; nothing written.")

if __name__ == "__main__":
    main()
