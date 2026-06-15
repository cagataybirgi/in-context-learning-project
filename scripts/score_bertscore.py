"""
scripts/score_bertscore.py — opt-in BERTScore augmentation
==========================================================

Computes BERTScore F1 for each row of one or more `run_*.csv` files and
writes the result back as a new `bertscore_f1` column.  Run after a normal
run finishes — does not block the experiment pipeline.

BERTScore is computed against the **full gold text** (the GSM8K reasoning
chain or the StrategyQA yes/no label), matching `runner.py`'s scope for
`token_f1` and `rouge_l`.  Read the score as a *generation-quality* signal,
not as answer correctness; EM remains the primary metric.

This script lazy-imports `bert_score` so the package is **not** a hard
dependency of the project.  Install only if you intend to use BERTScore:

    pip install bert_score

Usage
-----
    python -m scripts.score_bertscore results/run_*.csv
    python -m scripts.score_bertscore results/run_20260607_173356.csv \\
        --model_type microsoft/deberta-xlarge-mnli --batch_size 32
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd


def _expand_paths(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        matched = glob.glob(p)
        if not matched and Path(p).exists():
            matched = [p]
        out.extend(Path(m) for m in matched)
    return out


def _try_import_bertscore():
    try:
        from bert_score import score  # type: ignore
        return score
    except ImportError as e:
        print(
            "[ERROR] `bert_score` is not installed.  Install with:\n"
            "        pip install bert_score\n"
            f"Original ImportError: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Augment run CSVs with BERTScore F1.")
    parser.add_argument("paths", nargs="+", help="Run CSV files (globs allowed).")
    parser.add_argument(
        "--model_type", default="microsoft/deberta-xlarge-mnli",
        help="HuggingFace model used by bert_score (default is the recommended "
             "English-language model in the bert_score docs).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--lang", default="en",
        help="Falls back to bert_score's per-language defaults if --model_type "
             "is left at its default.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-score rows that already have a non-empty `bertscore_f1` value.",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.paths)
    if not paths:
        print("[ERROR] No input files matched.", file=sys.stderr)
        sys.exit(1)

    score = _try_import_bertscore()

    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}", file=sys.stderr)
            continue

        if "generation" not in df.columns or "gold_answer" not in df.columns:
            print(f"[WARN] {p} is missing generation/gold_answer columns; skipping.")
            continue

        if "bertscore_f1" not in df.columns:
            df["bertscore_f1"] = pd.NA

        # Decide which rows still need scoring.
        if args.overwrite:
            mask = pd.Series([True] * len(df))
        else:
            mask = df["bertscore_f1"].isna()
        to_score = df[mask]
        if to_score.empty:
            print(f"[skip] {p.name}: all rows already scored.")
            continue

        # BERTScore compares two strings.  We use the full generation vs the
        # full gold text — for GSM8K that's the gold reasoning chain, for
        # StrategyQA it's just the yes/no label (low information, but
        # included for consistency so the column is non-null everywhere).
        cands = to_score["generation"].astype(str).tolist()
        refs  = to_score["gold_answer"].astype(str).tolist()

        print(f"[bertscore] {p.name}: scoring {len(cands)} rows "
              f"with {args.model_type} (batch={args.batch_size}) …")
        _, _, f1 = score(
            cands, refs,
            model_type=args.model_type,
            batch_size=args.batch_size,
            lang=args.lang,
            verbose=False,
        )
        df.loc[mask, "bertscore_f1"] = [round(float(x), 4) for x in f1.tolist()]
        df.to_csv(p, index=False)
        mean_f1 = df.loc[mask, "bertscore_f1"].astype(float).mean()
        print(f"[bertscore] {p.name}: mean F1 over scored rows = {mean_f1:.4f}")


if __name__ == "__main__":
    main()
