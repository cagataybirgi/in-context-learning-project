"""
scripts/backfill_metrics.py — retro-score existing run CSVs
============================================================

Adds the new augmented metric columns — `substring_match`, `is_numeric_close`,
`token_f1`, `rouge_l` — to run CSVs that pre-date `runner.py`'s metric
augmentation.  Operates from the already-stored `generation` and
`gold_answer` columns, so it never re-queries the API.

BERTScore is **not** computed here; use `scripts/score_bertscore.py` for that.

Usage
-----
    python -m scripts.backfill_metrics results/run_*.csv
    python -m scripts.backfill_metrics results/run_20260604_181515.csv --overwrite
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from scripts.metrics import (
    substring_match,
    is_numeric_close,
    token_f1,
    rouge_l,
)


NEW_COLS = ["substring_match", "is_numeric_close", "token_f1", "rouge_l"]


def _expand_paths(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        matched = glob.glob(p)
        if not matched and Path(p).exists():
            matched = [p]
        out.extend(Path(m) for m in matched)
    return out


def _score_row(row) -> dict:
    dataset = row.get("dataset", "")
    generation = row.get("generation", "") or ""
    gold       = row.get("gold_answer", "")
    predicted  = row.get("predicted_answer", "") or ""
    # token_f1 / rouge_l compare full generation vs gold_answer.  We don't
    # have the original sample["answer"] (full gold reasoning chain) on
    # historical CSVs, so we fall back to comparing against the stored
    # gold_answer scalar — that's a faithful rescoring of what's on disk
    # and is consistent with the StrategyQA case.
    return {
        "substring_match":  int(substring_match(predicted or str(generation), str(gold))),
        "is_numeric_close": int(is_numeric_close(predicted, gold)) if dataset == "gsm8k" else 0,
        "token_f1":         round(token_f1(str(generation), str(gold)), 4),
        "rouge_l":          round(rouge_l(str(generation), str(gold)), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Backfill augmented metrics into existing CSVs.")
    parser.add_argument("paths", nargs="+", help="Run CSV files (globs allowed).")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Recompute even for rows that already have non-null values.",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.paths)
    if not paths:
        print("[ERROR] No input files matched.", file=sys.stderr)
        sys.exit(1)

    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}", file=sys.stderr)
            continue

        for col in NEW_COLS:
            if col not in df.columns:
                df[col] = pd.NA

        # Decide which rows still need scoring.
        if args.overwrite:
            mask = pd.Series([True] * len(df), index=df.index)
        else:
            mask = df["substring_match"].isna()
        to_score = df[mask]
        if to_score.empty:
            print(f"[skip] {p.name}: all rows already have metric columns populated.")
            continue

        print(f"[backfill] {p.name}: scoring {len(to_score)} rows …")
        scored = to_score.apply(_score_row, axis=1, result_type="expand")
        df.loc[mask, NEW_COLS] = scored[NEW_COLS].values
        df.to_csv(p, index=False)
        rates = {
            "sub":  df["substring_match"].astype(float).mean() * 100,
            "nc":   df["is_numeric_close"].astype(float).mean() * 100,
            "f1":   df["token_f1"].astype(float).mean(),
            "rL":   df["rouge_l"].astype(float).mean(),
        }
        print(f"[backfill] {p.name}: substring={rates['sub']:.1f}%  "
              f"num_close={rates['nc']:.1f}%  "
              f"F1={rates['f1']:.3f}  ROUGE-L={rates['rL']:.3f}")


if __name__ == "__main__":
    main()
