"""
============================================================

Reads one or more per-sample run CSVs and classifies every incorrect row into
one of three categories that the progress report calls out in Section 8:

  * truncation : the generation looks cut off — no terminal punctuation, no
                 explicit answer cue, and the completion_tokens column
                 (when present) is at or above the configured max_tokens.
  * extraction : the generation contains the gold answer somewhere in the
                 text, but the extractor returned something else (or nothing).
                 This is the "right answer, wrong format" case described in
                 the report.
  * reasoning  : everything else — the model produced a clean output that
                 does not contain the gold answer.

Output: a `errors.csv` file with one row per misclassified sample plus a
`errors_summary.csv` table of per-strategy counts.  Both are written next to
the inputs so they can be folded into the final report appendix.

Usage
-----
    python -m scripts.error_analysis results/run_*.csv
"""

from __future__ import annotations

import argparse
import glob
import re
import sys
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from scripts.metrics import extract_gsm8k_answer, extract_strategyqa_answer


_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _gold_present(generation: str, gold: str, dataset: str) -> bool:
    """Return True iff the gold answer appears literally inside the generation."""
    if not isinstance(generation, str) or not gold:
        return False
    gen = generation.lower()
    gold = str(gold).lower().strip()
    if dataset == "strategyqa":
        return bool(re.search(rf"\b{re.escape(gold)}\b", gen))
    # gsm8k: compare numeric tokens after stripping commas.
    target = gold.replace(",", "")
    for tok in _NUM_RE.findall(gen):
        if tok.replace(",", "") == target:
            return True
    return False


def _looks_truncated(generation: str, dataset: str, completion_tokens, max_tokens_hint=None) -> bool:
    if not isinstance(generation, str) or not generation.strip():
        return False
    stripped = generation.rstrip()
    ends_clean = stripped.endswith((".", "!", "?", '"', "'", ")", "]"))
    has_answer_cue = bool(re.search(r"answer\s*[:=]", stripped, re.IGNORECASE))
    if dataset == "gsm8k":
        has_answer_cue = has_answer_cue or "####" in stripped
    if ends_clean or has_answer_cue:
        return False
    # If completion_tokens is recorded and at/above the prompt's likely
    # max_tokens budget, treat as a truncation.  Fall back to a permissive
    # heuristic if the column is missing.
    if completion_tokens is not None and not pd.isna(completion_tokens):
        try:
            ct = int(completion_tokens)
            if max_tokens_hint and ct >= max_tokens_hint - 4:
                return True
            if ct >= 240:  # generic "looks long" cutoff
                return True
        except (TypeError, ValueError):
            pass
    # Generic fallback: long output, no clean ending → likely truncated.
    return len(stripped) > 600


def categorise_row(row, dataset: str, max_tokens_hint=None) -> str:
    gen = row.get("generation", "")
    gold = row.get("gold_answer", "")
    completion_tokens = row.get("completion_tokens", None)

    if _looks_truncated(gen, dataset, completion_tokens, max_tokens_hint):
        return "truncation"
    if _gold_present(gen, gold, dataset):
        return "extraction"
    return "reasoning"


def _expand_paths(patterns: Iterable[str]) -> List[Path]:
    out: List[Path] = []
    for p in patterns:
        matched = glob.glob(p)
        if not matched and Path(p).exists():
            matched = [p]
        out.extend(Path(m) for m in matched)
    return out


def main():
    parser = argparse.ArgumentParser(description="Categorise failures by type.")
    parser.add_argument("paths", nargs="+",
                        help="One or more `run_*.csv` files (globs allowed).")
    parser.add_argument("--out_dir", default="results/error_analysis")
    parser.add_argument(
        "--max_tokens_hint", type=int, default=1500,
        help="The max_tokens used during generation, for truncation detection. "
             "Override per-file via the `condition` column if present.",
    )
    args = parser.parse_args()

    paths = _expand_paths(args.paths)
    if not paths:
        print("[ERROR] No input files matched.", file=sys.stderr)
        sys.exit(1)

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df["__source_file"] = p.name
            frames.append(df)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}", file=sys.stderr)
    if not frames:
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)

    # Keep only failures (incorrect or parse-failed).
    failures = df[(df["is_correct"] == 0) | (df["parse_failed"] == 1)].copy()
    if failures.empty:
        print("No failures found — nothing to categorise.")
        return

    # Use the `condition` column (if present) to recover the max_tokens used
    # for that row; falls back to the CLI hint.
    def _row_max_tokens(row):
        cond = row.get("condition", "")
        if isinstance(cond, str):
            m = re.search(r"mt=(\d+)", cond)
            if m:
                return int(m.group(1))
        return args.max_tokens_hint

    failures["category"] = failures.apply(
        lambda r: categorise_row(r, r["dataset"], _row_max_tokens(r)), axis=1
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_path = out_dir / "errors.csv"
    failures.to_csv(errors_path, index=False)

    summary = (
        failures.groupby(["dataset", "strategy", "category"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
    )
    summary_path = out_dir / "errors_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {len(failures)} failure rows → {errors_path}")
    print(f"Wrote per-strategy summary → {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
