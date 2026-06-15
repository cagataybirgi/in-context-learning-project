"""
runner.py — Main experiment script
===================================

Runs prompting strategies across datasets using the NVIDIA Integrate API.
Configured for nvidia/nemotron-3-super-120b-a12b by default.

The runner now exposes the ablation dimensions described in Sections 6 and 7
of the progress report:

  --structured                  Append an explicit `Answer:` instruction.
  --k_shot K                    Override the per-strategy default k-shot count.
  --cot_trigger {default,careful,none}
  --persona_variant {revised,original,generic}
  --self_consistency N          Sample N reasoning paths and majority-vote
                                (requires --temperature > 0).
"""

import os
import sys
import time
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

from openai import OpenAI
from openai import AuthenticationError, PermissionDeniedError, NotFoundError
import pandas as pd
from tqdm import tqdm


class FatalAPIError(RuntimeError):
    """Raised for un-retryable provider errors (auth / model-not-found)."""

from scripts.data_loader import load_evaluation_datasets
from scripts.metrics import (
    extract_gsm8k_answer,
    extract_strategyqa_answer,
    calculate_exact_match,
    majority_vote,
    substring_match,
    is_numeric_close,
    token_f1,
    rouge_l,
)
from prompts.templates import get_prompt

RESULTS_DIR     = Path("results")
TIMESTAMP       = datetime.now().strftime("%Y%m%d_%H%M%S")

DATASETS    = ["gsm8k", "strategyqa"]
STRATEGIES  = ["standard_few_shot", "zero_shot_cot", "persona_prompting"]

# NVIDIA NIM trial-tier constraints
# ----------------------------------
#   * Hard cap: 40 requests / minute  → 1.50 s minimum spacing.
#   * Context window (input + output): 16,384 tokens for
#     nvidia/nemotron-3-super-120b-a12b.
# Sleep slightly above the floor to leave headroom for API latency jitter and
# avoid sliding-window edge cases (39 quick requests + 1 slow request would
# otherwise straddle the 60-s boundary).
NVIDIA_RPM_CAP    = 40
RATE_LIMIT_SLEEP  = 60.0 / NVIDIA_RPM_CAP + 0.1   # ≈ 1.6 s
MODEL_CONTEXT_WINDOW = 16384

# Per-sample CSV schema.  `condition` records the active ablation cell so the
# same results file can hold rows from many sweeps without ambiguity.
CSV_FIELDS = [
    "dataset", "strategy", "sample_idx",
    "question", "gold_answer", "generation",
    "predicted_answer", "is_correct", "parse_failed",
    # Augmented metrics (see scripts/metrics.py).  substring_match and
    # is_numeric_close compare extracted scalars; token_f1 and rouge_l
    # compare the full generation against the full gold text.
    "substring_match", "is_numeric_close",
    "token_f1", "rouge_l",
    "prompt_tokens", "completion_tokens", "total_tokens", "latency_s",
    "condition",
]


def call_llm(
    client: OpenAI,
    prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    n: int = 1,
):
    """
    Returns (texts, prompt_tokens, completion_tokens, latency_s).

    `texts` is a list of length `n`.  When `n == 1` the list has a single
    element; callers that don't need self-consistency can take `texts[0]`.
    """
    for attempt in range(3):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            latency = time.time() - t0
            texts = [(c.message.content or "") for c in response.choices]
            # Some providers return < n choices if the upstream sampler stalled.
            while len(texts) < n:
                texts.append("")
            usage = getattr(response, "usage", None)
            prompt_tokens     = getattr(usage, "prompt_tokens",     0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            return texts, prompt_tokens, completion_tokens, latency
        except (AuthenticationError, PermissionDeniedError, NotFoundError) as e:
            # 401 / 403 / 404 — re-trying won't help; abort the whole run so
            # the user fixes their key (or model name) before burning through
            # 100 samples × 14 s of useless retries.
            raise FatalAPIError(
                f"{type(e).__name__}: {e}.  Re-check NVIDIA_API_KEY and the "
                f"--model name; aborting before more samples are wasted."
            ) from e
        except Exception as e:
            wait = 2 ** attempt * 2
            print(f"\n  [API Error] {e} | Waiting {wait}s before retry {attempt+1}/3 …")
            if attempt == 2:
                return [""] * n, 0, 0, 0.0
            time.sleep(wait)
    return [""] * n, 0, 0, 0.0


def get_reference(sample: dict, dataset: str) -> str:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(sample["answer"])
    else:
        raw = sample["answer"]
        return "yes" if raw is True or str(raw).lower() == "true" else "no"


def extract_prediction(generation: str, dataset: str) -> str:
    if dataset == "gsm8k":
        return extract_gsm8k_answer(generation)
    else:
        return extract_strategyqa_answer(generation)


def _condition_label(args) -> str:
    """
    Compact ablation tag stored on each row for grouping in analysis.

    `model=` is included so that runs against multiple LLMs (e.g. Nemotron
    vs Llama) sharing the same `run_*.csv` can be split unambiguously.
    Tolerates an `args` object without `.model` (older callers): in that
    case the field is emitted as `model=unknown`.
    """
    model = getattr(args, "model", None) or "unknown"
    bits = [
        f"model={model}",
        f"k={args.k_shot if args.k_shot is not None else 'def'}",
        f"trig={args.cot_trigger}",
        f"persona={args.persona_variant}",
        f"struct={'on' if args.structured else 'off'}",
        f"mt={args.max_tokens}",
        f"T={args.temperature}",
        f"sc={args.self_consistency}",
    ]
    return ",".join(bits)


def run_experiment(
    client: OpenAI,
    dataset_name: str,
    dataset,
    strategy: str,
    max_samples: Optional[int],
    results_path: Path,
    already_done: set,
    model: str,
    max_tokens: int,
    temperature: float,
    *,
    structured: bool,
    k_shot: Optional[int],
    cot_trigger: str,
    persona_variant: str,
    self_consistency: int,
    condition: str,
) -> dict:
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    n = len(samples)

    predictions, references = [], []
    parse_failures = 0

    write_header = not results_path.exists()
    csv_file = open(results_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    print(f"\n{'='*60}")
    print(f"  Dataset  : {dataset_name.upper()}")
    print(f"  Strategy : {strategy}")
    print(f"  Samples  : {n}")
    print(f"  Condition: {condition}")
    print(f"{'='*60}")

    for idx, sample in enumerate(tqdm(samples, desc=f"{dataset_name}/{strategy}", ncols=72)):
        row_key = (dataset_name, strategy, idx, condition)
        if row_key in already_done:
            continue

        question  = sample["question"]
        gold      = get_reference(sample, dataset_name)
        prompt    = get_prompt(
            dataset_name,
            strategy,
            question,
            k_shot=k_shot,
            cot_trigger=cot_trigger,
            persona_variant=persona_variant,
            structured=structured,
        )

        generations, ptok, ctok, latency = call_llm(
            client, prompt, model, max_tokens, temperature, n=self_consistency
        )
        time.sleep(RATE_LIMIT_SLEEP)

        if self_consistency > 1:
            candidate_answers = [extract_prediction(g, dataset_name) for g in generations]
            predicted = majority_vote(candidate_answers)
            generation = " ||| ".join(generations)  # keep raw paths for analysis
        else:
            generation = generations[0]
            predicted = extract_prediction(generation, dataset_name)

        is_correct = (predicted == gold and predicted != "")
        failed = predicted == ""

        if failed:
            parse_failures += 1

        predictions.append(generation)
        references.append(sample["answer"])

        # Augmented metrics.
        # substring_match: gold scalar appears verbatim somewhere in the
        # extracted predicted scalar (or the raw generation as a fallback).
        # is_numeric_close: only meaningful for GSM8K.
        # token_f1 / rouge_l: full generation vs full gold text.
        gold_text = str(sample["answer"]) if dataset_name == "gsm8k" else gold
        sm = substring_match(predicted or generation, gold)
        nc = is_numeric_close(predicted, gold) if dataset_name == "gsm8k" else False
        f1 = token_f1(generation, gold_text)
        rl = rouge_l(generation, gold_text)

        writer.writerow({
            "dataset":          dataset_name,
            "strategy":         strategy,
            "sample_idx":       idx,
            "question":         question,
            "gold_answer":      gold,
            "generation":       str(generation).replace("\n", " "),
            "predicted_answer": predicted,
            "is_correct":       int(is_correct),
            "parse_failed":     int(failed),
            "substring_match":  int(sm),
            "is_numeric_close": int(nc),
            "token_f1":         round(f1, 4),
            "rouge_l":          round(rl, 4),
            "prompt_tokens":    ptok,
            "completion_tokens": ctok,
            "total_tokens":     ptok + ctok,
            "latency_s":        round(latency, 2),
            "condition":        condition,
        })
        csv_file.flush()

    csv_file.close()

    if n == 0:
        return {"dataset": dataset_name, "strategy": strategy, "n": 0, "em": 0.0, "parse_failure_rate": 0.0}

    # Re-read the CSV so resumed runs aggregate correctly.
    sm_rate = nc_rate = avg_f1 = avg_rouge = 0.0
    try:
        df_full = pd.read_csv(results_path)
        df_run = df_full[
            (df_full["dataset"] == dataset_name)
            & (df_full["strategy"] == strategy)
            & (df_full.get("condition", condition) == condition)
        ]
        completed_n = len(df_run)
        correct_n   = int(df_run["is_correct"].sum())
        pfail_n     = int(df_run["parse_failed"].sum())
        em  = (correct_n / completed_n) * 100 if completed_n > 0 else 0.0
        pfr = (pfail_n  / completed_n) * 100 if completed_n > 0 else 0.0
        avg_prompt_tokens     = float(df_run["prompt_tokens"].mean())
        avg_completion_tokens = float(df_run["completion_tokens"].mean())
        avg_total_tokens      = float(df_run["total_tokens"].mean())
        total_tokens_used     = int(df_run["total_tokens"].sum())
        avg_latency           = float(df_run["latency_s"].mean())
        # New per-cell metric aggregates.  Guarded so old CSVs without the
        # columns don't crash the resume path.
        if "substring_match" in df_run.columns:
            sm_rate = float(df_run["substring_match"].mean()) * 100
        if "is_numeric_close" in df_run.columns:
            nc_rate = float(df_run["is_numeric_close"].mean()) * 100
        if "token_f1" in df_run.columns:
            avg_f1 = float(df_run["token_f1"].mean())
        if "rouge_l" in df_run.columns:
            avg_rouge = float(df_run["rouge_l"].mean())
    except Exception:
        em  = calculate_exact_match(predictions, [s["answer"] for s in samples], dataset_name)
        pfr = (parse_failures / n) * 100
        avg_prompt_tokens = avg_completion_tokens = avg_total_tokens = avg_latency = 0.0
        total_tokens_used = 0

    print(f"\n  ✓ Executed: {n} | EM = {em:.1f}% | Parse-failure rate = {pfr:.1f}%")
    print(f"     Substring = {sm_rate:.1f}% | NumClose = {nc_rate:.1f}% | F1 = {avg_f1:.3f} | ROUGE-L = {avg_rouge:.3f}")
    print(f"  ⏱ Avg latency: {avg_latency:.1f}s | Avg total tokens: {avg_total_tokens:.0f} | Total tokens used: {total_tokens_used}")

    return {
        "dataset":               dataset_name,
        "strategy":              strategy,
        "condition":             condition,
        "n":                     n,
        "em":                    em,
        "parse_failure_rate":    pfr,
        "substring_match_rate":  round(sm_rate, 1),
        "numeric_close_rate":    round(nc_rate, 1),
        "avg_token_f1":          round(avg_f1, 4),
        "avg_rouge_l":           round(avg_rouge, 4),
        "avg_prompt_tokens":     round(avg_prompt_tokens, 1),
        "avg_completion_tokens": round(avg_completion_tokens, 1),
        "avg_total_tokens":      round(avg_total_tokens, 1),
        "total_tokens_used":     total_tokens_used,
        "avg_latency_s":         round(avg_latency, 2),
    }


def load_already_done(results_dir: Path) -> set:
    """
    Build the set of (dataset, strategy, sample_idx, condition) keys already
    captured across every `run_*.csv` file in `results_dir`.  Older runs
    without a `condition` column are credited to the default condition.
    """
    done = set()
    for f in results_dir.glob("run_*.csv"):
        try:
            df = pd.read_csv(f)
            cond_col = df["condition"] if "condition" in df.columns else None
            for i, row in df.iterrows():
                cond = cond_col.iloc[i] if cond_col is not None else "legacy"  # pyright: ignore[reportArgumentType, reportCallIssue]
                done.add((row["dataset"], row["strategy"], int(row["sample_idx"]), cond))
        except Exception:
            pass
    return done


def save_summary(summary_rows: List[dict], results_dir: Path):
    df = pd.DataFrame(summary_rows)
    summary_path = results_dir / "summary.csv"
    df.to_csv(summary_path, index=False)

    print("\n" + "="*60)
    print("  EXPERIMENT SUMMARY")
    print("="*60)
    try:
        pivot = df.pivot_table(index="strategy", columns="dataset", values="em", aggfunc="mean")
        print(pivot.to_string())
    except Exception:
        print(df.to_string(index=False))
    print(f"\nFull summary saved → {summary_path}")

    json_path = results_dir / "summary.json"
    json_path.write_text(json.dumps(summary_rows, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompting strategy experiments.")
    parser.add_argument("--datasets",    nargs="+", default=DATASETS,   choices=DATASETS)
    parser.add_argument("--strategies",  nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--max_samples", type=int,  default=None)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3-super-120b-a12b")
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0)

    # New ablation knobs.
    parser.add_argument("--structured", action="store_true",
                        help="Append the explicit Answer: instruction (Section 6).")
    parser.add_argument("--k_shot", type=int, default=None,
                        help="Override the per-strategy default number of demonstrations.")
    parser.add_argument("--cot_trigger", choices=["default", "careful", "none"], default="default",
                        help="Choose which CoT trigger phrase to use.")
    parser.add_argument("--persona_variant", choices=["revised", "original", "generic"],
                        default="revised", help="Persona phrasing variant (Section 6 revision).")
    parser.add_argument("--self_consistency", type=int, default=1,
                        help="Number of sampled reasoning paths to majority-vote over.")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.self_consistency > 1 and args.temperature == 0.0:
        print(
            "[WARN] --self_consistency > 1 with temperature=0.0 will produce "
            "identical samples; set --temperature 0.7 (or similar) for it to "
            "have any effect."
        )

    # NVIDIA NIM context-window guard.  The 16,384-token cap covers prompt +
    # completion; if the requested completion budget alone is too high we will
    # waste API calls on context-overflow errors.  Reserve at least 2 k tokens
    # for the prompt (few-shot demos + persona preamble + question).
    if args.max_tokens > MODEL_CONTEXT_WINDOW - 2048:
        print(
            f"[WARN] --max_tokens={args.max_tokens} leaves <2k tokens of the "
            f"{MODEL_CONTEXT_WINDOW}-token context window for the prompt; "
            f"expect overflow errors on long few-shot configurations."
        )

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("[ERROR] NVIDIA_API_KEY environment variable not set.")
        sys.exit(1)

    # Per-request timeout of 90 s.  Empirically the slowest legitimate
    # generation we observed (strategyqa / zero_shot_cot at max_tokens=1500)
    # tops out around 30 s; anything past 90 s is almost certainly a hung
    # connection that should be retried by `call_llm`'s backoff loop
    # instead of blocking on the SDK's 600-s default.
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=90.0,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"run_{TIMESTAMP}.csv"
    already_done = load_already_done(RESULTS_DIR) if args.resume else set()

    gsm8k, strategyqa = load_evaluation_datasets()
    dataset_map = {"gsm8k": gsm8k, "strategyqa": strategyqa}

    condition = _condition_label(args)

    summary_rows = []
    try:
        for dataset_name in args.datasets:
            for strategy in args.strategies:
                result = run_experiment(
                    client          = client,
                    dataset_name    = dataset_name,
                    dataset         = dataset_map[dataset_name],
                    strategy        = strategy,
                    max_samples     = args.max_samples,
                    results_path    = results_path,
                    already_done    = already_done,
                    model           = args.model,
                    max_tokens      = args.max_tokens,
                    temperature     = args.temperature,
                    structured      = args.structured,
                    k_shot          = args.k_shot,
                    cot_trigger     = args.cot_trigger,
                    persona_variant = args.persona_variant,
                    self_consistency= max(1, args.self_consistency),
                    condition       = condition,
                )
                summary_rows.append(result)
    except FatalAPIError as e:
        print(f"\n[FATAL] {e}", file=sys.stderr)
        print(
            "Fix the cause and rerun with `--resume` — completed rows are "
            "already on disk and will be skipped.",
            file=sys.stderr,
        )
        sys.exit(2)

    if summary_rows:
        save_summary(summary_rows, RESULTS_DIR)


if __name__ == "__main__":
    main()
