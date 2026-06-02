"""
runner.py — Main experiment script
===================================
Runs prompting strategies across datasets using the OpenRouter API.
Configured for nvidia/nemotron-3-super-120b-a12b:free.
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
import pandas as pd
from tqdm import tqdm

from scripts.data_loader import load_evaluation_datasets
from scripts.metrics import extract_gsm8k_answer, extract_strategyqa_answer, calculate_exact_match
from prompts.templates import get_prompt

RESULTS_DIR     = Path("results")
TIMESTAMP       = datetime.now().strftime("%Y%m%d_%H%M%S")

DATASETS    = ["gsm8k", "strategyqa"]
STRATEGIES  = ["standard_few_shot", "zero_shot_cot", "persona_prompting"]
RATE_LIMIT_SLEEP = 0.5  

def call_openrouter(client: OpenAI, prompt: str, model: str, max_tokens: int, temperature: float):
    """Returns (text, prompt_tokens, completion_tokens, latency_s)."""
    for attempt in range(3):
        try:
            t0 = time.time()
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            latency = time.time() - t0
            text = response.choices[0].message.content or ""
            # OpenRouter returns usage in the same shape as OpenAI
            usage = getattr(response, "usage", None)
            prompt_tokens     = getattr(usage, "prompt_tokens",     0) if usage else 0
            completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
            return text, prompt_tokens, completion_tokens, latency
        except Exception as e:
            wait = 2 ** attempt * 2
            print(f"\n  [API Error] {e} | Waiting {wait}s before retry {attempt+1}/3 …")
            if attempt == 2:
                return "", 0, 0, 0.0
            time.sleep(wait)
    return "", 0, 0, 0.0

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

def run_experiment(
    client: OpenAI,
    dataset_name: str,
    dataset,
    strategy: str,
    max_samples: Optional[int],
    results_path: Path,
    already_done: set,
    model: str,          # NEW
    max_tokens: int,     # NEW
    temperature: float,  # NEW
) -> dict:
    samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))
    n = len(samples)

    predictions, references = [], []
    parse_failures = 0

    write_header = not results_path.exists()
    csv_file = open(results_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=[
    "dataset", "strategy", "sample_idx",
    "question", "gold_answer", "generation",
    "predicted_answer", "is_correct", "parse_failed",
    "prompt_tokens", "completion_tokens", "total_tokens", "latency_s"   # NEW
    ])
    if write_header:
        writer.writeheader()

    print(f"\n{'='*60}")
    print(f"  Dataset : {dataset_name.upper()}")
    print(f"  Strategy: {strategy}")
    print(f"  Samples : {n}")
    print(f"{'='*60}")

    for idx, sample in enumerate(tqdm(samples, desc=f"{dataset_name}/{strategy}", ncols=72)):
        row_key = (dataset_name, strategy, idx)
        if row_key in already_done:
            continue

        question  = sample["question"]
        gold      = get_reference(sample, dataset_name)
        prompt    = get_prompt(dataset_name, strategy, question)

        generation, ptok, ctok, latency = call_openrouter(client, prompt, model, max_tokens, temperature)
        time.sleep(RATE_LIMIT_SLEEP)

        predicted = extract_prediction(generation, dataset_name)
        is_correct = (predicted == gold and predicted != "")
        failed = predicted == ""

        if failed:
            parse_failures += 1

        predictions.append(generation)
        references.append(sample["answer"])

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
            "prompt_tokens":     ptok,            
            "completion_tokens": ctok,            
            "total_tokens":      ptok + ctok,     
            "latency_s":         round(latency, 2),  

        })
        csv_file.flush()

    csv_file.close()

    if n == 0:
        return {"dataset": dataset_name, "strategy": strategy, "n": 0, "em": 0.0, "parse_failure_rate": 0.0}

    # Re-read the full CSV for this (dataset, strategy) pair so EM is correct
    # even after a resumed run where `predictions` only holds newly-run rows.
    try:
        df_full = pd.read_csv(results_path)
        df_run  = df_full[
            (df_full["dataset"] == dataset_name) &
            (df_full["strategy"] == strategy)
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
    except Exception:
        # Fallback: in-memory calc (only accurate on a fresh non-resumed run)
        em  = calculate_exact_match(predictions, [s["answer"] for s in samples], dataset_name)
        pfr = (parse_failures / n) * 100
        avg_prompt_tokens = avg_completion_tokens = avg_total_tokens = avg_latency = 0.0
        total_tokens_used = 0

    print(f"\n  ✓ Executed: {n} | EM = {em:.1f}% | Parse-failure rate = {pfr:.1f}%")
    print(f"  ⏱ Avg latency: {avg_latency:.1f}s | Avg total tokens: {avg_total_tokens:.0f} | Total tokens used: {total_tokens_used}")
    
    return {
    "dataset":             dataset_name,
    "strategy":            strategy,
    "n":                   n,
    "em":                  em,
    "parse_failure_rate":  pfr,
    "avg_prompt_tokens":     round(avg_prompt_tokens, 1),       # NEW
    "avg_completion_tokens": round(avg_completion_tokens, 1),   # NEW
    "avg_total_tokens":      round(avg_total_tokens, 1),        # NEW
    "total_tokens_used":     total_tokens_used,                 # NEW
    "avg_latency_s":         round(avg_latency, 2),             # NEW
    }

def load_already_done(results_dir: Path) -> set:
    done = set()
    for f in results_dir.glob("run_*.csv"):
        try:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                done.add((row["dataset"], row["strategy"], int(row["sample_idx"])))
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
    pivot = df.pivot_table(index="strategy", columns="dataset", values="em", aggfunc="mean")
    print(pivot.to_string())
    print(f"\nFull summary saved → {summary_path}")

    json_path = results_dir / "summary.json"
    json_path.write_text(json.dumps(summary_rows, indent=2))
    
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run prompting strategy experiments.")
    parser.add_argument("--datasets",    nargs="+", default=DATASETS,   choices=DATASETS)
    parser.add_argument("--strategies",  nargs="+", default=STRATEGIES, choices=STRATEGIES)
    parser.add_argument("--max_samples", type=int,  default=None)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3-super-120b-a12b:free")
    parser.add_argument("--max_tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()

def main():
    args = parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY environment variable not set.")
        sys.exit(1)
        
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"run_{TIMESTAMP}.csv"
    already_done = load_already_done(RESULTS_DIR) if args.resume else set()

    gsm8k, strategyqa = load_evaluation_datasets()
    dataset_map = {"gsm8k": gsm8k, "strategyqa": strategyqa}

    summary_rows = []
    for dataset_name in args.datasets:
        for strategy in args.strategies:
            result = run_experiment(
            client       = client,
            dataset_name = dataset_name,
            dataset      = dataset_map[dataset_name],
            strategy     = strategy,
            max_samples  = args.max_samples,
            results_path = results_path,
            already_done = already_done,
            model        = args.model,        # NEW
            max_tokens   = args.max_tokens,   # NEW
            temperature  = args.temperature,  # NEW
            )
            summary_rows.append(result)

    if summary_rows:
        save_summary(summary_rows, RESULTS_DIR)

if __name__ == "__main__":
    main()