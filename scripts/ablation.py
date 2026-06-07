"""
================================================

Builds the strategy × decoding × dataset matrix called for in Section 7 of
the progress report.  Rather than spawning subprocesses, each cell is invoked
in-process via `runner.run_experiment`, which streams to the same
`run_<timestamp>.csv` and aggregates per-condition summaries.

The sweep dimensions are:

  * max_tokens         ∈ {256, 512, 1024, 1500}   (decoding-parameter ablation)
  * k_shot             ∈ {0, 1, 3, 5}             (few-shot count ablation)
  * cot_trigger        ∈ {default, careful, none} (trigger-phrase ablation)
  * persona_variant    ∈ {revised, original, generic}
  * structured         ∈ {True, False}            (structured-output revision)

By default the script runs only the dimensions explicitly enumerated in
Section 11's "Plan Through Final Submission" — namely the structured-output
revision and the decoding-parameter sweep.  Pass --full to run every
dimension above.

Example
-------
    python -m scripts.ablation --max_samples 50 --dims max_tokens structured
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List

from openai import OpenAI

from runner import (
    RESULTS_DIR,
    FatalAPIError,
    load_already_done,
    run_experiment,
    save_summary,
    _condition_label,
)
from scripts.data_loader import load_evaluation_datasets


DIM_VALUES: Dict[str, list] = {
    "max_tokens":      [256, 512, 1024, 1500],
    "k_shot":          [0, 1, 3, 5],
    "cot_trigger":     ["default", "careful", "none"],
    "persona_variant": ["revised", "original", "generic"],
    "structured":      [False, True],
}

DEFAULT_DIMS = ["max_tokens", "structured"]


class _Args:
    """Lightweight stand-in for argparse.Namespace, fed to `_condition_label`."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _iter_conditions(active_dims, base):
    """Yield (overrides_dict,) for the Cartesian product of active dims."""
    values = [DIM_VALUES[d] for d in active_dims]
    for combo in product(*values):
        overrides = dict(zip(active_dims, combo))
        cfg = dict(base)
        cfg.update(overrides)
        yield cfg


def main():
    parser = argparse.ArgumentParser(description="Section 7 ablation sweep.")
    parser.add_argument("--datasets",   nargs="+", default=["gsm8k", "strategyqa"])
    parser.add_argument("--strategies", nargs="+",
                        default=["standard_few_shot", "zero_shot_cot", "persona_prompting"])
    parser.add_argument("--max_samples", type=int, default=50,
                        help="Samples per (strategy, condition) cell.")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3-super-120b-a12b")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dims", nargs="+", default=DEFAULT_DIMS, choices=list(DIM_VALUES.keys()),
        help="Which dimensions to sweep.  Defaults to the Section 11 plan: "
             "max_tokens and structured.",
    )
    parser.add_argument("--full", action="store_true",
                        help="Sweep every dimension in DIM_VALUES.")
    args = parser.parse_args()

    active_dims = list(DIM_VALUES.keys()) if args.full else args.dims

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        print("[ERROR] NVIDIA_API_KEY environment variable not set.")
        sys.exit(1)
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key,
        timeout=90.0,   # see runner.main() for rationale
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"run_ablation_{timestamp}.csv"
    already_done = load_already_done(RESULTS_DIR) if args.resume else set()

    gsm8k, strategyqa = load_evaluation_datasets()
    dataset_map = {"gsm8k": gsm8k, "strategyqa": strategyqa}

    # Sensible defaults for fields that are not part of the sweep.
    base = {
        "max_tokens":      1500,
        "k_shot":          None,
        "cot_trigger":     "default",
        "persona_variant": "revised",
        "structured":      True,   # Section 6 revision: on by default for the sweep.
        "temperature":     0.0,
        "self_consistency": 1,
    }

    summary_rows: List[dict] = []
    try:
        for cfg in _iter_conditions(active_dims, base):
            condition = _condition_label(_Args(**cfg))
            for dataset_name in args.datasets:
                for strategy in args.strategies:
                    result = run_experiment(
                        client            = client,
                        dataset_name      = dataset_name,
                        dataset           = dataset_map[dataset_name],
                        strategy          = strategy,
                        max_samples       = args.max_samples,
                        results_path      = results_path,
                        already_done      = already_done,
                        model             = args.model,
                        max_tokens        = cfg["max_tokens"],
                        temperature       = cfg["temperature"],
                        structured        = cfg["structured"],
                        k_shot            = cfg["k_shot"],
                        cot_trigger       = cfg["cot_trigger"],
                        persona_variant   = cfg["persona_variant"],
                        self_consistency  = cfg["self_consistency"],
                        condition         = condition,
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
