# In-Context Learning: Chain-of-Thought vs Few-Shot Prompting

Benchmarks three prompting strategies (`standard_few_shot`, `zero_shot_cot`,
`persona_prompting`) on two reasoning datasets (GSM8K, StrategyQA) using the
OpenRouter API (default model: `nvidia/nemotron-3-super-120b-a12b:free`).

The repository implements the full progress-report plan, including the
structured-output revision, decoding-parameter ablation, k-shot count
ablation, CoT-trigger ablation, persona revision, self-consistency
decoding, and categorised error analysis.

---

## Project Structure

```
project-root/
├── runner.py                       ← Main entry point
├── requirements.txt
├── README.md
│
├── prompts/
│   └── templates.py                ← Strategies + ablation knobs
│                                     (k-shot pool, CoT triggers,
│                                     persona variants, structured tail)
│
├── scripts/
│   ├── data_loader.py              ← Loads GSM8K + StrategyQA (HuggingFace)
│   ├── metrics.py                  ← Answer extraction + EM + majority_vote
│   ├── ablation.py                 ← Section 7 sweep driver
│   └── error_analysis.py           ← Section 8 failure categoriser
│
├── tests/
│   └── test_metrics.py             ← Extractor + template + vote tests
│
└── results/                        ← Auto-created on first run
    ├── run_<timestamp>.csv         ← Per-sample rows (one per condition)
    ├── run_ablation_<ts>.csv       ← Sweep outputs
    ├── summary.csv                 ← Per-condition EM / PFR
    ├── summary.json
    └── error_analysis/             ← Created by error_analysis.py
        ├── errors.csv
        └── errors_summary.csv
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires **Python 3.9+**.

### 2. Get an OpenRouter API key

1. Go to [https://openrouter.ai](https://openrouter.ai) and create a free account.
2. Navigate to **Keys → Create Key**.
3. Copy the key (starts with `sk-or-...`).

### 3. Set the API key

**macOS / Linux:**
```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
```

To make it permanent, add the export line to your `~/.bashrc` / `~/.zshrc`,
or use `setx OPENROUTER_API_KEY ...` on Windows.

### 4. Create package init files (first checkout only)

```bash
touch scripts/__init__.py
touch prompts/__init__.py
```

---

## Running Experiments

Always run from the **project root**.

### Baseline runs

```bash
# Smoke test — 10 samples per (strategy × dataset)
python runner.py --max_samples 10

# Full run — all datasets, all strategies, all samples
python runner.py

# Single dataset / single strategy
python runner.py --datasets gsm8k
python runner.py --strategies zero_shot_cot

# Combined filter
python runner.py --datasets gsm8k --strategies zero_shot_cot --max_samples 50

# Resume an interrupted run (skips already-completed (dataset, strategy,
# sample_idx, condition) tuples across all run_*.csv files)
python runner.py --resume
```

### Ablation flags (Sections 6 & 7)

| Flag | Effect |
|---|---|
| `--structured` | Append the explicit `Answer: ...` instruction. Primary mitigation for the residual 10% parse-failure rate on StrategyQA. |
| `--k_shot N` | Override the default per-strategy demonstration count (`0` / `1` / `3` / `5`). |
| `--cot_trigger {default,careful,none}` | Swap the CoT trigger phrase to measure trigger sensitivity. |
| `--persona_variant {revised,original,generic}` | Persona phrasing variant. `revised` (default) is the concise version called for in Section 6; `original` reproduces the initial-checkpoint prompt. |
| `--self_consistency N` | Sample `N` reasoning paths per prompt and majority-vote the answer. Requires `--temperature > 0`. |
| `--max_tokens N` | Output length budget. Sweep target in the decoding-parameter ablation. |
| `--temperature F` | Sampling temperature. Use `0.0` (default) for greedy / deterministic runs; raise for self-consistency. |

**Recommended final-submission configuration (n = 100 per cell):**

```bash
python runner.py --structured --max_samples 100
```

**Self-consistency example (5 sampled paths, GSM8K only):**

```bash
python runner.py \
  --structured \
  --datasets gsm8k --strategies zero_shot_cot \
  --self_consistency 5 --temperature 0.7 \
  --max_samples 100
```

### Ablation sweep (Section 7 matrix)

`scripts/ablation.py` runs the strategy × decoding × dataset matrix in-process
and streams every cell to one `run_ablation_<timestamp>.csv` so the resume
logic still applies.

```bash
# Section 11 plan: max_tokens × structured(on/off) sweep
python -m scripts.ablation --max_samples 50

# Sweep a specific subset of dimensions
python -m scripts.ablation --dims k_shot cot_trigger --max_samples 50

# Sweep every dimension (max_tokens × k_shot × cot_trigger × persona × structured)
python -m scripts.ablation --max_samples 50 --full
```

Available dims: `max_tokens`, `k_shot`, `cot_trigger`, `persona_variant`, `structured`.

### Error analysis (Section 8)

Categorises every incorrect row into one of three buckets:

* **truncation** — the generation looks cut off (no terminal punctuation, no
  `Answer:` cue, and the recorded `completion_tokens` is at the budget).
* **extraction** — the gold answer appears in the generation but the
  extractor returned something else (the "right answer, wrong format" case).
* **reasoning** — the model produced a clean output that does not contain
  the gold answer.

```bash
# Categorise failures from one or more run files (globs allowed)
python -m scripts.error_analysis results/run_*.csv

# Output:
#   results/error_analysis/errors.csv          ← one row per failure
#   results/error_analysis/errors_summary.csv  ← per-strategy counts
```

---

## Prompting Strategies

| Strategy | Description |
|---|---|
| `standard_few_shot` | Up to *k* Q/A demonstrations before the test question. `k = 1` by default; override with `--k_shot`. |
| `zero_shot_cot` | No examples; appends *"Let's think step by step."* (or the chosen `--cot_trigger`). |
| `persona_prompting` | Expert-persona preamble + CoT trigger. `revised` variant (concise, with explicit-answer requirement) is the new default. |

---

## Datasets

| Dataset | Task | Split | Size | Metric |
|---|---|---|---|---|
| GSM8K | Grade-school math word problems | `test` | 1,319 | Exact Match (numeric) |
| StrategyQA (`ChilleD/StrategyQA`) | Multi-hop commonsense QA | `test` | 687 | Exact Match (yes / no) |

Both are downloaded automatically from HuggingFace on first run.

---

## Output

| File | Contents |
|---|---|
| `results/run_<timestamp>.csv` | One row per sample, including `condition` (compact ablation tag), token counts, and latency. |
| `results/run_ablation_<timestamp>.csv` | Same schema, written by `scripts/ablation.py`. |
| `results/summary.csv` | EM (%) and parse-failure rate per (dataset × strategy × condition). |
| `results/summary.json` | Same as `summary.csv` in JSON form. |
| `results/error_analysis/errors.csv` | Failures classified as truncation / extraction / reasoning. |
| `results/error_analysis/errors_summary.csv` | Counts of each failure category per strategy. |

The `parse_failed` column flags rows where the extractor returned an empty
string — useful for separating format compliance failures from factual
mistakes. The `condition` column records the ablation cell (k-shot count,
CoT trigger, persona variant, structured on/off, max_tokens, temperature,
self-consistency `n`) so a single CSV can hold many sweep cells.

---

## Reproducibility

- Default model: `nvidia/nemotron-3-super-120b-a12b:free` (override with `--model`).
- Default decoding: greedy (`temperature=0.0`); deterministic across runs.
- Per-sample rows are streamed to CSV row-by-row; interrupted runs resume
  with `--resume`, which de-duplicates on
  `(dataset, strategy, sample_idx, condition)`.
- Few-shot demonstrations are drawn deterministically from a fixed pool
  (`FEW_SHOT_POOL` in `prompts/templates.py`).

---

## Tests

```bash
python -m pytest tests/ -q
```

The suite covers the priority-chain extractor (structured `Answer:` line,
`####` marker, conclusion keywords, last-number fallback), majority-vote
self-consistency, and the prompt-template construction (k-shot pool length,
zero-shot example absence, persona-variant divergence).
