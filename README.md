# In-Context Learning: Chain-of-Thought vs Few-Shot Prompting

Benchmarks three prompting strategies across two reasoning datasets using the OpenRouter API (free tier, `nvidia/nemotron-3-super-120b-a12b`).

---

## Project Structure

```
project-root/
├── runner.py                  ← Entry point — run this
├── requirements.txt
├── README.md
│
├── scripts/
│   ├── __init__.py
│   ├── data_loader.py         ← Loads GSM8K + StrategyQA from HuggingFace
│   └── metrics.py             ← Answer extraction + Exact Match scoring
│
├── prompts/
│   ├── __init__.py
│   └── templates.py           ← Prompt templates for all three strategies
│
└── results/                   ← Auto-created on first run
    ├── run_<timestamp>.csv    ← Per-sample predictions + correctness
    ├── summary.csv            ← Pivot table: strategy × dataset EM scores
    └── summary.json
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires **Python 3.9+**.

### 2. Get an OpenRouter API key

1. Go to [https://openrouter.ai](https://openrouter.ai) and create a free account
2. Navigate to **Keys** → **Create Key**
3. Copy the key (starts with `sk-or-...`)

### 3. Set the API key

**macOS / Linux:**
```bash
export OPENROUTER_API_KEY="sk-or-..."
```

**Windows (PowerShell):**
```powershell
$env:OPENROUTER_API_KEY="sk-or-..."
```

To make it permanent, add the export line to your `~/.bashrc` or `~/.zshrc`.

### 4. Create package init files

```bash
touch scripts/__init__.py
touch prompts/__init__.py
```

---

## Running Experiments

Always run from the **project root** directory.

```bash
# Smoke-test — 10 samples per strategy/dataset combination (recommended first)
python runner.py --max_samples 10

# Full run — all datasets, all strategies
python runner.py

# Single dataset
python runner.py --datasets gsm8k

# Single strategy
python runner.py --strategies zero_shot_cot

# Combined filter
python runner.py --datasets gsm8k --strategies zero_shot_cot --max_samples 50

# Resume an interrupted run (skips already-completed rows)
python runner.py --resume
```

### Final-submission ablation flags (Sections 6 & 7 of the progress report)

| Flag | Effect |
|---|---|
| `--structured` | Append an explicit `Answer: ...` instruction. Mitigation for the residual 10% parse-failure rate on StrategyQA. |
| `--k_shot N` | Override the default per-strategy demonstration count. Used for the 0/1/3/5-shot ablation. |
| `--cot_trigger {default,careful,none}` | Swap the CoT trigger phrase to measure trigger sensitivity. |
| `--persona_variant {revised,original,generic}` | Persona phrasing variant — `revised` (default) is the concise version called for in Section 6. |
| `--self_consistency N` | Sample `N` reasoning paths per prompt and majority-vote the answer (requires `--temperature > 0`). |

Example — recommended final-submission configuration on 100 samples:

```bash
python runner.py --structured --max_samples 100 --persona_variant revised
```

### Ablation sweep

Run the strategy × decoding × dataset matrix described in Section 7:

```bash
# Section 11 plan: max_tokens sweep + structured-on/off
python -m scripts.ablation --max_samples 50

# All ablation dimensions (k-shot, CoT trigger, persona, structured, max_tokens)
python -m scripts.ablation --max_samples 50 --full
```

### Error analysis

Categorise failures into truncation / reasoning / extraction (Section 8):

```bash
python -m scripts.error_analysis results/run_*.csv
```

---

## Prompting Strategies

| Strategy | Description |
|---|---|
| `standard_few_shot` | One Q/A demonstration before the test question |
| `zero_shot_cot` | Appends *"Let's think step by step."* — no examples |
| `persona_prompting` | Expert persona (mathematician / logician) + CoT trigger |

---

## Datasets

| Dataset | Task | Split | Size | Metric |
|---|---|---|---|---|
| GSM8K | Grade-school math word problems | test | 1,319 | Exact Match (numeric) |
| StrategyQA | Multi-hop commonsense QA | test | ~490 | Exact Match (yes / no) |

Datasets are downloaded automatically from HuggingFace on first run.

---

## Output

| File | Contents |
|---|---|
| `results/run_<timestamp>.csv` | One row per sample: question, gold answer, full generation, extracted answer, `is_correct`, `parse_failed` |
| `results/summary.csv` | EM (%) and parse-failure rate per strategy × dataset |
| `results/summary.json` | Same as summary.csv in JSON format |

The `parse_failed` column flags rows where the answer extractor returned an empty string — useful for error analysis separate from factual mistakes.

---

## Reproducibility

- Model: `nvidia/nemotron-3-super-120b-a12b:free` via OpenRouter
- Decoding: greedy (`temperature=0.0`, deterministic)
- All results are streamed to CSV row-by-row; interrupted runs can be resumed with `--resume`
