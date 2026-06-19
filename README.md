# In-Context Learning: Chain-of-Thought vs Few-Shot Prompting

A multi-model comparative study of three prompting strategies
(`standard_few_shot`, `zero_shot_cot`, `persona_prompting`) on two reasoning
datasets (GSM8K, StrategyQA), evaluated on **two open-weight models** through
the NVIDIA NIM API:

| Model | `--model` string |
|---|---|
| Meta Llama 3.3 70B Instruct | `meta/llama-3.3-70b-instruct` |
| NVIDIA Nemotron-3 Super 120B (default) | `nvidia/nemotron-3-super-120b-a12b` |

Both run through the same endpoint (`https://integrate.api.nvidia.com/v1`) so
the inference infrastructure is held constant across models.

The repository implements the full progress-report plan: the structured-output
revision, decoding-parameter ablation, k-shot count ablation, CoT-trigger
ablation, persona revision, self-consistency decoding, augmented evaluation
metrics, and categorised error analysis.

**Report (Overleaf, read-only):** <https://www.overleaf.com/read/rvqttrjjvwpw#0864ca>

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
│   ├── metrics.py                  ← Extraction, EM, majority_vote, and the
│   │                                 augmented metrics (substring, numeric,
│   │                                 token-F1, ROUGE-L)
│   ├── ablation.py                 ← Strategy × decoding × dataset sweep
│   ├── error_analysis.py           ← Failure categoriser (truncation /
│   │                                 extraction / reasoning)
│   ├── backfill_metrics.py         ← Retro-score old CSVs with the new metrics
│   └── score_bertscore.py          ← Opt-in BERTScore augmentation
│
├── tests/
│   └── test_metrics.py             ← Extractor, metric, template + vote tests
│
└── results/                        ← Auto-created on first run
    ├── run_<timestamp>.csv         ← Per-sample rows (one per condition)
    ├── run_ablation_<ts>.csv       ← Sweep outputs
    ├── summary.csv / summary.json  ← Per-condition aggregates (latest run)
    ├── summary_nemotron.csv/.json  ← Per-model summary (Nemotron)
    ├── summary_llama.csv/.json     ← Per-model summary (Llama)
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

> Requires **Python 3.9+**. `bert_score` is **not** in `requirements.txt` — it
> is only needed for the optional BERTScore step (see below).

### 2. Get an NVIDIA API key

1. Go to [https://build.nvidia.com](https://build.nvidia.com) and sign in.
2. Open the model page for either model and click **Get API Key**.
3. Copy the key (starts with `nvapi-...`).

### 3. Set the API key

**macOS / Linux:**
```bash
export NVIDIA_API_KEY="nvapi-..."
```

**Windows (PowerShell):**
```powershell
$env:NVIDIA_API_KEY="nvapi-..."
```

To make it permanent, add the export line to your `~/.bashrc` / `~/.zshrc`,
or use `setx NVIDIA_API_KEY ...` on Windows.

The runner uses the OpenAI-compatible NVIDIA endpoint:

```python
client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ["NVIDIA_API_KEY"],
    timeout=90.0,   # fail fast on hung connections; call_llm retries
)
```

An invalid key or unknown model surfaces as a `FatalAPIError` and aborts the
run immediately (rather than burning samples on un-retryable 4xx errors).

### NVIDIA NIM trial-tier limits

The trial tier enforces, for both models:

| Limit | Value | How the project handles it |
|---|---|---|
| Requests per minute | **40 RPM** (hard cap) | `RATE_LIMIT_SLEEP` in `runner.py` is `60 / 40 + 0.1 ≈ 1.6 s`, giving ≈ 37 RPM steady-state with headroom for latency jitter. |
| Context window (input + output) | **16,384 tokens** | The runner warns if `--max_tokens` leaves < 2 k tokens for the prompt. Keep `--max_tokens ≤ 1500` (the default) for the n = 100 runs. |

> **Throughput note.** At ≈37 RPM, one model's full grid (`2 datasets × 3
> strategies × 100 samples = 600 requests`) takes roughly **17 minutes**.
> Both models is ~34 minutes; self-consistency multiplies by the number of
> sampled paths.

### 4. Create package init files (first checkout only)

```bash
touch scripts/__init__.py
touch prompts/__init__.py
```

---

## Running Experiments

Always run from the **project root**.

### Per-model runs

```bash
# Nemotron (default model)
python runner.py --structured --max_samples 100

# Llama 3.3 70B — same command, different --model
python runner.py --structured --max_samples 100 --model "meta/llama-3.3-70b-instruct"
```

Each row records the active model inside its `condition` tag (see **Output**),
so runs from different models can share one CSV and still be separated cleanly.

### Other run patterns

```bash
# Smoke test — 10 samples per (strategy × dataset)
python runner.py --max_samples 10

# Single dataset / single strategy
python runner.py --datasets gsm8k
python runner.py --strategies zero_shot_cot

# Resume an interrupted run (skips already-completed
# (dataset, strategy, sample_idx, condition) tuples across all run_*.csv files)
python runner.py --structured --max_samples 100 --resume
```

### Flags

| Flag | Effect |
|---|---|
| `--model STR` | Model string (default `nvidia/nemotron-3-super-120b-a12b`). Set to `meta/llama-3.3-70b-instruct` for the Llama run. |
| `--structured` | Append the explicit `Answer: ...` instruction. Primary mitigation for StrategyQA parse failures. |
| `--k_shot N` | Override the default per-strategy demonstration count (`0` / `1` / `3` / `5`). |
| `--cot_trigger {default,careful,none}` | Swap the CoT trigger phrase. |
| `--persona_variant {revised,original,generic}` | Persona phrasing. `revised` (default) is the concise version; `original` reproduces the initial-checkpoint prompt. |
| `--self_consistency N` | Sample `N` reasoning paths per prompt and majority-vote. Requires `--temperature > 0`. |
| `--max_tokens N` | Output length budget (default 1500). Sweep target for the decoding ablation. |
| `--temperature F` | Sampling temperature. `0.0` (default) is greedy/deterministic; raise for self-consistency. |

### Ablation sweep

`scripts/ablation.py` runs the strategy × decoding × dataset matrix in-process,
streaming every cell to one `run_ablation_<timestamp>.csv`.

```bash
# Default: max_tokens × structured(on/off) sweep
python -m scripts.ablation --max_samples 50

# A subset of dimensions, or all of them
python -m scripts.ablation --dims k_shot cot_trigger --max_samples 50
python -m scripts.ablation --max_samples 50 --full

# Sweep on Llama instead of the default model
python -m scripts.ablation --max_samples 50 --model "meta/llama-3.3-70b-instruct"
```

Available dims: `max_tokens`, `k_shot`, `cot_trigger`, `persona_variant`, `structured`.

### Error analysis

Categorises every incorrect row as **truncation** (cut off before the answer),
**extraction** (gold answer present but parser missed it), or **reasoning**
(wrong conclusion). When the input spans multiple models, the summary table is
broken down by model.

```bash
python -m scripts.error_analysis results/run_*.csv
# → results/error_analysis/errors.csv   (one row per failure)
#   results/error_analysis/errors_summary.csv  (counts per cell)
```

---

## Evaluation Metrics

Exact Match (EM) is the primary metric; the rest are robustness checks logged
on every row.

| Metric | What it measures | Scope |
|---|---|---|
| `is_correct` (EM) | Extracted prediction equals gold after normalization. | Scalar vs scalar. |
| `parse_failed` | Extractor returned an empty string. | Scalar. |
| `substring_match` | Gold appears verbatim inside the prediction. | Scalar vs scalar — relaxes EM (e.g. `Answer: 36.36` vs gold `36`). |
| `is_numeric_close` | `|pred − gold| ≤ 0.5`, GSM8K only. | Numeric — separates rounding from real arithmetic errors. |
| `token_f1` | Word-level F1 of token multisets. | Full generation vs full gold text. |
| `rouge_l` | F1 of the longest common subsequence. | Same scope as `token_f1`. |
| `bertscore_f1` (opt-in) | Contextual-embedding similarity. | Same scope; computed by `score_bertscore.py`. |

> `token_f1` / `rouge_l` / `bertscore_f1` are **generation-quality** signals,
> not answer-correctness measures — on scalar/boolean gold they degenerate to
> a length-penalized presence signal. EM (with substring + numeric-tolerance
> as sanity checks) is the metric to read for accuracy.

### Backfilling metrics onto older CSVs

```bash
python -m scripts.backfill_metrics results/run_*.csv
```

Recomputes `substring_match`, `is_numeric_close`, `token_f1`, `rouge_l` from
the stored `generation` / `gold_answer` columns — no API calls. (On historical
CSVs the comparison is against the extracted scalar, since the full gold chain
isn't stored there.)

### BERTScore (opt-in)

```bash
pip install bert_score
python -m scripts.score_bertscore results/run_*.csv
```

Writes a `bertscore_f1` column back into each CSV; skips already-scored rows
unless `--overwrite` is passed.

---

## Prompting Strategies

| Strategy | Description |
|---|---|
| `standard_few_shot` | Up to *k* Q/A demonstrations before the test question. `k = 1` by default; override with `--k_shot`. |
| `zero_shot_cot` | No examples; appends *"Let's think step by step."* (or the chosen `--cot_trigger`). |
| `persona_prompting` | Expert-persona preamble + CoT trigger. `revised` variant (concise, with explicit-answer requirement) is the default. |

---

## Datasets

| Dataset | Task | Split | Size | Metric |
|---|---|---|---|---|
| GSM8K (`openai/gsm8k`, `main`) | Grade-school math word problems | `test` | 1,319 | Exact Match (numeric) |
| StrategyQA (`ChilleD/StrategyQA`) | Multi-hop commonsense QA | `test` | 687 | Exact Match (yes / no) |

The first 100 samples of each split are evaluated per condition, selected
deterministically. Both download automatically from HuggingFace on first run.
(The originally planned `lucasmccabe-lmi/strategyqa` mirror was removed from
the Hub mid-project; `ChilleD/StrategyQA` exposes an identical schema.)

---

## Output

| File | Contents |
|---|---|
| `results/run_<timestamp>.csv` | One row per sample: prediction, EM, the four augmented metrics, token counts, latency, and the `condition` tag. |
| `results/run_ablation_<timestamp>.csv` | Same schema, written by `scripts/ablation.py`. |
| `results/summary.csv` / `.json` | Per-condition aggregates for the latest `runner.py` invocation. |
| `results/summary_nemotron.{csv,json}` | Per-cell summary for the Nemotron run (EM, PFR, substring/numeric rates, token-F1/ROUGE-L, tokens, latency). |
| `results/summary_llama.{csv,json}` | Same, for the Llama run. |
| `results/error_analysis/errors.csv` | Failures classified as truncation / extraction / reasoning. |
| `results/error_analysis/errors_summary.csv` | Per-cell (and per-model when applicable) failure counts. |

The `condition` column is a compact tag recording the full ablation cell —
**including the model** — so a single CSV can hold many cells unambiguously:

```
model=meta/llama-3.3-70b-instruct,k=def,trig=default,persona=revised,struct=on,mt=1500,T=0.0,sc=1
```

---

## Reproducibility

- Two models, one endpoint: `nvidia/nemotron-3-super-120b-a12b` (default) and
  `meta/llama-3.3-70b-instruct`, both via the NVIDIA NIM API
  (`https://integrate.api.nvidia.com/v1`); switch with `--model`.
- Default decoding: greedy (`temperature=0.0`); deterministic across runs.
- 16,384-token context window and 40 RPM trial-tier cap; see *NVIDIA NIM
  trial-tier limits* above.
- Per-sample rows stream to CSV row-by-row; interrupted runs resume with
  `--resume`, de-duplicating on `(dataset, strategy, sample_idx, condition)` —
  and since `condition` carries the model, resuming never conflates models.
- Few-shot demonstrations are drawn deterministically from a fixed pool
  (`FEW_SHOT_POOL` in `prompts/templates.py`).

---

## Tests

```bash
python -m pytest tests/ -q
```

33 tests covering the priority-chain extractor (structured `Answer:` line,
`####` marker, conclusion keywords, last-number fallback), the augmented
metrics (substring, numeric-tolerance, token-F1, ROUGE-L edge cases),
majority-vote self-consistency, and prompt-template construction (k-shot pool
length, zero-shot example absence, persona-variant divergence).
