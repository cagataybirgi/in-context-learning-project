# prompts/templates.py
"""
Prompt templates for the in-context learning benchmark.

This module supports the three baseline strategies described in the progress
report (standard_few_shot, zero_shot_cot, persona_prompting) plus the ablation
dimensions enumerated in Sections 6 and 7 of the report:

  * Structured-output constraint  ("Answer: ..." line) — primary mitigation
    for the residual ~10% parse-failure rate on StrategyQA.
  * Few-shot example count (k in {0, 1, 3, 5}) drawn from a small pool.
  * CoT trigger phrase variants ("step by step", "carefully", or none).
  * Revised persona that asks for concise reasoning and an explicit
    answer line — addresses the verbosity / truncation failures.

`get_prompt(...)` is backwards compatible with the original three-argument
signature used by `runner.py`.
"""

from typing import List, Optional

# ---------------------------------------------------------------------------
# Few-shot example pools
# ---------------------------------------------------------------------------
# Five fixed demonstrations per dataset.  `k`-shot prompting selects a prefix
# of length k from the pool, preserving order so runs are deterministic.

FEW_SHOT_POOL = {
    "gsm8k": [
        {
            "question": (
                "There are 15 trees in the grove. Grove workers will plant trees "
                "in the grove today. After they are done, there will be 21 trees. "
                "How many trees did the grove workers plant today?"
            ),
            "answer": "6",
        },
        {
            "question": (
                "If there are 3 cars in the parking lot and 2 more cars arrive, "
                "how many cars are in the parking lot?"
            ),
            "answer": "5",
        },
        {
            "question": (
                "Leah had 32 chocolates and her sister had 42. If they ate 35, "
                "how many pieces do they have left in total?"
            ),
            "answer": "39",
        },
        {
            "question": (
                "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
                "has 12 lollipops. How many lollipops did Jason give to Denny?"
            ),
            "answer": "8",
        },
        {
            "question": (
                "Shawn has five toys. For Christmas, he got two toys each from his "
                "mom and dad. How many toys does he have now?"
            ),
            "answer": "9",
        },
    ],
    "strategyqa": [
        {"question": "Do hamsters provide food for any animals?", "answer": "Yes"},
        {"question": "Could a llama birth twice during War in Vietnam (1945-46)?", "answer": "No"},
        {"question": "Would a pear sink in water?", "answer": "No"},
        {"question": "Is the language used in Saint Vincent and the Grenadines rooted in English?", "answer": "Yes"},
        {"question": "Did Aristotle use a laptop?", "answer": "No"},
    ],
}

# ---------------------------------------------------------------------------
# CoT trigger variants
# ---------------------------------------------------------------------------

COT_TRIGGERS = {
    "default":  "Let's think step by step.",
    "careful":  "Let's solve this carefully.",
    "none":     "",
}

# ---------------------------------------------------------------------------
# Structured-output instruction
# ---------------------------------------------------------------------------
# Appended after the question (and CoT trigger) when `structured=True`.

_STRUCTURED_TAIL = {
    "gsm8k": (
        "\n\nWhen you are done reasoning, end your response with exactly one line "
        'of the form "Answer: <number>" (digits only, no units, no commas).'
    ),
    "strategyqa": (
        "\n\nWhen you are done reasoning, end your response with exactly one line "
        'of the form "Answer: Yes" or "Answer: No".'
    ),
}

# ---------------------------------------------------------------------------
# Persona preambles
# ---------------------------------------------------------------------------
# Two variants per dataset.  `original` is the version used in the initial
# 10-sample evaluation; `revised` is the concise version called for in
# Section 6 of the progress report.

PERSONA_PREAMBLES = {
    "gsm8k": {
        "original": (
            "You are an advanced mathematician recognized for precise, logical "
            "calculations. Solve the following mathematical problem with absolute "
            "accuracy."
        ),
        "revised": (
            "You are a careful mathematician. Solve the problem with brief, "
            "numbered steps (no more than five). Do not restate the problem and "
            "do not add commentary."
        ),
        "generic": (
            "You are an expert. Solve the problem accurately."
        ),
    },
    "strategyqa": {
        "original": (
            "You are an advanced logician with deep general knowledge. Evaluate "
            "the following question and provide a logically sound answer."
        ),
        "revised": (
            "You are a careful reasoner. Use at most three short steps to decide "
            "the answer. Do not add background commentary."
        ),
        "generic": (
            "You are an expert. Answer the question accurately."
        ),
    },
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _format_few_shot_block(dataset: str, k: int) -> str:
    """Return a Q/A demonstration block of length `k` (possibly empty)."""
    if k <= 0:
        return ""
    pool = FEW_SHOT_POOL[dataset]
    examples = pool[: min(k, len(pool))]
    return "\n\n".join(f"Q: {ex['question']}\nA: {ex['answer']}" for ex in examples)


def build_prompt(
    dataset: str,
    strategy: str,
    question: str,
    *,
    k_shot: Optional[int] = None,
    cot_trigger: str = "default",
    persona_variant: str = "revised",
    structured: bool = False,
) -> str:
    """
    Construct a prompt with all ablation dimensions exposed.

    Parameters
    ----------
    dataset      : 'gsm8k' or 'strategyqa'.
    strategy     : 'standard_few_shot', 'zero_shot_cot', or 'persona_prompting'.
    question     : The test-time question.
    k_shot       : Number of demonstrations.  Defaults to 1 for
                   `standard_few_shot` and 0 otherwise.
    cot_trigger  : One of `COT_TRIGGERS` keys.  Only used by the two
                   CoT-bearing strategies.
    persona_variant : One of `PERSONA_PREAMBLES[dataset]` keys.
    structured   : If True, append the explicit-answer-line instruction.
    """
    dataset = dataset.lower()
    strategy = strategy.lower()

    if dataset not in FEW_SHOT_POOL:
        raise ValueError(f"Unknown dataset '{dataset}'.")
    if strategy not in {"standard_few_shot", "zero_shot_cot", "persona_prompting"}:
        raise ValueError(f"Unknown strategy '{strategy}'.")

    if k_shot is None:
        k_shot = 1 if strategy == "standard_few_shot" else 0

    parts: List[str] = []

    if strategy == "persona_prompting":
        parts.append(PERSONA_PREAMBLES[dataset][persona_variant])

    fs_block = _format_few_shot_block(dataset, k_shot)
    if fs_block:
        parts.append(fs_block)

    trigger = COT_TRIGGERS.get(cot_trigger, COT_TRIGGERS["default"])
    if strategy == "standard_few_shot":
        # Few-shot baseline: no CoT trigger; the demonstration sets the format.
        question_block = f"Q: {question}\nA:"
    else:
        # zero_shot_cot and persona_prompting both end with an answer cue plus
        # (optionally) the trigger phrase.
        if trigger:
            question_block = f"Q: {question}\nA: {trigger}"
        else:
            question_block = f"Q: {question}\nA:"
    parts.append(question_block)

    prompt = "\n\n".join(p for p in parts if p)

    if structured:
        prompt += _STRUCTURED_TAIL[dataset]

    return prompt


def get_prompt(dataset: str, strategy: str, question: str, **kwargs) -> str:
    """
    Backwards-compatible entry point used by `runner.py`.

    Passes any additional keyword arguments straight through to
    `build_prompt`, so the runner can opt in to the structured-output or
    ablation variants by supplying them.
    """
    return build_prompt(dataset, strategy, question, **kwargs)


# Kept for callers that still introspect the original mapping.  These are the
# *exact* templates used to produce the initial-checkpoint numbers; do not
# edit without bumping the report's results section.
PROMPT_TEMPLATES = {
    ds: {
        "standard_few_shot": build_prompt(ds, "standard_few_shot", "{question}").replace(
            "{question}", "{question}"
        ),
        "zero_shot_cot": build_prompt(ds, "zero_shot_cot", "{question}"),
        "persona_prompting": build_prompt(
            ds, "persona_prompting", "{question}", persona_variant="original"
        ),
    }
    for ds in ("gsm8k", "strategyqa")
}
