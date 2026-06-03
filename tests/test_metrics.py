from scripts.metrics import (
    extract_gsm8k_answer,
    extract_strategyqa_answer,
    majority_vote,
)


# ---------------------------------------------------------------------------
# GSM8K extractor
# ---------------------------------------------------------------------------

def test_gsm8k_hash_marker():
    assert extract_gsm8k_answer("Step 1: ... #### 42") == "42"


def test_gsm8k_prefers_answer_keyword_over_last_number():
    text = "The answer is 45. John drove for 4 hours."
    assert extract_gsm8k_answer(text) == "45"


def test_gsm8k_comma_handling():
    assert extract_gsm8k_answer("#### 1,000") == "1000"


def test_gsm8k_structured_answer_line_wins():
    # Structured Answer: line takes priority even when other digits appear later.
    text = (
        "Step 1: 12 + 30 = 42. Step 2: This equals 42 cookies.\n"
        "Answer: 42\nSome trailing chatter mentioning 99 unrelated things."
    )
    assert extract_gsm8k_answer(text) == "42"


def test_gsm8k_structured_overrides_hash_marker():
    text = "Working out... #### 7\nAnswer: 8"
    assert extract_gsm8k_answer(text) == "8"


# ---------------------------------------------------------------------------
# StrategyQA extractor
# ---------------------------------------------------------------------------

def test_strategyqa_final_word():
    assert extract_strategyqa_answer("Long reasoning ... so the answer is yes.") == "yes"


def test_strategyqa_empty_on_no_match():
    assert extract_strategyqa_answer("I cannot determine this.") == ""


def test_strategyqa_structured_answer_line():
    text = "Long winding reasoning that ends inconclusively.\nAnswer: No"
    assert extract_strategyqa_answer(text) == "no"


def test_strategyqa_structured_overrides_trailing_yes():
    # The structured line should beat a final "yes" later in the chain.
    text = "Therefore yes seems plausible.\nAnswer: No"
    assert extract_strategyqa_answer(text) == "no"


# ---------------------------------------------------------------------------
# Majority vote (self-consistency)
# ---------------------------------------------------------------------------

def test_majority_vote_picks_majority():
    assert majority_vote(["yes", "no", "yes"]) == "yes"


def test_majority_vote_ignores_empty_candidates():
    assert majority_vote(["", "", "no"]) == "no"


def test_majority_vote_all_empty_returns_empty():
    assert majority_vote(["", "", ""]) == ""


# ---------------------------------------------------------------------------
# Prompt templates (k-shot / structured / persona variants)
# ---------------------------------------------------------------------------

def test_structured_prompt_adds_answer_instruction():
    from prompts.templates import build_prompt
    prompt = build_prompt("gsm8k", "zero_shot_cot", "What is 2+2?", structured=True)
    assert "Answer:" in prompt


def test_k_shot_pool_length():
    from prompts.templates import build_prompt
    prompt = build_prompt("gsm8k", "standard_few_shot", "Q?", k_shot=3)
    # Three demonstration "Q:" markers plus the test-time question = 4.
    assert prompt.count("Q:") == 4


def test_zero_shot_has_no_examples():
    from prompts.templates import build_prompt
    prompt = build_prompt("strategyqa", "zero_shot_cot", "Does X imply Y?")
    assert prompt.count("Q:") == 1


def test_persona_variants_differ():
    from prompts.templates import build_prompt
    orig = build_prompt("gsm8k", "persona_prompting", "Q", persona_variant="original")
    rev  = build_prompt("gsm8k", "persona_prompting", "Q", persona_variant="revised")
    assert orig != rev
