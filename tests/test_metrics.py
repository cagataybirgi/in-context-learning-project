from scripts.metrics import extract_gsm8k_answer, extract_strategyqa_answer

def test_gsm8k_hash_marker():
    assert extract_gsm8k_answer("Step 1: ... #### 42") == "42"

def test_gsm8k_prefers_answer_keyword_over_last_number():
    # The exact failure mode you debugged: "John is 45 miles from home after 4 hours"
    text = "The answer is 45. John drove for 4 hours."
    assert extract_gsm8k_answer(text) == "45"

def test_gsm8k_comma_handling():
    assert extract_gsm8k_answer("#### 1,000") == "1000"

def test_strategyqa_final_word():
    assert extract_strategyqa_answer("Long reasoning ... so the answer is yes.") == "yes"

def test_strategyqa_empty_on_no_match():
    assert extract_strategyqa_answer("I cannot determine this.") == ""