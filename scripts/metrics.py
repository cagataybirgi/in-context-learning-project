# scripts/metrics.py
"""
Answer extraction and Exact Match scoring.

Section 6 of the progress report calls for a structured-output prompt revision
that asks the model to end with an explicit `Answer: <value>` line, paired
with an extractor that prioritises that line.  This module implements that
priority chain while preserving the previous fall-back heuristics, so old
generations (and old test fixtures) still parse correctly.
"""

import re
from collections import Counter
from typing import Iterable, List, Optional

# Regex used by the structured-output mode.  Matches the *last* occurrence of
# `Answer: <value>` in the generation (a model that reasons out loud will
# often produce a draft "Answer:" mid-chain and overwrite it at the end).
_ANSWER_LINE_RE = re.compile(r"answer\s*[:=]\s*([^\n\r]+)", re.IGNORECASE)


def _structured_answer(text: str) -> Optional[str]:
    """Return the content of the final `Answer:` line, or None if absent."""
    matches = _ANSWER_LINE_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip().rstrip(".!?, ").strip()


def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the final numerical answer from a GSM8K generation.

    Priority chain:
      1. Structured `Answer: <number>` line (new — Section 6 mitigation).
      2. Reference-style `#### <number>` marker (GSM8K gold format).
      3. Number adjacent to a conclusion keyword (answer/total/therefore).
      4. Final number anywhere in the text.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1) Structured answer line.
    structured = _structured_answer(text)
    if structured:
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", structured)
        if nums:
            return nums[-1].replace(",", "")

    # 2) GSM8K reference marker.
    if "####" in text:
        tail = text.split("####")[-1].strip()
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", tail)
        if nums:
            return nums[-1].replace(",", "")

    # 3) Number near a conclusion keyword.
    keyword_hits = re.findall(
        r"(?:answer|total|therefore|so)[^\d-]{0,15}([-+]?\d[\d,]*(?:\.\d+)?)",
        text.lower(),
    )
    if keyword_hits:
        return keyword_hits[-1].replace(",", "")

    # 4) Last number anywhere.
    nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if nums:
        return nums[-1].replace(",", "")
    return ""


def extract_strategyqa_answer(text: str) -> str:
    """
    Extracts the boolean Yes/No answer from a StrategyQA generation.

    Priority chain:
      1. Structured `Answer: Yes|No` line (Section 6 mitigation).
      2. Explicit conclusion statements ("answer is yes", "therefore, no").
      3. Final word of the generation.
      4. Last yes/no token anywhere.
    """
    if not isinstance(text, str):
        text = str(text)
    raw_lower = text.lower().strip()

    # 1) Structured answer line.
    structured = _structured_answer(text)
    if structured is not None:
        s = structured.lower().strip().rstrip(".!?, ")
        first_token = s.split()[0] if s.split() else ""
        if first_token in {"yes", "true"}:
            return "yes"
        if first_token in {"no", "false"}:
            return "no"

    # 2) Explicit conclusion statements.
    if "answer is yes" in raw_lower or "therefore, yes" in raw_lower:
        return "yes"
    if "answer is no" in raw_lower or "therefore, no" in raw_lower:
        return "no"

    # 3) Final word.
    words = raw_lower.rstrip(".!?, ").split()
    if words:
        final_word = words[-1]
        if final_word == "yes":
            return "yes"
        if final_word == "no":
            return "no"

    # 4) Fallback — last yes/no anywhere.
    tokens = re.findall(r"\b(yes|no)\b", raw_lower)
    if tokens:
        return tokens[-1]
    return ""


def calculate_exact_match(predictions: list, references: list, dataset_name: str) -> float:
    """
    Calculates the Exact Match (EM) score between model predictions and references.
    """
    if not predictions or not references or len(predictions) != len(references):
        return 0.0

    matches = 0
    total = len(references)

    for pred, ref in zip(predictions, references):
        if dataset_name.lower() == "gsm8k":
            pred_ans = extract_gsm8k_answer(str(pred))
            ref_ans = extract_gsm8k_answer(str(ref))
        elif dataset_name.lower() == "strategyqa":
            pred_ans = extract_strategyqa_answer(str(pred))
            ref_ans = "yes" if ref is True or str(ref).lower() == "true" else "no"
        else:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. Choose 'gsm8k' or 'strategyqa'."
            )

        if pred_ans == ref_ans and pred_ans != "":
            matches += 1

    return (matches / total) * 100


def majority_vote(answers: Iterable[str]) -> str:
    """
    Self-consistency aggregator (Section 6).  Returns the most common
    non-empty answer; ties are broken by first occurrence.  Returns "" if
    every candidate failed to parse.
    """
    cleaned: List[str] = [a for a in answers if a]
    if not cleaned:
        return ""
    counts = Counter(cleaned)
    top = counts.most_common(1)[0][1]
    for ans in cleaned:
        if counts[ans] == top:
            return ans
    return cleaned[0]
