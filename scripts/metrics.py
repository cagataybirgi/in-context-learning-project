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


# ---------------------------------------------------------------------------
# Additional metrics
# ---------------------------------------------------------------------------
# Section-5 evaluation augmentation.  The four metrics below complement EM
# along different axes:
#
#   * substring_match   — relaxed EM: gold appears literally inside prediction
#   * is_numeric_close  — numeric tolerance, captures "36.36 vs 36" rounding
#                         disagreements on GSM8K
#   * token_f1          — word-level precision / recall harmonic mean
#   * rouge_l           — longest-common-subsequence based F1
#
# token_f1 and rouge_l are computed in `runner.py` against the *full*
# generation vs the *full* gold text — degenerate when applied to one-token
# extracted scalars, so callers should pass texts, not labels.  BERTScore is
# scored by `scripts/score_bertscore.py` after a run completes (lazy-loaded).

_WORD_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    """Lowercase word-tokenization shared by F1 and ROUGE-L."""
    if not isinstance(text, str):
        text = str(text)
    return _WORD_RE.findall(text.lower())


def substring_match(prediction: str, reference: str) -> bool:
    """
    Binary: is `reference` a substring of `prediction` (case-insensitive,
    whitespace-normalised)?  Returns False if either side is empty.
    """
    if prediction is None or reference is None:
        return False
    p = str(prediction).strip().lower()
    r = str(reference).strip().lower()
    if not p or not r:
        return False
    return r in p


def is_numeric_close(prediction, reference, abs_tol: float = 0.5) -> bool:
    """
    Numeric tolerance match.  True iff both sides parse as floats and
    |prediction - reference| <= abs_tol.

    Default `abs_tol = 0.5` corresponds to "rounds to the same integer" when
    the gold is an integer (the typical GSM8K case).  This recovers the
    failure mode where the model writes "Answer: 36.36" but the gold is 36.
    Pass `abs_tol=0.01` for strict (1%-of-1) tolerance.
    """
    def _to_float(x):
        try:
            return float(str(x).replace(",", "").strip())
        except (TypeError, ValueError):
            return None

    p = _to_float(prediction)
    r = _to_float(reference)
    if p is None or r is None:
        return False
    return abs(p - r) <= abs_tol


def token_f1(prediction: str, reference: str) -> float:
    """
    Word-level F1 between two strings.  Tokenisation is lowercase + `\\w+`.
    Returns 0.0 if either side is empty.  Computes the multiset intersection
    of tokens (a token contributes at most min(count_pred, count_ref) hits).
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    """Length of the longest common subsequence of two token lists."""
    if not a or not b:
        return 0
    m, n = len(a), len(b)
    # Two-row DP saves memory on long GSM8K chains.
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(m):
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(curr[j], prev[j + 1])
        prev, curr = curr, prev
        for k in range(n + 1):
            curr[k] = 0
    return prev[n]


def rouge_l(prediction: str, reference: str) -> float:
    """
    ROUGE-L F1 score based on the longest common subsequence of word
    tokens.  Returns 0.0 if either side is empty.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall    = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)
