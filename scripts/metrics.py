# scripts/metrics.py
import re

def extract_gsm8k_answer(text: str) -> str:
    """
    Extracts the final numerical answer from a GSM8K generation.
    Assumes standard CoT format where the answer appears after '####' 
    or extracts the last number in the text.
    """
    # GSM8K reference answers often use '#### [answer]'
    if "####" in text:
        text = text.split("####")[-1].strip()
    
    # Extract the last number found in the remaining text
    numbers = re.findall(r'[-+]?\d*[\.,]?\d+', text)
    if numbers:
        # Clean commas for consistent exact matching (e.g., "1,000" -> "1000")
        return numbers[-1].replace(',', '')
    return ""

def extract_strategyqa_answer(text: str) -> str:
    """
    Extracts the boolean Yes/No answer from a StrategyQA generation.
    """
    text = str(text).lower()
    
    # Check for explicit statements common in reasoning traces
    if "answer is yes" in text or "therefore, yes" in text:
        return "yes"
    elif "answer is no" in text or "therefore, no" in text:
        return "no"
    
    # Fallback to the last occurrence of yes or no
    words = re.findall(r'\b(yes|no)\b', text)
    if words:
        return words[-1]
    return ""

def calculate_exact_match(predictions: list, references: list, dataset_name: str) -> float:
    """
    Calculates the exact match (EM) score between model predictions and references.
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
            # StrategyQA references are often boolean (True/False) in the raw dataset
            ref_ans = "yes" if ref is True or str(ref).lower() == "true" else "no"
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}. Choose 'gsm8k' or 'strategyqa'.")
            
        # Ensure exact string match and avoid matching empty parsing failures
        if pred_ans == ref_ans and pred_ans != "":
            matches += 1
            
    return (matches / total) * 100