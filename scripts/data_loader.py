from datasets import load_dataset

def load_evaluation_datasets():
    print("Loading GSM8K (Mathematical Reasoning)...")
    # GSM8K uses the 'main' subset
    gsm8k = load_dataset("openai/gsm8k", "main", split="test")
    print("Loading StrategyQA (Commonsense Reasoning)...")
    # Using an open-source Hugging Face mirror for StrategyQA
    strategyqa = load_dataset("ChilleD/StrategyQA", split="test")
    return gsm8k, strategyqa

def inspect_datasets(gsm8k, strategyqa):
    print("\nSample from GSM8K:")
    sample_gsm = gsm8k[0]
    
    print(f"Question: {sample_gsm['question']}")
    print(f"Target Answer: {sample_gsm['answer']}")
    print("\n--- StrategyQA Sample ---")
    sample_sqa = strategyqa[0]
    print(f"Question: {sample_sqa['question']}")
    print(f"Target Answer: {sample_sqa['answer']}")
    
if __name__ == "__main__":
    gsm8k, strategyqa = load_evaluation_datasets()
    inspect_datasets(gsm8k, strategyqa)