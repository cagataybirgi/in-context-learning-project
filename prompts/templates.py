# prompts/templates.py

PROMPT_TEMPLATES = {
    "gsm8k": {
        "standard_few_shot": """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: 6

Q: {question}
A:""",
        
        "zero_shot_cot": """Q: {question}
A: Let's think step by step.""",
        
        "persona_prompting": """You are an advanced mathematician recognized for precise, logical calculations. Solve the following mathematical problem with absolute accuracy.

Q: {question}
A: Let's think step by step."""
    },
    
    "strategyqa": {
        "standard_few_shot": """Q: Do hamsters provide food for any animals?
A: Yes

Q: {question}
A:""",
        
        "zero_shot_cot": """Q: {question}
A: Let's think step by step.""",
        
        "persona_prompting": """You are an advanced logician with deep general knowledge. Evaluate the following question and provide a logically sound answer.

Q: {question}
A: Let's think step by step."""
    }
}

def get_prompt(dataset: str, strategy: str, question: str) -> str:
    """
    Retrieves the formatted prompt template based on the dataset and strategy.
    """
    template = PROMPT_TEMPLATES.get(dataset.lower(), {}).get(strategy.lower(), "")
    if not template:
        raise ValueError(f"Template not found for dataset '{dataset}' and strategy '{strategy}'.")
    
    return template.format(question=question)