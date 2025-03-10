import random
from typing import Optional

def mask_expression(expression: str, seed: Optional[int] = None) -> tuple:    
    if seed is not None:
        random.seed(seed)
    tokens = expression.split()
    mask_idx = random.choice(range(len(tokens)))
    original_token = tokens[mask_idx]
    tokens[mask_idx] = '?'
    return original_token, ' '.join(tokens)
