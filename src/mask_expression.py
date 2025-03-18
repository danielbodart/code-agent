import random
from typing import Optional
from .split_expression import split_expression

def mask_expression(expression: str, seed: Optional[int] = None) -> tuple:    
    if seed is not None:
        random.seed(seed)
    tokens = split_expression(expression)
    mask_idx = random.choice(range(len(tokens)))
    original_token = tokens[mask_idx]
    tokens[mask_idx] = '<mask>'
    return original_token, ' '.join(tokens)
