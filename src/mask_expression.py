import random
from typing import Optional

def mask_expression(expression: str, seed: Optional[int] = None) -> tuple:
    """
    Mask one non-whitespace token in the expression with '?' and inject randomness. Return the original token and masked expression.

    >>> mask_expression("3 + 5 = 8", seed=0)
    ('=', '3 + 5 ? 8')
    """
    
    if seed is not None:
        random.seed(seed)
    tokens = expression.split()
    mask_idx = random.choice(range(len(tokens)))
    original_token = tokens[mask_idx]
    tokens[mask_idx] = '?' * len(tokens[mask_idx])
    return original_token, ' '.join(tokens)
