import math
from typing import Iterator, Tuple

def loglinear_pairs(a: float = 2, b: float = 0.25) -> Iterator[Tuple[int, int]]:
    """
    Generate a loglinear sequence of pairs.
    
    Args:
        a: Base value (default: 2)
        b: Growth rate (default: 0.25)
    
    Yields:
        Consecutive loglinear pairs with exponential growth
    """
    step = 0
    current = max(1, int(a * math.exp(b * step)))
    
    while True:
        step += 1
        next_val = max(1, int(a * math.exp(b * step)))
        yield current, next_val
        current = next_val


def calculate_tokens_per_step(tokens: int) -> Iterator[int]:
    """
    Calculate how many tokens to unmask at each step using a Fibonacci schedule.
    
    Args:
        tokens: The target total number of tokens to unmask
        
    Returns:
        A list of integers representing the number of tokens to unmask at each step,
        with values always increasing or staying the same.
    """
    if tokens <= 0:
        return
    
    # For a single token, just return it
    if tokens == 1:
        yield 1
        return

    sum = 0
    for (a, b) in loglinear_pairs():
        if a == 0:
            continue
        sum += a
        if b + sum > tokens:
            yield a + (tokens - sum)
            break
        yield a
    


# Examples for different token counts
if __name__ == "__main__":
    examples = [
        512,  # Large number of tokens
        100,  # Medium number of tokens
        10,   # Small number of tokens
        5,
        4,
        3,
        2,    # Very small number of tokens
        1,    # Single token
    ]
    
    for token_count in examples:
        schedule = list(calculate_tokens_per_step(token_count))
        print(f"\nTokens: {token_count}")
        print(f"Schedule: {schedule}")
        print(f"Steps: {len(schedule)}")
        print(f"Total tokens: {sum(schedule)}")
        
        # Verify the progression
        is_increasing = all(schedule[i] >= schedule[i-1] for i in range(1, len(schedule)))
        print(f"Always increasing: {is_increasing}")
