import math
from typing import Iterator, Tuple
from itertools import pairwise

def loglinear(base: float = 2, growth: float = 0.25) -> Iterator[int]:
    """
    Generate a loglinear sequence of values.
    
    Yields:
        Sequence of integers with exponential growth
    """
    step = 0
    while True:
        current = max(1, int(base * math.exp(growth * step)))
        yield current
        step += 1

def fibonacci() -> Iterator[int]:
    """
    Generate a Fibonacci sequence of values.
    
    Yields:
        Sequence of integers with Fibonacci growth
    """
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b


def noise_schedule(tokens: int, generator = loglinear) -> Iterator[int]:
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
    for (a, b) in pairwise(generator()):
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
        schedule = list(noise_schedule(token_count))
        print(f"\nTokens: {token_count}")
        print(f"Schedule: {schedule}")
        print(f"Steps: {len(schedule)}")
        print(f"Total tokens: {sum(schedule)}")
        
        # Verify the progression
        is_increasing = all(schedule[i] >= schedule[i-1] for i in range(1, len(schedule)))
        print(f"Always increasing: {is_increasing}")
