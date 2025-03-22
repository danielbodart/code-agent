import random
from src.reasoning_example import ReasoningExample

def generate_addition_example(r = random, max_number=1000):
    a = r.randint(0, max_number)
    b = r.randint(0, max_number)
    return ReasoningExample(f"What is {a} + {b}?", [
                f"If {a} = {number_as_ones(a)}", 
                f"And {b} = {number_as_ones(b)}",
                f"Then {a} + {b} = {number_as_ones(a + b)}", 
                f"Therefore {a} + {b} = {a + b}"
            ], 
            f"{a + b}")

def number_as_ones(n):
    if n == 0:
        return "0"
    return " + ".join(["1"] * n)


