import random

def generate_addition_example(r = random):
    a = r.randint(0, 10)
    b = r.randint(0, 10)
    return [f"Question: What is {a} + {b}?", 
            f"Reasoning: If {a} = {number_as_ones(a)},\nAnd {b} = {number_as_ones(b)}\nThen {a} + {b} = {number_as_ones(a + b)}\nTherefore {a} + {b} = {a + b}", 
            f"Answer: {a + b}"]

def number_as_ones(n):
    if n == 0:
        return "0"
    return " + ".join(["1"] * n)


