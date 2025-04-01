from src.data_generator import generate_addition_example
from random import Random

def test_generate_addition_example():
    example = generate_addition_example(Random(1), 10)
    assert example.question == "What is 2 + 9?"
    assert example.answer == "11"
