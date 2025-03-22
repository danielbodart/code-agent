from src.data_generator import generate_addition_example
import unittest
from random import Random

class TestDataGenerator(unittest.TestCase):
    def test_generate_addition_example(self):
        example = generate_addition_example(Random(1))
        self.assertEqual(example[0], "Question: What is 2 + 9?")
        self.assertEqual(example[1], "Reasoning: If 2 = 1 + 1,\nAnd 9 = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1\nThen 2 + 9 = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1\nTherefore 2 + 9 = 11")
        self.assertEqual(example[2], "Answer: 11")

if __name__ == '__main__':
    unittest.main()
