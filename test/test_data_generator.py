from src.data_generator import generate_addition_example
import unittest
from random import Random

class TestDataGenerator(unittest.TestCase):
    def test_generate_addition_example(self):
        example = generate_addition_example(Random(1), 10)
        self.assertEqual(example.question, "What is 2 + 9?")
        self.assertEqual(example.reasoning_steps, ["If 2 = 1 + 1", "And 9 = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1", "Then 2 + 9 = 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 1","Therefore 2 + 9 = 11"])
        self.assertEqual(example.answer, "11")

if __name__ == '__main__':
    unittest.main()
