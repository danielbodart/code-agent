from src.data_generator import generate_addition_example
import unittest
from src.split_expression import split_expression

class TestDataGenerator(unittest.TestCase):
    def test_generate_addition_example(self):
        example = generate_addition_example()
        self.assertIsInstance(example, str)
        tokens = split_expression(example)
        self.assertEqual(tokens[1], '+')
        self.assertEqual(tokens[3], '=')
        a, b, c = int(tokens[0]), int(tokens[2]), int(tokens[4])
        self.assertEqual(a + b, c)

if __name__ == '__main__':
    unittest.main()
