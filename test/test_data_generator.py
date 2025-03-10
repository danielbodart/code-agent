import unittest
from src.data_generator import DataGenerator

class TestDataGenerator(unittest.TestCase):
    def test_generate_addition_example(self):
        example = DataGenerator.generate_addition_example()
        # Check if the generated example is a string
        self.assertIsInstance(example, str)
        # Check if the example matches the expected format
        parts = example.split()
        self.assertEqual(parts[1], '+')
        self.assertEqual(parts[3], '=')
        # Ensure the equation is correct
        a, b, c = int(parts[0]), int(parts[2]), int(parts[4])
        self.assertEqual(a + b, c)

if __name__ == '__main__':
    unittest.main()
