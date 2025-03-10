from src.data_generator import generate_addition_example
import unittest

class TestDataGenerator(unittest.TestCase):
    def test_generate_addition_example(self):
        example = generate_addition_example()
        self.assertIsInstance(example, str)
        parts = example.split()
        self.assertEqual(parts[1], '+')
        self.assertEqual(parts[3], '=')
        a, b, c = int(parts[0]), int(parts[2]), int(parts[4])
        self.assertEqual(a + b, c)

if __name__ == '__main__':
    unittest.main()
