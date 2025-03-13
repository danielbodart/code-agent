import unittest
from src.reward_function import reward_function

class TestRewardFunction(unittest.TestCase):
    def test_reward_function_all_correct(self):
        self.assertEqual(reward_function("123 + 456 = 579", "123 + 456 = 579"), 1.0)

    def test_reward_function_some_correct(self):
        self.assertAlmostEqual(reward_function("123 + 456 = 579", "123 + 456 = 580"), 0.8, delta=0.1)  # Example with partial correctness

    def test_reward_function_all_masked(self):
        self.assertEqual(reward_function("123 + 456 = 579", "123 + ? = 579"), -1.0)
        self.assertEqual(reward_function("123 + 456 = 579", "123 ? 456 = 579"), -1.0)

if __name__ == '__main__':
    unittest.main()
