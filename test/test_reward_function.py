import unittest
from src.reward_function import reward_function

class TestRewardFunction(unittest.TestCase):
    def test_reward_function_correct_operator(self):
        self.assertEqual(reward_function("+", "+"), 1.0)

    def test_reward_function_incorrect_operator(self):
        self.assertEqual(reward_function("+", "-"), 0.5)

    def test_reward_function_correct_integer(self):
        self.assertEqual(reward_function("42", "42"), 1.0)

    def test_reward_function_close_integer(self):
        self.assertAlmostEqual(reward_function("42", "41"), 0.98, places=2)
        self.assertAlmostEqual(reward_function("42", "43"), 0.98, places=2)

    def test_reward_function_incorrect_integer(self):
        self.assertAlmostEqual(reward_function("42", "50"), 0.81, places=2)
        self.assertAlmostEqual(reward_function("42", "30"), 0.72, places=2)

    def test_reward_function_non_integer_guess(self):
        self.assertEqual(reward_function("42", "forty"), -1.0)

if __name__ == '__main__':
    unittest.main()
