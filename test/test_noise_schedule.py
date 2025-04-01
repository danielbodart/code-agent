import unittest
from src.noise_schedule import noise_schedule, loglinear


class TestNoiseSchedule(unittest.TestCase):
    
    def test_specific_examples(self):
        for tokens in [512, 100, 10, 5, 4, 3, 2, 1]:
            schedule = list(noise_schedule(tokens))
            self.assertEqual(schedule[0], 1)
            self.assertEqual(sum(schedule), tokens)
            for i in range(1, len(schedule)):
                self.assertGreaterEqual(schedule[i], schedule[i-1], f"Failed with tokens={tokens}, schedule={schedule}")

    def test_zero_tokens(self):
        schedule = list(noise_schedule(0))
        self.assertEqual(schedule, [])
    
    def test_negative_values(self):
        schedule = list(noise_schedule(-100))
        self.assertEqual(schedule, [])


if __name__ == "__main__":
    unittest.main()
