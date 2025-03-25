import unittest
from src.token_schedule import calculate_tokens_per_step


class TestTokenSchedule(unittest.TestCase):
    
    def test_specific_examples(self):
        """Test specific examples with known expected outputs."""
        # Test with 512 tokens
        schedule = list(calculate_tokens_per_step(512))
        self.assertEqual(schedule, [2, 2, 3, 4, 5, 6, 8, 11, 14, 18, 24, 31, 40, 51, 66, 85, 142])
        self.assertEqual(sum(schedule), 512)
        
        # Test with 100 tokens
        schedule = list(calculate_tokens_per_step(100))
        self.assertEqual(schedule, [2, 2, 3, 4, 5, 6, 8, 11, 14, 18, 27])
        self.assertEqual(sum(schedule), 100)
        
        # Test with 10 tokens
        schedule = list(calculate_tokens_per_step(10))
        self.assertEqual(schedule, [2, 2, 6])
        self.assertEqual(sum(schedule), 10)
        
        # Test with 5 tokens
        schedule = list(calculate_tokens_per_step(5))
        self.assertEqual(schedule, [2, 3])
        self.assertEqual(sum(schedule), 5)
    
    def test_small_values(self):
        """Test with very small token counts."""
        # Test with 4 tokens
        schedule = list(calculate_tokens_per_step(4))
        self.assertEqual(schedule, [2, 2])
        self.assertEqual(sum(schedule), 4)
        
        # Test with 3 tokens
        schedule = list(calculate_tokens_per_step(3))
        self.assertEqual(schedule, [3])
        self.assertEqual(sum(schedule), 3)
        
        # Test with 2 tokens
        schedule = list(calculate_tokens_per_step(2))
        self.assertEqual(schedule, [2])
        self.assertEqual(sum(schedule), 2)
        
        # Test with 1 token
        schedule = list(calculate_tokens_per_step(1))
        self.assertEqual(schedule, [1])
        self.assertEqual(sum(schedule), 1)
    
    def test_always_increasing(self):
        """Test that values are always increasing or staying the same."""
        # Test with various token counts
        for tokens in [512, 100, 10, 5, 4, 3, 2, 1]:
            schedule = list(calculate_tokens_per_step(tokens))
            # Skip if only one step
            if len(schedule) > 1:
                for i in range(1, len(schedule)):
                    self.assertGreaterEqual(schedule[i], schedule[i-1], 
                                           f"Failed with tokens={tokens}, schedule={schedule}")
    
    def test_zero_tokens(self):
        schedule = list(calculate_tokens_per_step(0))
        self.assertEqual(schedule, [])
    
    def test_negative_values(self):
        schedule = list(calculate_tokens_per_step(-100))
        self.assertEqual(schedule, [])


if __name__ == "__main__":
    unittest.main()
