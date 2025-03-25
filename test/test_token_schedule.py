import unittest
import math
from src.token_schedule import calculate_tokens_per_step


class TestTokenSchedule(unittest.TestCase):
    
    def test_basic_functionality(self):
        """Test the basic functionality with standard parameters."""
        # Test with 512 tokens and 15 steps
        schedule = calculate_tokens_per_step(15, 512)
        
        # Check that we have the right number of steps
        self.assertEqual(len(schedule), 15)
        
        # Check that the total is exactly the requested amount
        self.assertEqual(sum(schedule), 512)
        
        # Check that the schedule follows a loglinear progression (each step should be >= the previous)
        for i in range(1, len(schedule)):
            self.assertGreaterEqual(schedule[i], schedule[i-1])
    
    def test_single_step(self):
        """Test with a single step."""
        schedule = calculate_tokens_per_step(1, 100)
        self.assertEqual(schedule, [100])
        self.assertEqual(sum(schedule), 100)
    
    def test_low_token_count(self):
        """Test with a low number of tokens."""
        # Test with 5 tokens and 5 steps
        schedule = calculate_tokens_per_step(5, 5)
        self.assertEqual(len(schedule), 5)
        self.assertEqual(sum(schedule), 5)
        
        # With low token counts, we expect a distribution that sums to the token count
        self.assertEqual(sum(schedule), 5)
    
    def test_tokens_less_than_steps(self):
        """Test when the number of tokens is less than the number of steps."""
        # Test with 3 tokens and 10 steps
        schedule = calculate_tokens_per_step(10, 3)
        
        # We should have 10 steps
        self.assertEqual(len(schedule), 10)
        
        # The total should be exactly 3
        self.assertEqual(sum(schedule), 3)
        
        # Each step should have at least 0 tokens
        for tokens in schedule:
            self.assertGreaterEqual(tokens, 0)
    
    def test_tokens_equal_steps(self):
        """Test when the number of tokens equals the number of steps."""
        # Test with 10 tokens and 10 steps
        schedule = calculate_tokens_per_step(10, 10)
        
        # We should have 10 steps
        self.assertEqual(len(schedule), 10)
        
        # The total should be exactly 10
        self.assertEqual(sum(schedule), 10)
    
    def test_large_values(self):
        """Test with large numbers of tokens and steps."""
        # Test with 10000 tokens and 100 steps
        schedule = calculate_tokens_per_step(100, 10000)
        
        # We should have 100 steps
        self.assertEqual(len(schedule), 100)
        
        # The total should be exactly 10000
        self.assertEqual(sum(schedule), 10000)
        
        # The schedule should follow a loglinear progression
        for i in range(1, len(schedule)):
            self.assertGreaterEqual(schedule[i], schedule[i-1])
    
    def test_zero_tokens(self):
        """Test with zero tokens (should raise ValueError)."""
        with self.assertRaises(ValueError):
            calculate_tokens_per_step(5, 0)
    
    def test_zero_steps(self):
        """Test with zero steps (should raise ValueError)."""
        with self.assertRaises(ValueError):
            calculate_tokens_per_step(0, 100)
    
    def test_negative_values(self):
        """Test with negative values (should raise ValueError)."""
        with self.assertRaises(ValueError):
            calculate_tokens_per_step(-5, 100)
        
        with self.assertRaises(ValueError):
            calculate_tokens_per_step(5, -100)


if __name__ == "__main__":
    unittest.main()
