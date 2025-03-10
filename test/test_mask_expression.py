import unittest
from src.mask_expression import mask_expression

class TestMaskExpression(unittest.TestCase):
    def test_mask_expression(self):
        expression = "3 + 5 = 8"
        original_token, masked = mask_expression(expression, seed=0)
        self.assertEqual(masked, '3 + 5 ? 8')
        self.assertEqual(original_token, "=")

if __name__ == '__main__':
    unittest.main()
