import unittest
from src.reasoning_example import ReasoningExample


class TestReasoningExample(unittest.TestCase):
    def test_creation(self):
        example = ReasoningExample("Question", "Answer")
        
        self.assertEqual(example.question, "Question")
        self.assertEqual(example.answer, "Answer")
    
    def test_iteration(self):
        example = ReasoningExample("Question", "Answer")

        self.assertEqual(list(example), ["Question", "Answer"])

    def test_str(self):
        example = ReasoningExample("A", "D")
        
        self.assertEqual(str(example), "A[SEP]D")

