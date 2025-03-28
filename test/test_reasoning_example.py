import unittest
from src.reasoning_example import ReasoningExample


class TestReasoningExample(unittest.TestCase):
    def test_creation(self):
        example = ReasoningExample(
            "Question",
            ["Reasoning Step 1", "Reasoning Step 2"],
            "Answer"
        )
        
        self.assertEqual(example.question, "Question")
        self.assertEqual(len(example.reasoning_steps), 2)
        self.assertEqual(example.answer, "Answer")
    
    def test_iteration(self):
        example = ReasoningExample(
            "Question",
            ["Reasoning Step 1", "Reasoning Step 2"],
            "Answer"
        )
        
        items = list(example)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[0], "Question")
        self.assertEqual(items[1], "Reasoning Step 1\nReasoning Step 2")
        self.assertEqual(items[2], "Answer")

    def test_str(self):
        example = ReasoningExample(
            "A",
            ["B", "C"],
            "D"
        )
        
        self.assertEqual(str(example), "A[SEP]B\nC[SEP]D")

