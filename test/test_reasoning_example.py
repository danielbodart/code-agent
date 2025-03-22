"""
Tests for the ReasoningExample and TokenizedExample classes.
"""
import unittest
import torch
from src.reasoning_example import ReasoningExample, TokenizedExample


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

from transformers import AutoTokenizer
from src.reasoning_example import tokenize


class TestTokenizedExample(unittest.TestCase):
    def test_tokenization(self):
        example = ReasoningExample(
            "Question",
            ["Reasoning Step 1", "Reasoning Step 2"],
            "Answer"
        )
        
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
        tokenized = tokenize(example, tokenizer)
        
        # Check question tokens                                                     "Question"                                                                     "Question"
        self.assertEqual(tokenized.token_ids[0, :3].tolist(), [tokenizer.cls_token_id, 23433, tokenizer.sep_token_id])
        self.assertEqual(tokenized.attention_mask[0, :3].tolist(), [1, 1, 1])

        # Check reasoning tokens                             "Reason", "ing"                                          "Reasoning Step 1"
        self.assertEqual(tokenized.token_ids[0, 3:5].tolist(), [40722, 272])
        self.assertEqual(tokenized.attention_mask[0, 3:5].tolist(), [1, 1])

        # Ignores the padding
        self.assertEqual(tokenized.token_ids[0, 14:16].tolist(), [tokenizer.sep_token_id, tokenizer.pad_token_id])
        self.assertEqual(tokenized.attention_mask[0, 14:16].tolist(), [1, 0])


if __name__ == '__main__':
    unittest.main()
