"""
Tests for the ReasoningExample and TokenizedExample classes.
"""
import unittest
from src.reasoning_example import ReasoningExample, tokenize
import torch


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
        self.assertEqual(items[0], "<question>Question</question>")
        self.assertEqual(items[1], "<reasoning>Reasoning Step 1\nReasoning Step 2</reasoning>")
        self.assertEqual(items[2], "<answer>Answer</answer>")

from transformers import AutoTokenizer
from src.tokens import open_tag, close_tag

class TestTokenizedExample(unittest.TestCase):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    question = 19751
    reasoning = [10752, 272]
 
    def test_tokenization(self):
        example = ReasoningExample(
            "Question",
            ["Reasoning Step 1", "Reasoning Step 2"],
            "Answer"
        )

        tokenized = tokenize([example], self.tokenizer)
 

        self.assertEqual(tokenized.token_ids[0, :9].tolist(), [self.tokenizer.cls_token_id, *open_tag(self.question), 23433, *close_tag(self.question), self.tokenizer.sep_token_id])
        self.assertEqual(tokenized.attention_mask[0, :3].tolist(), [1, 1, 1])

        self.assertEqual(tokenized.token_ids[0, 9:15].tolist(), [*open_tag(self.reasoning), 40722, 272])
        self.assertEqual(tokenized.attention_mask[0, 9:15].tolist(), [1, 1, 1, 1, 1, 1])

        # Ignores the padding
        self.assertEqual(tokenized.token_ids[0, 34:36].tolist(), [self.tokenizer.sep_token_id, self.tokenizer.pad_token_id])
        self.assertEqual(tokenized.attention_mask[0, 34:36].tolist(), [1, 0])

    def test_can_mask_tokens(self):
        example = ReasoningExample(
            "Should NOT get masked",
            ["Should get masked"],
            "Should get masked"
        )

        tokenized = tokenize([example], self.tokenizer)

        masked = next(iter(tokenized.mask(percentage=1)))

        self.assertEqual(masked, "[CLS]<question>Should NOT get masked</question>[SEP]<reasoning>[MASK][MASK][MASK]</reasoning>[SEP]<answer>[MASK][MASK][MASK]</answer>[SEP]")

    def test_maskable(self):
        # all tokens in the question should not be maskable
        # all xml tags should not be maskable
        # all tokens in the reasoning should be maskable
        # all tokens in the answer should be maskable
        # all special tokens should not be maskable
        example = ReasoningExample(
            "ignore",
            ["one", "two"],
            "three"
        )

        tokenized = tokenize([example], self.tokenizer)
        
        # Only compare up to the actual length of the example
        actual_length = tokenized.lengths[0].item()
        maskable_tokens = tokenized.maskable[0, :actual_length].tolist()
        
        self.assertEqual(maskable_tokens, [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0]) 
        
    def test_lengths(self):
        """Test that the lengths property correctly calculates the length of each example."""
        example1 = ReasoningExample(
            "short question",
            ["short reasoning"],
            "short answer"
        )
        
        example2 = ReasoningExample(
            "longer question with more tokens",
            ["first reasoning step", "second reasoning step"],
            "longer answer with more tokens too"
        )
        
        tokenized = tokenize([example1, example2], self.tokenizer)
        
        lengths = tokenized.lengths
        
        self.assertEqual(lengths.shape, (2,))
        self.assertEqual(lengths[0].item(), 30)
        self.assertEqual(lengths[1].item(), 44)

    def test_masking_percentage(self):
        """Test that masking with percentage produces reasonable results."""
        example = ReasoningExample(
            "This is a question",
            ["This is reasoning step one"],
            "This is the answer"
        )
        
        tokenized = tokenize([example], self.tokenizer)
        
        masked = tokenized.mask(percentage=0.5)
        mask_count = (masked.token_ids == self.tokenizer.mask_token_id).sum().item()
        
        self.assertGreater(mask_count, 0, "Should mask at least some tokens")
        max_maskable = (tokenized.maskable == 1).sum().item()
        self.assertLess(mask_count, max_maskable, "Should not mask 100% of maskable tokens")
        
    def test_labels(self):
        example = ReasoningExample(
            "This is a question",
            ["This is reasoning"],
            "This is an answer"
        )
        
        tokenized = tokenize([example], self.tokenizer)
        
        labels = tokenized.labels
        maskable = tokenized.maskable
        
        # Check that unmaskable tokens have label -100
        self.assertTrue(torch.all(labels[maskable == 0] == -100))
        
        # Check that maskable tokens have their original token IDs
        self.assertTrue(torch.all(labels[maskable == 1] == tokenized.token_ids[maskable == 1]))