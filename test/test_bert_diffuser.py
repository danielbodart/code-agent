import unittest

import torch
from transformers import AutoTokenizer

from src.bert_diffuser import BERTDiffuser
from src.reasoning_example import ReasoningExample
from src.tokens import open_tag, close_tag


class TestBERTDiffuser(unittest.TestCase):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    question = 19751
    reasoning = [10752, 272]

    def test_tokenization(self):
        example = ReasoningExample(
            "Question",
            ["Reasoning Step 1", "Reasoning Step 2"],
            "Answer"
        )

        tokenized = BERTDiffuser.create([example], self.tokenizer)


        self.assertEqual(tokenized.input_ids[0, :9].tolist(), [self.tokenizer.cls_token_id, *open_tag(self.question), 23433, *close_tag(self.question), self.tokenizer.sep_token_id])
        self.assertEqual(tokenized.attention_mask[0, :3].tolist(), [1, 1, 1])

        self.assertEqual(tokenized.input_ids[0, 9:15].tolist(), [*open_tag(self.reasoning), 40722, 272])
        self.assertEqual(tokenized.attention_mask[0, 9:15].tolist(), [1, 1, 1, 1, 1, 1])

        # Ignores the padding
        self.assertEqual(tokenized.input_ids[0, 34:36].tolist(), [self.tokenizer.sep_token_id, self.tokenizer.pad_token_id])
        self.assertEqual(tokenized.attention_mask[0, 34:36].tolist(), [1, 0])

    def test_can_mask_tokens(self):
        example = ReasoningExample(
            "Should get masked",
            ["Should get masked"],
            "Should get masked"
        )

        tokenized = BERTDiffuser.create([example], self.tokenizer)
        
        # With the new implementation, all tokens are maskable
        # So we need to check that tokens are being masked based on the percentage
        
        # First, let's verify that with 0% masking, nothing gets masked
        no_masking = next(iter(tokenized.mask(percentage=0)))
        self.assertNotIn("[MASK]", no_masking)
        
        # Then, with 100% masking, everything should be masked
        full_masking = next(iter(tokenized.mask(percentage=1)))
        
        # The output should be all [MASK] tokens
        # Count the number of [MASK] tokens
        mask_count = full_masking.count("[MASK]")
        
        # Get the total number of tokens (excluding padding)
        total_tokens = tokenized.attention_mask[0].sum().item()
        
        # All tokens should be masked
        self.assertEqual(mask_count, total_tokens)

    def test_maskable(self):
        # All tokens should be maskable
        example = ReasoningExample(
            "ignore",
            ["one", "two"],
            "three"
        )

        tokenized = BERTDiffuser.create([example], self.tokenizer)
        
        # Check that all tokens are maskable (all ones)
        self.assertTrue(torch.all(tokenized.maskable == 1))

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

        tokenized = BERTDiffuser.create([example1, example2], self.tokenizer)

        lengths = tokenized.lengths

        self.assertEqual(lengths.shape, (2,))
        self.assertEqual(lengths[0].item(), 30)
        self.assertEqual(lengths[1].item(), 44)

    def test_mask_percentage(self):
        """Test that masking with percentage produces reasonable results."""
        example = ReasoningExample(
            "This is a question",
            ["This is reasoning step one"],
            "This is the answer"
        )

        tokenized = BERTDiffuser.create([example], self.tokenizer)

        masked = tokenized.mask(percentage=0.5)
        mask_count = masked.masked.sum().item()

        self.assertEqual((masked.labels == self.tokenizer.mask_token_id).sum().item(), mask_count)

        self.assertGreater(mask_count, 0, "Should mask at least some tokens")
        max_maskable = (tokenized.maskable == 1).sum().item()
        self.assertLess(mask_count, max_maskable, "Should not mask 100% of maskable tokens")



    def test_update(self):
        example = ReasoningExample(
            "This is a question",
            ["This is reasoning"],
            "This is an answer"
        )

        tokenized = BERTDiffuser.create([example], self.tokenizer)
        masked = tokenized.mask(percentage=1.0)

        predicted_ids = torch.randint(1000, 5000, (masked.input_ids.shape[0], masked.input_ids.shape[1]))
        updated = masked.update(predicted_ids)

        self.assertTrue(torch.all(updated.input_ids[masked.maskable == 1] == predicted_ids.view(-1)[:masked.maskable.sum()]))
        self.assertTrue(torch.all(updated.input_ids[masked.maskable == 0] == masked.input_ids[masked.maskable == 0]))
