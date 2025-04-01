import unittest

import torch
from transformers import AutoTokenizer

from src.masked_diffusion_state import MaskedDiffusionState, tokenize
from src.reasoning_example import ReasoningExample
from pytorch_lightning import seed_everything

seed_everything(42)

class TestMaskedDiffusionState(unittest.TestCase):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

    def test_tokenization(self):
        example = ReasoningExample("Question", "Answer")

        tokenized = MaskedDiffusionState.from_batch(self.tokenizer, tokenize(self.tokenizer, [str(example)], max_length=6))
        decoded = self.tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=False)

        self.assertEqual("[CLS]Question[SEP]Answer[SEP][PAD]", decoded)
        self.assertEqual(tokenized.attention_mask[0].tolist(), [1, 1, 1, 1, 1, 0])

    def test_can_mask_tokens(self):
        example = ReasoningExample("Can get masked", "Can get masked")

        tokenized = MaskedDiffusionState.from_batch(self.tokenizer, tokenize(self.tokenizer, [str(example)], max_length=10))
        
        no_masking = tokenized.mask(percentage=0)
        no_masking_decoded = self.tokenizer.decode(no_masking.input_ids[0], skip_special_tokens=False)
        self.assertEqual("[CLS]Can get masked[SEP]Can get masked[SEP][PAD]", no_masking_decoded)

        # half_masking = tokenized.mask(percentage=0.5)
        # half_masking_decoded = self.tokenizer.decode(half_masking.input_ids[0], skip_special_tokens=False)
        # self.assertEqual("[CLS]Can get masked[SEP][MASK] get masked[MASK][PAD]", half_masking_decoded)

        full_masking = tokenized.mask(percentage=1)
        full_masking_decoded = self.tokenizer.decode(full_masking.input_ids[0], skip_special_tokens=False)
        self.assertEqual("[MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK]", full_masking_decoded)


    def test_maskable(self):
        # All tokens should be maskable
        example = ReasoningExample(
            "ignore",
            "three"
        )

        tokenized = MaskedDiffusionState.from_batch(self.tokenizer, tokenize(self.tokenizer, [str(example)]))
        
        # Check that all tokens are maskable (all ones)
        self.assertTrue(torch.all(tokenized.maskable == 1))

    def test_lengths(self):
        """Test that the lengths property correctly calculates the length of each example."""
        example1 = ReasoningExample(
            "short question",
            "short answer"
        )

        example2 = ReasoningExample(
            "longer question with more tokens",
            "longer answer with more tokens too"
        )

        tokenized = MaskedDiffusionState.from_batch(self.tokenizer, tokenize(self.tokenizer, [str(example1), str(example2)]))

        lengths = tokenized.lengths

        self.assertEqual(lengths.shape, (2,))
        self.assertEqual(lengths[0].item(), 7)
        self.assertEqual(lengths[1].item(), 16)

    def test_mask_percentage(self):
        """Test that masking with percentage produces reasonable results."""
        example = ReasoningExample(
            "This is a question",
            "This is the answer"
        )

        tokenized = MaskedDiffusionState.from_batch(self.tokenizer, tokenize(self.tokenizer, [str(example)]))

        masked = tokenized.mask(percentage=0.5)
        mask_count = masked.masked.sum().item()

        self.assertEqual((masked.labels == -100).sum().item(), masked.input_ids.size(1) - mask_count)

        self.assertGreater(mask_count, 0, "Should mask at least some tokens")
        max_maskable = (tokenized.maskable == 1).sum().item()
        self.assertLess(mask_count, max_maskable, "Should not mask 100% of maskable tokens")



    def test_update(self):
        example = ReasoningExample(
            "This is a question",
            "This is an answer"
        )

        tokenized = MaskedDiffusionState.from_batch(self.tokenizer, tokenize(self.tokenizer, [str(example)]))
        masked = tokenized.mask(percentage=1.0)

        predicted_ids = torch.randint(1000, 5000, (masked.input_ids.shape[0], masked.input_ids.shape[1]))
        updated = masked.update(predicted_ids)

        self.assertTrue(torch.all(updated.input_ids[masked.maskable == 1] == predicted_ids.view(-1)[:masked.maskable.sum()]))
        self.assertTrue(torch.all(updated.input_ids[masked.maskable == 0] == masked.input_ids[masked.maskable == 0]))
