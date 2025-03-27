from dataclasses import dataclass, field
from functools import cached_property
from typing import List

import torch
from transformers import PreTrainedTokenizerFast, BatchEncoding

def tokenize(tokenizer:PreTrainedTokenizerFast, examples:List[str], max_length=512) -> BatchEncoding:
    """
    Tokenize a list of examples.

    Args:
        tokenizer: HuggingFace tokenizer
        examples: List of examples, where each example is an iterator of strings
        max_length: Maximum sequence length

    Returns:
        A dictionary containing token IDs and attention mask
    """
    return tokenizer(
        text=examples,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True,
    )


@dataclass(frozen=True)
class MaskedDiffusionState:
    """Collection of tokenized examples."""

    tokenizer:PreTrainedTokenizerFast
    input_ids:torch.Tensor
    attention_mask:torch.Tensor
    original_ids:torch.Tensor

    @classmethod
    def from_batch(cls, tokenizer:PreTrainedTokenizerFast, batch:BatchEncoding):
        """
        Create a BERTDiffuser instance from a batch of tokenized examples.

        Args:
            tokenizer: HuggingFace tokenizer
            batch: Batch of tokenized examples

        Returns:
            A new BERTDiffuser instance
        """
        return cls(tokenizer, batch["input_ids"], batch["attention_mask"], batch["input_ids"].clone())


    @classmethod
    def from_tensors(cls, tokenizer:PreTrainedTokenizerFast, input_ids:torch.Tensor, attention_mask:torch.Tensor):
        """
        Create a BERTDiffuser instance from a list of examples.

        Args:
            tokenizer: HuggingFace tokenizer
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            A new BERTDiffuser instance
        """
        return cls(tokenizer, input_ids, attention_mask, input_ids.clone())


    def __str__(self):
        """Return a string representation of all examples in the batch."""
        return "\n".join(list(self))

    def __iter__(self):
        """Iterate over each example in the batch, yielding decoded text."""
        batch_size = self.input_ids.size(0)
        for i in range(batch_size):
            # Get actual length of this example
            length = self.lengths[i].item()
            # Only decode up to the actual length
            example_tokens = self.input_ids[i, :length].tolist()
            yield self.tokenizer.decode(example_tokens)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index], "labels": self.labels[index]}

    @cached_property
    def labels(self):
        return torch.where(self.input_ids == self.tokenizer.mask_token_id, self.original_ids, -100)

    @cached_property
    def lengths(self):
        """
        Returns the actual length (number of tokens) for each example in the batch.

        Returns:
            torch.Tensor: A 1D tensor containing the length of each example
        """
        return self.attention_mask.sum(dim=1)

    @cached_property
    def masked(self):
        """
        Returns a mask indicating which tokens are masked.

        Returns:
            torch.Tensor: A 1D tensor containing the count of masked tokens for each example
        """
        return self.input_ids == self.tokenizer.mask_token_id

    @cached_property
    def maskable(self):
        """
        Returns a mask indicating which tokens can be masked.
        
        In this implementation, all tokens are considered maskable,
        including special tokens, following the MDLM approach.
        
        Returns:
            torch.Tensor: A tensor of the same shape as input_ids, where 1 indicates maskable tokens
        """
        # All tokens are maskable
        return torch.ones_like(self.input_ids)

    def mask(self, percentage: float):
        """
        Masks a percentage of the tokens in the collection.
        Only masks tokens that are marked as maskable.

        Args:
            percentage: Percentage of maskable tokens to mask (0.0 to 1.0)

        Returns:
            A new BERTDiffuser instance with masked tokens
        """
        # Only consider tokens that are maskable
        mask = (torch.rand_like(self.input_ids.float()) < percentage) & (self.maskable == 1)

        # Create masked inputs
        masked_inputs = self.input_ids.clone()
        masked_inputs[mask] = self.tokenizer.mask_token_id

        return MaskedDiffusionState(self.tokenizer, masked_inputs, self.attention_mask, self.original_ids)

    def update(self, predicted_ids, update_mask=None):
        """
        Updates tokens with predicted token IDs based on an optional update mask.

        Args:
            predicted_ids: Tensor of token IDs to replace tokens
            update_mask: Optional boolean mask indicating which positions to update.
                        If None, defaults to self.maskable

        Returns:
            A new BERTDiffuser instance with updated input_ids
        """
        # Create a copy of input_ids to update
        updated_input_ids = self.input_ids.clone()

        # Use provided update_mask or default to maskable
        mask = update_mask if update_mask is not None else (self.maskable == 1)
        
        # Update only the positions specified by the mask
        updated_input_ids[mask] = predicted_ids[mask]

        # Return a new BERTDiffuser instance with updated input_ids
        return MaskedDiffusionState(self.tokenizer, updated_input_ids, self.attention_mask, self.original_ids)
