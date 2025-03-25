from dataclasses import dataclass
from functools import cached_property
from typing import List, Iterator

import torch
from transformers import AutoTokenizer

from src.tokens import open_tag, close_tag, find_sequences_batch, tag_start, tag_end, close_tag_start


@dataclass(frozen=True)
class BERTDiffuser:
    """Collection of tokenized examples."""

    tokenizer:AutoTokenizer
    input_ids:torch.Tensor
    attention_mask:torch.Tensor
    labels:torch.Tensor
    original_ids:torch.Tensor

    @classmethod
    def create(cls, examples:List[Iterator[str]], tokenizer:AutoTokenizer, max_length=512):
        """
        Create a BERTDiffuser instance from a list of examples.

        Args:
            examples: List of examples, where each example is an iterator of strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length

        Returns:
            A new BERTDiffuser instance
        """
        encoded = tokenizer(
            text=["[SEP]".join(example) for example in examples],
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
        )

        return cls.from_tensors(tokenizer, encoded['input_ids'], encoded['attention_mask'])

    @classmethod
    def from_tensors(cls, tokenizer:AutoTokenizer, input_ids:torch.Tensor, attention_mask:torch.Tensor):
        """
        Create a BERTDiffuser instance from a list of examples.

        Args:
            tokenizer: HuggingFace tokenizer
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            A new BERTDiffuser instance
        """
        labels = torch.full_like(input_ids, -100)
        original_ids = input_ids.clone()

        return cls(tokenizer, input_ids, attention_mask, labels, original_ids)

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
        
    @cached_property
    def timesteps(self):
        return self.input_ids.size(1)
        
    @cached_property
    def timestep(self):
        return (self.input_ids == self.tokenizer.mask_token_id).sum(dim=1)

    def mask(self, percentage: float):
        """
        Masks a percentage of the tokens in the collection.
        Only masks tokens that are marked as maskable.
        Does not modify labels.

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

        return BERTDiffuser(self.tokenizer, masked_inputs, self.attention_mask, self.labels, self.original_ids)

    def unmask(self, percentage: float):
        """
        Sets labels for a percentage of masked tokens to their original values.
        This allows the model to learn to predict these tokens.
        Only operates on tokens that are currently masked.

        Args:
            percentage: Percentage of masked tokens to unmask (0.0 to 1.0)

        Returns:
            A new BERTDiffuser instance with updated labels
        """
        # Only unmask a percentage of the masked tokens
        unmask = (torch.rand_like(self.input_ids.float()) < percentage) & self.masked

        labels = self.labels.clone()
        labels[unmask] = self.input_ids[unmask]

        return BERTDiffuser(self.tokenizer, self.input_ids, self.attention_mask, labels, self.original_ids)

    def update(self, predicted_ids):
        """
        Updates the maskable tokens with predicted token IDs.

        Args:
            predicted_ids: Tensor of token IDs to replace maskable tokens

        Returns:
            A new BERTDiffuser instance with updated input_ids
        """
        # Create a copy of input_ids to update
        updated_input_ids = self.input_ids.clone()

        # Simple update approach
        mask = self.maskable == 1
        updated_input_ids[mask] = predicted_ids.view(-1)[:mask.sum()]

        # Return a new BERTDiffuser instance with updated input_ids
        return BERTDiffuser(self.tokenizer, updated_input_ids, self.attention_mask, self.labels, self.original_ids)
