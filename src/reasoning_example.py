"""
ReasoningExample class for representing examples with question, reasoning, and answer.
"""
from typing import List, Iterator
from dataclasses import dataclass
from functools import cached_property

@dataclass(frozen=True)
class ReasoningExample:
    """
    Represents a reasoning example with a question, reasoning steps, and an answer.
    
    This class provides a clean interface for working with reasoning examples,
    including methods for creating, formatting, and masking examples.
    """
    question: str
    reasoning_steps: List[str]
    answer: str
    
    def __iter__(self) -> Iterator[str]:
        """
        Makes the ReasoningExample iterable, yielding [question, reasoning, answer].
        This allows unpacking like: question, reasoning, answer = example
        """
        yield "<question>" + self.question + "</question>"
        yield "<reasoning>" + "\n".join(self.reasoning_steps) + "</reasoning>"
        yield "<answer>" + self.answer + "</answer>"

from transformers import AutoTokenizer
import torch
from src.tokens import tag_start, tag_end, close_tag_start, find_sequences_batch, open_tag, close_tag


@dataclass(frozen=True)
class TokenizedExamples:
    """Collection of tokenized examples."""
    
    tokenizer:AutoTokenizer
    input_ids:torch.Tensor
    attention_mask:torch.Tensor
    labels:torch.Tensor
    original_ids:torch.Tensor = None

    def __post_init__(self):
        # If original_ids is not provided, use input_ids as the original
        if self.original_ids is None:
            object.__setattr__(self, 'original_ids', self.input_ids.clone())

    @classmethod
    def create(cls, examples:List[Iterator[str]], tokenizer:AutoTokenizer, max_length=512):
        """
        Create a TokenizedExamples instance from a list of examples.
        
        Args:
            examples: List of examples, where each example is an iterator of strings
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            
        Returns:
            A new TokenizedExamples instance
        """
        encoded = tokenizer(
            text=["[SEP]".join(example) for example in examples],
            max_length=max_length, 
            truncation=True, 
            padding='max_length',
            return_tensors='pt',
            return_attention_mask=True,
        )
        
        # Initialize labels with -100 (ignore index for loss calculation)
        labels = torch.full_like(encoded['input_ids'], -100)
        
        # Original IDs are the same as input_ids at creation time
        original_ids = encoded['input_ids'].clone()
        
        return cls(tokenizer, encoded['input_ids'], encoded['attention_mask'], labels, original_ids)

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
        
        Rules:
        - Question tokens are not maskable (0)
        - XML tags are not maskable (0)
        - Reasoning tokens are maskable (1)
        - Answer tokens are maskable (1)
        - Special tokens are not maskable (0)
        - Already masked tokens are considered maskable (1)
        - Padding tokens are not maskable (0)
        
        Returns:
            torch.Tensor: A tensor of the same shape as input_ids, where 1 indicates maskable tokens
        """
        batch_size = self.input_ids.size(0)
        
        # Initialize a tensor of zeros (not maskable)
        maskable_tensor = torch.zeros_like(self.input_ids)
        
        # Define tags and their maskability
        tags = {
            "question": False,  # question tokens are not maskable
            "reasoning": True,  # reasoning tokens are maskable
            "answer": True      # answer tokens are maskable
        }
        
        # Create dictionaries to store patterns and positions
        open_patterns = {}
        close_patterns = {}
        open_positions = {}
        close_positions = {}
        
        # Build patterns and find positions for each tag
        for tag_name, is_maskable in tags.items():
            # Get token IDs for the tag
            tag_tokens = self.tokenizer.encode(tag_name, add_special_tokens=False)
            
            # Create patterns
            open_patterns[tag_name] = open_tag(tag_tokens)
            close_patterns[tag_name] = close_tag(tag_tokens)
            
            # Find positions
            open_positions[tag_name] = find_sequences_batch(self.input_ids, open_patterns[tag_name])
            close_positions[tag_name] = find_sequences_batch(self.input_ids, close_patterns[tag_name])
        
        # Special tokens that are never maskable
        special_token_ids = [
            self.tokenizer.cls_token_id, 
            self.tokenizer.sep_token_id, 
            self.tokenizer.pad_token_id,
            tag_start, tag_end, close_tag_start
        ]
        
        for i in range(batch_size):
            # Process each tag
            for tag_name, is_maskable in tags.items():
                if is_maskable and len(open_positions[tag_name][i]) > 0 and len(close_positions[tag_name][i]) > 0:
                    # Get the content between opening and closing tags
                    start = open_positions[tag_name][i][0] + len(open_patterns[tag_name])
                    end = close_positions[tag_name][i][0]
                    
                    # Make these tokens maskable (1)
                    maskable_tensor[i, start:end] = 1
            
            # Make special tokens not maskable (0)
            for token_id in special_token_ids:
                maskable_tensor[i] = maskable_tensor[i] * (self.input_ids[i] != token_id).int()
        
        # Apply attention mask to ensure only tokens with attention are maskable
        maskable_tensor = maskable_tensor * self.attention_mask
        
        return maskable_tensor

    def mask(self, percentage: float):
        """
        Masks a percentage of the tokens in the collection. 
        Only masks tokens that are marked as maskable.
        Does not modify labels.
        
        Args:
            percentage: Percentage of maskable tokens to mask (0.0 to 1.0)
        
        Returns:
            A new TokenizedExamples instance with masked tokens
        """
        # Only consider tokens that are maskable
        mask = (torch.rand_like(self.input_ids.float()) < percentage) & (self.maskable == 1)
        
        # Create masked inputs
        masked_inputs = self.input_ids.clone()
        masked_inputs[mask] = self.tokenizer.mask_token_id

        return TokenizedExamples(self.tokenizer, masked_inputs, self.attention_mask, self.labels, self.original_ids)
        
    def unmask(self, percentage: float):
        """
        Sets labels for a percentage of masked tokens to their original values.
        This allows the model to learn to predict these tokens.
        Only operates on tokens that are currently masked.
        
        Args:
            percentage: Percentage of masked tokens to unmask (0.0 to 1.0)
        
        Returns:
            A new TokenizedExamples instance with updated labels
        """
        # Only unmask a percentage of the masked tokens
        unmask = (torch.rand_like(self.input_ids.float()) < percentage) & self.masked
        
        labels = self.labels.clone()
        labels[unmask] = self.original_ids[unmask]
        
        return TokenizedExamples(self.tokenizer, self.input_ids, self.attention_mask, labels, self.original_ids)
