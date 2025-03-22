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


def tokenize(examples:List[ReasoningExample], tokenizer:AutoTokenizer, max_length=512):
    """
    Tokenize the example using the provided tokenizer.
    
    Args:
        tokenizer: HuggingFace tokenizer
        return_tensors: Format of tensors to return ('pt', 'tf', 'np', or None for lists)
        max_length: Maximum sequence length
        
    Returns:
        self (for method chaining)
    """
    encoded = tokenizer(
        text=["[SEP]".join(example) for example in examples],
        max_length=max_length, 
        truncation=True, 
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True,
    )
    
    return TokenizedExamples(tokenizer, encoded['input_ids'], encoded['attention_mask'])

@dataclass(frozen=True)
class TokenizedExamples:
    """Collection of tokenized examples."""
    
    tokenizer:AutoTokenizer
    token_ids:torch.Tensor
    attention_mask:torch.Tensor

    def __str__(self):
        """Return a string representation of all examples in the batch."""
        return "\n".join(list(self))

    def __iter__(self):
        """Iterate over each example in the batch, yielding decoded text."""
        batch_size = self.token_ids.size(0)
        for i in range(batch_size):
            # Get actual length of this example
            length = self.lengths[i].item()
            # Only decode up to the actual length
            example_tokens = self.token_ids[i, :length].tolist()
            yield self.tokenizer.decode(example_tokens)

    @cached_property
    def lengths(self):
        """
        Returns the actual length (number of tokens) for each example in the batch.
        
        Returns:
            torch.Tensor: A 1D tensor containing the length of each example
        """
        return self.attention_mask.sum(dim=1)

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
            List of integer lists, one per example in the batch, where 1 indicates maskable tokens
        """
        batch_size = self.token_ids.size(0)
        results = []
        
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
            open_positions[tag_name] = find_sequences_batch(self.token_ids, open_patterns[tag_name])
            close_positions[tag_name] = find_sequences_batch(self.token_ids, close_patterns[tag_name])
        
        # Special tokens that are never maskable
        special_token_ids = [
            self.tokenizer.cls_token_id, 
            self.tokenizer.sep_token_id, 
            self.tokenizer.pad_token_id,
            tag_start, tag_end, close_tag_start
        ]
        
        for i in range(batch_size):
            # Start with all tokens not maskable (0)
            maskable = torch.zeros_like(self.token_ids[i], dtype=torch.int)
            
            # Process each tag
            for tag_name, is_maskable in tags.items():
                if is_maskable and len(open_positions[tag_name][i]) > 0 and len(close_positions[tag_name][i]) > 0:
                    # Get the content between opening and closing tags
                    start = open_positions[tag_name][i][0] + len(open_patterns[tag_name])
                    end = close_positions[tag_name][i][0]
                    
                    # Make these tokens maskable (1)
                    maskable[start:end] = 1
            
            # Make special tokens not maskable (0)
            for token_id in special_token_ids:
                maskable = maskable * (self.token_ids[i] != token_id).int()
            
            # Only include tokens up to the actual length in the result
            seq_len = self.lengths[i].item()
            results.append(maskable[:seq_len].tolist())
        
        return results

    def mask(self, percentage: float):
        """
        Masks a percentage of the tokens in the collection. 
        Only masks tokens that are marked as maskable.
        
        Args:
            percentage: Percentage of maskable tokens to mask (0.0 to 1.0)
        
        Returns:
            A new TokenizedExamples instance with masked tokens
        """
        # Convert maskable lists to tensor
        maskable_tensor = torch.zeros_like(self.token_ids)
        for i, maskable_list in enumerate(self.maskable):
            maskable_tensor[i, :len(maskable_list)] = torch.tensor(maskable_list, device=self.token_ids.device)
        
        # Only consider tokens that are maskable and have attention
        mask = (torch.rand_like(self.token_ids.float()) < percentage) & (maskable_tensor == 1)
        
        # Create masked inputs
        masked_inputs = self.token_ids.clone()
        masked_inputs[mask] = self.tokenizer.mask_token_id
        
        return TokenizedExamples(self.tokenizer, masked_inputs, self.attention_mask)
