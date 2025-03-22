"""
ReasoningExample class for representing examples with question, reasoning, and answer.
"""
from typing import List, Iterator


class ReasoningExample:
    """
    Represents a reasoning example with a question, reasoning steps, and an answer.
    
    This class provides a clean interface for working with reasoning examples,
    including methods for creating, formatting, and masking examples.
    """
    
    def __init__(self, question: str, reasoning_steps: List[str], answer: str):
        """
        Create a new reasoning example.
        
        Args:
            question: The question being asked
            reasoning_steps: The reasoning steps as a list of strings
            answer: The final answer
        """
        self.question = question
        self.reasoning_steps = reasoning_steps
        self.answer = answer
    
    def __iter__(self) -> Iterator[str]:
        """
        Makes the ReasoningExample iterable, yielding [question, reasoning, answer].
        This allows unpacking like: question, reasoning, answer = example
        """
        yield self.question
        yield "\n".join(self.reasoning_steps)
        yield self.answer

from transformers import AutoTokenizer


def tokenize(example:ReasoningExample, tokenizer:AutoTokenizer, max_length=512):
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
        text="[SEP]".join(example),
        max_length=max_length, 
        truncation=True, 
        padding='max_length',
        return_tensors='pt',
        return_attention_mask=True,
    )
    
    return TokenizedExample(example, tokenizer, encoded['input_ids'], encoded['attention_mask'])


class TokenizedExample(ReasoningExample):
    """ReasoningExample with tokenization information for model input."""
    
    def __init__(self, example: ReasoningExample, tokenizer:AutoTokenizer, token_ids, attention_mask):
        """
        Create a tokenized example.
        
        Args:
            example: Original ReasoningExample
            token_ids: Token IDs from tokenizer
            attention_mask: Attention mask from tokenizer
        """
        super().__init__(
            example.question,
            example.reasoning_steps,
            example.answer
        )
        self.tokenizer = tokenizer
        self.token_ids = token_ids
        self.attention_mask = attention_mask
        
  
    
    
