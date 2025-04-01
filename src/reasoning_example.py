"""
ReasoningExample class for representing examples with question, reasoning, and answer.
"""
from typing import List, Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class ReasoningExample:
    """
    Represents a reasoning example with a question, reasoning steps, and an answer.
    
    This class provides a clean interface for working with reasoning examples,
    including methods for creating, formatting, and masking examples.
    """
    question: str
    answer: str
    
    def __iter__(self) -> Iterator[str]:
        """
        Makes the ReasoningExample iterable, yielding [question, reasoning, answer].
        This allows unpacking like: question, reasoning, answer = example
        """
        yield self.question
        yield self.answer

    def __str__(self) -> str:
        return "[SEP]".join(self)
        


