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


