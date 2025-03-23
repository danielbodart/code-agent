import torch
from torch.utils.data import Dataset
from functools import cached_property
import random
from src.data_generator import generate_addition_example
from src.reasoning_example import TokenizedExamples

class AdditionReasoningDataset(Dataset):
    def __init__(self, tokenizer, num_examples=10000, max_number=1000, r=random):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_number = max_number
        self.r = r
    
    @cached_property
    def examples(self):
        return [generate_addition_example(r=self.r, max_number=self.max_number) for _ in range(self.num_examples)]
    
    @cached_property
    def tokenized(self):
        return TokenizedExamples.create(self.examples, self.tokenizer)
    
    def __len__(self):
        return self.num_examples
    
    def __getitem__(self, idx):
        return self.tokenized[idx]
