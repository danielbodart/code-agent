from torch.utils.data import Dataset
from functools import lru_cache
import random
from src.data_generator import generate_addition_example
from src.masked_diffusion_state import MaskedDiffusionState, tokenize

class AdditionReasoningDataset(Dataset):
    def __init__(self, tokenizer, num_examples=10000, max_number=1000, start_index=0, max_tokens=512):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.max_number = max_number
        self.start_index = start_index
        self.max_tokens = max_tokens
    
    def __len__(self):
        return self.num_examples
    
    @lru_cache(maxsize=128)
    def __getitem__(self, idx):
        example = generate_addition_example( r=random.Random(idx + self.start_index), max_number=self.max_number )
        return MaskedDiffusionState.from_batch(self.tokenizer, tokenize(self.tokenizer, [str(example)], max_length=self.max_tokens))[0]
