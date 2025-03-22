from typing import Union, List
import torch

tag_start = 29 # <
tag_end = 31 # >
close_tag_start = 870 # </

def open_tag(tag: Union[int, List[int]]):
    return [tag_start, *([tag] if isinstance(tag, int) else list(tag)), tag_end]

def close_tag(tag: Union[int, List[int]]):
    return [close_tag_start, *([tag] if isinstance(tag, int) else list(tag)), tag_end]

def find_sequence(tensor, pattern):
    """
    Find all occurrences of a sequence pattern in a tensor.
    
    Args:
        tensor: A 1D tensor to search in
        pattern: A list or tensor of token IDs to find
    
    Returns:
        A tensor of indices where the pattern starts
    """
    # Convert pattern to tensor if needed
    if not isinstance(pattern, torch.Tensor):
        pattern = torch.tensor(pattern, device=tensor.device)
    
    pattern_len = len(pattern)
    seq_len = tensor.size(0)
    
    # If pattern is longer than sequence, it can't be found
    if pattern_len > seq_len:
        return torch.tensor([], device=tensor.device)
    
    # Use unfold to create sliding windows
    windows = tensor.unfold(0, pattern_len, 1)
    
    # Compare each window with pattern
    matches = (windows == pattern.unsqueeze(0)).all(dim=1)
    match_indices = torch.where(matches)[0]
    
    return match_indices


def find_sequences_batch(batch_tensor, pattern):
    """
    Find all occurrences of a sequence pattern in a batch of tensors.
    
    Args:
        batch_tensor: A 2D tensor [batch_size, seq_len] to search in
        pattern: A list or tensor of token IDs to find
    
    Returns:
        A list of tensors, each containing indices where the pattern starts in the corresponding batch item
    """
    batch_size = batch_tensor.size(0)
    results = []
    
    for i in range(batch_size):
        results.append(find_sequence(batch_tensor[i], pattern))
    
    return results

