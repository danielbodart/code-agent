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
    if not isinstance(pattern, torch.Tensor):
        pattern = torch.tensor(pattern, device=tensor.device)
    
    pattern_len = len(pattern)
    seq_len = tensor.size(0)
    
    # If pattern is longer than sequence, it can't be found
    if pattern_len > seq_len:
        return torch.tensor([], dtype=torch.long, device=tensor.device)
    
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

def create_tag_mask(tensor, tag_tokens):
    """
    Create a mask where all XML tags are marked as False (not maskable).
    
    Args:
        tensor: The token tensor to analyze
        tag_tokens: List of token IDs that indicate tags (like tag_start, tag_end)
    
    Returns:
        A boolean tensor where True indicates non-tag tokens
    """
    mask = torch.ones_like(tensor, dtype=torch.bool)
    
    # Mark positions with tag tokens as False
    for token in tag_tokens:
        mask = mask & (tensor != token)
    
    # Find tag spans
    in_tag = False
    for i in range(tensor.size(0)):
        if tensor[i] == tag_start:
            in_tag = True
        elif tensor[i] == tag_end:
            in_tag = False
            continue
        
        if in_tag:
            mask[i] = False
    
    return mask
