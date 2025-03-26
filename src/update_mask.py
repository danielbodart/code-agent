import torch
import torch.nn.functional as F
from src.masked_diffusion_state import MaskedDiffusionState

def gumbel_max_sampling(logits):
    """Gumbel-max sampling for categorical distribution"""
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    return (logits + gumbel_noise).argmax(dim=-1)


def select_top_confidence_positions(state: MaskedDiffusionState, logits: torch.Tensor, to_unmask: int) -> torch.Tensor:
    """
    Calculate which positions to update based on model confidence.
    
    Args:
        state: Current MaskedDiffusionState
        logits: Model logits
        to_unmask: Number of positions to unmask
        
    Returns:
        Boolean mask indicating which positions to update
    """
    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Calculate confidence for each position (max probability)
    confidence = probs.max(dim=-1).values
    
    # Zero out confidence for already unmasked positions
    confidence = confidence * state.masked
    
    # Find the positions with highest confidence
    _, top_positions = torch.topk(confidence, to_unmask)
    
    # Create a mask for positions to update
    update_mask = torch.zeros_like(state.masked)
    update_mask.scatter_(1, top_positions, 1)
    
    return update_mask
