import torch
from src.update_mask import select_top_confidence_positions

def test_select_top_confidence_positions_batch():
    batch_size = 2
    seq_len = 4
    vocab_size = 3
    
    masked = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    masked[0, 1:3] = True  # Mask positions 1,2 in first sequence
    masked[1, 2:4] = True  # Mask positions 2,3 in second sequence
    
    logits = torch.zeros((batch_size, seq_len, vocab_size))
    logits[0, 1] = torch.tensor([0.0, 1.0, 0.0])  # Lower confidence
    logits[0, 2] = torch.tensor([0.0, 2.0, 0.0])  # Higher confidence
    logits[1, 2] = torch.tensor([0.0, 1.0, 0.0])  # Lower confidence
    logits[1, 3] = torch.tensor([0.0, 2.0, 0.0])  # Higher confidence
    
    result = select_top_confidence_positions(masked, logits, to_unmask=1)
    
    assert result[0, 2].item()  # Should select higher confidence position in first sequence
    assert result[1, 3].item()  # Should select higher confidence position in second sequence
    assert result.sum().item() == 2  # Should unmask 1 position per sequence
