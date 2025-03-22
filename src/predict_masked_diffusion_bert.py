#!/usr/bin/env python
# modern_bert_masked_diffusion_iterative.py

"""
Minimal single-file example of using ModernBERT from Hugging Face
with a masked diffusion style training loop via PyTorch Lightning,
plus an iterative inference procedure that un-masks tokens step by step.

"""
import torch
from masked_diffusion_bert import MaskedDiffusionBERT


def main():
    # Set random seed for reproducibility
    # seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    # Initialize the model
    model = MaskedDiffusionBERT()
    
    # Test the model on a sample text with device handling
    test_text = """Question: What is the capital of Panama? Answer: [MASK][MASK]"""
    print(f"\nTest input: {test_text}")
    
    # Use the model's predict method with device-aware tensors
    for result in model.unmask(test_text):
        print(f"Prediction: {result}")
    

if __name__ == "__main__":
    main()
