#!/usr/bin/env python
# modern_bert_masked_diffusion_iterative.py

"""
Minimal single-file example of using ModernBERT from Hugging Face
with a masked diffusion style training loop via PyTorch Lightning,
plus an iterative inference procedure that un-masks tokens step by step.

"""
from masked_diffusion_model import MaskedDiffusionModel
from src.setup import setup
from pytorch_lightning import seed_everything
from src.checkpoints import get_latest_checkpoint

seed_everything(42)
setup()

checkpoint = get_latest_checkpoint()

if checkpoint is None:
    model = MaskedDiffusionModel()
    print("No checkpoint found, using vanilla model.")
else:
    model = MaskedDiffusionModel.load_from_checkpoint(checkpoint)
    print(f"Loading checkpoint: {checkpoint}")

print(model.generate("What is 2 + 2?[SEP][MASK]"))
print(model.generate("What is 324 + 5324?[SEP][MASK][MASK]"))



