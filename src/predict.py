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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', const='', default=None, type=str, nargs='?', help='Path to specific checkpoint to load')
args = parser.parse_args()

setup()

checkpoint = args.checkpoint if args.checkpoint is not None else get_latest_checkpoint("lightning_logs")

if not checkpoint:
    model = MaskedDiffusionModel()
    print("No checkpoint found, using vanilla model.")
else:
    model = MaskedDiffusionModel.load_from_checkpoint(checkpoint)
    print(f"Loading checkpoint: {checkpoint}")

print(model.generate("What is 2 + 2? [MASK]"))
print(model.generate("What is 4 + 9? [MASK]"))
print(model.generate("What is 9 + 18? [MASK]"))
print(model.generate("What is 45 + 24? [MASK]"))
print(model.generate("What is 31 + 12? [MASK]"))
print(model.generate("What is 99 + 99? [MASK]"))

print(model.generate("What is [MASK] + [MASK]? 3"))
print(model.generate("What is [MASK] + [MASK]? 27"))
print(model.generate("What is [MASK] + [MASK]? 100"))
