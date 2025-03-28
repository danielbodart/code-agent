import os
import torch
import random
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from masked_diffusion_model import MaskedDiffusionModel
from addition_reasoning_dataset import AdditionReasoningDataset
from src.masked_diffusion_state import MaskedDiffusionState
from pytorch_lightning import seed_everything
from src.setup import setup

seed_everything(42)
setup()

model = MaskedDiffusionModel(lr=1e-4)
tokenizer = model.tokenizer

dataset = AdditionReasoningDataset(tokenizer, num_examples=10000, max_number=1000)
dataloader = DataLoader(dataset, batch_size=32)

trainer = Trainer(
    max_epochs=10,
    accumulate_grad_batches=8, 
    precision="bf16-mixed",
    accelerator="gpu"
)

trainer.fit(model, dataloader)

model.model.cuda()

print(model.generate("What is 2 + 2?[SEP][MASK]"))
print(model.generate("What is 324 + 5324?[SEP][MASK][MASK]"))
