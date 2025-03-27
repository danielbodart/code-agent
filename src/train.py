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

model = MaskedDiffusionModel()
tokenizer = model.tokenizer

dataset = AdditionReasoningDataset(tokenizer, num_examples=10000, max_number=1000)

dataloader = DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=True,
    num_workers=min(os.cpu_count() or 1, 16),
    pin_memory=True
)

trainer = Trainer( max_epochs=6, default_root_dir="checkpoints", accumulate_grad_batches=8, precision="bf16-mixed" )

print("Starting training...")
trainer.fit(model, dataloader)

# Test the model on a few examples
print("\nTesting model on a few examples:")
test_batch = next(iter(dataloader))
with torch.no_grad():
    # Create a BERTDiffuser instance directly since we already have tokenized data
    masked = MaskedDiffusionState.from_batch(tokenizer, test_batch).mask(0.1)
    for i, (updated_examples) in enumerate(model.predict(masked)):
        decoded = tokenizer.decode(updated_examples.input_ids[0], skip_special_tokens=False)
        print(f"Example {i+1}:\n{decoded}\n")
