import os
import torch
import random
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from masked_diffusion_bert import MaskedDiffusionBERT
from addition_reasoning_dataset import AdditionReasoningDataset
from reasoning_example import TokenizedExamples

model = MaskedDiffusionBERT()
tokenizer = model.tokenizer

dataset = AdditionReasoningDataset(tokenizer, num_examples=100, max_number=100, r=random)

dataloader = DataLoader(
    dataset, 
    batch_size=2, 
    shuffle=True,
    num_workers=min(os.cpu_count() or 1, 16),
    pin_memory=True
)

trainer = Trainer( max_epochs=5, default_root_dir="checkpoints", accumulate_grad_batches=4, precision="bf16-mixed" )

print("Starting training...")
trainer.fit(model, dataloader)

# Test the model on a few examples
print("\nTesting model on a few examples:")
test_batch = next(iter(dataloader))
with torch.no_grad():
    # Create a TokenizedExamples instance directly since we already have tokenized data
    masked = TokenizedExamples.from_tensors(tokenizer, test_batch["input_ids"], test_batch["attention_mask"]).mask(0.2)
    for i, (updated_ids, _) in enumerate(model.predict(masked)):
        decoded = tokenizer.decode(updated_ids[0], skip_special_tokens=True)
        print(f"Example {i+1}:\n{decoded}\n")
