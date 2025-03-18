#!/usr/bin/env python
# modern_bert_masked_diffusion_iterative.py

"""
Minimal single-file example of using ModernBERT from Hugging Face
with a masked diffusion style training loop via PyTorch Lightning,
plus an iterative inference procedure that un-masks tokens step by step.

"""

import random
import math

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from lightning_fabric.utilities.seed import seed_everything
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
)


###############################
# 2) LightningModule
###############################


class MaskedDiffusionBERT(pl.LightningModule):
    def __init__(self, model_name="answerdotai/ModernBERT-base", lr=1e-4):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.lr = lr

    def forward(self, input_ids, attention_mask):
        """ Forward pass: Takes masked inputs, predicts original tokens. """
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits

    def training_step(self, batch, batch_idx):
        """ Training step: 
            1. Randomly mask tokens
            2. Pass masked sequence through model
            3. Compute loss only on masked positions
        """
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        
        # 1) Apply noise (randomly mask some tokens)
        p = torch.rand(1).item()  # Random mask fraction in [0,1)
        mask = (torch.rand_like(input_ids.float()) < p) & (input_ids != self.tokenizer.pad_token_id)
        masked_inputs = input_ids.clone()
        masked_inputs[mask] = self.mask_token_id  # Replace tokens with [MASK]

        # 2) Forward pass
        logits = self.forward(masked_inputs, attention_mask)

        # 3) Compute loss only for masked positions
        loss = F.cross_entropy(
            logits.view(-1, self.model.config.vocab_size)[mask.view(-1)], 
            input_ids.view(-1)[mask.view(-1)]
        )
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def predict(self, input_ids, attention_mask):
        """
        Single-step unmasking: Predicts missing tokens in one forward pass.
        Returns the most likely token for each masked position.
        """
        logits = self.forward(input_ids, attention_mask)
        predicted_ids = logits.argmax(dim=-1)  # Get top-1 prediction per token
        return predicted_ids

    def unmask(self, input_text, num_steps=5, fraction_per_step=0.1, max_length=16):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Yields intermediate steps as an iterator.
        - Uses `predict()` for single-step fills.
        """
        tokens = self.tokenizer(
            input_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length'
        )
        input_ids = tokens['input_ids'].clone()
        attention_mask = tokens['attention_mask']

        # Identify which tokens are masked
        still_masked = (input_ids == self.mask_token_id)

        for step in range(num_steps):
            if not still_masked.any():
                break  # Stop early if no more [MASK] tokens

            # Decide how many to unmask this step (e.g., 10% of still-masked tokens)
            masked_indices = still_masked.nonzero(as_tuple=False)
            num_masked = masked_indices.size(0)
            unmask_count = max(1, int(fraction_per_step * num_masked))

            # Select random subset of masked tokens to reveal
            chosen_indices = masked_indices[torch.randperm(num_masked)[:unmask_count]]

            # Single-step predict
            predicted_ids = self.predict(input_ids, attention_mask)

            # Replace only the chosen masked positions with predicted tokens
            for (b_idx, t_idx) in chosen_indices:
                input_ids[b_idx, t_idx] = predicted_ids[b_idx, t_idx]

            # Update mask tracking
            still_masked = (input_ids == self.mask_token_id)

            # Yield intermediate result
            yield self.tokenizer.decode(input_ids[0], skip_special_tokens=True)

    def generate(self, input_text, num_steps=5, fraction_per_step=0.1, max_length=16):
        """
        Final output: Calls `unmask()` and returns the last generated result.
        """
        final_text = None
        for text in self.unmask(input_text, num_steps, fraction_per_step, max_length):
            final_text = text  # Keep updating to get the last step
        return final_text




###############################
# 4) Main Function
###############################
def main():
    seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    # Load dataset
    print("Loading facebook/natural_reasoning dataset...")
    raw_dataset = load_dataset("facebook/natural_reasoning", split="train")
    
    # Filter to only include examples with a reference answer
    print("Filtering dataset...")
    filtered_dataset = raw_dataset.filter(
        lambda example: bool(example["reference_answer"]),
        desc="Filtering examples with reference answers"
    )
    
    # Limit to first 10,000 examples (or any other number)
    max_examples = 10000
    if len(filtered_dataset) > max_examples:
        print(f"Limiting to {max_examples} examples...")
        filtered_dataset = filtered_dataset.select(range(max_examples))
    
    # Combine question and reference_answer in one step
    print("Processing dataset...")
    processed_dataset = filtered_dataset.map(
        lambda example: {"text": f"{example['question']} {example['reference_answer']}"},
        remove_columns=filtered_dataset.column_names,  # Remove original columns
        desc="Combining question and answer"
    )
    
    print(f"Processed dataset size: {len(processed_dataset)}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    
    # Tokenize the dataset in batches for efficiency
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = processed_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=1000,  # Process 1000 examples at once
        desc="Tokenizing dataset",
        remove_columns=["text"]  # Remove the original text column
    )
    
    # Format dataset to return PyTorch tensors
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # Create DataLoader
    import os
    num_workers = min(os.cpu_count(), 16)  # Use available cores but cap at 16 to avoid excessive overhead
    print(f"Using {num_workers} workers for data loading")
    
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=8, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # This can speed up data transfer to GPU
    )
    
    # Initialize the model
    model = MaskedDiffusionBERT()
    
    # PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=3
    )
    
    # Train
    print("Starting training...")
    trainer.fit(model, dataloader)
    
    # Save the trained model
    model_path = "masked_diffusion_model"
    model.save_hyperparameters()
    trainer.save_checkpoint(f"{model_path}.ckpt")
    print(f"Model saved to {model_path}.ckpt")
    
    # Test the model on a sample text
    test_text = "What is the formula for kinetic energy? [MASK][MASK][MASK][MASK][MASK][MASK]"
    print(f"\nTest input: {test_text}")
    print("Generating prediction...")
    prediction = model.generate(test_text)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    main()
