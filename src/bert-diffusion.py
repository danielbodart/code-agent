#!/usr/bin/env python
# modern_bert_masked_diffusion_iterative.py

"""
Minimal single-file example of using ModernBERT from Hugging Face
with a masked diffusion style training loop via PyTorch Lightning,
plus an iterative inference procedure that un-masks tokens step by step.

"""

import math
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from lightning_fabric.utilities.seed import seed_everything
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)


def sample_logits(logits, temperature=1.0):
    """
    Takes a [batch_size, seq_len, vocab_size] logits tensor.
    If temperature=0, do greedy decoding (argmax).
    Else, apply temperature and sample from softmax.
    """
    if temperature == 0:
        # Greedy decode
        return logits.argmax(dim=-1)
    else:
        # Sample with temperature
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        
        # Reshape for multinomial sampling
        batch_size, seq_len, vocab_size = probs.size()
        flat_probs = probs.view(-1, vocab_size)
        
        # Sample from the flattened distribution
        samples = torch.multinomial(flat_probs, num_samples=1)
        
        # Reshape back to [batch_size, seq_len]
        return samples.view(batch_size, seq_len)


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

        # 2) Get the last logits from the predict generator
        last_logits = None
        for _, logits in self.predict(masked_inputs, attention_mask, fraction_per_step=0.1, temperature=1.0):
            last_logits = logits
        
        # 3) Compute loss only for masked positions using the last logits
        loss = F.cross_entropy(
            last_logits.view(-1, self.model.config.vocab_size)[mask.view(-1)], 
            input_ids.view(-1)[mask.view(-1)]
        )
        
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def predict(self, input_ids, attention_mask, fraction_per_step=0.1, temperature=1.0):
        """
        Iterative unmasking generator: Takes masked input_ids and gradually fills them in.
        Yields (updated_input_ids, logits) tuples at each step of the unmasking process.
        
        Args:
            input_ids: Tensor of token IDs, with some positions containing mask tokens
            attention_mask: Tensor indicating which tokens to attend to (1) vs ignore (0)
            fraction_per_step: Fraction of masked tokens to unmask in each step
            temperature: Controls randomness in token selection (higher = more random)
            
        Yields:
            tuple: (updated_input_ids, logits)
                - updated_input_ids: Current state of tokens with some positions unmasked
                - logits: Raw logits from the model for all positions
        """
        input_ids = input_ids.clone()
        
        # Create a mask to track which tokens were originally masked
        original_mask = (input_ids == self.mask_token_id)
        
        # Calculate the number of masked tokens
        num_masked_tokens = original_mask.sum().item()
        
        # Add 50% extra steps for refinement
        # For 4 tokens: 4 + 4*0.5 = 6 steps total
        total_steps = max(2, int(num_masked_tokens * 1.5))
        
        # Calculate how many tokens to unmask per step (10% of original masked tokens)
        unmask_per_step = max(1, int(fraction_per_step * num_masked_tokens))
        
        # Track which tokens have been unmasked so far
        unmasked_so_far = torch.zeros_like(original_mask)
        
        for _ in range(total_steps):
            # Identify which tokens are still masked
            still_masked = original_mask & (~unmasked_so_far)
            
            # Get indices of still masked tokens
            masked_indices = still_masked.nonzero(as_tuple=False)
            
            # Select random subset of masked tokens to reveal (up to unmask_per_step)
            indices_to_unmask = masked_indices[torch.randperm(masked_indices.size(0))[:min(unmask_per_step, masked_indices.size(0))]]
            
            # Get indices of previously unmasked tokens
            unmasked_indices = (original_mask & unmasked_so_far).nonzero(as_tuple=False)
            
            # Combine both sets of indices to update
            all_indices_to_update = torch.cat([indices_to_unmask, unmasked_indices], dim=0)

            # Forward pass to get logits for current state
            logits = self.forward(input_ids, attention_mask)
            
            # Apply temperature and sample or take argmax
            predicted_ids = sample_logits(logits, temperature)
            
            # Update all eligible tokens
            for (b_idx, t_idx) in all_indices_to_update:
                input_ids[b_idx, t_idx] = predicted_ids[b_idx, t_idx]
                # Mark newly unmasked tokens
                if still_masked[b_idx, t_idx]:
                    unmasked_so_far[b_idx, t_idx] = True
            
            # Yield both the updated input_ids and the logits
            yield input_ids.clone(), logits

    def unmask(self, input_text, fraction_per_step=0.1, temperature=1.0, max_length=512):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Yields intermediate steps as an iterator.
        """
        tokens = self.tokenizer(
            input_text, return_tensors='pt', max_length=max_length, truncation=True, padding='max_length'
        )
        input_ids = tokens['input_ids'].clone()
        attention_mask = tokens['attention_mask']
        
        # Use predict generator for iterative unmasking
        for updated_ids, _ in self.predict(input_ids, attention_mask, fraction_per_step, temperature):
            # Yield the decoded text at each step
            yield self.tokenizer.decode(updated_ids[0], skip_special_tokens=True)

    def generate(self, input_text, fraction_per_step=0.1, temperature=1.0, max_length=512):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Returns only the final result.
        """
        # Use the unmask generator but only return the last item
        final_result = None
        for result in self.unmask(input_text, fraction_per_step, temperature, max_length):
            final_result = result
        return final_result



###############################
# 4) Main Function
###############################
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or use a masked diffusion BERT model')
    parser.add_argument('--train', action='store_true', help='Force training even if checkpoint exists')
    args = parser.parse_args()

    # Set random seed for reproducibility
    # seed_everything(42)
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
            max_length=512,
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
    
    # Define model checkpoint path
    model_path = "masked_diffusion_model"
    checkpoint_path = f"{model_path}.ckpt"
    
    # Check if checkpoint exists and load it
    should_train = args.train or not os.path.exists(checkpoint_path)
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        model = MaskedDiffusionBERT.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
        print("Checkpoint loaded successfully")
    else:
        print("No checkpoint found, training from scratch")
        should_train = True
    
    # PyTorch Lightning Trainer
    trainer = Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=5
    )

    should_train = False

    # Train only if needed
    if should_train:
        print("Starting training...")
        trainer.fit(model, dataloader)
        
        # Save the trained model
        model.save_hyperparameters()
        trainer.save_checkpoint(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    else:
        print("Skipping training as checkpoint exists and --train flag was not provided")
    
    # Test the model on a sample text
    test_text = """Question: How many people live in [MASK][MASK]? Answer: [MASK][MASK]"""
    print(f"\nTest input: {test_text}")
    
    print("\nWith temperature=1.0 (balanced):")
    for result in model.unmask(test_text, temperature=1.0):
        print(f"Prediction: {result}")
    
    print("\nWith temperature=0.5 (more focused):")
    for result in model.unmask(test_text, temperature=0.5):
        print(f"Prediction: {result}")
    
    print("\nWith temperature=1.5 (more random):")
    for result in model.unmask(test_text, temperature=1.5):
        print(f"Prediction: {result}")


if __name__ == "__main__":
    main()
