import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from lightning_fabric.utilities.seed import seed_everything
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
)
from src.reasoning_example import TokenizedExamples

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
    def __init__(self, model_name="answerdotai/ModernBERT-large", lr=1e-5):
        super().__init__()
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.model.train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.lr = lr

    def forward(self, input_ids, attention_mask, labels):
        """ Forward pass: Takes masked inputs, predicts original tokens. """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

    def training_step(self, batch, batch_idx: int):
        """ Training step: 
            1. Randomly mask tokens
            2. Pass masked sequence through model
            3. Compute loss only on masked positions
        """
        # 1) Apply noise (randomly mask some tokens)
        tokenized = TokenizedExamples(self.tokenizer, batch["input_ids"], batch["attention_mask"])
        
        # Use a masking probability between 0.2 and 1.0
        mask_prob = 0.2 + 0.8 * torch.rand(1).item()
        
        masked = tokenized.mask(mask_prob)
        
        # 2) Forward pass
        # Use predict generator for iterative unmasking
        first_logits = None
        for _, logits in self.predict(masked.input_ids, masked.attention_mask, masked.labels):
            first_logits = logits
            break

        # 3) Compute loss only for masked positions
        loss_indices = masked.maskable.view(-1)
        loss = F.cross_entropy(
            first_logits.view(-1, self.model.config.vocab_size)[loss_indices], 
            masked.input_ids.view(-1)[loss_indices]
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss
        

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def predict(self, input_ids, attention_mask, labels, fraction_per_step=0.1, temperature=1.0):
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
            logits = self.forward(input_ids, attention_mask, labels)
            
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
        for updated_ids, _ in self.predict(input_ids, attention_mask, input_ids, fraction_per_step, temperature):
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
