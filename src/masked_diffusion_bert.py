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
from src.bert_diffuser import BERTDiffuser
import math
from typing import Iterator, Tuple, Any

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
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).logits

    def training_step(self, batch, batch_idx: int):
        """ Training step: 
            1. Randomly mask tokens
            2. Pass masked sequence through model
            3. Compute loss only on masked positions
        """
        # 1) Apply noise (randomly mask some tokens)
        tokenized = BERTDiffuser.from_tensors(self.tokenizer, batch["input_ids"], batch["attention_mask"])
        
        # Use a masking probability between 0 and 1.0
        mask_prob = torch.rand(1).item()
        masked = tokenized.mask(mask_prob)
        
        # 2) Forward pass
        unmasked = masked.unmask(1)
        logits = self.forward(unmasked.input_ids, unmasked.attention_mask, unmasked.labels)
        
        # 3) Compute loss only for unmasked labels that are not -100
        loss_indices = unmasked.labels != -100
        loss = F.cross_entropy(
            logits.view(-1, self.model.config.vocab_size)[loss_indices], 
            unmasked.original_ids.view(-1)[loss_indices]
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss
        

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def predict(self, tokenized_examples, fraction_per_step=0.1, temperature=1.0) -> Iterator[torch.Tensor]:
        """
        Iterative unmasking generator: Takes masked BERTDiffuser and gradually fills them in.
        Yields (updated_examples, logits) tuples at each step of the unmasking process.
        
        Args:
            tokenized_examples: BERTDiffuser instance with masked tokens
            fraction_per_step: Fraction of masked tokens to unmask in each step
            temperature: Controls randomness in token selection (higher = more random)
            
        Yields:
            tuple: (updated_examples, logits)
                - updated_examples: Current state of BERTDiffuser with some positions unmasked
                - logits: Raw logits from the model for all positions
        """
        total_steps = math.ceil(1.5 / fraction_per_step)
        total_steps = max(1, total_steps)  # Ensure at least 1 step
        
        current_examples = tokenized_examples
        
        for _ in range(total_steps):
            current_examples = current_examples.unmask(1) #fraction_per_step

            # Forward pass to get logits for current state
            logits = self.forward(
                current_examples.input_ids, 
                current_examples.attention_mask, 
                current_examples.labels
            )
            
            # Apply temperature and sample or take argmax
            predicted_ids = sample_logits(logits, temperature)
            
            # Update the examples with the predicted IDs
            current_examples = current_examples.update(predicted_ids)
            
            # Yield both the updated examples and the logits
            yield current_examples


    def unmask(self, input_text, fraction_per_step=0.1, temperature=1.0, max_length=512):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Yields intermediate steps as an iterator.
        """
        tokenized_examples = BERTDiffuser.create([input_text], self.tokenizer, max_length)
        for updated_examples in self.predict(tokenized_examples, fraction_per_step, temperature):
            # Yield the decoded text at each step
            yield self.tokenizer.decode(updated_examples.input_ids[0], skip_special_tokens=True)


    def generate(self, input_text, fraction_per_step=0.1, temperature=1.0, max_length=512):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Returns only the final result.
        """
        final_result = None
        for result in self.unmask(input_text, fraction_per_step, temperature, max_length):
            final_result = result
        return final_result
