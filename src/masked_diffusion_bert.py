import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Iterator
from transformers import AutoTokenizer
from src.token_schedule import calculate_tokens_per_step
from src.update_mask import calculate_update_mask, gumbel_max_sampling

from transformers import (
    AutoModelForMaskedLM,
)
from src.bert_diffuser import BERTDiffuser


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

    def forward(self, state: BERTDiffuser):
        return self.model(input_ids=state.input_ids, attention_mask=state.attention_mask, labels=state.labels).logits

    def training_step(self, batch, batch_idx: int):
        """ Training step: 
            1. Randomly mask tokens
            2. Pass masked sequence through model
            3. Compute loss only on masked positions
        """
        # 1) Apply noise (randomly mask some tokens)
        state = BERTDiffuser.from_tensors(self.tokenizer, batch["input_ids"], batch["attention_mask"])
        
        # Use a masking probability between 0 and 1.0
        mask_prob = torch.rand(1).item()
        masked = state.mask(mask_prob)
        
        # 2) Forward pass
        logits = self.forward(masked)
        
        # 3) Compute loss only for masked tokens
        loss_indices = masked.masked.view(-1)  # Flatten the mask to match flattened logits
        loss = F.cross_entropy(
            logits.view(-1, self.model.config.vocab_size)[loss_indices], 
            masked.original_ids.view(-1)[loss_indices]
        )

        self.log("train_loss", loss, prog_bar=True)
        return loss
        

    def predict_step(self, state, to_unmask):
        """
        Performs one step of the prediction process, unmasking the most confident positions.
        
        Args:
            state: Current BERTDiffuser state
            to_unmask: Number of positions to unmask
            
        Returns:
            Updated BERTDiffuser state
        """
        # Forward pass to get logits
        logits = self.forward(state)
        
        # Calculate which positions to update
        update_mask = calculate_update_mask(state, logits, to_unmask)
        
        # Sample from the model's distribution
        sampled_tokens = gumbel_max_sampling(logits)
        
        # Update state with new tokens using the update mask
        return state.update(sampled_tokens, update_mask)


    def predict(self, state: BERTDiffuser) -> Iterator[BERTDiffuser]:
        """
        Iterative unmasking generator: Takes masked BERTDiffuser and gradually fills them in.
        Yields BERTDiffuser instances at each step of the unmasking process.
        
        Args:
            tokenized_examples: BERTDiffuser instance with masked tokens
        Yields:
            BERTDiffuser: Current state of BERTDiffuser with some positions unmasked
        """
        current_examples = state
        number_of_masks = current_examples.masked.sum().item()
        
        for to_unmask in calculate_tokens_per_step(number_of_masks):
            current_examples = self.predict_step(current_examples, to_unmask)
            yield current_examples


    def unmask(self, input_text, max_length=512):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Yields intermediate steps as an iterator.
        """
        state = BERTDiffuser.create([input_text], self.tokenizer, max_length)
        for updated in self.predict(state):
            # Yield the decoded text at each step
            yield self.tokenizer.decode(updated.input_ids[0], skip_special_tokens=True)


    def generate(self, input_text, max_length=512):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Returns only the final result.
        """
        final_result = None
        for result in self.unmask(input_text, max_length):
            final_result = result
        return final_result

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
