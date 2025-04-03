import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Iterator
from transformers import AutoTokenizer
from src.noise_schedule import noise_schedule
from src.update_mask import select_top_confidence_positions, gumbel_max_sampling

from transformers import ModernBertForMaskedLM
from src.masked_diffusion_state import MaskedDiffusionState, tokenize

class MaskedDiffusionModel(pl.LightningModule):
    def __init__(self, model_name="answerdotai/ModernBERT-large", lr=1e-5):
        super().__init__()
        self.model = ModernBertForMaskedLM.from_pretrained(model_name).cuda()
        self.model.train()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.lr = lr

    def forward(self, state: MaskedDiffusionState):
        return self.model(input_ids=state.input_ids, attention_mask=state.attention_mask, labels=state.labels)

    def training_step(self, batch, batch_idx: int):
        """ Training step: 
            1. Randomly mask tokens
            2. Pass masked sequence through model
            3. Compute loss only on masked positions
        """
        state = MaskedDiffusionState.from_batch(self.tokenizer, batch)
        
        # 1) Apply noise (randomly mask some tokens)
        mask_prob = torch.rand(1).item()
        masked = state.mask(mask_prob)
        
        # 2) Forward pass and compute loss
        loss = self.forward(masked).loss
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
        

    def predict(self, state: MaskedDiffusionState) -> Iterator[MaskedDiffusionState]:
        """
        Iterative unmasking generator: Takes masked MaskedDiffusionState and gradually fills them in.
        Yields MaskedDiffusionState instances at each step of the unmasking process.
        
        Args:
            state: MaskedDiffusionState with masked tokens
        Yields:
            MaskedDiffusionState: Current state of MaskedDiffusionState with some positions unmasked
        """
        current_state = state
        max_masked_tokens = current_state.masked.sum(dim=1).max().item()

        for masks_in_step in noise_schedule(max_masked_tokens):
            # Forward pass to get logits
            logits = self.forward(current_state).logits
            
            # Calculate which positions to update
            update_mask = select_top_confidence_positions(current_state.masked, logits, masks_in_step)
            
            # Sample from the model's distribution
            sampled_tokens = gumbel_max_sampling(logits)
            
            # Update state with new tokens using the update mask
            current_state = current_state.update(sampled_tokens, update_mask)
            yield current_state


    def unmask(self, input_text, max_length=512, skip_special_tokens=True):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Yields intermediate steps as an iterator.
        """
        state = MaskedDiffusionState.from_batch(self.tokenizer, tokenize(self.tokenizer, [str(input_text)], max_length=max_length))
        for updated in self.predict(state):
            # Yield the decoded text at each step
            yield self.tokenizer.decode(updated.input_ids[0], skip_special_tokens=skip_special_tokens)


    def generate(self, input_text, max_length=512, skip_special_tokens=True):
        """
        Iterative unmasking: Takes an input text with [MASK] tokens and gradually fills it in.
        - Returns only the final result.
        """
        final_result = None
        for result in self.unmask(input_text, max_length, skip_special_tokens):
            final_result = result
        return final_result

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
