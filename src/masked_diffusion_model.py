import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl

class MaskedDiffusionModel(pl.LightningModule):
    def __init__(self, vocab_size=100, embedding_dim=64, T=1000, hidden_dim=128, num_layers=2, mask_token_id=0):
        """
        Initialize the masked diffusion model based on the MDLM approach.
        
        Args:
            vocab_size (int): Size of the vocabulary (number of unique tokens).
            embedding_dim (int): Dimension of the token embeddings.
            T (int): Number of diffusion timesteps.
            hidden_dim (int): Hidden dimension of the transformer's feedforward layers.
            num_layers (int): Number of transformer encoder layers.
            mask_token_id (int): Token ID for the mask token (e.g., 0 for '[MASK]').
        """
        super().__init__()
        
        # Embedding layer for tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Timestep embedding (confirmed in MDLM paper, Section 3.2)
        self.time_embedding = nn.Embedding(T, embedding_dim)
        
        # Transformer encoder with visible layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,      # Input/output dimension
            nhead=4,                    # Number of attention heads
            dim_feedforward=hidden_dim, # Feedforward layer dimension
            batch_first=True            # Input shape: (batch_size, seq_len, embedding_dim)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embedding_dim)
        )
        
        # Output layer to predict token probabilities
        self.out = nn.Linear(embedding_dim, vocab_size)
        
        # Diffusion parameters
        self.T = T
        self.mask_token_id = mask_token_id
        
        # Linear masking schedule (simplified; MDLM uses a schedule to control masking)
        self.mask_prob_schedule = torch.linspace(0, 1, T)  # 0% to 100% masking over T steps

    def get_masked_sequence(self, tokens, t):
        """
        Apply the masking-based forward process based on timestep t.
        
        Args:
            tokens (Tensor): Original token indices, shape (batch_size, seq_len)
            t (Tensor): Timesteps, shape (batch_size,)
        
        Returns:
            Tensor: Masked token sequence, shape (batch_size, seq_len)
        """
        # Ensure t is on the same device as self.mask_prob_schedule
        t = t.to(self.mask_prob_schedule.device)
        batch_size, seq_len = tokens.size()
        mask_prob = self.mask_prob_schedule[t].view(batch_size, 1).to(tokens.device)  # Ensure mask_prob is on the same device as tokens
        mask = torch.rand(batch_size, seq_len, device=tokens.device) < mask_prob
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        return masked_tokens

    def forward(self, x_t, t):
        """
        Forward pass to predict token probabilities from the masked sequence.
        
        Args:
            x_t (Tensor): Masked token indices, shape (batch_size, seq_len)
            t (Tensor): Timesteps, shape (batch_size,)
        
        Returns:
            Tensor: Predicted token logits, shape (batch_size, seq_len, vocab_size)
        """
        # Embed the masked tokens
        x_emb = self.embedding(x_t)  # (batch_size, seq_len, embedding_dim)
        
        # Embed the timestep and add it to token embeddings (MDLM approach)
        time_emb = self.time_embedding(t)  # (batch_size, embedding_dim)
        time_emb = time_emb.unsqueeze(1).expand(-1, x_emb.size(1), -1)  # (batch_size, seq_len, embedding_dim)
        x_emb = x_emb + time_emb
        
        # Process through transformer
        transformer_out = self.transformer(x_emb)  # (batch_size, seq_len, embedding_dim)
        
        # Predict token probabilities
        logits = self.out(transformer_out)  # (batch_size, seq_len, vocab_size)
        return logits

    def training_step(self, batch, batch_idx):
        """
        Training step: mask tokens and predict the original tokens at masked positions.
        
        Args:
            batch (Tensor): Token indices, shape (batch_size, seq_len)
            batch_idx (int): Batch index
        
        Returns:
            Tensor: Cross-entropy loss
        """
        tokens = batch[0]  # (batch_size, seq_len)
        batch_size = tokens.size(0)
        
        # Sample random timesteps
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        
        # Apply masking to create the noisy input
        x_t = self.get_masked_sequence(tokens, t)
        
        # Predict token probabilities
        logits = self(x_t, t)  # (batch_size, seq_len, vocab_size)
        
        # Compute loss only on masked positions (MDLM objective)
        mask = (x_t == self.mask_token_id)
        logits_flat = logits[mask]  # (num_masked, vocab_size)
        targets_flat = tokens[mask]  # (num_masked)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits_flat, targets_flat)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure the optimizer."""
        return optim.Adam(self.parameters(), lr=1e-4)