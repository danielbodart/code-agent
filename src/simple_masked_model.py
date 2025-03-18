import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional
import random
import os
from src.data_generator import generate_addition_example

class SimpleTokenizer:
    """A minimal tokenizer for text data."""
    def __init__(self, examples=None):
        # Define special tokens
        self.mask_token = "<MASK>"
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        
        # Special token IDs
        self.mask_token_id = 0
        self.pad_token_id = 1
        self.unk_token_id = 2
        
        # Initialize with special tokens
        self.special_tokens = [self.mask_token, self.pad_token, self.unk_token]
        
        # Build vocabulary from examples if provided
        if examples is not None:
            self.build_vocab_from_examples(examples)
        else:
            # Fallback to a minimal default vocabulary
            self.vocab = self.special_tokens + ["the", "a", "an", "in", "on", "at", "is", 
                     "and", "of", "to", "it", "that", "with", "for", "as", "was", "by"]
            self.token2id = {token: i for i, token in enumerate(self.vocab)}
            self.id2token = {i: token for i, token in enumerate(self.vocab)}
            self.vocab_size = len(self.vocab)
    
    def build_vocab_from_examples(self, examples):
        """Build vocabulary from a list of example texts."""
        # Start with special tokens
        vocab = set(self.special_tokens)
        
        # Add all tokens from examples
        for example in examples:
            # For mathematical expressions, tokenize by spaces and also separate operators
            tokens = []
            for token in example.split():
                # Try to convert to number
                try:
                    num = int(token)
                    tokens.append(token)  # Keep numbers as tokens
                except ValueError:
                    # If not a number, add as is
                    tokens.append(token)
            
            vocab.update(tokens)
        
        # Convert to list and sort for deterministic ordering
        self.vocab = self.special_tokens + sorted(list(vocab - set(self.special_tokens)))
        
        # Create token to ID mapping
        self.token2id = {token: i for i, token in enumerate(self.vocab)}
        self.id2token = {i: token for i, token in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        print(f"Built vocabulary with {self.vocab_size} tokens")
    
    def encode(self, text: str, max_length: int = 32) -> List[int]:
        """Convert text to token IDs."""
        # For mathematical expressions, tokenize by spaces and also separate operators
        tokens = []
        for token in text.split():
            # Handle operators and special characters separately
            if token in ['+', '-', '*', '/', '=']:
                tokens.append(token)
            else:
                # Try to convert to number
                try:
                    num = int(token)
                    tokens.append(token)  # Keep numbers as tokens
                except ValueError:
                    # If not a number, add as is
                    tokens.append(token)
        
        # Convert to token IDs
        token_ids = [self.token2id.get(token, self.unk_token_id) for token in tokens[:max_length]]
        
        # Pad to max_length
        if len(token_ids) < max_length:
            token_ids = token_ids + [self.pad_token_id] * (max_length - len(token_ids))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        tokens = [self.id2token[id] for id in token_ids if id not in [self.pad_token_id]]
        
        # For mathematical expressions, we want to keep the original spacing
        result = " ".join(tokens)
        
        # Remove mask tokens from the result
        result = result.replace(self.mask_token + " ", "").replace(" " + self.mask_token, "").replace(self.mask_token, "")
        
        return result


class TextDataset(Dataset):
    """Simple dataset for text examples."""
    def __init__(self, texts: List[str], tokenizer: SimpleTokenizer, max_length: int = 32):
        self.examples = [torch.tensor(tokenizer.encode(text, max_length)) for text in texts]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class SimpleMaskedModel(nn.Module):
    """A simple transformer for masked language modeling."""
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        
        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Simple transformer (extremely simplified)
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        # Output projection
        self.output = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, x):
        # Create embeddings
        x = self.embedding(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Project to vocabulary
        logits = self.output(x)
        
        return logits


class MaskedDiffusion:
    """Diffusion model for masked language modeling."""
    
    def __init__(self, model: nn.Module, tokenizer: SimpleTokenizer, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def diffuse(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """Forward diffusion process - masks tokens randomly with probability t."""
        mask = torch.rand_like(x.float()) < t
        x_t = x.clone()
        x_t[mask] = self.tokenizer.mask_token_id
        return x_t
    
    def train_step(self, x: torch.Tensor, optimizer) -> float:
        """Single training step with random timestep t."""
        optimizer.zero_grad()
        
        # Sample random timestep t between 0 and 1
        t = random.random()
        
        # Apply forward diffusion (masking)
        x_t = self.diffuse(x, t)
        
        # Get model predictions
        logits = self.model(x_t)
        
        # Compute loss only on masked tokens
        mask = (x_t == self.tokenizer.mask_token_id)
        
        # Skip if no tokens are masked
        if not mask.any():
            return 0.0
        
        # The key insight from the papers - simplified loss formulation
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1))[mask.view(-1)], 
            x.view(-1)[mask.view(-1)]
        )
        
        # Check for NaN loss
        if torch.isnan(loss):
            print("Warning: NaN loss detected, skipping backward pass")
            return 0.0
            
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def sample(self, seq_length: int, steps: int = 10) -> torch.Tensor:
        """Generate a new sequence through the reverse diffusion process."""
        # Start with all masked tokens
        x = torch.full((1, seq_length), self.tokenizer.mask_token_id, device=self.device)
        
        print("Sampling: ", end="")
        # Gradually unmask from t=1 to t=0
        for i in range(steps):
            print(".", end="", flush=True)
            t = 1.0 - (i / steps)
            
            with torch.no_grad():
                # Predict token probabilities
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
            
            # Determine which tokens to unmask at this step
            remaining_masks = (x == self.tokenizer.mask_token_id)
            
            # Simple strategy: unmask a fixed percentage at each step
            num_to_unmask = max(1, int(remaining_masks.sum().item() / (steps - i)))
            
            if remaining_masks.any():
                # Get probabilities for masked positions
                masked_probs = probs[remaining_masks].view(-1, probs.size(-1))
                
                # Sample from predicted distribution
                categorical = torch.distributions.Categorical(masked_probs)
                sampled = categorical.sample()
                
                # Get indices of tokens to unmask
                mask_indices = torch.nonzero(remaining_masks.view(-1), as_tuple=True)[0]
                
                # Limit to num_to_unmask
                if len(mask_indices) > num_to_unmask:
                    perm = torch.randperm(len(mask_indices))
                    mask_indices = mask_indices[perm[:num_to_unmask]]
                
                # Update tokens
                x.view(-1)[mask_indices] = sampled[:len(mask_indices)]
        
        print(" Done")
        return x
    
    def complete_masked(self, template: str, steps: int = 10) -> str:
        """Complete a template with masked tokens.
        
        Args:
            template: A string with <MASK> tokens to be completed
            steps: Number of steps for the reverse diffusion process
            
        Returns:
            The completed template with predictions for masked tokens
        """
        self.model.eval()
        
        # Tokenize the template
        tokens = template.split()
        token_ids = []
        
        for token in tokens:
            if token == self.tokenizer.mask_token:
                token_ids.append(self.tokenizer.mask_token_id)
            else:
                token_ids.append(self.tokenizer.token2id.get(token, self.tokenizer.unk_token_id))
        
        # Convert to tensor
        x = torch.tensor([token_ids], device=self.device)
        
        # Gradually unmask the <MASK> tokens
        for i in range(steps):
            # Calculate timestep - going from t=1.0 to t=0.0
            t = 1.0 - (i / steps)
            
            with torch.no_grad():
                # Predict token probabilities
                logits = self.model(x)
                probs = F.softmax(logits, dim=-1)
            
            # Identify masked positions
            mask_positions = (x == self.tokenizer.mask_token_id)
            
            # Skip if no masks remain
            if not mask_positions.any():
                break
                
            # Use timestep to determine how many tokens to unmask
            # At the beginning (t close to 1), unmask fewer tokens
            # At the end (t close to 0), unmask more tokens
            remaining_masks = mask_positions.sum().item()
            unmask_ratio = (1.0 - t) * 0.5  # Adjust this factor as needed
            num_to_unmask = max(1, int(remaining_masks * unmask_ratio))
            
            # Get probabilities for masked positions
            masked_probs = probs[mask_positions].view(-1, probs.size(-1))
            
            # Sample from predicted distribution
            categorical = torch.distributions.Categorical(masked_probs)
            sampled = categorical.sample()
            
            # Get indices of tokens to unmask
            mask_indices = torch.nonzero(mask_positions.view(-1), as_tuple=True)[0]
            
            # Limit to num_to_unmask
            if len(mask_indices) > num_to_unmask:
                perm = torch.randperm(len(mask_indices))
                mask_indices = mask_indices[perm[:num_to_unmask]]
            
            # Update tokens
            x.view(-1)[mask_indices] = sampled[:len(mask_indices)]
        
        # Decode the completed template
        completed = self.tokenizer.decode(x[0].cpu().tolist())
        return completed
    
    def generate(self, max_length: int = 32) -> str:
        """Generate and return a text sample."""
        self.model.eval()
        sample = self.sample(max_length)
        return self.tokenizer.decode(sample[0].cpu().tolist())
    
    def train(self, dataloader: DataLoader, epochs: int = 5, lr: float = 1e-4):
        """Train the diffusion model."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            print(f"Epoch {epoch+1}/{epochs} - Processing {len(dataloader)} batches...")
            
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 10 == 0:
                    print(f"  Batch {batch_idx}/{len(dataloader)}", end="\r")
                
                batch = batch.to(self.device)
                loss = self.train_step(batch, optimizer)
                total_loss += loss
                num_batches += 1
            
            avg_loss = total_loss/num_batches if num_batches > 0 else float('nan')
            print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")
    
    def save_model(self, path: str):
        """Save the model and tokenizer to the specified path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state
        model_state = {
            'model_state_dict': self.model.state_dict(),
            'vocab_size': self.tokenizer.vocab_size,
            'embedding_dim': self.model.embedding_dim if hasattr(self.model, 'embedding_dim') else 128,
            'hidden_dim': self.model.hidden_dim if hasattr(self.model, 'hidden_dim') else 256
        }
        
        # Save tokenizer state
        tokenizer_state = {
            'token2id': self.tokenizer.token2id,
            'id2token': self.tokenizer.id2token,
            'vocab_size': self.tokenizer.vocab_size,
            'mask_token': self.tokenizer.mask_token,
            'unk_token': self.tokenizer.unk_token,
            'pad_token': self.tokenizer.pad_token
        }
        
        # Save both states
        torch.save({
            'model': model_state,
            'tokenizer': tokenizer_state
        }, path)
        
        print(f"Model and tokenizer saved to {path}")
    
    @classmethod
    def load_model(cls, path: str):
        """Load a model and tokenizer from the specified path."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model found at {path}")
        
        # Load the saved state
        state = torch.load(path)
        
        # Extract model and tokenizer states
        model_state = state['model']
        tokenizer_state = state['tokenizer']
        
        # Recreate the tokenizer
        tokenizer = SimpleTokenizer([])  # Create with empty examples
        tokenizer.token2id = tokenizer_state['token2id']
        tokenizer.id2token = tokenizer_state['id2token']
        tokenizer.vocab_size = tokenizer_state['vocab_size']
        tokenizer.mask_token = tokenizer_state['mask_token']
        tokenizer.unk_token = tokenizer_state['unk_token']
        tokenizer.pad_token = tokenizer_state['pad_token']
        
        # Reconstruct the vocabulary from token2id
        tokenizer.vocab = [tokenizer.id2token[i] for i in range(tokenizer.vocab_size)]
        
        # Set token IDs
        tokenizer.mask_token_id = tokenizer.token2id[tokenizer.mask_token]
        tokenizer.unk_token_id = tokenizer.token2id[tokenizer.unk_token]
        tokenizer.pad_token_id = tokenizer.token2id[tokenizer.pad_token]
        
        print(f"Loaded tokenizer with vocabulary size: {tokenizer.vocab_size}")
        
        # Recreate the model
        model = SimpleMaskedModel(
            model_state['vocab_size'],
            embedding_dim=model_state['embedding_dim'],
            hidden_dim=model_state['hidden_dim']
        )
        
        # Load the model weights
        model.load_state_dict(model_state['model_state_dict'])
        
        # Create and return the diffusion model
        diffusion = cls(model, tokenizer)
        print(f"Model and tokenizer loaded from {path}")
        return diffusion


# Example usage function
def run_minimal_example():
    # Define model path
    model_path = "models/masked_model.pt"
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        diffusion = MaskedDiffusion.load_model(model_path)
    else:
        # Generate a reasonable number of examples
        print("No existing model found. Training new model...")
        print("Generating examples...")
        texts = [generate_addition_example() for _ in range(20000)]
        
        # Initialize components
        print("Building tokenizer...")
        tokenizer = SimpleTokenizer(texts)
        dataset = TextDataset(texts, tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Create model with larger capacity
        print("Creating model...")
        model = SimpleMaskedModel(
            tokenizer.vocab_size,
            embedding_dim=256,
            hidden_dim=512
        )
        diffusion = MaskedDiffusion(model, tokenizer)
        
        # Train for more epochs
        print("Training model...")
        diffusion.train(dataloader, epochs=15, lr=1e-4)
        
        # Save the trained model
        print("Saving model...")
        diffusion.save_model(model_path)

    print("\nGenerating a sample:")
    print(diffusion.generate())
    
    # Complete masked templates
    print("\nCompleting masked templates:")
    templates = [
        "2 + <MASK> = 4",
        "<MASK> + 3 = 8",
        "10 + 20 = <MASK>",
        "<MASK> + <MASK> = 10",
        "15 + <MASK> = 25",
        "<MASK> + 7 = 12",
        "5 + 8 = <MASK>",
        "<MASK> + <MASK> = 15"
    ]
    
    for template in templates:
        print(f"Template: {template}")
        print(f"Completed: {diffusion.complete_masked(template)}")
        print()


# Run the example
if __name__ == "__main__":
    run_minimal_example()
