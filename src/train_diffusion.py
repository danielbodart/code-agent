import torch
import pytorch_lightning as pl
from src.masked_diffusion_model import MaskedDiffusionModel
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning.loggers import CSVLogger
from src.data_generator import generate_addition_example
from src.vocab import build_vocab
import os

torch.set_float32_matmul_precision('medium')

# Generate examples using the data generator
examples = [generate_addition_example() for _ in range(100000)]

# Calculate max tokens
max_tokens = 7

# Build vocabulary
vocab = build_vocab()
vocab_size = len(vocab)

# Tokenize sentences
tokenized = []
for example in examples:
    # convert to int using vocab
    tokens = [vocab[token] for token in example.split()]
    # Pad or truncate
    if len(tokens) < max_tokens:
        tokens += [vocab['<pad>']] * (max_tokens - len(tokens))
    else:
        # throw error
        raise ValueError(f"Example {example} has more than {max_tokens} tokens")
    tokenized.append(tokens)

data = torch.tensor(tokenized)  # (num_examples, seq_len)

# Create DataLoader
dataset = TensorDataset(data)
dataloader: DataLoader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=os.cpu_count() or 4)

# Initialize model
model = MaskedDiffusionModel(
    vocab_size=vocab_size,
    embedding_dim=256,
    timesteps=1000,
    hidden_dim=256,
    num_layers=10
)

# Train the model
logger = CSVLogger("logs", name="my_experiment")
trainer = pl.Trainer(max_epochs=20, accelerator='auto', logger=logger, log_every_n_steps=50)
trainer.fit(model, dataloader)
# Save the model to disk after training
model_path = "masked_diffusion_model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")