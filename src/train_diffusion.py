import torch
import pytorch_lightning as pl
from src.masked_diffusion_model import MaskedDiffusionModel
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_lightning.loggers import CSVLogger

torch.set_float32_matmul_precision('medium')

# Simple dataset (for demonstration)
sentences = [
    "the cat sat on the mat",
    "a dog ran in the park",
    "birds fly over the sky"
]
# Build a basic vocabulary
words = set()
for s in sentences:
    words.update(s.split())
vocab = {word: idx for idx, word in enumerate(words, 1)}  # Start indices at 1
vocab['<pad>'] = 0  # Padding token
vocab_size = len(vocab)

# Tokenize sentences
max_len = 6  # Fixed sequence length
tokenized = []
for s in sentences:
    tokens = [vocab[word] for word in s.split()]
    # Pad or truncate
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    tokenized.append(tokens)
data = torch.tensor(tokenized)  # (num_sentences, seq_len)

# Create DataLoader
dataset = TensorDataset(data)
dataloader: DataLoader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)

# Initialize model
model = MaskedDiffusionModel(
    vocab_size=vocab_size,
    embedding_dim=64,
    T=1000,
    hidden_dim=128,
    num_layers=2
)

# Train the model
logger = CSVLogger("logs", name="my_experiment")
trainer = pl.Trainer(max_epochs=10, accelerator='auto', logger=logger, log_every_n_steps=1)
trainer.fit(model, dataloader)