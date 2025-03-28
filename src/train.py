from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from src.masked_diffusion_model import MaskedDiffusionModel
from src.addition_reasoning_dataset import AdditionReasoningDataset
from pytorch_lightning import seed_everything
from src.setup import setup
from src.checkpoints import get_latest_checkpoint
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', const='', default=None, type=str, nargs='?', help='Path to specific checkpoint to load')
args = parser.parse_args()

seed_everything(42)
setup()

model = MaskedDiffusionModel(lr=1e-5)
tokenizer = model.tokenizer
max_tokens = 16
batch_size = 128

train_dataset = AdditionReasoningDataset(tokenizer, num_examples=90000, max_number=100, max_tokens=max_tokens)
val_dataset = AdditionReasoningDataset(tokenizer, num_examples=10000, max_number=100, start_index=90000, max_tokens=max_tokens)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

trainer = Trainer(
    max_time="00:01:50:00",
    accumulate_grad_batches=4, 
    precision="bf16-mixed",
    accelerator="gpu",
)

checkpoint = args.checkpoint if args.checkpoint is not None else get_latest_checkpoint("lightning_logs")
if not checkpoint:
    print("No checkpoint found, starting from scratch.")
else:
    print(f"Training from checkpoint: {checkpoint}")

trainer.fit(model, train_loader, val_loader, ckpt_path=checkpoint)

model.model.cuda()

print(model.generate("What is 2 + 2? [MASK]"))
print(model.generate("What is 4 + 9? [MASK]"))
print(model.generate("What is 9 + 18? [MASK]"))
print(model.generate("What is 45 + 24? [MASK]"))
print(model.generate("What is 31 + 12? [MASK]"))
print(model.generate("What is 99 + 99? [MASK]"))

print(model.generate("What is [MASK] + [MASK]? 3"))
print(model.generate("What is [MASK] + [MASK]? 27"))
print(model.generate("What is [MASK] + [MASK]? 100"))
