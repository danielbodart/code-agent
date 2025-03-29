from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from src.masked_diffusion_model import MaskedDiffusionModel
from src.addition_reasoning_dataset import AdditionReasoningDataset
from pytorch_lightning import seed_everything
from src.setup import setup

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
    max_epochs=30,
    accumulate_grad_batches=4, 
    precision="bf16-mixed",
    accelerator="gpu",
    # callbacks=[EarlyStopping(monitor='train_loss', patience=10, mode='min')]
)

trainer.fit(model, train_loader, val_loader)

model.model.cuda()

print(model.generate("What is 2 + 2?[SEP][MASK]"))
print(model.generate("What is 324 + 5324?[SEP][MASK][MASK]"))
