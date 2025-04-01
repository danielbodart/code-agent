import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from src.addition_reasoning_dataset import AdditionReasoningDataset
from pytorch_lightning import seed_everything
from src.masked_diffusion_model import MaskedDiffusionModel
from src.setup import setup

seed_everything(42)
setup()

def test_overfit_batch():
    model = MaskedDiffusionModel()
    tokenizer = model.tokenizer
    
    dataset = AdditionReasoningDataset(tokenizer, num_examples=100, max_number=100, max_tokens=16)
    dataloader = DataLoader(dataset, batch_size=32)
    
    trainer = Trainer(
        fast_dev_run=True,
        overfit_batches=1,
        accumulate_grad_batches=8, 
        precision="bf16-mixed",
        accelerator="gpu"
    )
    
    trainer.fit(model, dataloader)

    # predictions = trainer.predict(model, dataloader)

    # print(predictions)

    model.model.cuda()
    
    with torch.no_grad():
        # print(model.generate("What is 2 + 2?[SEP][MASK]"))
        # print(model.generate("What is 3 + 5?[SEP][MASK]"))
        # print(model.generate("What is 10 + 7?[SEP][MASK]"))
        # print(model.generate("What is 45 + 24?[SEP][MASK]"))
        # print(model.generate("What is 31 + 12?[SEP][MASK]"))
        for text in model.unmask("What is 99 + 99?[SEP][MASK]", max_length=16, skip_special_tokens=False):
            print(text)
    
def test_generate():
    """ Check we haven't broken the underlying Modern BERT model """
    model = MaskedDiffusionModel()
    
    with torch.no_grad():
        assert model.generate("The capital of France is [MASK].") == "The capital of France is Paris."
