import unittest
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from src.masked_diffusion_bert import MaskedDiffusionBERT
from src.bert_diffuser import BERTDiffuser
from src.addition_reasoning_dataset import AdditionReasoningDataset
from pytorch_lightning import seed_everything


class TestMaskedDiffusionBERT(unittest.TestCase):
    def manual_test_overfit_batch(self):
        seed_everything(42)

        model = MaskedDiffusionBERT()
        tokenizer = model.tokenizer
        
        dataset = AdditionReasoningDataset(tokenizer, num_examples=100, max_number=100)
        dataloader = DataLoader(dataset, batch_size=2, pin_memory=True)
        
        trainer = Trainer(
            fast_dev_run=True,
            overfit_batches=1,
            accumulate_grad_batches=8, 
            precision="bf16-mixed",
        )
        
        trainer.fit(model, dataloader)
        
        with torch.no_grad():
            print(model.generate("<question>What is 2 + 2?</question><answer>[MASK]</answer>"))


if __name__ == '__main__':
    unittest.main()
