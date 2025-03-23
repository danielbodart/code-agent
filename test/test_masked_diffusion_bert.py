import unittest
import torch
import random
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from src.masked_diffusion_bert import MaskedDiffusionBERT
from src.reasoning_example import TokenizedExamples
from src.addition_reasoning_dataset import AdditionReasoningDataset

class TestMaskedDiffusionBERT(unittest.TestCase):
    def test_overfit_batch(self):
        model = MaskedDiffusionBERT()
        tokenizer = model.tokenizer
        
        r = random.Random(42)
        dataset = AdditionReasoningDataset(tokenizer, num_examples=100, max_number=100, r=r)
        dataloader = DataLoader(dataset, batch_size=4)
        
        trainer = Trainer(
            fast_dev_run=True,
            overfit_batches=1,
            accumulate_grad_batches=4, 
            precision="bf16-mixed" 
        )
        
        trainer.fit(model, dataloader)
        
        batch = next(iter(dataloader))
        examples = TokenizedExamples.from_tensors(tokenizer=tokenizer, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        with torch.no_grad():
            for updated_examples, _ in model.predict(examples):
                decoded = tokenizer.decode(updated_examples.input_ids[0], skip_special_tokens=True)
                print(decoded)


if __name__ == '__main__':
    unittest.main()
