import unittest
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from src.masked_diffusion_bert import MaskedDiffusionBERT
from src.reasoning_example import TokenizedExamples
from src.addition_reasoning_dataset import AdditionReasoningDataset
from pytorch_lightning import seed_everything


class TestMaskedDiffusionBERT(unittest.TestCase):
    def test_overfit_batch(self):
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
        
        batch = next(iter(dataloader))
        examples = TokenizedExamples.from_tensors(tokenizer=tokenizer, input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        with torch.no_grad():
            for updated_examples, _ in model.predict(examples):
                decoded = tokenizer.decode(updated_examples.input_ids[0], skip_special_tokens=True)
                print(decoded)


if __name__ == '__main__':
    unittest.main()
