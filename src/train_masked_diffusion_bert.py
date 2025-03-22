import os
import argparse
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from masked_diffusion_bert import MaskedDiffusionBERT


# Parse command line arguments
parser = argparse.ArgumentParser(description='Train or use a masked diffusion BERT model')
parser.add_argument('--train', action='store_true', help='Force training even if checkpoint exists')
args = parser.parse_args()



# Define model checkpoint path
model_path = "masked_diffusion_bert"
checkpoint_path = f"{model_path}.ckpt"

# Check if checkpoint exists and load it
should_train = args.train or not os.path.exists(checkpoint_path)

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    model = MaskedDiffusionBERT.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer)
    print("Checkpoint loaded successfully")
else:
    print("No checkpoint found, training from scratch")
    model = MaskedDiffusionBERT()
    should_train = True


# Filter to only include examples with a reference answer
print("Filtering dataset...")
filtered_dataset = raw_dataset.filter(
    lambda example: bool(example["reference_answer"]),
    desc="Filtering examples with reference answers"
)

# Limit to first 10,000 examples (or any other number)
max_examples = 10000
if len(filtered_dataset) > max_examples:
    print(f"Limiting to {max_examples} examples...")
    filtered_dataset = filtered_dataset.select(range(max_examples))

# Combine question and reference_answer in one step
print("Processing dataset...")
processed_dataset = filtered_dataset.map(
    lambda example: {"text": f"{example['question']} {example['reference_answer']}"},
    remove_columns=filtered_dataset.column_names,  # Remove original columns
    desc="Combining question and answer"
)

print(f"Processed dataset size: {len(processed_dataset)}")

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

# Tokenize the dataset in batches for efficiency
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

print("Tokenizing dataset...")
tokenized_dataset = processed_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=1000,  # Process 1000 examples at once
    desc="Tokenizing dataset",
    remove_columns=["text"]  # Remove the original text column
)

# Format dataset to return PyTorch tensors
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# Create DataLoader
import os
num_workers = min(os.cpu_count(), 16)  # Use available cores but cap at 16 to avoid excessive overhead
print(f"Using {num_workers} workers for data loading")

dataloader = DataLoader(
    tokenized_dataset, 
    batch_size=8, 
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True  # This can speed up data transfer to GPU
)


# PyTorch Lightning Trainer
trainer = Trainer(
    accelerator="cuda" if torch.cuda.is_available() else "cpu",
    devices=1,
    max_epochs=5
)


print("Starting training...")
trainer.fit(model, dataloader)

# Save the trained model
model.save_hyperparameters()
trainer.save_checkpoint(checkpoint_path)
print(f"Model saved to {checkpoint_path}")
