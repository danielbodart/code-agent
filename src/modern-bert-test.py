import os
import logging
from transformers import AutoTokenizer, ModernBertForMaskedLM
import torch

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR
logging.getLogger('tensorflow').setLevel(logging.ERROR)

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = ModernBertForMaskedLM.from_pretrained("answerdotai/ModernBERT-base")

# Let's do iterative unmasking with multiple masked tokens
print("\n--- Iterative Unmasking Example ---")
inputs = tokenizer("Question: [MASK][MASK][MASK][MASK][MASK][MASK][MASK] Answer: Paris", return_tensors="pt")
original_input_ids = inputs.input_ids.clone()

# Find all mask token indices
mask_token_indices = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
print(f"Found {len(mask_token_indices)} masked tokens")

# Store the original sentence for reference
original_sentence = tokenizer.decode(original_input_ids[0], skip_special_tokens=True)
print(f"Original: {original_sentence}")

# Iteratively unmask one token at a time
current_input_ids = original_input_ids.clone()
for iteration in range(len(mask_token_indices)):
    # Get the current sentence with some tokens potentially already unmasked
    current_sentence = tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
    print(f"\nIteration {iteration+1}:")
    print(f"Current: {current_sentence}")
    
    # Find remaining mask tokens
    remaining_mask_indices = (current_input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    
    # If no masks left, we're done
    if len(remaining_mask_indices) == 0:
        break
    
    # Randomly select one mask to predict (you could also use a specific strategy)
    import random
    mask_idx_to_predict = remaining_mask_indices[random.randint(0, len(remaining_mask_indices)-1)]
    
    # Get prediction for this mask
    with torch.no_grad():
        logits = model(**{"input_ids": current_input_ids}).logits
    
    predicted_token_id = logits[0, mask_idx_to_predict].argmax(axis=-1)
    predicted_token = tokenizer.decode(predicted_token_id)
    
    # Update the input_ids with the predicted token
    current_input_ids[0, mask_idx_to_predict] = predicted_token_id
    
    # Show what was predicted
    print(f"Unmasked token at position {mask_idx_to_predict.item()}: {predicted_token}")
    
    # Show the updated sentence
    updated_sentence = tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
    print(f"Updated: {updated_sentence}")

print("\n--- Final Result ---")
print(f"Original: {original_sentence}")
print(f"Final: {tokenizer.decode(current_input_ids[0], skip_special_tokens=True)}")
