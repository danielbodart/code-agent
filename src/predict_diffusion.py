import torch
from src.masked_diffusion_model import MaskedDiffusionModel
from src.data_generator import generate_addition_example
from src.vocab import build_vocab, tokenize, detokenize
from src.mask_expression import mask_expression

# Load the model
model_path = "masked_diffusion_model.pth"
vocab = build_vocab()

model = MaskedDiffusionModel(
    vocab_size=len(vocab),
    embedding_dim=256,  
    timesteps=1000,
    hidden_dim=256,  
    num_layers=10  
)
model.load_state_dict(torch.load(model_path))
model.eval()

# Generate and predict examples
examples = [generate_addition_example() for _ in range(10)]

for example in examples:
    original_token, masked_example = mask_expression(example)
    tokens = tokenize(masked_example, vocab)
    # Convert tokens to tensor
    input_tensor = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension
    # Predict using the model's predict method
    output = model.predict(input_tensor)
    # Detokenize output
    predicted_tokens = output.argmax(dim=-1).squeeze().tolist()
    prediction = detokenize(predicted_tokens, vocab)
    print(f"Original: {example} | Masked: {masked_example} | Prediction: {prediction}")
