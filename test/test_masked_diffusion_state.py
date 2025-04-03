import torch
from transformers import AutoTokenizer

from src.masked_diffusion_state import MaskedDiffusionState, tokenize
from src.reasoning_example import ReasoningExample
from pytorch_lightning import seed_everything

seed_everything(42)

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")

def test_tokenization():
    example = ReasoningExample("Question", "Answer")

    tokenized = MaskedDiffusionState.from_batch(tokenizer, tokenize(tokenizer, [str(example)], max_length=6))
    decoded = tokenizer.decode(tokenized.input_ids[0], skip_special_tokens=False)

    assert decoded == "[CLS]Question[SEP]Answer[SEP][PAD]"
    assert tokenized.attention_mask[0].tolist() == [1, 1, 1, 1, 1, 0]

def test_can_mask_tokens():
    example = ReasoningExample("Can get masked", "Can get masked")

    tokenized = MaskedDiffusionState.from_batch(tokenizer, tokenize(tokenizer, [str(example)], max_length=10))
    
    no_masking = tokenized.mask(percentage=0)
    no_masking_decoded = tokenizer.decode(no_masking.input_ids[0], skip_special_tokens=False)
    assert no_masking_decoded == "[CLS]Can get masked[SEP]Can get masked[SEP][PAD]"
    assert (no_masking.input_ids == tokenizer.mask_token_id).sum().item() == 0

    half_masking = tokenized.mask(percentage=0.5)
    num_maskable = tokenized.maskable[0].sum().item()
    expected_masked = int(num_maskable * 0.5)
    assert (half_masking.input_ids[0] == tokenizer.mask_token_id).sum().item() == expected_masked

    full_masking = tokenized.mask(percentage=1)
    full_masking_decoded = tokenizer.decode(full_masking.input_ids[0], skip_special_tokens=False)
    assert full_masking_decoded == "[MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK][MASK]"
    assert (full_masking.input_ids == tokenizer.mask_token_id).sum().item() == num_maskable

def test_maskable():
    example = ReasoningExample("ignore", "three")

    tokenized = MaskedDiffusionState.from_batch(tokenizer, tokenize(tokenizer, [str(example)]))
    
    assert torch.all(tokenized.maskable == 1)

def test_lengths():
    example1 = ReasoningExample("short question", "short answer")

    example2 = ReasoningExample("longer question with more tokens", "longer answer with more tokens too")

    tokenized = MaskedDiffusionState.from_batch(tokenizer, tokenize(tokenizer, [str(example1), str(example2)]))

    lengths = tokenized.lengths

    assert lengths.shape == (2,)
    assert lengths[0].item() == 7
    assert lengths[1].item() == 16

def test_mask_percentage():
    """Test that masking with percentage produces reasonable results."""
    example = ReasoningExample(
        "This is a question",
        "This is the answer"
    )

    tokenized = MaskedDiffusionState.from_batch(tokenizer, tokenize(tokenizer, [str(example)]))

    masked = tokenized.mask(percentage=0.5)
    mask_count = masked.masked.sum().item()

    assert (masked.labels == -100).sum().item() == masked.input_ids.size(1) - mask_count

    assert mask_count > 0, "Should mask at least some tokens"
    max_maskable = (tokenized.maskable == 1).sum().item()
    assert mask_count < max_maskable, "Should not mask 100% of maskable tokens"


def test_update():
    example = ReasoningExample(
        "This is a question",
        "This is an answer"
    )

    tokenized = MaskedDiffusionState.from_batch(tokenizer, tokenize(tokenizer, [str(example)]))
    masked = tokenized.mask(percentage=1.0)

    predicted_ids = torch.randint(1000, 5000, (masked.input_ids.shape[0], masked.input_ids.shape[1]))
    updated = masked.update(predicted_ids)

    assert torch.all(updated.input_ids[masked.maskable == 1] == predicted_ids.view(-1)[:masked.maskable.sum()])
    assert torch.all(updated.input_ids[masked.maskable == 0] == masked.input_ids[masked.maskable == 0])

def test_can_mask_tokens_batch():
    examples = [ReasoningExample("short", "answer"), ReasoningExample("longer question", "longer answer")]
    tokenized = MaskedDiffusionState.from_batch(tokenizer, tokenize(tokenizer, [str(e) for e in examples], max_length=10))
    
    no_masking = tokenized.mask(percentage=0)
    assert (no_masking.input_ids == tokenizer.mask_token_id).sum(dim=1).tolist() == [0, 0]

    half_masking = tokenized.mask(percentage=0.5)
    expected_masked = (tokenized.maskable.sum(dim=1) * 0.5).long()
    assert torch.all((half_masking.input_ids == tokenizer.mask_token_id).sum(dim=1) == expected_masked)

    full_masking = tokenized.mask(percentage=1)
    assert torch.all((full_masking.input_ids == tokenizer.mask_token_id).sum(dim=1) == tokenized.maskable.sum(dim=1))
