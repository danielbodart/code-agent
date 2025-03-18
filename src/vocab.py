# Vocabulary and tokenization utilities

def build_vocab():
    vocab = {'<mask>': 0}  # Mask token
    vocab.update({str(i): i + 1 for i in range(200)})  # Digits 0-199
    vocab.update({'+': 201, '-': 202, '*': 203, '/': 204, '=': 205})
    vocab['<pad>'] = 206
    return vocab


def tokenize(example, vocab):
    return [vocab[token] for token in example.split() if token in vocab]


def detokenize(tokens, vocab):
    reverse_vocab = {v: k for k, v in vocab.items()}
    return ' '.join(reverse_vocab[token] for token in tokens if token in reverse_vocab)
