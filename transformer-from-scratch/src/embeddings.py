import numpy as np


class TokenEmbedding:
    """
    Lookup table from token IDs to d_model-dimensional vectors.

    The original Transformer scales embeddings by sqrt(d_model)
    before adding positional encoding. That scaling is left
    explicit in main.py so it is easy to see.
    """

    def __init__(self, vocab_size, d_model, seed=None):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(loc=0.0, scale=0.1, size=(vocab_size, d_model))

    def forward(self, token_ids):
        return self.weights[token_ids]


def positional_encoding(sequence_length, d_model):
    """
    Sinusoidal positional encoding from Attention Is All You Need.

        PE(pos, 2i)     = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i + 1) = cos(pos / 10000^(2i / d_model))
    """

    pe = np.zeros((sequence_length, d_model))
    positions = np.arange(sequence_length)[:, np.newaxis]
    even_dims = np.arange(0, d_model, 2)

    div_term = 10000 ** (even_dims / d_model)

    pe[:, 0::2] = np.sin(positions / div_term)
    pe[:, 1::2] = np.cos(positions / div_term)

    return pe
