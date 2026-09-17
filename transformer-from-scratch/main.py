import numpy as np

from src.attention import MultiHeadAttention, create_causal_mask
from src.embeddings import TokenEmbedding, positional_encoding
from src.normalization import LayerNorm


sentence = "i love machine learning"
tokens = sentence.split()

vocab = {
    "i": 0,
    "love": 1,
    "machine": 2,
    "learning": 3,
}

token_ids = np.array([vocab[token] for token in tokens])

vocab_size = 4
d_model = 4
num_heads = 2

embedding_layer = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, seed=0)
embeddings = embedding_layer.forward(token_ids)

scaled_embeddings = embeddings * np.sqrt(d_model)

pe = positional_encoding(sequence_length=len(token_ids), d_model=d_model)

X = scaled_embeddings + pe

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=42)
mask = create_causal_mask(sequence_length=len(token_ids))

attention_output, attention_weights = mha.forward(X, mask=mask)

# Residual connection: keep the original X and add the attention result.
residual = X + attention_output

# LayerNorm normalizes each token's features independently.
layer_norm = LayerNorm(d_model)
normalized_output = layer_norm.forward(residual)

token_means = np.mean(normalized_output, axis=-1)
token_variances = np.var(normalized_output, axis=-1)


print("X before attention:")
print(X)

print("\nAttention output:")
print(attention_output)

print("\nResidual result (X + attention_output):")
print(residual)

print("\nNormalized output:")
print(normalized_output)

print("\nMean of each token after LayerNorm:")
print(token_means)

print("\nVariance of each token after LayerNorm:")
print(token_variances)

print("\nNormalized output shape:")
print(normalized_output.shape)


assert X.shape == (4, 4)
assert attention_output.shape == (4, 4)
assert residual.shape == (4, 4)
assert normalized_output.shape == (4, 4)

assert np.allclose(residual, X + attention_output)

assert np.allclose(token_means, 0.0, atol=1e-6)
assert np.allclose(token_variances, 1.0, atol=1e-4)

assert attention_weights.shape == (2, 4, 4)
for head_weights in attention_weights:
    assert np.allclose(head_weights.sum(axis=1), 1.0)
    assert np.allclose(head_weights[0, 1:], 0.0)
    assert np.allclose(head_weights[1, 2:], 0.0)
    assert np.allclose(head_weights[2, 3], 0.0)
