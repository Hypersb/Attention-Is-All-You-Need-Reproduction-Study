import numpy as np

from src.attention import MultiHeadAttention, create_causal_mask
from src.embeddings import TokenEmbedding, positional_encoding
from src.feed_forward import FeedForward, relu
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
d_ff = 8

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
attention_normalized = layer_norm.forward(residual)

ffn = FeedForward(d_model=d_model, d_ff=d_ff, seed=1)
ffn_output = ffn.forward(attention_normalized)

# Second residual: keep the attention-normalized representation and add the FFN result.
ffn_residual = attention_normalized + ffn_output

second_layer_norm = LayerNorm(d_model)
final_output = second_layer_norm.forward(ffn_residual)

final_means = np.mean(final_output, axis=-1)
final_variances = np.var(final_output, axis=-1)


print("Attention-normalized input:")
print(attention_normalized)

print("\nFFN output:")
print(ffn_output)

print("\nFFN residual result:")
print(ffn_residual)

print("\nFinal normalized output:")
print(final_output)

print("\nFinal mean of each token:")
print(final_means)

print("\nFinal variance of each token:")
print(final_variances)

print("\nFinal output shape:")
print(final_output.shape)


assert attention_normalized.shape == (4, 4)
assert ffn_output.shape == (4, 4)
assert ffn_residual.shape == (4, 4)
assert final_output.shape == (4, 4)

assert np.allclose(ffn_residual, attention_normalized + ffn_output)

assert np.allclose(final_means, 0.0, atol=1e-6)
assert np.allclose(final_variances, 1.0, atol=1e-4)

assert ffn.linear1.W.shape == (4, 8)
assert ffn.linear1.b.shape == (8,)
assert ffn.linear2.W.shape == (8, 4)
assert ffn.linear2.b.shape == (4,)

assert np.array_equal(relu(np.array([-2, -1, 0, 1, 2])), np.array([0, 0, 0, 1, 2]))

assert attention_weights.shape == (2, 4, 4)
for head_weights in attention_weights:
    assert np.allclose(head_weights.sum(axis=1), 1.0)
    assert np.allclose(head_weights[0, 1:], 0.0)
    assert np.allclose(head_weights[1, 2:], 0.0)
    assert np.allclose(head_weights[2, 3], 0.0)
