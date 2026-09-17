import numpy as np

from src.attention import MultiHeadAttention, create_causal_mask
from src.embeddings import TokenEmbedding, positional_encoding


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

output, attention_weights = mha.forward(X, mask=mask)


print("Tokens:")
print(tokens)

print("\nToken IDs:")
print(token_ids)

print("\nRaw token embeddings:")
print(embeddings)
print("Raw embedding shape:")
print(embeddings.shape)

print("\nPositional encoding:")
print(pe)
print("Positional encoding shape:")
print(pe.shape)

print("\nFinal X:")
print(X)
print("Final X.shape:")
print(X.shape)

print("\nAttention weights for Head 1:")
print(attention_weights[0])

print("\nAttention weights for Head 2:")
print(attention_weights[1])

print("\nFinal multi-head output:")
print(output)
print("Final output shape:")
print(output.shape)


assert token_ids.shape == (4,)
assert embeddings.shape == (4, 4)
assert pe.shape == (4, 4)
assert X.shape == (4, 4)
assert attention_weights.shape == (2, 4, 4)
assert output.shape == (4, 4)

for head_weights in attention_weights:
    assert np.allclose(head_weights.sum(axis=1), 1.0)
    assert np.allclose(head_weights[0, 1:], 0.0)
    assert np.allclose(head_weights[1, 2:], 0.0)
    assert np.allclose(head_weights[2, 3], 0.0)

assert np.allclose(pe[0], np.array([0.0, 1.0, 0.0, 1.0]))
