import numpy as np

from src.embeddings import TokenEmbedding, positional_encoding
from src.encoder import Encoder


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
num_layers = 2

embedding_layer = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, seed=0)
embeddings = embedding_layer.forward(token_ids)

scaled_embeddings = embeddings * np.sqrt(d_model)

pe = positional_encoding(sequence_length=len(token_ids), d_model=d_model)

X = scaled_embeddings + pe

encoder = Encoder(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    seed=42,
)

encoder_output, attention_weights = encoder.forward(X, mask=None)

final_means = np.mean(encoder_output, axis=-1)
final_variances = np.var(encoder_output, axis=-1)


print("Encoder input X:")
print(X)
print("Encoder input shape:")
print(X.shape)

print("\nNumber of encoder layers:")
print(encoder.num_layers)

print("\nAttention weights shape:")
print(attention_weights.shape)

print("\nLayer 1 - Head 1 attention weights:")
print(attention_weights[0, 0])

print("\nLayer 1 - Head 2 attention weights:")
print(attention_weights[0, 1])

print("\nLayer 2 - Head 1 attention weights:")
print(attention_weights[1, 0])

print("\nLayer 2 - Head 2 attention weights:")
print(attention_weights[1, 1])

print("\nFinal encoder output:")
print(encoder_output)
print("Final encoder output shape:")
print(encoder_output.shape)

print("\nMean of each token in final output:")
print(final_means)

print("\nVariance of each token in final output:")
print(final_variances)


assert X.shape == (4, 4)
assert encoder_output.shape == (4, 4)
assert attention_weights.shape == (2, 2, 4, 4)

for layer_weights in attention_weights:
    for head_weights in layer_weights:
        assert np.allclose(head_weights.sum(axis=1), 1.0)

assert np.allclose(final_means, 0.0, atol=1e-6)
assert np.allclose(final_variances, 1.0, atol=1e-4)

layer_1 = encoder.layers[0]
layer_2 = encoder.layers[1]

assert layer_1 is not layer_2
assert layer_1.self_attention is not layer_2.self_attention
assert layer_1.feed_forward is not layer_2.feed_forward
assert layer_1.self_attention.heads[0].W_Q is not layer_2.self_attention.heads[0].W_Q
assert layer_1.feed_forward.linear1 is not layer_2.feed_forward.linear1

assert not np.array_equal(
    layer_1.self_attention.heads[0].W_Q.W,
    layer_2.self_attention.heads[0].W_Q.W,
)
assert not np.array_equal(
    layer_1.feed_forward.linear1.W,
    layer_2.feed_forward.linear1.W,
)
