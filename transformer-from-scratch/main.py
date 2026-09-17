import numpy as np

from src.attention import create_causal_mask
from src.decoder import Decoder
from src.embeddings import TokenEmbedding, positional_encoding
from src.encoder import Encoder


source_sentence = "i love machine learning"
target_sentence = "machine learning is fun"

source_tokens = source_sentence.split()
target_tokens = target_sentence.split()

vocab = {
    "i": 0,
    "love": 1,
    "machine": 2,
    "learning": 3,
    "is": 4,
    "fun": 5,
}

source_ids = np.array([vocab[token] for token in source_tokens])
target_ids = np.array([vocab[token] for token in target_tokens])

vocab_size = 6
d_model = 4
num_heads = 2
d_ff = 8
num_layers = 2

embedding_layer = TokenEmbedding(vocab_size=vocab_size, d_model=d_model, seed=0)

source_embeddings = embedding_layer.forward(source_ids)
target_embeddings = embedding_layer.forward(target_ids)

scaled_source = source_embeddings * np.sqrt(d_model)
scaled_target = target_embeddings * np.sqrt(d_model)

source_pe = positional_encoding(sequence_length=len(source_ids), d_model=d_model)
target_pe = positional_encoding(sequence_length=len(target_ids), d_model=d_model)

source_X = scaled_source + source_pe
target_X = scaled_target + target_pe

encoder = Encoder(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    seed=42,
)

decoder = Decoder(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    seed=100,
)

encoder_output, encoder_attention = encoder.forward(source_X, mask=None)

causal_mask = create_causal_mask(sequence_length=len(target_ids))

decoder_output, decoder_self_attention, decoder_cross_attention = decoder.forward(
    target_X,
    encoder_output,
    causal_mask,
)

final_means = np.mean(decoder_output, axis=-1)
final_variances = np.var(decoder_output, axis=-1)


print("Source tokens:")
print(source_tokens)

print("\nTarget tokens:")
print(target_tokens)

print("\nEncoder output shape:")
print(encoder_output.shape)

print("\nDecoder input shape:")
print(target_X.shape)

print("\nCausal mask:")
print(causal_mask)

print("\nDecoder self-attention shape:")
print(decoder_self_attention.shape)

print("\nDecoder cross-attention shape:")
print(decoder_cross_attention.shape)

print("\nDecoder Layer 1 - Self-Attention Head 1:")
print(decoder_self_attention[0, 0])

print("\nDecoder Layer 1 - Cross-Attention Head 1:")
print(decoder_cross_attention[0, 0])

print("\nFinal decoder output:")
print(decoder_output)

print("\nFinal decoder output shape:")
print(decoder_output.shape)

print("\nMean of each token:")
print(final_means)

print("\nVariance of each token:")
print(final_variances)


assert encoder_output.shape == (4, 4)
assert decoder_output.shape == (4, 4)
assert decoder_self_attention.shape == (2, 2, 4, 4)
assert decoder_cross_attention.shape == (2, 2, 4, 4)

for layer_weights in decoder_self_attention:
    for head_weights in layer_weights:
        assert np.allclose(head_weights.sum(axis=1), 1.0)
        assert np.allclose(head_weights[0, 1:], 0.0)
        assert np.allclose(head_weights[1, 2:], 0.0)
        assert np.allclose(head_weights[2, 3], 0.0)

for layer_weights in decoder_cross_attention:
    for head_weights in layer_weights:
        assert np.allclose(head_weights.sum(axis=1), 1.0)

assert np.allclose(final_means, 0.0, atol=1e-6)
assert np.allclose(final_variances, 1.0, atol=1e-4)

layer = decoder.layers[0]
assert layer.masked_self_attention is not layer.cross_attention
assert layer.masked_self_attention.heads[0].W_Q is not layer.cross_attention.heads[0].W_Q
assert not np.array_equal(
    layer.masked_self_attention.heads[0].W_Q.W,
    layer.cross_attention.heads[0].W_Q.W,
)
assert not np.array_equal(
    layer.masked_self_attention.heads[0].W_K.W,
    layer.cross_attention.heads[0].W_K.W,
)
assert not np.array_equal(
    layer.masked_self_attention.heads[0].W_V.W,
    layer.cross_attention.heads[0].W_V.W,
)
