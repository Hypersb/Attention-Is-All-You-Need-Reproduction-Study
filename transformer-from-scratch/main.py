import numpy as np

from src.transformer import Transformer


vocab = {
    "i": 0,
    "love": 1,
    "machine": 2,
    "learning": 3,
    "is": 4,
    "fun": 5,
}

inv_vocab = {
    0: "i",
    1: "love",
    2: "machine",
    3: "learning",
    4: "is",
    5: "fun",
}

source_tokens = "i love machine learning".split()
target_tokens = "machine learning is fun".split()

source_token_ids = np.array([vocab[token] for token in source_tokens])
target_token_ids = np.array([vocab[token] for token in target_tokens])

vocab_size = 6
d_model = 4
num_heads = 2
d_ff = 8
num_encoder_layers = 2
num_decoder_layers = 2

transformer = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    num_heads=num_heads,
    d_ff=d_ff,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    max_sequence_length=10,
    seed=42,
)

(
    logits,
    probabilities,
    encoder_attention,
    decoder_self_attention,
    decoder_cross_attention,
) = transformer.forward(source_token_ids, target_token_ids)

predicted_ids = np.argmax(probabilities, axis=-1)
predicted_tokens = [inv_vocab[int(token_id)] for token_id in predicted_ids]


print("Source tokens:")
print(source_tokens)
print("Source token IDs:")
print(source_token_ids)

print("\nTarget tokens:")
print(target_tokens)
print("Target token IDs:")
print(target_token_ids)

print("\nLogits:")
print(logits)
print("Logits shape:")
print(logits.shape)

print("\nProbabilities:")
print(probabilities)
print("Probabilities shape:")
print(probabilities.shape)

print("\nProbability row sums:")
print(probabilities.sum(axis=1))

print("\nProbability distribution at each target position:")
for position, row in enumerate(probabilities, start=1):
    print(f"\nPosition {position}:")
    for token, token_id in vocab.items():
        print(f"    {token:<10}: {row[token_id]:.2f}")

print("\nPredicted token at each position:")
print(predicted_tokens)

print("\nModel is untrained; predictions are expected to be random.")


assert logits.shape == (4, 6)
assert probabilities.shape == (4, 6)
assert np.allclose(probabilities.sum(axis=1), 1.0)
assert np.all(probabilities >= 0)
assert np.all(probabilities <= 1)

assert encoder_attention.shape == (2, 2, 4, 4)
assert decoder_self_attention.shape == (2, 2, 4, 4)
assert decoder_cross_attention.shape == (2, 2, 4, 4)

assert transformer.output_projection.W.shape == (4, 6)
assert transformer.output_projection.b.shape == (6,)

assert predicted_ids.shape == (4,)
assert np.all((predicted_ids >= 0) & (predicted_ids < vocab_size))
