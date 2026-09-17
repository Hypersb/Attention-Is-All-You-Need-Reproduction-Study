import numpy as np

from src.attention import MultiHeadAttention, create_causal_mask


sequence_length = 4
d_model = 4
num_heads = 2

X = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads, seed=42)
mask = create_causal_mask(sequence_length=sequence_length)

output, attention_weights = mha.forward(X, mask=mask)


print("X:")
print(X)
print("X.shape:")
print(X.shape)

print("\nNumber of heads:")
print(mha.num_heads)
print("Dimension per head:")
print(mha.d_k)

print("\nAttention weights for Head 1:")
print(attention_weights[0])

print("\nAttention weights for Head 2:")
print(attention_weights[1])

print("\nRow sums for Head 1:")
print(attention_weights[0].sum(axis=1))

print("\nRow sums for Head 2:")
print(attention_weights[1].sum(axis=1))

print("\nFinal multi-head output:")
print(output)
print("Final output shape:")
print(output.shape)


assert output.shape == (4, 4)
assert attention_weights.shape == (2, 4, 4)

for head_weights in attention_weights:
    assert np.allclose(head_weights.sum(axis=1), 1.0)
    assert np.allclose(head_weights[0, 1:], 0.0)
    assert np.allclose(head_weights[1, 2:], 0.0)
    assert np.allclose(head_weights[2, 3], 0.0)

head_1 = mha.heads[0]
head_2 = mha.heads[1]

assert not np.array_equal(head_1.W_Q.W, head_2.W_Q.W)
assert not np.array_equal(head_1.W_K.W, head_2.W_K.W)
assert not np.array_equal(head_1.W_V.W, head_2.W_V.W)
