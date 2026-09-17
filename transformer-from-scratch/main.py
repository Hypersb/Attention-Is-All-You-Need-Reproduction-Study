import numpy as np

from src.attention import create_causal_mask, scaled_dot_product_attention


Q = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
])

K = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [0.5, 0.5],
])

V = np.array([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0],
    [1.0, 1.0],
])

mask = create_causal_mask(sequence_length=4)

output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)


print("Raw QK^T scores:")
print(Q @ K.T)

print("\nCausal mask:")
print(mask)

print("\nAttention weights:")
print(weights)

print("\nRow sums of attention weights:")
print(weights.sum(axis=1))

print("\nFinal attention output:")
print(output)


assert np.allclose(weights.sum(axis=1), 1.0)

# Token 1 cannot look at tokens 2, 3, or 4.
assert np.allclose(weights[0, 1:], 0.0)

# Token 2 cannot look at tokens 3 or 4.
assert np.allclose(weights[1, 2:], 0.0)

# Token 3 cannot look at token 4.
assert np.allclose(weights[2, 3], 0.0)
