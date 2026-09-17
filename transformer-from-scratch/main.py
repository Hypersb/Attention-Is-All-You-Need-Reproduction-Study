import numpy as np

from src.attention import QKVProjection, create_causal_mask, scaled_dot_product_attention


d_model = 4
d_k = 2
d_v = 2

X = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
])

projection = QKVProjection(d_model=d_model, d_k=d_k, d_v=d_v, seed=0)
Q, K, V = projection.forward(X)

mask = create_causal_mask(sequence_length=4)
output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)


print("X:")
print(X)
print("X.shape:")
print(X.shape)

print("\nQ:")
print(Q)
print("Q.shape:")
print(Q.shape)

print("\nK:")
print(K)
print("K.shape:")
print(K.shape)

print("\nV:")
print(V)
print("V.shape:")
print(V.shape)

print("\nAttention weights:")
print(weights)

print("\nRow sums:")
print(weights.sum(axis=1))

print("\nAttention output:")
print(output)


assert X.shape == (4, 4)

assert Q.shape == (4, 2)
assert K.shape == (4, 2)
assert V.shape == (4, 2)

assert weights.shape == (4, 4)
assert output.shape == (4, 2)

assert np.allclose(weights.sum(axis=1), 1.0)

# Token 1 cannot look at tokens 2, 3, or 4.
assert np.allclose(weights[0, 1:], 0.0)

# Token 2 cannot look at tokens 3 or 4.
assert np.allclose(weights[1, 2:], 0.0)

# Token 3 cannot look at token 4.
assert np.allclose(weights[2, 3], 0.0)
