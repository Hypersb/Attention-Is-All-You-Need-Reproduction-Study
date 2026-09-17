import numpy as np

from src.attention import scaled_dot_product_attention


Q = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

K = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

V = np.array([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])


output, weights = scaled_dot_product_attention(Q, K, V)


print("Q:")
print(Q)

print("\nK:")
print(K)

print("\nQK^T:")
print(Q @ K.T)

print("\nAttention weights:")
print(weights)

print("\nOutput:")
print(output)
