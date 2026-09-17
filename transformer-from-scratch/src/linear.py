import numpy as np


class Linear:
    """
    Simple linear projection: Y = X @ W + b

    These weights will later be learned. For now W is
    randomly initialized with small values and b is zeros.
    """

    def __init__(self, input_dim, output_dim, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=0.1, size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)

    def forward(self, X):
        return X @ self.W + self.b
