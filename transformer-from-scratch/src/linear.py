import numpy as np


class Linear:
    """
    Simple linear projection: Y = X @ W

    These weights will later be learned. For now they are
    randomly initialized with small values.
    """

    def __init__(self, input_dim, output_dim, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=0.1, size=(input_dim, output_dim))

    def forward(self, X):
        return X @ self.W
