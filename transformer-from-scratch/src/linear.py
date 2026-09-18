import numpy as np


class Linear:
    """
    Simple linear projection: Y = X @ W + b
    """

    def __init__(self, input_dim, output_dim, seed=None):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=0.1, size=(input_dim, output_dim))
        self.b = np.zeros(output_dim)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self.cache_X = None

    def forward(self, X):
        self.cache_X = X
        return X @ self.W + self.b

    def backward(self, d_output):
        self.dW += self.cache_X.T @ d_output
        self.db += np.sum(d_output, axis=0)
        dX = d_output @ self.W.T
        return dX

    def parameters_and_gradients(self):
        return [(self.W, self.dW), (self.b, self.db)]
