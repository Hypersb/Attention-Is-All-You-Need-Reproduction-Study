import numpy as np


class LayerNorm:
    """
    Layer Normalization over the feature dimension of each token.
    """

    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1, keepdims=True)

        X_normalized = (X - mean) / np.sqrt(variance + self.eps)

        output = self.gamma * X_normalized + self.beta

        return output
