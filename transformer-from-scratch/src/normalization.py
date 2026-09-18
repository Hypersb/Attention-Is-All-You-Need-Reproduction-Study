import numpy as np


class LayerNorm:
    """
    Layer Normalization over the feature dimension of each token.
    """

    def __init__(self, d_model, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
        self.dgamma = np.zeros(d_model)
        self.dbeta = np.zeros(d_model)
        self.cache = None

    def forward(self, X):
        mean = np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1, keepdims=True)
        std = np.sqrt(variance + self.eps)
        X_normalized = (X - mean) / std
        output = self.gamma * X_normalized + self.beta

        self.cache = {
            "X_normalized": X_normalized,
            "std": std,
        }

        return output

    def backward(self, d_output):
        X_normalized = self.cache["X_normalized"]
        std = self.cache["std"]
        n_features = d_output.shape[-1]

        self.dgamma += np.sum(d_output * X_normalized, axis=0)
        self.dbeta += np.sum(d_output, axis=0)

        dX_normalized = d_output * self.gamma
        dX = (
            (1.0 / n_features)
            / std
            * (
                n_features * dX_normalized
                - np.sum(dX_normalized, axis=-1, keepdims=True)
                - X_normalized * np.sum(dX_normalized * X_normalized, axis=-1, keepdims=True)
            )
        )

        return dX

    def parameters_and_gradients(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]
