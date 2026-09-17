import numpy as np

from src.linear import Linear


def relu(x):
    return np.maximum(0, x)


class FeedForward:
    """
    Position-wise feed-forward network from Attention Is All You Need.

        FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2
    """

    def __init__(self, d_model, d_ff, seed=None):
        if seed is None:
            seed1, seed2 = None, None
        else:
            seed1, seed2 = seed, seed + 1

        self.linear1 = Linear(d_model, d_ff, seed=seed1)
        self.linear2 = Linear(d_ff, d_model, seed=seed2)

    def forward(self, X):
        hidden = self.linear1.forward(X)
        activated = relu(hidden)
        output = self.linear2.forward(activated)

        return output
