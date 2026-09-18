import numpy as np

from src.linear import Linear


def relu(x):
    return np.maximum(0, x)


def relu_backward(d_output, hidden):
    return d_output * (hidden > 0)


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
        self.cache_hidden = None

    def forward(self, X):
        hidden = self.linear1.forward(X)
        self.cache_hidden = hidden
        activated = relu(hidden)
        output = self.linear2.forward(activated)

        return output

    def backward(self, d_output):
        d_activated = self.linear2.backward(d_output)
        d_hidden = relu_backward(d_activated, self.cache_hidden)
        dX = self.linear1.backward(d_hidden)
        return dX

    def parameters_and_gradients(self):
        params = []
        params.extend(self.linear1.parameters_and_gradients())
        params.extend(self.linear2.parameters_and_gradients())
        return params
