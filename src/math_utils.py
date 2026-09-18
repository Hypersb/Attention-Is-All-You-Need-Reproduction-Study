import numpy as np


def softmax(x):
    """
    Convert raw scores into probabilities.

    Each row will sum to 1.
    """

    x_shifted = x - np.max(x, axis=-1, keepdims=True)

    exp_x = np.exp(x_shifted)

    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
