import numpy as np

from src.math_utils import softmax


def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.

    Parameters
    ----------
    Q : np.ndarray
        Query matrix.

    K : np.ndarray
        Key matrix.

    V : np.ndarray
        Value matrix.

    Returns
    -------
    output : np.ndarray
        Attention output.

    attention_weights : np.ndarray
        Probability assigned to each key.
    """

    d_k = Q.shape[-1]

    scores = Q @ K.T

    scaled_scores = scores / np.sqrt(d_k)

    attention_weights = softmax(scaled_scores)

    output = attention_weights @ V

    return output, attention_weights
