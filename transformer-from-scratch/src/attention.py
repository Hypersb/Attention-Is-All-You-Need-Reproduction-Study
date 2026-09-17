import numpy as np

from src.math_utils import softmax


def create_causal_mask(sequence_length):
    """
    Create a decoder-style look-ahead mask.

    True marks future tokens that must not be attended to.
    """

    return np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)


def scaled_dot_product_attention(Q, K, V, mask=None):
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

    mask : np.ndarray, optional
        Boolean mask. True means "block this position"
        (typically future tokens in a causal mask).

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

    if mask is not None:
        scaled_scores = np.where(mask, -np.inf, scaled_scores)

    attention_weights = softmax(scaled_scores)

    output = attention_weights @ V

    return output, attention_weights
