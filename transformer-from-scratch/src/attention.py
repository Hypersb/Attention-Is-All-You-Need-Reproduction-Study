import numpy as np

from src.linear import Linear
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


class QKVProjection:
    """
    Create Query, Key, and Value matrices from input X.

        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V

    W_Q, W_K, and W_V are independent linear projections.
    """

    def __init__(self, d_model, d_k, d_v, seed=None):
        if seed is None:
            q_seed, k_seed, v_seed = None, None, None
        else:
            q_seed, k_seed, v_seed = seed, seed + 1, seed + 2

        self.W_Q = Linear(d_model, d_k, seed=q_seed)
        self.W_K = Linear(d_model, d_k, seed=k_seed)
        self.W_V = Linear(d_model, d_v, seed=v_seed)

    def forward(self, query_input, key_value_input=None):
        if key_value_input is None:
            key_value_input = query_input

        Q = self.W_Q.forward(query_input)
        K = self.W_K.forward(key_value_input)
        V = self.W_V.forward(key_value_input)

        return Q, K, V


class MultiHeadAttention:
    """
    Multi-head attention from Attention Is All You Need.

        head_i = Attention(X @ W_Q_i, X @ W_K_i, X @ W_V_i)

        MultiHead(X) = Concat(head_1, ..., head_h) @ W_O
    """

    def __init__(self, d_model, num_heads, seed=None):
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        self.heads = []
        for i in range(num_heads):
            head_seed = None if seed is None else seed + i * 3
            self.heads.append(
                QKVProjection(d_model, self.d_k, self.d_v, seed=head_seed)
            )

        wo_seed = None if seed is None else seed + num_heads * 3
        self.W_O = Linear(num_heads * self.d_v, d_model, seed=wo_seed)

    def forward(self, query_input, key_value_input=None, mask=None):
        head_outputs = []
        attention_weights = []

        for projection in self.heads:
            Q, K, V = projection.forward(query_input, key_value_input)
            head_output, weights = scaled_dot_product_attention(Q, K, V, mask)
            head_outputs.append(head_output)
            attention_weights.append(weights)

        concatenated = np.concatenate(head_outputs, axis=-1)
        output = self.W_O.forward(concatenated)
        attention_weights = np.stack(attention_weights, axis=0)

        return output, attention_weights
