import numpy as np

from src.linear import Linear
from src.math_utils import softmax


def create_causal_mask(sequence_length):
    """
    Create a decoder-style look-ahead mask.

    True marks future tokens that must not be attended to.
    """

    return np.triu(np.ones((sequence_length, sequence_length), dtype=bool), k=1)


def create_key_padding_mask(token_ids, pad_id):
    """Block attention to PAD keys. Shape: (query_len, key_len) with query_len == key_len."""

    key_is_pad = token_ids == pad_id
    return np.repeat(key_is_pad[np.newaxis, :], len(token_ids), axis=0)


def create_cross_padding_mask(query_length, key_token_ids, pad_id):
    """Block cross-attention to PAD keys in the encoder output."""

    key_is_pad = key_token_ids == pad_id
    return np.repeat(key_is_pad[np.newaxis, :], query_length, axis=0)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.
    """

    d_k = Q.shape[-1]

    scores = Q @ K.T
    scaled_scores = scores / np.sqrt(d_k)

    if mask is not None:
        scaled_scores = np.where(mask, -np.inf, scaled_scores)

    attention_weights = softmax(scaled_scores)
    output = attention_weights @ V

    return output, attention_weights


def scaled_dot_product_attention_backward(d_output, cache):
    Q = cache["Q"]
    K = cache["K"]
    V = cache["V"]
    attention_weights = cache["attention_weights"]
    d_k = cache["d_k"]

    dV = attention_weights.T @ d_output
    d_attention = d_output @ V.T

    d_scaled_scores = attention_weights * (
        d_attention - np.sum(d_attention * attention_weights, axis=-1, keepdims=True)
    )
    d_scores = d_scaled_scores / np.sqrt(d_k)

    dQ = d_scores @ K
    dK = d_scores.T @ Q

    return dQ, dK, dV


class QKVProjection:
    """
    Create Query, Key, and Value matrices.

        Q = query_input @ W_Q
        K = key_value_input @ W_K
        V = key_value_input @ W_V
    """

    def __init__(self, d_model, d_k, d_v, seed=None):
        if seed is None:
            q_seed, k_seed, v_seed = None, None, None
        else:
            q_seed, k_seed, v_seed = seed, seed + 1, seed + 2

        self.W_Q = Linear(d_model, d_k, seed=q_seed)
        self.W_K = Linear(d_model, d_k, seed=k_seed)
        self.W_V = Linear(d_model, d_v, seed=v_seed)
        self._is_cross = False

    def forward(self, query_input, key_value_input=None):
        self._is_cross = key_value_input is not None
        if key_value_input is None:
            key_value_input = query_input

        Q = self.W_Q.forward(query_input)
        K = self.W_K.forward(key_value_input)
        V = self.W_V.forward(key_value_input)

        return Q, K, V

    def backward(self, dQ, dK, dV):
        d_query = self.W_Q.backward(dQ)
        d_from_k = self.W_K.backward(dK)
        d_from_v = self.W_V.backward(dV)

        if self._is_cross:
            return d_query, d_from_k + d_from_v

        return d_query + d_from_k + d_from_v, None

    def parameters_and_gradients(self):
        params = []
        params.extend(self.W_Q.parameters_and_gradients())
        params.extend(self.W_K.parameters_and_gradients())
        params.extend(self.W_V.parameters_and_gradients())
        return params


class MultiHeadAttention:
    """
    Multi-head attention from Attention Is All You Need.
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
        self._is_cross = False
        self._head_caches = []

    def forward(self, query_input, key_value_input=None, mask=None):
        self._is_cross = key_value_input is not None
        self._head_caches = []
        head_outputs = []
        attention_weights = []

        for projection in self.heads:
            Q, K, V = projection.forward(query_input, key_value_input)
            head_output, weights = scaled_dot_product_attention(Q, K, V, mask)
            head_outputs.append(head_output)
            attention_weights.append(weights)
            self._head_caches.append(
                {
                    "Q": Q,
                    "K": K,
                    "V": V,
                    "attention_weights": weights,
                    "mask": mask,
                    "d_k": self.d_k,
                }
            )

        concatenated = np.concatenate(head_outputs, axis=-1)
        output = self.W_O.forward(concatenated)
        attention_weights = np.stack(attention_weights, axis=0)

        return output, attention_weights

    def backward(self, d_output):
        d_concat = self.W_O.backward(d_output)
        d_heads = np.split(d_concat, self.num_heads, axis=-1)

        d_query_total = None
        d_kv_total = None

        for projection, d_head, cache in zip(self.heads, d_heads, self._head_caches):
            dQ, dK, dV = scaled_dot_product_attention_backward(d_head, cache)
            d_query, d_kv = projection.backward(dQ, dK, dV)

            d_query_total = d_query if d_query_total is None else d_query_total + d_query
            if self._is_cross:
                d_kv_total = d_kv if d_kv_total is None else d_kv_total + d_kv

        if self._is_cross:
            return d_query_total, d_kv_total

        return d_query_total

    def parameters_and_gradients(self):
        params = []
        for head in self.heads:
            params.extend(head.parameters_and_gradients())
        params.extend(self.W_O.parameters_and_gradients())
        return params
