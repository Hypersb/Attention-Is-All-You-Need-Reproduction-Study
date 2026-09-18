import numpy as np

from src.attention import MultiHeadAttention
from src.feed_forward import FeedForward
from src.normalization import LayerNorm


class DecoderLayer:
    """
    One Transformer decoder layer:

        masked self-attention -> residual + LayerNorm
        encoder-decoder cross-attention -> residual + LayerNorm
        feed-forward -> residual + LayerNorm
    """

    def __init__(self, d_model, num_heads, d_ff, seed=None):
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.norm1 = LayerNorm(d_model)

        cross_seed = None if seed is None else seed + 200
        self.cross_attention = MultiHeadAttention(d_model, num_heads, seed=cross_seed)
        self.norm2 = LayerNorm(d_model)

        ffn_seed = None if seed is None else seed + 400
        self.feed_forward = FeedForward(d_model, d_ff, seed=ffn_seed)
        self.norm3 = LayerNorm(d_model)

    def forward(self, X, encoder_output, causal_mask, encoder_pad_mask=None):
        self_attention_output, self_attention_weights = (
            self.masked_self_attention.forward(X, mask=causal_mask)
        )

        X1 = X + self_attention_output
        X1 = self.norm1.forward(X1)

        # Decoder queries come from the decoder representation.
        # Keys and values come from the encoder output.
        # This lets the decoder retrieve information from the encoded source.
        cross_attention_output, cross_attention_weights = self.cross_attention.forward(
            X1,
            key_value_input=encoder_output,
            mask=encoder_pad_mask,
        )

        X2 = X1 + cross_attention_output
        X2 = self.norm2.forward(X2)

        ffn_output = self.feed_forward.forward(X2)

        X3 = X2 + ffn_output
        output = self.norm3.forward(X3)

        return output, self_attention_weights, cross_attention_weights

    def backward(self, d_output):
        d_x3 = self.norm3.backward(d_output)
        d_x2 = d_x3 + self.feed_forward.backward(d_x3)

        d_residual_cross = self.norm2.backward(d_x2)
        d_query, d_encoder = self.cross_attention.backward(d_residual_cross)
        d_x1 = d_residual_cross + d_query

        d_residual_self = self.norm1.backward(d_x1)
        d_x = d_residual_self + self.masked_self_attention.backward(d_residual_self)

        return d_x, d_encoder

    def parameters_and_gradients(self):
        params = []
        params.extend(self.masked_self_attention.parameters_and_gradients())
        params.extend(self.norm1.parameters_and_gradients())
        params.extend(self.cross_attention.parameters_and_gradients())
        params.extend(self.norm2.parameters_and_gradients())
        params.extend(self.feed_forward.parameters_and_gradients())
        params.extend(self.norm3.parameters_and_gradients())
        return params


class Decoder:
    """
    Stack of independent DecoderLayer objects.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, seed=None):
        self.num_layers = num_layers
        self.layers = []

        for i in range(num_layers):
            layer_seed = None if seed is None else seed + i * 1000
            self.layers.append(
                DecoderLayer(d_model, num_heads, d_ff, seed=layer_seed)
            )

    def forward(self, X, encoder_output, causal_mask, encoder_pad_mask=None):
        all_self_attention_weights = []
        all_cross_attention_weights = []

        for layer in self.layers:
            X, self_attention_weights, cross_attention_weights = layer.forward(
                X,
                encoder_output,
                causal_mask,
                encoder_pad_mask,
            )
            all_self_attention_weights.append(self_attention_weights)
            all_cross_attention_weights.append(cross_attention_weights)

        all_self_attention_weights = np.stack(all_self_attention_weights, axis=0)
        all_cross_attention_weights = np.stack(all_cross_attention_weights, axis=0)

        return X, all_self_attention_weights, all_cross_attention_weights

    def backward(self, d_output):
        d_x = d_output
        d_encoder_total = None

        for layer in reversed(self.layers):
            d_x, d_encoder = layer.backward(d_x)
            d_encoder_total = d_encoder if d_encoder_total is None else d_encoder_total + d_encoder

        return d_x, d_encoder_total

    def parameters_and_gradients(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters_and_gradients())
        return params
