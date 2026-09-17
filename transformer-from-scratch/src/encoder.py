import numpy as np

from src.attention import MultiHeadAttention
from src.feed_forward import FeedForward
from src.normalization import LayerNorm


class EncoderLayer:
    """
    One Transformer encoder layer:

        X -> Multi-Head Self-Attention -> residual + LayerNorm
          -> Feed-Forward -> residual + LayerNorm
    """

    def __init__(self, d_model, num_heads, d_ff, seed=None):
        self.self_attention = MultiHeadAttention(d_model, num_heads, seed=seed)
        self.norm1 = LayerNorm(d_model)

        ffn_seed = None if seed is None else seed + 100
        self.feed_forward = FeedForward(d_model, d_ff, seed=ffn_seed)
        self.norm2 = LayerNorm(d_model)

    def forward(self, X, mask=None):
        attention_output, attention_weights = self.self_attention.forward(X, mask=mask)

        X1 = X + attention_output
        X1 = self.norm1.forward(X1)

        ffn_output = self.feed_forward.forward(X1)

        X2 = X1 + ffn_output
        output = self.norm2.forward(X2)

        return output, attention_weights


class Encoder:
    """
    Stack of independent EncoderLayer objects.
    """

    def __init__(self, num_layers, d_model, num_heads, d_ff, seed=None):
        self.num_layers = num_layers
        self.layers = []

        for i in range(num_layers):
            layer_seed = None if seed is None else seed + i * 1000
            self.layers.append(
                EncoderLayer(d_model, num_heads, d_ff, seed=layer_seed)
            )

    def forward(self, X, mask=None):
        all_attention_weights = []

        for layer in self.layers:
            X, attention_weights = layer.forward(X, mask)
            all_attention_weights.append(attention_weights)

        all_attention_weights = np.stack(all_attention_weights, axis=0)

        return X, all_attention_weights
