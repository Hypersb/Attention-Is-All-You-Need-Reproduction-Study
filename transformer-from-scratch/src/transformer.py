import numpy as np

from src.attention import create_causal_mask
from src.decoder import Decoder
from src.embeddings import TokenEmbedding, positional_encoding
from src.encoder import Encoder
from src.linear import Linear
from src.math_utils import softmax


class Transformer:
    """
    Full Transformer encoder-decoder from Attention Is All You Need.

    This class only implements the forward pass.
    """

    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_encoder_layers,
        num_decoder_layers,
        max_sequence_length,
        seed=None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

        src_seed = seed
        tgt_seed = None if seed is None else seed + 1
        enc_seed = None if seed is None else seed + 2
        dec_seed = None if seed is None else seed + 10000
        proj_seed = None if seed is None else seed + 20000

        self.source_embedding = TokenEmbedding(vocab_size, d_model, seed=src_seed)
        self.target_embedding = TokenEmbedding(vocab_size, d_model, seed=tgt_seed)

        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            seed=enc_seed,
        )
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            seed=dec_seed,
        )

        self.output_projection = Linear(d_model, vocab_size, seed=proj_seed)

    def encode(self, source_token_ids):
        embeddings = self.source_embedding.forward(source_token_ids)
        scaled_embeddings = embeddings * np.sqrt(self.d_model)
        pe = positional_encoding(len(source_token_ids), self.d_model)
        X = scaled_embeddings + pe

        encoder_output, encoder_attention_weights = self.encoder.forward(X, mask=None)

        return encoder_output, encoder_attention_weights

    def decode(self, target_token_ids, encoder_output):
        embeddings = self.target_embedding.forward(target_token_ids)
        scaled_embeddings = embeddings * np.sqrt(self.d_model)
        pe = positional_encoding(len(target_token_ids), self.d_model)
        X = scaled_embeddings + pe

        causal_mask = create_causal_mask(len(target_token_ids))

        decoder_output, decoder_self_attention, decoder_cross_attention = (
            self.decoder.forward(X, encoder_output, causal_mask)
        )

        return decoder_output, decoder_self_attention, decoder_cross_attention

    def forward(self, source_token_ids, target_token_ids):
        encoder_output, encoder_attention = self.encode(source_token_ids)

        decoder_output, decoder_self_attention, decoder_cross_attention = self.decode(
            target_token_ids,
            encoder_output,
        )

        logits = self.output_projection.forward(decoder_output)
        probabilities = softmax(logits)

        return (
            logits,
            probabilities,
            encoder_attention,
            decoder_self_attention,
            decoder_cross_attention,
        )
