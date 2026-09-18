import numpy as np

from src.attention import (
    create_causal_mask,
    create_cross_padding_mask,
    create_key_padding_mask,
)
from src.decoder import Decoder
from src.embeddings import TokenEmbedding, positional_encoding
from src.encoder import Encoder
from src.linear import Linear
from src.math_utils import softmax


class Transformer:
    """
    Full Transformer encoder-decoder from Attention Is All You Need.
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
        use_positional_encoding=True,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.use_positional_encoding = use_positional_encoding

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
        self.cache_source_ids = None
        self.cache_target_ids = None
        self.cache_pad_id = None

    def _add_positional_encoding(self, scaled_embeddings, sequence_length):
        if self.use_positional_encoding:
            return scaled_embeddings + positional_encoding(sequence_length, self.d_model)
        return scaled_embeddings + np.zeros((sequence_length, self.d_model))

    def encode(self, source_token_ids, pad_id=None):
        self.cache_source_ids = source_token_ids
        self.cache_pad_id = pad_id

        embeddings = self.source_embedding.forward(source_token_ids)
        scaled_embeddings = embeddings * np.sqrt(self.d_model)
        X = self._add_positional_encoding(scaled_embeddings, len(source_token_ids))

        mask = None
        if pad_id is not None:
            mask = create_key_padding_mask(source_token_ids, pad_id)

        encoder_output, encoder_attention_weights = self.encoder.forward(X, mask=mask)

        return encoder_output, encoder_attention_weights

    def decode(self, target_token_ids, encoder_output, pad_id=None):
        self.cache_target_ids = target_token_ids

        embeddings = self.target_embedding.forward(target_token_ids)
        scaled_embeddings = embeddings * np.sqrt(self.d_model)
        X = self._add_positional_encoding(scaled_embeddings, len(target_token_ids))

        causal_mask = create_causal_mask(len(target_token_ids))
        if pad_id is not None:
            causal_mask = causal_mask | create_key_padding_mask(target_token_ids, pad_id)

        encoder_pad_mask = None
        if pad_id is not None and self.cache_source_ids is not None:
            encoder_pad_mask = create_cross_padding_mask(
                len(target_token_ids),
                self.cache_source_ids,
                pad_id,
            )

        decoder_output, decoder_self_attention, decoder_cross_attention = (
            self.decoder.forward(X, encoder_output, causal_mask, encoder_pad_mask)
        )

        return decoder_output, decoder_self_attention, decoder_cross_attention

    def forward(self, source_token_ids, target_token_ids, pad_id=None):
        encoder_output, encoder_attention = self.encode(source_token_ids, pad_id=pad_id)

        decoder_output, decoder_self_attention, decoder_cross_attention = self.decode(
            target_token_ids,
            encoder_output,
            pad_id=pad_id,
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

    def backward(self, d_logits):
        d_decoder_output = self.output_projection.backward(d_logits)
        d_target_x, d_encoder_output = self.decoder.backward(d_decoder_output)

        scale = np.sqrt(self.d_model)
        self.target_embedding.backward(self.cache_target_ids, d_target_x * scale)

        d_source_x = self.encoder.backward(d_encoder_output)
        self.source_embedding.backward(self.cache_source_ids, d_source_x * scale)

    def parameters_and_gradients(self):
        params = []
        params.extend(self.source_embedding.parameters_and_gradients())
        params.extend(self.target_embedding.parameters_and_gradients())
        params.extend(self.encoder.parameters_and_gradients())
        params.extend(self.decoder.parameters_and_gradients())
        params.extend(self.output_projection.parameters_and_gradients())

        seen = set()
        unique = []
        for param, grad in params:
            param_id = id(param)
            if param_id in seen:
                raise ValueError("Duplicate parameter reference collected.")
            seen.add(param_id)
            unique.append((param, grad))

        return unique

    def zero_grad(self):
        for _, grad in self.parameters_and_gradients():
            grad.fill(0.0)
