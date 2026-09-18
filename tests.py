import numpy as np

from src.attention import (
    MultiHeadAttention,
    create_causal_mask,
    scaled_dot_product_attention,
)
from src.embeddings import TokenEmbedding, positional_encoding
from src.feed_forward import FeedForward, relu
from src.grad_check import run_gradient_checks
from src.loss import cross_entropy_with_gradient
from src.math_utils import softmax
from src.normalization import LayerNorm
from src.transformer import Transformer
from train import build_dataset, create_model, predict_example, train_model


def test_softmax_rows_sum_to_one():
    scores = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 1.0]])
    probs = softmax(scores)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_causal_masking():
    Q = np.eye(4)
    K = np.eye(4)
    V = np.arange(16, dtype=float).reshape(4, 4)
    mask = create_causal_mask(4)
    _, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    assert np.allclose(np.triu(weights, k=1), 0.0)
    for i in range(4):
        assert np.allclose(weights[i, i + 1 :], 0.0)


def test_attention_dimensions():
    Q = np.random.default_rng(0).normal(size=(5, 3))
    K = np.random.default_rng(1).normal(size=(5, 3))
    V = np.random.default_rng(2).normal(size=(5, 4))
    output, weights = scaled_dot_product_attention(Q, K, V)
    assert output.shape == (5, 4)
    assert weights.shape == (5, 5)


def test_multi_head_dimensions():
    X = np.random.default_rng(0).normal(size=(4, 8))
    mha = MultiHeadAttention(d_model=8, num_heads=2, seed=0)
    output, weights = mha.forward(X)
    assert output.shape == (4, 8)
    assert weights.shape == (2, 4, 4)


def test_positional_encoding_position_zero():
    pe = positional_encoding(sequence_length=4, d_model=4)
    assert np.allclose(pe[0], np.array([0.0, 1.0, 0.0, 1.0]))


def test_layer_norm_mean_variance():
    X = np.random.default_rng(0).normal(size=(4, 8))
    ln = LayerNorm(8)
    Y = ln.forward(X)
    assert np.allclose(np.mean(Y, axis=-1), 0.0, atol=1e-6)
    assert np.allclose(np.var(Y, axis=-1), 1.0, atol=1e-4)


def test_encoder_decoder_cross_attention_dimensions():
    model = Transformer(
        vocab_size=10,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_sequence_length=8,
        seed=0,
    )
    source = np.array([3, 4, 5, 6], dtype=int)
    target = np.array([1, 3, 4, 5], dtype=int)
    logits, probs, enc, dec_self, dec_cross = model.forward(source, target)
    assert logits.shape == (4, 10)
    assert probs.shape == (4, 10)
    assert enc.shape == (1, 2, 4, 4)
    assert dec_self.shape == (1, 2, 4, 4)
    assert dec_cross.shape == (1, 2, 4, 4)


def test_probability_distributions():
    logits = np.array([[0.1, 0.2, 0.3], [1.0, -1.0, 0.0]])
    probs = softmax(logits)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_finite_loss_and_training_smoke():
    examples, vocab, inv_vocab, pad_id, eos_id, _ = build_dataset()
    model = create_model(vocab_size=len(vocab), seed=1)
    result = train_model(model, examples, pad_id, num_epochs=5, verbose=False)
    assert np.isfinite(result["initial_loss"])
    assert np.isfinite(result["final_loss"])
    prediction = predict_example(model, examples[0], inv_vocab, pad_id, eos_id)
    assert prediction["probabilities"].shape[1] == len(vocab)


def test_embeddings_and_ffn_shapes():
    emb = TokenEmbedding(6, 4, seed=0)
    out = emb.forward(np.array([0, 2, 1]))
    assert out.shape == (3, 4)

    ffn = FeedForward(4, 8, seed=0)
    y = ffn.forward(np.ones((2, 4)))
    assert y.shape == (2, 4)
    assert np.array_equal(relu(np.array([-1.0, 0.0, 2.0])), np.array([0.0, 0.0, 2.0]))


def test_cross_entropy_gradient_shape():
    logits = np.random.default_rng(0).normal(size=(4, 6))
    targets = np.array([0, 1, 2, 3])
    loss, probs, d_logits = cross_entropy_with_gradient(logits, targets)
    assert np.isfinite(loss)
    assert probs.shape == logits.shape
    assert d_logits.shape == logits.shape


def run_all_tests():
    print("Running architectural and mathematical tests...\n")

    test_softmax_rows_sum_to_one()
    print("PASS: softmax rows sum to 1")

    test_causal_masking()
    print("PASS: causal masking")

    test_attention_dimensions()
    print("PASS: attention dimensions")

    test_multi_head_dimensions()
    print("PASS: multi-head dimensions")

    test_positional_encoding_position_zero()
    print("PASS: positional encoding position 0")

    test_layer_norm_mean_variance()
    print("PASS: LayerNorm mean/variance")

    test_encoder_decoder_cross_attention_dimensions()
    print("PASS: encoder/decoder/cross-attention dimensions")

    test_probability_distributions()
    print("PASS: probability distributions")

    test_finite_loss_and_training_smoke()
    print("PASS: finite loss and short training smoke test")

    test_embeddings_and_ffn_shapes()
    print("PASS: embeddings and FFN shapes")

    test_cross_entropy_gradient_shape()
    print("PASS: cross-entropy gradient shape")

    print("\nRunning numerical gradient checks...")
    run_gradient_checks()

    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    run_all_tests()
