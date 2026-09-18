import numpy as np

from src.loss import cross_entropy_with_gradient
from src.transformer import Transformer


def _loss_from_forward(model, source_ids, decoder_input_ids, expected_ids, pad_id=None):
    logits, _, _, _, _ = model.forward(source_ids, decoder_input_ids, pad_id=pad_id)
    loss, _, _ = cross_entropy_with_gradient(logits, expected_ids, ignore_index=pad_id)
    return float(loss)


def numerical_gradient(param, index, loss_fn, epsilon=1e-5):
    original = param.flat[index]
    param.flat[index] = original + epsilon
    loss_plus = loss_fn()
    param.flat[index] = original - epsilon
    loss_minus = loss_fn()
    param.flat[index] = original
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def relative_error(analytical, numerical):
    denom = max(abs(analytical) + abs(numerical), 1e-8)
    return abs(analytical - numerical) / denom


def check_parameter(name, param, grad, loss_fn, n_elements=3, epsilon=1e-5, max_rel_error=5e-4):
    rng = np.random.default_rng(0)
    n = param.size
    indices = rng.choice(n, size=min(n_elements, n), replace=False)

    results = []
    for index in indices:
        numerical = numerical_gradient(param, int(index), loss_fn, epsilon=epsilon)
        analytical = float(grad.flat[int(index)])

        if not np.isfinite(numerical) or not np.isfinite(analytical):
            raise ValueError(f"{name}: non-finite gradient (ana={analytical}, num={numerical})")

        error = relative_error(analytical, numerical)
        results.append((int(index), analytical, numerical, error))

    worst = max(results, key=lambda item: item[3])
    _, ana, num, error = worst

    status = "PASS" if error < max_rel_error else "FAIL"
    print(
        f"{status}  {name}: relative error={error:.3e}  "
        f"analytical={ana:.6e}  numerical={num:.6e}"
    )

    if error >= max_rel_error:
        raise AssertionError(
            f"Gradient check failed for {name}: relative error {error:.3e} exceeds {max_rel_error}"
        )

    return error


def run_gradient_checks():
    print("Running numerical gradient checks...")

    model = Transformer(
        vocab_size=8,
        d_model=4,
        num_heads=2,
        d_ff=8,
        num_encoder_layers=1,
        num_decoder_layers=1,
        max_sequence_length=8,
        seed=0,
    )

    source_ids = np.array([3, 4, 5], dtype=int)
    decoder_input_ids = np.array([1, 3, 4], dtype=int)
    expected_ids = np.array([3, 4, 2], dtype=int)

    def loss_fn():
        return _loss_from_forward(model, source_ids, decoder_input_ids, expected_ids)

    model.zero_grad()
    logits, _, _, _, _ = model.forward(source_ids, decoder_input_ids)
    _, _, d_logits = cross_entropy_with_gradient(logits, expected_ids)
    model.backward(d_logits)

    linear_w = model.output_projection.W
    linear_dw = model.output_projection.dW
    ffn_w = model.encoder.layers[0].feed_forward.linear1.W
    ffn_dw = model.encoder.layers[0].feed_forward.linear1.dW
    attn_w = model.encoder.layers[0].self_attention.heads[0].W_Q.W
    attn_dw = model.encoder.layers[0].self_attention.heads[0].W_Q.dW

    errors = [
        check_parameter("Linear weight (output projection)", linear_w, linear_dw, loss_fn),
        check_parameter("FFN weight (encoder linear1)", ffn_w, ffn_dw, loss_fn),
        check_parameter("Attention projection weight (W_Q)", attn_w, attn_dw, loss_fn),
    ]

    print("All gradient checks passed.")
    return errors
