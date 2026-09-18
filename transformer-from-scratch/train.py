import numpy as np

from src.loss import cross_entropy_with_gradient
from src.optim import Adam, clip_grad_norm
from src.transformer import Transformer


PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"


def build_dataset():
    pairs = [
        ("i love cats", "cats are cute"),
        ("i love dogs", "dogs are cute"),
        ("i like cats", "cats are nice"),
        ("i like dogs", "dogs are nice"),
        ("cats are cute", "i love cats"),
        ("dogs are cute", "i love dogs"),
    ]

    specials = [PAD, BOS, EOS]
    words = []
    for source, target in pairs:
        words.extend(source.split())
        words.extend(target.split())

    vocab = {token: index for index, token in enumerate(specials)}
    for word in words:
        if word not in vocab:
            vocab[word] = len(vocab)

    inv_vocab = {index: token for token, index in vocab.items()}
    pad_id = vocab[PAD]
    bos_id = vocab[BOS]
    eos_id = vocab[EOS]

    encoded = []
    for source, target in pairs:
        source_ids = [vocab[token] for token in source.split()]
        target_ids = [vocab[token] for token in target.split()]
        encoded.append((source_ids, target_ids, source, target))

    max_source = max(len(item[0]) for item in encoded)
    max_target = max(len(item[1]) for item in encoded)

    examples = []
    for source_ids, target_ids, source_text, target_text in encoded:
        source_padded = source_ids + [pad_id] * (max_source - len(source_ids))
        decoder_input = [bos_id] + target_ids + [pad_id] * (max_target - len(target_ids))
        expected = target_ids + [eos_id] + [pad_id] * (max_target - len(target_ids))
        examples.append(
            {
                "source_ids": np.array(source_padded, dtype=int),
                "decoder_input_ids": np.array(decoder_input, dtype=int),
                "expected_ids": np.array(expected, dtype=int),
                "source_text": source_text,
                "target_text": target_text,
            }
        )

    return examples, vocab, inv_vocab, pad_id, eos_id, bos_id


def encode_source(text, vocab, pad_id, length):
    ids = [vocab[token] for token in text.split()]
    return np.array(ids + [pad_id] * (length - len(ids)), dtype=int)


def ids_to_tokens(ids, inv_vocab, pad_id, eos_id):
    tokens = []
    for token_id in ids:
        token_id = int(token_id)
        if token_id == pad_id:
            break
        tokens.append(inv_vocab[token_id])
        if token_id == eos_id:
            break
    return tokens


def create_model(
    vocab_size,
    d_model=8,
    num_heads=2,
    d_ff=16,
    num_encoder_layers=1,
    num_decoder_layers=1,
    seed=7,
    use_positional_encoding=True,
):
    return Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        max_sequence_length=16,
        seed=seed,
        use_positional_encoding=use_positional_encoding,
    )


def train_model(
    model,
    examples,
    pad_id,
    learning_rate=0.02,
    num_epochs=120,
    max_grad_norm=1.0,
    verbose=True,
):
    parameters = model.parameters_and_gradients()
    optimizer = Adam(parameters, learning_rate=learning_rate)

    history = []
    initial_loss = None
    last_grad_norm = 0.0

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0

        for example in examples:
            optimizer.zero_grad()
            logits, _, _, _, _ = model.forward(
                example["source_ids"],
                example["decoder_input_ids"],
                pad_id=pad_id,
            )
            loss, _, d_logits = cross_entropy_with_gradient(
                logits,
                example["expected_ids"],
                ignore_index=pad_id,
            )

            if not np.isfinite(loss):
                raise ValueError(f"Non-finite loss at epoch {epoch}: {loss}")

            model.backward(d_logits)
            last_grad_norm = clip_grad_norm(parameters, max_norm=max_grad_norm)
            optimizer.step()
            epoch_loss += float(loss)

        epoch_loss /= len(examples)
        if initial_loss is None:
            initial_loss = epoch_loss

        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss,
                "gradient_norm": float(last_grad_norm),
            }
        )

        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == num_epochs):
            print(
                f"Epoch {epoch:<4}  Loss: {epoch_loss:.4f}  "
                f"Grad norm: {last_grad_norm:.4f}"
            )

    return {
        "history": history,
        "initial_loss": initial_loss,
        "final_loss": history[-1]["training_loss"],
    }


def predict_example(model, example, inv_vocab, pad_id, eos_id):
    _, probabilities, encoder_attention, decoder_self, decoder_cross = model.forward(
        example["source_ids"],
        example["decoder_input_ids"],
        pad_id=pad_id,
    )
    predicted_ids = np.argmax(probabilities, axis=-1)
    predicted_tokens = ids_to_tokens(predicted_ids, inv_vocab, pad_id, eos_id)
    target_tokens = ids_to_tokens(example["expected_ids"], inv_vocab, pad_id, eos_id)
    return {
        "probabilities": probabilities,
        "encoder_attention": encoder_attention,
        "decoder_self_attention": decoder_self,
        "decoder_cross_attention": decoder_cross,
        "predicted_ids": predicted_ids,
        "predicted_tokens": predicted_tokens,
        "target_tokens": target_tokens,
        "correct": predicted_tokens == target_tokens,
    }


def run_regression_tests(model, example, pad_id):
    logits, probabilities, encoder_attention, decoder_self, decoder_cross = model.forward(
        example["source_ids"],
        example["decoder_input_ids"],
        pad_id=pad_id,
    )

    assert np.all(np.isfinite(logits))
    assert np.all(np.isfinite(probabilities))
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)

    source_len = len(example["source_ids"])
    target_len = len(example["decoder_input_ids"])
    num_encoder_layers = len(model.encoder.layers)
    num_decoder_layers = len(model.decoder.layers)
    num_heads = model.encoder.layers[0].self_attention.num_heads

    assert encoder_attention.shape == (num_encoder_layers, num_heads, source_len, source_len)
    assert decoder_self.shape == (num_decoder_layers, num_heads, target_len, target_len)
    assert decoder_cross.shape == (num_decoder_layers, num_heads, target_len, source_len)

    for layer_weights in encoder_attention:
        for head_weights in layer_weights:
            assert np.allclose(head_weights.sum(axis=1), 1.0)
            future = head_weights[0, 1:]
            assert np.any(future > 0.0)

    for layer_weights in decoder_self:
        for head_weights in layer_weights:
            assert np.allclose(head_weights.sum(axis=1), 1.0)
            assert np.allclose(np.triu(head_weights, k=1), 0.0)

    for layer_weights in decoder_cross:
        for head_weights in layer_weights:
            assert np.allclose(head_weights.sum(axis=1), 1.0)

    variances = np.var(
        model.encoder.layers[0].norm2.cache["X_normalized"],
        axis=-1,
    )
    assert np.allclose(variances, 1.0, atol=1e-3)
    print("Regression tests passed.")


def train():
    print("\nStarting training...\n")

    examples, vocab, inv_vocab, pad_id, eos_id, _ = build_dataset()
    model = create_model(vocab_size=len(vocab), seed=7)
    result = train_model(model, examples, pad_id, verbose=True)

    print("\nInitial loss:", f"{result['initial_loss']:.4f}")
    print("Final loss:", f"{result['final_loss']:.4f}")

    assert np.isfinite(result["initial_loss"])
    assert np.isfinite(result["final_loss"])
    assert result["final_loss"] < result["initial_loss"] * 0.7

    print("\nPredictions after training:")
    for example in examples[:4]:
        prediction = predict_example(model, example, inv_vocab, pad_id, eos_id)
        print(f"  source    : {example['source_text']}")
        print(f"  target    : {' '.join(prediction['target_tokens'])}")
        print(f"  predicted : {' '.join(prediction['predicted_tokens'])}")
        print()

    run_regression_tests(model, examples[0], pad_id)
    print("All training assertions passed.")


if __name__ == "__main__":
    train()
