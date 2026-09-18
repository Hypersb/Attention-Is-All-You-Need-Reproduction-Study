import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.attention import create_causal_mask
from src.embeddings import positional_encoding
from src.grad_check import run_gradient_checks
from src.math_utils import softmax
from train import (
    build_dataset,
    create_model,
    encode_source,
    ids_to_tokens,
    predict_example,
    train_model,
)


ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(ROOT, "plots")


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)


def save_attention_heatmap(matrix, row_labels, col_labels, title, path, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Key positions")
    ax.set_ylabel("Query positions")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["training_loss"], color="#1f77b4", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Transformer From Scratch - Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "training_loss.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history_df["epoch"], history_df["gradient_norm"], color="#d62728", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Transformer From Scratch - Gradient Norm")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "gradient_norm.png"), dpi=150)
    plt.close(fig)


def plot_positional_encoding():
    pe = positional_encoding(sequence_length=50, d_model=32)
    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(pe, aspect="auto", cmap="RdBu", vmin=-1.0, vmax=1.0)
    ax.set_xlabel("Embedding dimension")
    ax.set_ylabel("Token position")
    ax.set_title("Sinusoidal Positional Encoding")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "positional_encoding.png"), dpi=150)
    plt.close(fig)


def evaluate_training_set(model, examples, inv_vocab, pad_id, eos_id):
    rows = []
    for example in examples:
        prediction = predict_example(model, example, inv_vocab, pad_id, eos_id)
        rows.append(
            {
                "source": example["source_text"],
                "expected_target": " ".join(prediction["target_tokens"]),
                "predicted_target": " ".join(prediction["predicted_tokens"]),
                "correct": bool(prediction["correct"]),
            }
        )

    df = pd.DataFrame(rows)
    accuracy = float(df["correct"].mean())
    return df, accuracy


def greedy_decode(model, source_ids, bos_id, eos_id, pad_id, max_len):
    generated = [bos_id]
    for _ in range(max_len):
        decoder_input = np.array(
            generated + [pad_id] * (max_len - len(generated)),
            dtype=int,
        )
        _, probabilities, _, _, _ = model.forward(
            source_ids,
            decoder_input,
            pad_id=pad_id,
        )
        next_id = int(np.argmax(probabilities[len(generated) - 1]))
        generated.append(next_id)
        if next_id == eos_id:
            break
    return generated[1:]


def run_unseen_tests(model, examples, vocab, inv_vocab, pad_id, eos_id, bos_id):
    source_length = len(examples[0]["source_ids"])
    max_decoder_len = len(examples[0]["decoder_input_ids"])

    unseen_sources = [
        "cats are nice",
        "dogs are nice",
        "i like cute",
    ]

    rows = []
    for source_text in unseen_sources:
        source_ids = encode_source(source_text, vocab, pad_id, source_length)
        predicted_ids = greedy_decode(
            model,
            source_ids,
            bos_id,
            eos_id,
            pad_id,
            max_decoder_len,
        )
        predicted_tokens = ids_to_tokens(predicted_ids, inv_vocab, pad_id, eos_id)
        rows.append(
            {
                "source": source_text,
                "prediction": " ".join(predicted_tokens),
            }
        )

    return pd.DataFrame(rows)


def run_ablation_positional_encoding(examples, vocab_size, pad_id, num_epochs=80):
    with_pe = create_model(vocab_size=vocab_size, seed=7, use_positional_encoding=True)
    without_pe = create_model(vocab_size=vocab_size, seed=7, use_positional_encoding=False)

    result_with = train_model(
        with_pe,
        examples,
        pad_id,
        num_epochs=num_epochs,
        verbose=False,
    )
    result_without = train_model(
        without_pe,
        examples,
        pad_id,
        num_epochs=num_epochs,
        verbose=False,
    )

    df = pd.DataFrame(
        [
            {
                "configuration": "with_positional_encoding",
                "final_loss": result_with["final_loss"],
                "initial_loss": result_with["initial_loss"],
            },
            {
                "configuration": "without_positional_encoding",
                "final_loss": result_without["final_loss"],
                "initial_loss": result_without["initial_loss"],
            },
        ]
    )
    return df


def run_ablation_num_heads(examples, vocab_size, pad_id, num_epochs=80):
    rows = []
    for num_heads in (1, 2):
        model = create_model(
            vocab_size=vocab_size,
            d_model=8,
            num_heads=num_heads,
            d_ff=16,
            seed=7,
        )
        result = train_model(
            model,
            examples,
            pad_id,
            num_epochs=num_epochs,
            verbose=False,
        )
        rows.append(
            {
                "num_heads": num_heads,
                "final_loss": result["final_loss"],
                "initial_loss": result["initial_loss"],
            }
        )
    return pd.DataFrame(rows)


def verify_causal_masking(decoder_self_attention):
    for layer_weights in decoder_self_attention:
        for head_weights in layer_weights:
            assert np.allclose(np.triu(head_weights, k=1), 0.0)
            for i in range(head_weights.shape[0]):
                for j in range(i + 1, head_weights.shape[1]):
                    assert head_weights[i, j] == 0.0
    print("PASS: decoder causal masking verified")


def verify_probabilities(probabilities):
    assert np.all(probabilities >= 0.0)
    assert np.all(probabilities <= 1.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    print("PASS: output probabilities verified")


def label_tokens(token_ids, inv_vocab, pad_id):
    labels = []
    for token_id in token_ids:
        token = inv_vocab[int(token_id)]
        if int(token_id) == pad_id:
            labels.append("<PAD>")
        else:
            labels.append(token)
    return labels


def run_experiments():
    ensure_dirs()
    np.random.seed(7)

    print("=" * 60)
    print("STEP 12: Final Experiments")
    print("=" * 60)

    print("\n[1] Gradient checks")
    grad_errors = run_gradient_checks()

    print("\n[2] Training with history recording")
    examples, vocab, inv_vocab, pad_id, eos_id, bos_id = build_dataset()
    model = create_model(vocab_size=len(vocab), seed=7, num_heads=2)
    train_result = train_model(model, examples, pad_id, num_epochs=120, verbose=True)

    history_df = pd.DataFrame(train_result["history"])
    history_path = os.path.join(RESULTS_DIR, "training_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"Saved {history_path}")

    print("\n[3] Training and gradient-norm plots")
    plot_training_curves(history_df)

    print("\n[4] Positional encoding visualization")
    plot_positional_encoding()

    print("\n[5] Attention heatmaps")
    demo = examples[0]
    prediction = predict_example(model, demo, inv_vocab, pad_id, eos_id)

    source_labels = label_tokens(demo["source_ids"], inv_vocab, pad_id)
    target_labels = label_tokens(demo["decoder_input_ids"], inv_vocab, pad_id)

    encoder_attn = prediction["encoder_attention"][0, 0]
    decoder_self = prediction["decoder_self_attention"][0, 0]
    cross_attn = prediction["decoder_cross_attention"][0, 0]

    save_attention_heatmap(
        encoder_attn,
        source_labels,
        source_labels,
        "Encoder Self-Attention (Layer 1, Head 1)",
        os.path.join(PLOTS_DIR, "encoder_attention.png"),
    )
    save_attention_heatmap(
        decoder_self,
        target_labels,
        target_labels,
        "Decoder Masked Self-Attention (Layer 1, Head 1)",
        os.path.join(PLOTS_DIR, "decoder_self_attention.png"),
    )
    save_attention_heatmap(
        cross_attn,
        target_labels,
        source_labels,
        "Decoder-Encoder Cross-Attention (Layer 1, Head 1)",
        os.path.join(PLOTS_DIR, "cross_attention.png"),
    )

    print("\n[6] Multi-head comparison")
    head1 = prediction["encoder_attention"][0, 0]
    head2 = prediction["encoder_attention"][0, 1]
    head_diff = float(np.mean(np.abs(head1 - head2)))
    print(f"Mean absolute difference between Head 1 and Head 2: {head_diff:.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, matrix, title in (
        (axes[0], head1, "Encoder Head 1"),
        (axes[1], head2, "Encoder Head 2"),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(source_labels)))
        ax.set_yticks(range(len(source_labels)))
        ax.set_xticklabels(source_labels, rotation=45, ha="right")
        ax.set_yticklabels(source_labels)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Multi-Head Attention Comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "multi_head_comparison.png"), dpi=150)
    plt.close(fig)

    print("\n[7] Training-set evaluation")
    predictions_df, accuracy = evaluate_training_set(
        model, examples, inv_vocab, pad_id, eos_id
    )
    print(predictions_df.to_string(index=False))
    print(f"Exact-match training accuracy: {accuracy:.2%}")
    predictions_path = os.path.join(RESULTS_DIR, "training_predictions.csv")
    predictions_df.to_csv(predictions_path, index=False)

    print("\n[8] Unseen combination tests")
    unseen_df = run_unseen_tests(
        model, examples, vocab, inv_vocab, pad_id, eos_id, bos_id
    )
    print(unseen_df.to_string(index=False))
    unseen_path = os.path.join(RESULTS_DIR, "unseen_predictions.csv")
    unseen_df.to_csv(unseen_path, index=False)
    print(
        "Note: this is a qualitative probe on a tiny vocabulary. "
        "It does not demonstrate strong generalization."
    )

    print("\n[9] Ablation: positional encoding")
    pe_ablation = run_ablation_positional_encoding(
        examples, len(vocab), pad_id, num_epochs=80
    )
    print(pe_ablation.to_string(index=False))

    print("\n[10] Ablation: number of heads")
    heads_ablation = run_ablation_num_heads(
        examples, len(vocab), pad_id, num_epochs=80
    )
    print(heads_ablation.to_string(index=False))

    ablation_df = pd.concat(
        [
            pe_ablation.assign(experiment="positional_encoding"),
            heads_ablation.assign(experiment="num_heads"),
        ],
        ignore_index=True,
    )
    ablation_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    ablation_df.to_csv(ablation_path, index=False)

    print("\n[11] Final verification checks")
    verify_causal_masking(prediction["decoder_self_attention"])
    verify_probabilities(prediction["probabilities"])

    # Soft sanity for softmax helper still used by the project.
    demo_scores = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    demo_probs = softmax(demo_scores)
    assert np.allclose(demo_probs.sum(axis=1), 1.0)
    assert np.allclose(create_causal_mask(4)[0, 1:], True)

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Initial loss: {train_result['initial_loss']:.4f}")
    print(f"Final loss:   {train_result['final_loss']:.4f}")
    print(f"Training exact-match accuracy: {accuracy:.2%}")
    print(f"Multi-head mean abs difference: {head_diff:.6f}")
    print("Gradient-check relative errors:")
    print(f"  output Linear: {grad_errors[0]:.3e}")
    print(f"  FFN weight:    {grad_errors[1]:.3e}")
    print(f"  attention W_Q: {grad_errors[2]:.3e}")
    print("Positional-encoding ablation final losses:")
    print(pe_ablation[["configuration", "final_loss"]].to_string(index=False))
    print("Number-of-heads ablation final losses:")
    print(heads_ablation[["num_heads", "final_loss"]].to_string(index=False))
    print("Unseen predictions:")
    print(unseen_df.to_string(index=False))
    print("Generated plots:")
    for name in sorted(os.listdir(PLOTS_DIR)):
        if name.endswith(".png"):
            print(f"  plots/{name}")
    print("Generated results:")
    for name in sorted(os.listdir(RESULTS_DIR)):
        if name.endswith(".csv"):
            print(f"  results/{name}")

    return {
        "train_result": train_result,
        "accuracy": accuracy,
        "head_diff": head_diff,
        "grad_errors": grad_errors,
        "pe_ablation": pe_ablation,
        "heads_ablation": heads_ablation,
        "unseen_df": unseen_df,
        "predictions_df": predictions_df,
    }


if __name__ == "__main__":
    run_experiments()
