# Transformer From Scratch

An educational NumPy reproduction of the architecture introduced in
**Attention Is All You Need** (Vaswani et al., 2017).

This project implements the encoder-decoder Transformer end to end
without PyTorch, TensorFlow, Keras, JAX, or Hugging Face. Gradients,
backpropagation, and Adam are written manually in NumPy.

---

## 1. Overview

The goal is understanding, not WMT-scale performance. The repository
builds the Transformer piece by piece:

token embeddings → positional encoding → multi-head attention →
encoder/decoder stacks → vocabulary projection → cross-entropy training.

A tiny toy dataset is used so every mathematical step remains inspectable.

## 2. Motivation

Frameworks hide the internals of attention, LayerNorm, residuals, and
optimizer state. Reimplementing them from scratch makes the paper's
formulas concrete:

- why we scale by \(\sqrt{d_k}\)
- why decoder self-attention is causally masked
- how encoder-decoder cross-attention retrieves source information
- how residuals and LayerNorm stabilize deep stacks
- how analytical gradients compare with finite differences

## 3. Architecture

This educational model uses small dimensions for fast CPU training:

| Setting | Value |
|---|---|
| `d_model` | 8 |
| `num_heads` | 2 |
| `d_ff` | 16 |
| encoder layers | 1 |
| decoder layers | 1 |

The original paper used `d_model = 512`, 8 heads, and 6 layers per stack.
Those settings are intentionally **not** reproduced here.

## 4. Scaled Dot-Product Attention

\[
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Implemented in `src/attention.py` with optional boolean masking
(`True` = blocked position).

## 5. Multi-Head Attention

Each head has independent \(W_Q\), \(W_K\), and \(W_V\). Head outputs are
concatenated and projected by \(W_O\):

\[
\mathrm{MultiHead}(X)=\mathrm{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_h)W_O
\]

The same module supports:

- self-attention (encoder / decoder)
- cross-attention (decoder queries, encoder keys/values)

## 6. Positional Encoding

Sinusoidal positional encodings from the paper:

\[
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),\quad
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

Token embeddings are scaled by \(\sqrt{d_{model}}\) before adding PE.

## 7. Encoder

Each encoder layer:

1. multi-head self-attention
2. residual + LayerNorm
3. position-wise feed-forward
4. residual + LayerNorm

Encoder attention is bidirectional (non-causal), with optional PAD masking.

## 8. Decoder

Each decoder layer:

1. **masked** multi-head self-attention
2. residual + LayerNorm
3. encoder-decoder **cross-attention**
4. residual + LayerNorm
5. feed-forward
6. residual + LayerNorm

## 9. Cross-Attention

Decoder queries come from the decoder representation.
Keys and values come from the encoder output.

That is the mechanism that lets the decoder retrieve information from
the encoded source sequence.

## 10. Training From Scratch

Training components implemented manually in NumPy:

- numerically stable cross-entropy (`src/loss.py`)
- backward methods for Linear, ReLU, LayerNorm, attention, embeddings
- Adam with bias correction (`src/optim.py`)
- global gradient-norm clipping
- teacher forcing with `<BOS>` / `<EOS>` / `<PAD>`

Training command:

```bash
python train.py
```

## 11. Gradient Checking

Finite-difference checks validate analytical gradients for:

- output Linear weights
- FFN weights
- attention projection weights

Representative relative errors from the final suite:

| Parameter | Relative error |
|---|---|
| Output Linear weight | `4.070e-11` |
| FFN weight | `5.311e-08` |
| Attention \(W_Q\) | `6.022e-05` |

Exact values are printed by `python tests.py` and `python experiments.py`.

## 12. Experiments

Run:

```bash
python experiments.py
```

Selected results from the final educational run:

| Metric | Value |
|---|---|
| Initial training loss | `2.2711` |
| Final training loss (120 epochs) | `0.0002` |
| Training exact-match accuracy | `100%` |
| Multi-head mean abs difference | `0.371799` |
| Final loss with PE (80-epoch ablation) | `0.000480` |
| Final loss without PE (80-epoch ablation) | `0.000407` |
| Final loss with 1 head (80 epochs) | `0.125447` |
| Final loss with 2 heads (80 epochs) | `0.000480` |

On this tiny deterministic dataset, removing positional encoding did **not** hurt final loss. That is expected to be possible when sequences are very short and patterns are memorized; it should **not** be interpreted as evidence that positional encoding is unnecessary in general.

Recorded artifacts:

- `results/training_history.csv`
- `results/training_predictions.csv`
- `results/unseen_predictions.csv`
- `results/ablation_results.csv`

Experiments include:

1. training loss / gradient-norm curves
2. attention heatmaps
3. multi-head difference metric
4. training-set exact-match accuracy
5. qualitative unseen combination probes
6. positional-encoding ablation
7. number-of-heads ablation

## 13. Visualizations

Generated under `plots/`:

- `training_loss.png`
- `gradient_norm.png`
- `positional_encoding.png`
- `encoder_attention.png`
- `decoder_self_attention.png`
- `cross_attention.png`
- `multi_head_comparison.png`

## 14. Project Structure

```text
transformer-from-scratch/
├── src/
│   ├── math_utils.py
│   ├── linear.py
│   ├── attention.py
│   ├── embeddings.py
│   ├── normalization.py
│   ├── feed_forward.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── transformer.py
│   ├── loss.py
│   ├── optim.py
│   └── grad_check.py
├── results/
├── plots/
├── main.py
├── train.py
├── experiments.py
├── tests.py
├── gradient_check.py
├── requirements.txt
└── README.md
```

## 15. How to Run

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt

python main.py
python train.py
python tests.py
python experiments.py
```

## 16. Limitations

- Tiny educational dataset (a handful of short phrase pairs).
- Tiny model dimensions (`d_model=8`), far from the paper's 512-d model.
- Results are **not** comparable to original WMT translation numbers.
- High training-set accuracy mainly shows memorization of the toy pairs.
- Unseen probes are qualitative only; they do **not** prove broad language ability.
- Implementation prioritizes clarity over speed, batching, or production features.
- No beam search, length penalty, or large-scale tokenization.

## 17. Reference

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, Ł., & Polosukhin, I. (2017).
**Attention Is All You Need**.
*Advances in Neural Information Processing Systems (NeurIPS)*.
