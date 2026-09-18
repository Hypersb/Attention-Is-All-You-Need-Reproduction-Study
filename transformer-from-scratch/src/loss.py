import numpy as np

from src.math_utils import softmax


def log_softmax(logits):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    return shifted - log_sum_exp


def cross_entropy_loss(logits, target_ids, ignore_index=None):
    """
    Mean negative log-likelihood of the correct token at each position.
    """

    log_probs = log_softmax(logits)
    probabilities = np.exp(log_probs)
    n_positions = logits.shape[0]
    row_index = np.arange(n_positions)
    nll = -log_probs[row_index, target_ids]

    if ignore_index is None:
        loss = np.mean(nll)
    else:
        valid = target_ids != ignore_index
        valid_count = np.maximum(np.sum(valid), 1)
        loss = np.sum(nll * valid) / valid_count

    return loss, probabilities


def cross_entropy_with_gradient(logits, target_ids, ignore_index=None):
    loss, probabilities = cross_entropy_loss(logits, target_ids, ignore_index)

    n_positions = logits.shape[0]
    d_logits = probabilities.copy()
    d_logits[np.arange(n_positions), target_ids] -= 1

    if ignore_index is None:
        d_logits /= n_positions
    else:
        valid = target_ids != ignore_index
        d_logits *= valid[:, np.newaxis]
        valid_count = np.maximum(np.sum(valid), 1)
        d_logits /= valid_count

    return loss, probabilities, d_logits
