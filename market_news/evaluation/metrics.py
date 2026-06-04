"""
Evaluation metrics for financial news generation models.

Includes:
  - ``compute_perplexity`` — standard causal LM perplexity.
  - ``compute_sentiment_accuracy`` — how often predicted sentiment matches
    a reference label.
  - ``compute_coherence_score`` — simple lexical-coherence heuristic.
"""

import math
from typing import Dict, List, Optional

import torch
from torch.nn import functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 2048,
    stride: int = 512,
) -> float:
    """Compute the mean perplexity of a model on a list of texts.

    Uses a sliding-window approach for long sequences.

    Args:
        model: The language model (in ``eval()`` mode).
        tokenizer: The corresponding tokenizer.
        texts: List of text strings to evaluate.
        max_length: Maximum context window size.
        stride: Overlap stride for the sliding window.

    Returns:
        Mean perplexity across all texts.
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_length
        )
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        nll = 0.0  # negative log-likelihood
        n_tokens = 0

        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            chunk = input_ids[:, begin:end]
            if chunk.size(1) < 2:
                continue

            logits = model(chunk).logits  # (1, T, V)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = chunk[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )

            nll += loss.item()
            n_tokens += shift_labels.numel()
            prev_end = end

        if n_tokens > 0:
            total_loss += nll
            total_tokens += n_tokens

    if total_tokens == 0:
        return float("inf")

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# ---------------------------------------------------------------------------
# Sentiment accuracy
# ---------------------------------------------------------------------------

def compute_sentiment_accuracy(
    predicted_scores: List[float],
    ground_truth_scores: List[float],
    threshold: float = 0.5,
) -> float:
    """Compute classification-style sentiment accuracy.

    Scores are discretised into **positive** (>= threshold) or
    **negative** (< threshold) and compared.

    Args:
        predicted_scores: Model-predicted sentiment scores.
        ground_truth_scores: Reference sentiment scores.
        threshold: Decision boundary.

    Returns:
        Accuracy in [0.0, 1.0].
    """
    if len(predicted_scores) != len(ground_truth_scores):
        raise ValueError("Predicted and ground-truth lists must have the same length.")

    if not predicted_scores:
        return 0.0

    correct = 0
    for pred, gt in zip(predicted_scores, ground_truth_scores):
        pred_class = 1 if pred >= threshold else 0
        gt_class = 1 if gt >= threshold else 0
        if pred_class == gt_class:
            correct += 1

    return correct / len(predicted_scores)


# ---------------------------------------------------------------------------
# Coherence score (heuristic)
# ---------------------------------------------------------------------------

def compute_coherence_score(
    headlines: List[str],
    event_contexts: List[str],
) -> float:
    """Compute a simple lexical-coherence score between headlines and contexts.

    Uses Jaccard similarity of token sets (lowercased, punctuation-stripped)
    as a proxy for topical coherence.

    Args:
        headlines: Generated headlines.
        event_contexts: Corresponding event contexts.

    Returns:
        Mean Jaccard similarity across all pairs, in [0.0, 1.0].
    """
    if len(headlines) != len(event_contexts):
        raise ValueError("Headlines and contexts must have the same length.")

    if not headlines:
        return 0.0

    import re

    def _tokenize(text: str) -> set:
        return set(re.findall(r"\b[a-z]+\b", text.lower()))

    scores: List[float] = []
    for hl, ctx in zip(headlines, event_contexts):
        hl_tokens = _tokenize(hl)
        ctx_tokens = _tokenize(ctx)

        if not hl_tokens and not ctx_tokens:
            scores.append(1.0)
        elif not hl_tokens or not ctx_tokens:
            scores.append(0.0)
        else:
            intersection = hl_tokens & ctx_tokens
            union = hl_tokens | ctx_tokens
            scores.append(len(intersection) / len(union))

    return sum(scores) / len(scores)