"""Head aggregation for attention maps.

A cross attention layer has 8-40 heads and each one looks somewhere else. Plain
averaging blurs a word over all the places any head glanced at, which is the
main reason a DAAM map looks like a fog. Two things help:

* ``median`` - an average that ignores the one or two heads that fire on the
  background, and ``max`` - keeps the strongest head and throws away the rest;
* ``min_entropy`` - keep only the heads whose attention distribution is the
  most concentrated. A head that spreads its attention over the whole prompt
  (high entropy) contributes noise; a head that locks onto one word (low
  entropy) contributes shape. Selecting the lowest entropy quarter was the
  single most useful aggregation in the diploma experiments, and it is cheap:
  entropy is computed per head in the same pass.
"""

from __future__ import annotations

import torch

METHODS = ("mean", "median", "max", "min_entropy")
# Share of the heads kept by min_entropy: the most concentrated quarter.
ENTROPY_FRACTION = 0.25


def head_entropy(probs):
    """Normalised entropy of every head, averaged over the queries.

    ``probs`` is (batch, heads, queries, keys). The result is (heads,), each
    value in 0..1: 0 is a head that attends to a single key, 1 is a head that
    spreads evenly over all keys.
    """
    keys = probs.shape[-1]
    if keys < 2:
        return torch.zeros(probs.shape[1], device=probs.device)
    entropy = -(probs.clamp_min(1e-12).log() * probs).sum(-1)  # (batch, heads, queries)
    normaliser = float(torch.tensor(float(keys)).log())
    return entropy.mean((0, 2)) / normaliser


def select_heads(probs, fraction=ENTROPY_FRACTION):
    """Indices of the most concentrated heads, at least one."""
    heads = probs.shape[1]
    keep = max(1, int(round(heads * float(fraction))))
    entropy = head_entropy(probs)
    order = torch.argsort(entropy)  # lowest entropy first
    return order[:keep].sort().values


def aggregate_heads(probs, method="mean", fraction=ENTROPY_FRACTION):
    """(batch, heads, queries, keys) -> (batch, queries, keys).

    The result is renormalised over the key axis, so every query still reads as
    a distribution: a threshold means the same thing whatever the aggregation.
    """
    if method == "median":
        merged = probs.median(dim=1).values
    elif method == "max":
        merged = probs.max(dim=1).values
    elif method == "min_entropy":
        merged = probs[:, select_heads(probs, fraction)].mean(dim=1)
    else:
        merged = probs.mean(dim=1)
    return merged / merged.sum(-1, keepdim=True).clamp_min(1e-12)


def value_norm_weights(value, tokens):
    """Per token weight from the norm of its value vector, mean 1.

    Padding and the end-of-prompt token carry value vectors that are close to
    zero, so their attention maps are noise on a scale that still sums to one.
    Scaling the token's row by its value norm pushes them down without changing
    the row sum by more than the renormalisation that follows.
    """
    if value is None:
        return None
    norms = value.detach().float().norm(dim=-1)  # (batch, tokens)
    if norms.numel() == 0:
        return None
    mean = norms.mean(dim=-1, keepdim=True).clamp_min(1e-12)
    weights = norms / mean
    if weights.shape[-1] != tokens:  # value may be shared/expanded
        weights = weights[..., :tokens]
        if weights.shape[-1] != tokens:
            return None
    return weights


def apply_value_norm(merged, value):
    """Scale (batch, queries, tokens) by the value norm of each token."""
    weights = value_norm_weights(value, merged.shape[-1])
    if weights is None:
        return merged
    scaled = merged * weights[:, None, :]
    return scaled / scaled.sum(-1, keepdim=True).clamp_min(1e-12)
