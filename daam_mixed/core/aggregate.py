"""Combining the maps of several sampling steps.

The sampler can keep a map per step. Averaging them with equal weight, as the
diploma version and both public packs do, treats the diffuse first step (where
the model has only decided the composition) the same as the last one (where it
has decided the edges). Weighting by the inverse of sigma says what was actually
meant: the later the step, the sharper the attribution.

``inverse_sigma`` weights a step by 1/sigma and ``inverse_sigma_squared`` by
1/sigma^2, so with a schedule from 14 down to 0.1 the last step counts roughly
140x or 20000x as much as the first. ``none`` keeps the plain average, which is
what the older packs do.
"""

from __future__ import annotations

import torch

STEP_WEIGHTINGS = ("none", "inverse_sigma", "inverse_sigma_squared")
EPSILON = 1e-3


def step_weights(schedule, steps, mode="inverse_sigma"):
    """{step: weight}, normalised to sum 1 over the given steps.

    ``schedule`` is the sigma list of the sampling pass (length steps + 1), the
    same one the sampler compared against to find the step index.
    """
    steps = [int(step) for step in steps]
    if not steps:
        return {}

    if mode not in STEP_WEIGHTINGS or mode == "none" or not schedule:
        return {step: 1.0 / len(steps) for step in steps}

    weights = {}
    for step in steps:
        sigma = float(schedule[step]) if 0 <= step < len(schedule) else 0.0
        sigma = max(sigma, 0.0)
        value = 1.0 / (sigma + EPSILON)
        if mode == "inverse_sigma_squared":
            value = value * value
        weights[step] = value

    total = sum(weights.values()) or 1.0
    return {step: value / total for step, value in weights.items()}


def combine_steps(per_step, weights):
    """Weighted average of {step: {batch: {tokens: map}}} -> {batch: {tokens: map}}.

    Steps whose token count or grid differ are averaged over the intersection,
    so a prompt that changed length mid-schedule cannot corrupt the result.
    """
    accumulated = {}
    total = {}
    for step, by_batch in per_step.items():
        weight = float(weights.get(int(step), 0.0))
        if weight <= 0:
            continue
        for batch_index, by_tokens in by_batch.items():
            for tokens, tensor in by_tokens.items():
                key = (batch_index, tokens)
                if key in accumulated and accumulated[key].shape != tensor.shape:
                    continue
                if key in accumulated:
                    accumulated[key] = accumulated[key] + tensor.float() * weight
                else:
                    accumulated[key] = tensor.float() * weight
                    total[key] = 0.0
                total[key] += weight

    result = {}
    for (batch_index, tokens), tensor in accumulated.items():
        divisor = total.get((batch_index, tokens), 0.0) or 1.0
        result.setdefault(batch_index, {})[tokens] = tensor / divisor
    return result


def side_maps(heatmaps, side, step_weighting="none"):
    """The maps of one side, optionally collapsed over steps with SNR weights.

    Falls back to the step-averaged maps the sampler already produced when no
    per step data was kept.
    """
    if not heatmaps:
        return {}

    declared = heatmaps.get("side")
    if declared and declared != side:
        return {}
    if not declared and not heatmaps.get(side):
        return {}

    per_step = heatmaps.get("steps") or {}
    if step_weighting == "none" or not per_step:
        return heatmaps.get(side) or {}

    schedule = (heatmaps.get("sigmas") or {}).get("schedule") or []
    weights = step_weights(schedule, per_step.keys(), step_weighting)
    return combine_steps(per_step, weights)


def self_affinity(heatmaps, level="secondary", batch_index=0):
    """The self attention affinity matrix of one batch entry, or None."""
    self_maps = (heatmaps or {}).get("self") or {}
    if not self_maps:
        return None
    if level == "off":
        return None

    chosen = self_maps.get(level) or None
    if not chosen and level == "combined":
        chosen = self_maps.get("primary") or self_maps.get("secondary") or None
    if not chosen and level == "primary":
        chosen = self_maps.get("secondary") or None
    if not chosen:
        return None
    if batch_index in chosen:
        return chosen[batch_index]
    return chosen[min(chosen)] if chosen else None


def summarize(heatmaps):
    """A one line description for node output and for the console."""
    if not heatmaps:
        return "no maps"
    parts = []
    for side in ("pos", "neg"):
        by_batch = heatmaps.get(side) or {}
        if by_batch:
            counts = sorted({tokens for entry in by_batch.values() for tokens in entry})
            parts.append(f"{side}: {len(by_batch)} image(s), tokens {counts}")
    steps = heatmaps.get("steps") or {}
    if steps:
        parts.append("steps " + ",".join(str(step) for step in sorted(steps)))
    self_maps = heatmaps.get("self") or {}
    if self_maps:
        levels = [level for level in ("primary", "secondary", "combined") if self_maps.get(level)]
        parts.append("self: " + ",".join(levels))
    return "; ".join(parts) or "empty maps"


def diff(a, b):
    """Largest absolute difference of two maps, for the tests and the log."""
    if a is None or b is None:
        return None
    return float((torch.as_tensor(a).float() - torch.as_tensor(b).float()).abs().max())
