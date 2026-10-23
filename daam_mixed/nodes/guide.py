"""Turning raw attention maps into sharper ones, before they become masks.

Two things happen here, both optional:

* the maps of the recorded steps are combined with SNR weights instead of a
  plain average (``none`` keeps the old behaviour);
* each token map is pushed through the self attention affinity for a few
  iterations. Self attention is the model's own opinion about which pixels
  belong together, and it is the only signal in the pack that can move a border
  away from the 16 pixel cross attention cell.

Both are post-processing: nothing here needs a re-run of the sampler, so the
sliders can be dialled in on an existing render.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

from ..core import refine
from ..core.aggregate import self_affinity, side_maps, summarize

log = logging.getLogger("daam_mixed")

STEP_WEIGHTINGS = ["none", "inverse_sigma", "inverse_sigma_squared"]
SELF_LEVELS = ["off", "secondary", "primary", "combined"]
THRESHOLD_MODES = ["otsu", "gmm", "fixed", "percentile", "none"]
NORMALIZE_MODES = ["minmax", "zscore", "none"]


def _resize(maps, size):
    """(tokens, h, w) -> (tokens, *size), bilinear."""
    tensor = maps.float()[None]
    out = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return out[0]


def _row_normalize(matrix):
    tensor = torch.as_tensor(matrix, dtype=torch.float32)
    return tensor / tensor.sum(-1, keepdim=True).clamp_min(1e-8)


class DAAMGuide:
    """Refines attention maps with the self attention graph and SNR weights."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "heatmaps": (
                    "DAAM_HEATMAP",
                    {"tooltip": "pos_heatmaps or neg_heatmaps from a DAAM sampler."},
                ),
                "step_weighting": (
                    STEP_WEIGHTINGS,
                    {
                        "default": "none",
                        "tooltip": (
                            "How the recorded steps are combined. inverse_sigma and "
                            "inverse_sigma_squared favour the last, sharpest steps."
                        ),
                    },
                ),
                "self_level": (
                    SELF_LEVELS,
                    {
                        "default": "secondary",
                        "tooltip": (
                            "Which self attention grid to diffuse through. secondary "
                            "is latent/4 and costs little; primary is latent/2 and "
                            "needs the sampler to have captured it."
                        ),
                    },
                ),
                "iterations": (
                    "INT",
                    {
                        "default": 3,
                        "min": 0,
                        "max": 50,
                        "tooltip": "Passes of the map through the affinity; 0 just merges the steps.",
                    },
                ),
                "threshold_mode": (
                    THRESHOLD_MODES,
                    {
                        "default": "otsu",
                        "tooltip": "What is dropped after every pass; none keeps the soft map.",
                    },
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "percentile": (
                    "FLOAT",
                    {"default": 85.0, "min": 1.0, "max": 99.9, "step": 0.5},
                ),
                "normalize": (
                    NORMALIZE_MODES,
                    {"default": "minmax", "tooltip": "Per token renormalisation after each pass."},
                ),
                "blend": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "1 is fully refined, 0 is the untouched map.",
                    },
                ),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
            }
        }

    RETURN_TYPES = ("DAAM_HEATMAP", "STRING")
    RETURN_NAMES = ("heatmaps", "report")
    FUNCTION = "guide"
    CATEGORY = "DAAM/Mixed"
    DESCRIPTION = (
        "Merges the recorded steps with SNR weights and refines every token map "
        "along the self attention affinity, so borders can leave the 16px grid."
    )

    def guide(
        self,
        heatmaps,
        step_weighting,
        self_level,
        iterations,
        threshold_mode,
        threshold,
        percentile,
        normalize,
        blend,
        batch_index,
    ):
        if not heatmaps:
            raise RuntimeError("[DAAM] no attention maps to refine.")

        out = {}
        notes = []
        for side in ("pos", "neg"):
            maps_by_batch = side_maps(heatmaps, side, step_weighting)
            if not maps_by_batch:
                continue

            affinity = self_affinity(heatmaps, self_level, batch_index)
            if affinity is None and iterations > 0:
                notes.append(f"{side}: no self-attention map, kept the raw maps")
                affinity = None
            elif affinity is not None and affinity.shape[0] != int(round(affinity.shape[0] ** 0.5)) ** 2:
                notes.append(f"{side}: affinity is not square, kept the raw maps")
                affinity = None

            refined = {}
            for index, by_tokens in maps_by_batch.items():
                refined[index] = {}
                for tokens, tensor in by_tokens.items():
                    refined[index][tokens] = self._refine_one(
                        tensor, affinity, iterations, threshold_mode,
                        threshold, percentile, normalize, blend,
                    )
            out[side] = refined

        steps_before = len(heatmaps.get("steps") or {})
        if step_weighting != "none":
            notes.append(f"steps merged: {steps_before} -> 1 ({step_weighting})")
        if iterations > 0 and self_level != "off":
            notes.append(f"self-attention diffusion x{iterations} at {self_level}")
        report = "; ".join(notes) or "maps passed through"
        log.info("[DAAM] guide: %s | %s", report, summarize(heatmaps))
        return (out, report)

    @staticmethod
    def _refine_one(tensor, affinity, iterations, mode, threshold, percentile, normalize, blend):
        """One (tokens, h, w) map through the affinity, at the affinity's grid."""
        original_grid = tuple(tensor.shape[-2:])
        maps = tensor.float()

        if affinity is None or iterations <= 0:
            if blend >= 1.0:
                return maps
            return maps

        grid = (int(round(affinity.shape[0] ** 0.5)),) * 2
        if grid[0] * grid[1] != affinity.shape[0]:
            return maps  # not a square grid, nothing sane to do

        at_grid = _resize(maps, grid) if tuple(grid) != original_grid else maps
        rows = at_grid.reshape(at_grid.shape[0], -1).cpu().numpy()
        refined = refine.guided_diffuse(
            rows,
            _row_normalize(affinity).cpu().numpy(),
            iterations=iterations,
            mode=mode,
            threshold=threshold,
            percentile=percentile,
            normalize=normalize,
            blend=blend,
        )
        result = torch.from_numpy(refined).reshape((refined.shape[0], *grid))
        result = result * maps.max().clamp_min(1e-8)  # back to the attention scale
        if tuple(grid) != original_grid:
            result = _resize(result, original_grid)
        return result


class DAAMMerge:
    """Averages the maps of several runs - the multi-seed trick, done openly.

    Running one step with several seeds and averaging the maps lowers the
    stochastic part of the noise by 1/sqrt(K). It only makes sense when the
    mask is being built for a prompt rather than for one particular picture:
    different seeds give different compositions, and one mask cannot describe
    all of them. Keeping the averaging in its own node makes that explicit and
    leaves the sampler returning a single latent, as every other node expects.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "heatmaps_1": ("DAAM_HEATMAP", {}),
                "heatmaps_2": ("DAAM_HEATMAP", {}),
                "mode": (["mean", "median"], {"default": "mean"}),
            },
            "optional": {
                "heatmaps_3": ("DAAM_HEATMAP", {}),
                "heatmaps_4": ("DAAM_HEATMAP", {}),
            },
        }

    RETURN_TYPES = ("DAAM_HEATMAP", "STRING")
    RETURN_NAMES = ("heatmaps", "report")
    FUNCTION = "merge"
    CATEGORY = "DAAM/Mixed"
    DESCRIPTION = "Averages attention maps from several runs (multi-seed averaging)."

    def merge(self, heatmaps_1, heatmaps_2, mode, heatmaps_3=None, heatmaps_4=None):
        sources = [entry for entry in (heatmaps_1, heatmaps_2, heatmaps_3, heatmaps_4) if entry]
        out = {}
        for side in ("pos", "neg"):
            per_source = [entry.get(side) or {} for entry in sources]
            if not any(per_source):
                continue

            merged = {}
            for index in sorted({index for maps in per_source for index in maps}):
                token_counts = {tokens for maps in per_source for tokens in maps.get(index, {})}
                for tokens in token_counts:
                    tensors = [
                        maps[index][tokens].float()
                        for maps in per_source
                        if index in maps and tokens in maps[index]
                    ]
                    if not tensors:
                        continue
                    stacked = torch.stack(tensors)
                    value = stacked.median(0).values if mode == "median" else stacked.mean(0)
                    merged.setdefault(index, {})[tokens] = value
            out[side] = merged

        report = f"{len(sources)} map set(s) merged by {mode}"
        log.info("[DAAM] merge: %s", report)
        return (out, report)
