"""From collected attention maps to masks, overlays and a label map.

This is the second half of the pack: the sampler records where every prompt
token looked, and this node answers the question the maps exist for - which
pixels belong to which tag - at pixel resolution rather than at the 16px grid
the attention itself works on.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ..core import refine
from ..core.tags import flatten, pick_heat_map, split_tags, stream_key, tag_maps, tokenize_prompt

log = logging.getLogger("daam_mixed")


def _parse_labels(text):
    """The tag filter field: one tag per line, commas also separate."""
    labels = []
    for line in (text or "").replace(",", "\n").splitlines():
        line = line.strip()
        if line:
            labels.append(line)
    return labels


def _to_image(tensor):
    """(H, W, 3) 0..1 numpy from a ComfyUI image row."""
    array = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
    return np.clip(array.astype(np.float32), 0.0, 1.0)


class DAAMMasks:
    """Pixel level masks, one per prompt tag."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "heatmaps": (
                    "DAAM_HEATMAP",
                    {"tooltip": "pos_heatmaps or neg_heatmaps from a DAAM sampler."},
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The encoder the prompt was conditioned with."},
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "The exact prompt that was sampled, BREAK included.",
                    },
                ),
                "images": ("IMAGE", {"tooltip": "The decoded images; also the edge guide."}),
                "tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "One tag per line to output. Empty means every tag.",
                    },
                ),
                "side": (["positive", "negative"], {"default": "positive"}),
                "threshold_mode": (
                    ["auto", "fixed", "percentile"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "auto: Otsu, kept in a sane band. fixed: threshold. "
                            "percentile: relative to the map's own distribution."
                        ),
                    },
                ),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "percentile": ("FLOAT", {"default": 85.0, "min": 1.0, "max": 99.9, "step": 0.5}),
                "pixel_precision": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Upsample with a guided filter using the image as the "
                            "edge guide, so borders follow the silhouette instead of "
                            "the 16px attention grid."
                        ),
                    },
                ),
                "smooth": (
                    "INT",
                    {"default": 1, "min": 0, "max": 16, "tooltip": "Guided filter radius, in map cells."},
                ),
                "hysteresis": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Grow the strong region down to half the threshold. Keeps "
                            "faint parts of a tag attached to its core; turn off for a "
                            "harder cut."
                        ),
                    },
                ),
                "open_px": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "tooltip": "Speck removal, in pixels."},
                ),
                "close_px": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "tooltip": "Hole filling, in pixels."},
                ),
                "grow_px": (
                    "INT",
                    {"default": 0, "min": -64, "max": 64, "tooltip": "Grow (or shrink) the mask."},
                ),
                "min_area": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 65536,
                        "tooltip": "Drop islands smaller than this many pixels; 0 keeps them.",
                    },
                ),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 128}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
                "soft": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Output the continuous map instead of a thresholded mask.",
                    },
                ),
                "overlay_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("masks", "union", "overlay", "labels", "tag_names")
    OUTPUT_TOOLTIPS = (
        "One mask per tag, at image resolution.",
        "Every requested tag in one mask.",
        "The image with the maps painted over it, jet coloured.",
        "Each pixel painted with the tag that owns it.",
        "The tags that were used, as a '|' separated list.",
    )
    FUNCTION = "make"
    CATEGORY = "DAAM/Mixed"
    DESCRIPTION = (
        "Turns attention maps into pixel level masks: guided filter upsampling, "
        "per tag thresholding, optional hysteresis, and a label map of who owns "
        "what."
    )

    def make(
        self,
        heatmaps,
        clip,
        text,
        images,
        tags,
        side,
        threshold_mode,
        threshold,
        percentile,
        pixel_precision,
        smooth,
        hysteresis,
        open_px,
        close_px,
        grow_px,
        min_area,
        feather_px,
        batch_index,
        soft,
        overlay_strength,
    ):
        if not heatmaps:
            raise RuntimeError(
                "[DAAM] no attention maps. Connect the heatmap output of a DAAM "
                "sampler, and make sure it recorded at least one step."
            )

        prompt_tokens = tokenize_prompt(clip, text)
        key = stream_key(prompt_tokens)
        token_count = len(flatten(prompt_tokens, key))

        heat_map, _ = pick_heat_map(
            heatmaps, "pos" if side == "positive" else "neg", token_count, batch_index
        )
        if heat_map is None:
            raise RuntimeError(f"[DAAM] no {side} attention maps were collected.")

        all_tags = split_tags(clip, text)
        if not all_tags:
            raise RuntimeError("[DAAM] could not split the prompt into tags.")

        used, maps = tag_maps(all_tags, heat_map, _parse_labels(tags))
        if not maps:
            raise RuntimeError(
                f"[DAAM] none of the requested tags were found. Available: "
                f"{', '.join(tag['label'] for tag in all_tags)}"
            )

        image = _to_image(images[min(batch_index, images.shape[0] - 1)])
        image_np = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=-1)

        masks = []
        overlays = []
        for low_map in maps:
            mask = refine.attention_to_mask(
                low_map.numpy(),
                image=image_np,
                radius=smooth,
                pixel_precision=pixel_precision,
                mode=threshold_mode,
                use_hysteresis=hysteresis,
                threshold=threshold,
                percentile=percentile,
                open_px=open_px,
                close_px=close_px,
                grow_px=grow_px,
                min_area=min_area,
                feather_px=feather_px,
                soft=soft,
            )
            masks.append(mask.astype(np.float32))
            overlays.append(refine.overlay(image_np, mask, overlay_strength))

        mask_stack = torch.from_numpy(np.stack(masks))
        union = torch.from_numpy(np.clip(np.max(np.stack(masks), axis=0), 0.0, 1.0))
        overlay_stack = torch.from_numpy(np.stack(overlays))
        label = torch.from_numpy(refine.label_image([np.asarray(m) for m in masks]))

        return (mask_stack, union, overlay_stack, label, "|".join(used))


class DAAMMaskCleanup:
    """Speck removal, hole filling and growing for any mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {}),
                "open_px": ("INT", {"default": 0, "min": 0, "max": 64}),
                "close_px": ("INT", {"default": 0, "min": 0, "max": 64}),
                "grow_px": ("INT", {"default": 0, "min": -64, "max": 64}),
                "keep_main": ("BOOLEAN", {"default": False}),
                "hysteresis": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Grow the strong region down to half the threshold. Keeps "
                            "faint parts of a tag attached to its core; turn off for a "
                            "harder cut."
                        ),
                    },
                ),
                "open_px": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "tooltip": "Speck removal, in pixels."},
                ),
                "close_px": (
                    "INT",
                    {"default": 0, "min": 0, "max": 64, "tooltip": "Hole filling, in pixels."},
                ),
                "grow_px": (
                    "INT",
                    {"default": 0, "min": -64, "max": 64, "tooltip": "Grow (or shrink) the mask."},
                ),
                "min_area": (
                    "INT",
                    {
                        "default": 64,
                        "min": 0,
                        "max": 65536,
                        "tooltip": "Drop islands smaller than this many pixels; 0 keeps them.",
                    },
                ),
                "feather_px": ("INT", {"default": 0, "min": 0, "max": 128}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "run"
    CATEGORY = "DAAM/Mixed"
    DESCRIPTION = "Morphological cleanup of a mask, applied per batch entry."

    def run(self, mask, open_px, close_px, grow_px, keep_main, feather_px):
        cleaned = []
        for item in mask:
            values = item.detach().cpu().numpy()
            binary = values > 0.5
            binary = refine.clean(
                binary, open_px=open_px, close_px=close_px, grow_px=grow_px
            )
            if keep_main:
                binary = refine.keep_largest(binary)
            cleaned.append(refine.feather(binary, feather_px))
        return (torch.from_numpy(np.stack(cleaned)),)
