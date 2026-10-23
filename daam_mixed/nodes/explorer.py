"""The visual half: a node that hands the maps to an interactive panel.

The Python side does the cheap part - one PNG of the render and one small .npy
of the per tag maps at their native image/16 grid - and the browser does all
the interaction: hovering shows which tags claim a pixel, clicking a tag paints
its map, and the strength and smoothing sliders redraw without a round trip.

Shipping the native grid instead of full resolution maps is what keeps this
small: 20 tags at 64x64 is 320 kB, while the same maps at 1024px would be
80 MB.
"""

from __future__ import annotations

import io
import logging

import numpy as np
import torch
from PIL import Image

try:
    from server import PromptServer
except Exception:  # pragma: no cover - only absent outside ComfyUI
    PromptServer = None

from ..core.tags import flatten, pick_heat_map, split_tags, stream_key, tag_maps, tokenize_prompt

log = logging.getLogger("daam_mixed")

# node id -> (png bytes, npy bytes, tag names). Rerunning a node replaces its
# entry, so this never holds more than one payload per explorer node.
EXPLORER_RESULTS = {}


def _parse_labels(text):
    return [line.strip() for line in (text or "").replace(",", "\n").splitlines() if line.strip()]


class DAAMExplorer:
    """Shows the maps in an interactive panel on the node itself."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "heatmaps": (
                    "DAAM_HEATMAP",
                    {"tooltip": "pos_heatmaps or neg_heatmaps from a DAAM sampler."},
                ),
                "clip": ("CLIP", {"tooltip": "The encoder the prompt was conditioned with."}),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "The exact prompt that was sampled, BREAK included.",
                    },
                ),
                "images": ("IMAGE", {"tooltip": "The decoded images."}),
                "tags": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "One tag per line to show. Empty means every tag.",
                    },
                ),
                "side": (["positive", "negative"], {"default": "positive"}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "explore"
    OUTPUT_NODE = True
    CATEGORY = "DAAM/Mixed"
    DESCRIPTION = (
        "Interactive attention panel: hover the image for the tags that claim "
        "that pixel, click tags to paint their maps, drag the sliders to smooth."
    )

    def explore(self, heatmaps, clip, text, images, tags, side, batch_index, unique_id):
        if not heatmaps:
            raise RuntimeError(
                "[DAAM] no attention maps were collected. Check that the heatmap "
                "output of a DAAM sampler is connected and that at least one step "
                "was recorded."
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
        labels, maps = tag_maps(all_tags, heat_map, _parse_labels(tags))
        if not maps:
            raise RuntimeError(
                "[DAAM] none of the requested tags were found. Available: "
                f"{', '.join(tag['label'] for tag in all_tags)}"
            )

        stacked = torch.stack([m.float() for m in maps]).numpy()
        # One scale for every tag, stretched by the strongest map, so the bars
        # of the panel compare tags instead of each being its own maximum.
        top = float(np.percentile(stacked, 99.0)) or 1.0
        stacked = np.clip(stacked / top, 0.0, 1.0).astype(np.float32)

        image = images[min(int(batch_index), images.shape[0] - 1)]
        array = (image.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        image_bytes, map_bytes = io.BytesIO(), io.BytesIO()
        Image.fromarray(array).save(image_bytes, format="PNG", compress_level=1)
        np.save(map_bytes, stacked)

        EXPLORER_RESULTS[str(unique_id)] = {
            "image": image_bytes.getvalue(),
            "maps": map_bytes.getvalue(),
            "tags": labels,
        }

        return {
            "ui": {
                "daam": [
                    {
                        "node": str(unique_id),
                        "tags": labels,
                        "shape": [int(stacked.shape[1]), int(stacked.shape[2])],
                    }
                ]
            }
        }


if PromptServer is not None:
    routes = PromptServer.instance.routes

    @routes.get("/daam_mixed/{node_id}/{kind}")
    async def daam_mixed_result(request):
        from aiohttp import web

        entry = EXPLORER_RESULTS.get(request.match_info["node_id"])
        if entry is None:
            return web.Response(status=404)
        kind = request.match_info["kind"]
        headers = {"Cache-Control": "no-store"}
        if kind == "image.png":
            return web.Response(body=entry["image"], content_type="image/png", headers=headers)
        if kind == "maps.npy":
            return web.Response(
                body=entry["maps"], content_type="application/octet-stream", headers=headers
            )
        return web.Response(status=404)
