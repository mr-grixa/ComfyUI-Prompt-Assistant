"""Sampling nodes that record cross attention maps in a window of steps.

Two flavours on purpose: a KSampler look-alike for the common graph, and a
SamplerCustom look-alike for the sampler/sigmas chain. Both run a normal
sampling pass - the node is a drop-in replacement, it does not resample - and
both take the capture window as "how many of the last steps", which is the
setting that matters in practice: the last step is the sharpest attribution and
the cheapest to record, while early steps are noisy and slow.
"""

from __future__ import annotations

import logging

import comfy.sample
import comfy.samplers
import comfy.utils

from ..core.capture import (
    AttentionCollector,
    install_attention_capture,
    remove_attention_capture,
)

log = logging.getLogger("daam_mixed")

try:
    import latent_preview
except Exception:  # pragma: no cover - ComfyUI always ships it
    latent_preview = None

CAPTURE_INPUTS = {
    "capture_last_steps": (
        "INT",
        {
            "default": 1,
            "min": 0,
            "max": 10000,
            "tooltip": (
                "How many of the final sampling steps are recorded. 1 records "
                "only the last step, which is the sharpest and the fastest to "
                "record; 0 records every step."
            ),
        },
    ),
    "capture_end_offset": (
        "INT",
        {
            "default": 0,
            "min": 0,
            "max": 10000,
            "tooltip": (
                "Steps to leave out at the very end, e.g. 1 to record up to one "
                "step before the last."
            ),
        },
    ),
}


def output_connected(prompt, node_id, output_index):
    """Whether something consumes this node's output at ``output_index``."""
    if not prompt or node_id is None:
        return True
    for other_id, node in prompt.items():
        if str(other_id) == str(node_id):
            continue
        for value in node.get("inputs", {}).values():
            if isinstance(value, list) and len(value) == 2:
                if str(value[0]) == str(node_id) and value[1] == output_index:
                    return True
    return False


def _callback(model, steps):
    if latent_preview is None:
        return None
    try:
        return latent_preview.prepare_callback(model, steps)
    except Exception as error:  # previews are cosmetic
        log.debug("[DAAM] no latent preview: %s", error)
        return None


def _prepare(model, latent):
    samples = comfy.sample.fix_empty_latent_channels(model, latent["samples"])
    return samples


def _patched(model, collector):
    """A clone of the model with the collectors' attn2 replacement installed.

    Cloning keeps the patch off the shared patcher, so a cached model used by
    other nodes in the graph never sees it.
    """
    working = model.clone() if hasattr(model, "clone") else model
    keys = install_attention_capture(working, collector)
    if not keys:
        log.warning(
            "[DAAM] no cross attention blocks found: this pack reads the attn2 "
            "blocks of a UNet, so SDXL, SD1.5 and other UNet models work, while "
            "DiT models (Flux, SD3, Qwen-Image, ...) have no such blocks."
        )
    return working, keys


class DAAMKSampler:
    """KSampler, plus attention maps from the chosen steps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising."}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {}),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "latent_image": ("LATENT", {}),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                **CAPTURE_INPUTS,
            },
            "hidden": {"prompt": "PROMPT", "node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "DAAM_HEATMAP", "DAAM_HEATMAP")
    RETURN_NAMES = ("latent", "pos_heatmaps", "neg_heatmaps")
    OUTPUT_TOOLTIPS = (
        "The denoised latent, exactly as KSampler returns it.",
        "Positive prompt attention maps (image/16 grid).",
        "Negative prompt attention maps (image/16 grid).",
    )
    FUNCTION = "sample"
    CATEGORY = "DAAM/Mixed"
    DESCRIPTION = (
        "Drop-in KSampler that records cross attention maps for the chosen "
        "window of sampling steps. Leave both map outputs unconnected and it "
        "costs what KSampler costs."
    )

    def sample(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        capture_last_steps,
        capture_end_offset,
        prompt=None,
        node_id=None,
    ):
        latent = latent_image.copy()
        latent["samples"] = _prepare(model, latent)

        collect_pos = output_connected(prompt, node_id, 1)
        collect_neg = output_connected(prompt, node_id, 2)
        _, _, latent_h, latent_w = latent["samples"].shape

        collector = AttentionCollector(
            latent_h,
            latent_w,
            collect_pos=collect_pos,
            collect_neg=collect_neg,
            capture_last_steps=capture_last_steps,
            capture_end_offset=capture_end_offset,
        )

        working, keys = (model, [])
        if collect_pos or collect_neg:
            working, keys = _patched(model, collector)

        try:
            noise = comfy.sample.prepare_noise(
                latent["samples"], seed, latent.get("batch_index")
            )  # noise_inds: per image seeds of a batch, as KSampler passes it
            samples = comfy.sample.sample(
                working,
                noise,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive,
                negative,
                latent["samples"],
                denoise=denoise,
                noise_mask=latent.get("noise_mask"),
                callback=_callback(working, steps),
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=seed,
            )
        finally:
            if keys:
                remove_attention_capture(working, keys)

        out = latent.copy()
        out["samples"] = samples

        if collector.captured_steps:
            log.info(collector.summary())
        return (
            out,
            collector.heatmaps("pos") if collect_pos else None,
            collector.heatmaps("neg") if collect_neg else None,
        )


class DAAMSamplerCustom:
    """SamplerCustom, plus attention maps from the chosen steps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {}),
                "add_noise": ("BOOLEAN", {"default": True}),
                "noise_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True},
                ),
                "cfg": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01},
                ),
                "positive": ("CONDITIONING", {}),
                "negative": ("CONDITIONING", {}),
                "sampler": ("SAMPLER", {}),
                "sigmas": ("SIGMAS", {}),
                "latent_image": ("LATENT", {}),
                **CAPTURE_INPUTS,
            },
            "hidden": {"prompt": "PROMPT", "node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("LATENT", "LATENT", "DAAM_HEATMAP", "DAAM_HEATMAP")
    RETURN_NAMES = ("output", "denoised_output", "pos_heatmaps", "neg_heatmaps")
    FUNCTION = "sample"
    CATEGORY = "DAAM/Mixed"
    DESCRIPTION = (
        "Drop-in SamplerCustom that records cross attention maps for the chosen "
        "window of sampling steps."
    )

    def sample(
        self,
        model,
        add_noise,
        noise_seed,
        cfg,
        positive,
        negative,
        sampler,
        sigmas,
        latent_image,
        capture_last_steps,
        capture_end_offset,
        prompt=None,
        node_id=None,
    ):
        from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise

        latent = latent_image.copy()
        latent["samples"] = _prepare(model, latent)

        collect_pos = output_connected(prompt, node_id, 2)
        collect_neg = output_connected(prompt, node_id, 3)
        _, _, latent_h, latent_w = latent["samples"].shape

        collector = AttentionCollector(
            latent_h,
            latent_w,
            collect_pos=collect_pos,
            collect_neg=collect_neg,
            capture_last_steps=capture_last_steps,
            capture_end_offset=capture_end_offset,
        )

        working, keys = (model, [])
        if collect_pos or collect_neg:
            working, keys = _patched(model, collector)

        noise_source = Noise_RandomNoise(noise_seed) if add_noise else Noise_EmptyNoise()
        noise = noise_source.generate_noise(latent)

        x0_output = {}
        callback = None
        if latent_preview is not None:
            try:
                callback = latent_preview.prepare_callback(
                    working, sigmas.shape[-1] - 1, x0_output
                )
            except Exception:
                callback = None

        try:
            samples = comfy.sample.sample_custom(
                working,
                noise,
                cfg,
                sampler,
                sigmas,
                positive,
                negative,
                latent["samples"],
                noise_mask=latent.get("noise_mask"),
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=noise_seed,
            )
        finally:
            if keys:
                remove_attention_capture(working, keys)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            denoised = latent.copy()
            denoised["samples"] = working.model.process_latent_out(x0_output["x0"].cpu())
        else:
            denoised = out

        if collector.captured_steps:
            log.info(collector.summary())
        return (
            out,
            denoised,
            collector.heatmaps("pos") if collect_pos else None,
            collector.heatmaps("neg") if collect_neg else None,
        )
