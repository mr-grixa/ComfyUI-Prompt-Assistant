"""Cross-attention capture for DAAM style attribution maps.

Attention probabilities only exist while the sampler runs, so every ``attn2``
block of the UNet is replaced for the duration of one sampling pass. ComfyUI
calls a replacement with the already projected q, k, v and with
``extra_options`` - the transformer options of the current UNet call. Those
carry

* ``sigmas``        - the sigma of the call being made,
* ``sample_sigmas`` - the whole schedule of the sampling pass,
* ``cond_or_uncond``- which entries of the batch are conditional,
* ``block``         - ("input" | "middle" | "output", block index).

Comparing the current sigma with the schedule gives the step index without
counting calls, so a multi-evaluation sampler (dpmpp_2s, ...) cannot drift and
the capture window can be limited to, for example, the last step only. This is
the same trick ComfyUI itself uses for context windows.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

try:  # the fused kernels; only used to pass a step through untouched
    from comfy.ldm.modules.attention import optimized_attention
except Exception:  # pragma: no cover - the node still works without it
    optimized_attention = None

log = logging.getLogger("daam_mixed")


def raw_tensor(value):
    """The plain tensor inside a ComfyUI attention container, if there is one.

    Newer ComfyUI wraps q/k/v in ``AttentionTensorContainer`` and treats them
    as single-owner: reading ``.tensor`` leaves the ownership untouched, while
    ``.take()`` would hand the tensor over and break the backend call.
    """
    inner = getattr(value, "tensor", None)
    return inner if isinstance(inner, torch.Tensor) else value


def attention_with_probs(q, k, v, heads, mask=None):
    """Plain scaled dot product attention, returning (output, probabilities)."""
    batch, q_len, inner = q.shape
    dim_head = inner // heads

    def split(tensor):
        return tensor.view(tensor.shape[0], tensor.shape[1], heads, dim_head).transpose(1, 2)

    qh, kh, vh = split(q), split(k), split(v)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(dim_head)
    if mask is not None:
        scores = scores + mask
    probs = scores.softmax(dim=-1)
    out = torch.matmul(probs, vh).transpose(1, 2).reshape(batch, q_len, inner)
    return out, probs


def default_attention(q, k, v, heads, extra_options=None):
    """Whatever ComfyUI would have used, for the steps we are not recording.

    The containers are handed back to the fused kernel as they are, so a step
    outside the capture window costs exactly what it costs without this pack.
    """
    if optimized_attention is not None:
        try:
            if extra_options is not None:
                return optimized_attention(q, k, v, heads, transformer_options=extra_options)
            return optimized_attention(q, k, v, heads)
        except TypeError:  # older/newer signature: fall through to raw math
            pass
    return attention_with_probs(raw_tensor(q), raw_tensor(k), raw_tensor(v), heads)[0]


def locate_attn2_blocks(diffusion_model):
    """Every (block group, block index, transformer index) that has an attn2.

    The keys are the ones ComfyUI's ``set_model_patch_replace`` expects and the
    ones the UNet looks up while running: ("input" | "middle" | "output", block
    index, index of the transformer block inside that block's SpatialTransformer).
    Looking for the ``transformer_blocks`` attribute rather than for a class
    name keeps this working across model families that rename their modules.
    """
    tags = []
    groups = (
        ("input", getattr(diffusion_model, "input_blocks", ())),
        ("middle", [diffusion_model.middle_block] if getattr(diffusion_model, "middle_block", None) else ()),
        ("output", getattr(diffusion_model, "output_blocks", ())),
    )

    for name, blocks in groups:
        for index, block in enumerate(blocks):
            for module in [block, *block.modules()]:
                transformers = getattr(module, "transformer_blocks", None)
                if not transformers:
                    continue
                for transformer_index, transformer in enumerate(transformers):
                    if getattr(transformer, "attn2", None) is not None:
                        tags.append((name, index, transformer_index))
                break
    return tags


def install_attention_capture(model_patcher, collector):
    """Patch every attn2 block of one model patcher. Returns the patched keys."""
    diffusion_model = model_patcher.model.diffusion_model
    keys = locate_attn2_blocks(diffusion_model)
    for name, index, transformer_index in keys:
        model_patcher.set_model_patch_replace(
            collector.attn2, "attn2", name, index, transformer_index
        )
    return keys


def remove_attention_capture(model_patcher, keys):
    """Take the replacements back out of the model options.

    ``set_model_patch_replace(None, ...)`` only stores None under the key, and
    ComfyUI calls whatever it finds there - so the keys are deleted from a copy
    of the options instead of being overwritten.
    """
    options = getattr(model_patcher, "model_options", None)
    transformer_options = options.get("transformer_options") if isinstance(options, dict) else None
    patches_replace = transformer_options.get("patches_replace") if transformer_options else None

    if not isinstance(patches_replace, dict) or "attn2" not in patches_replace:
        for name, index, transformer_index in keys:
            model_patcher.set_model_patch_replace(None, "attn2", name, index, transformer_index)
        return

    attn2_patches = dict(patches_replace["attn2"])
    for name, index, transformer_index in keys:
        attn2_patches.pop((name, index, transformer_index), None)

    new_options = dict(options)
    new_options["transformer_options"] = dict(transformer_options)
    new_options["transformer_options"]["patches_replace"] = dict(patches_replace)
    new_options["transformer_options"]["patches_replace"]["attn2"] = attn2_patches
    model_patcher.model_options = new_options


class AttentionCollector:
    """Collects per token cross attention maps during a sampling pass.

    ``capture_last_steps`` and ``capture_end_offset`` describe the window in
    sampling steps: with 20 steps, ``capture_last_steps=1`` records the final
    step and ``capture_last_steps=0`` records all of them. ``capture_end_offset``
    shifts the window back from the end, e.g. 1 stops one step before the last.
    Both are resolved against the real schedule, so ``denoise`` and sliced
    sigmas behave the same way.
    """

    def __init__(
        self,
        latent_height,
        latent_width,
        collect_pos=True,
        collect_neg=True,
        capture_last_steps=1,
        capture_end_offset=0,
    ):
        self.latent_height = max(1, int(latent_height))
        self.latent_width = max(1, int(latent_width))
        self.enabled = {"pos": bool(collect_pos), "neg": bool(collect_neg)}
        self.capture_last_steps = max(0, int(capture_last_steps))
        self.capture_end_offset = max(0, int(capture_end_offset))

        # side -> (h, w) -> token count -> running sum (tokens, h, w)
        self.sums = {"pos": {}, "neg": {}}
        self.counts = {"pos": {}, "neg": {}}

        self.step = None
        self.steps_total = None
        self.captured_steps = set()
        self.calls = 0

    # ---------------------------------------------------------------- window
    def _schedule(self, extra_options):
        sigmas = extra_options.get("sample_sigmas") if extra_options else None
        current = extra_options.get("sigmas") if extra_options else None
        return sigmas, current

    def step_of(self, extra_options):
        """Sampling step index of this UNet call, or None if it cannot be told."""
        sigmas, current = self._schedule(extra_options)
        if not torch.is_tensor(sigmas) or not torch.is_tensor(current):
            return None
        if sigmas.numel() < 2:
            return None
        try:
            value = current.flatten()[0].to(device=sigmas.device, dtype=sigmas.dtype)
            distance = (sigmas - value).abs()
            return int(distance.argmin().item())
        except Exception:  # non numeric sigma, e.g. a list of timesteps
            return None

    def window(self, steps_total):
        """(first, last) step index to record, inclusive."""
        last = steps_total - 1 - self.capture_end_offset
        if self.capture_last_steps:
            first = steps_total - self.capture_last_steps - self.capture_end_offset
        else:
            first = 0
        first = max(0, first)
        last = min(steps_total - 1, last)
        return first, last

    def is_capturing(self, extra_options):
        step = self.step_of(extra_options)
        self.step = step
        if step is None:
            # No schedule to compare against: record everything rather than
            # silently returning empty maps.
            return True
        sigmas, _ = self._schedule(extra_options)
        steps_total = max(1, int(sigmas.numel()) - 1)
        self.steps_total = steps_total
        first, last = self.window(steps_total)
        if first <= step <= last:
            self.captured_steps.add(step)
            return True
        return False

    # ------------------------------------------------------------- attention
    def attn2(self, q, k, v, extra_options=None, mask=None):
        """The attn2 replacement ComfyUI calls for every cross attention block."""
        extra_options = extra_options or {}
        self.calls += 1

        try:
            capturing = self.enabled["pos"] or self.enabled["neg"]
            capturing = capturing and self.is_capturing(extra_options)
        except Exception as error:  # never break sampling over bookkeeping
            log.warning("[DAAM] step detection failed: %s", error)
            capturing = False

        if not capturing:
            return default_attention(q, k, v, extra_options.get("n_heads", 8), extra_options)

        heads = int(extra_options.get("n_heads", 8))
        out, probs = attention_with_probs(raw_tensor(q), raw_tensor(k), raw_tensor(v), heads, mask)

        try:
            with torch.no_grad():
                self._accumulate(probs.detach(), extra_options)
        except Exception as error:  # a broken map must not break the render
            log.warning("[DAAM] attention map skipped: %s", error)

        return out

    def _accumulate(self, probs, extra_options):
        batch, _, q_len, tokens = probs.shape
        grid = self._grid_for(q_len)
        if grid is None:
            return

        cond_or_uncond = extra_options.get("cond_or_uncond") or [0]
        groups = max(1, len(cond_or_uncond))
        group_size = max(1, batch // groups)

        for index in range(batch):
            side = "neg" if cond_or_uncond[min(index // group_size, groups - 1)] == 1 else "pos"
            if not self.enabled[side]:
                continue
            latent_index = index % group_size

            # (heads, q_len, tokens) -> (tokens, h, w), averaged over heads
            maps = probs[index].permute(2, 0, 1).reshape(tokens, -1, *grid).mean(1)

            by_grid = self.sums[side].setdefault(latent_index, {})
            store = by_grid.setdefault(grid, {})
            if tokens in store:
                store[tokens] += maps
            else:
                store[tokens] = maps.detach().clone()

            counts = self.counts[side].setdefault(latent_index, {}).setdefault(grid, {})
            counts[tokens] = counts.get(tokens, 0) + 1

    def _grid_for(self, q_len):
        """The (h, w) of the latent grid this attention call works on."""
        area = self.latent_height * self.latent_width
        if area <= 0:
            return None
        ratio = math.sqrt(q_len / area)
        height = int(round(self.latent_height * ratio))
        width = int(round(self.latent_width * ratio))
        if height <= 0 or width <= 0 or height * width != q_len:
            return None
        return height, width

    # ----------------------------------------------------------------- result
    def heatmaps(self, side):
        """{latent index: {token count: (tokens, h, w)}} at the finest grid."""
        if not self.enabled[side] or not self.sums[side]:
            return {}

        result = {}
        for latent_index, by_grid in self.sums[side].items():
            target = max(by_grid, key=lambda grid: grid[0] * grid[1])
            counts = self.counts[side][latent_index]
            merged = {}
            for grid, by_tokens in by_grid.items():
                for tokens, total in by_tokens.items():
                    averaged = (total / counts[grid][tokens]).float()
                    if grid != target:
                        averaged = F.interpolate(
                            averaged[None], size=target, mode="bicubic", align_corners=False
                        )[0]
                    merged.setdefault(tokens, []).append(averaged)
            result[latent_index] = {
                tokens: torch.stack(maps).mean(0).cpu() for tokens, maps in merged.items()
            }
        return result

    def summary(self):
        """A short line for the console, so a run says what it recorded."""
        if not self.captured_steps:
            return "[DAAM] no step matched the capture window"
        steps = sorted(self.captured_steps)
        return (
            f"[DAAM] captured {len(steps)} step(s) of "
            f"{self.steps_total or '?'}: {', '.join(str(s) for s in steps)}"
        )
