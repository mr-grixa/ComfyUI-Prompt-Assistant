"""Cross and self attention capture for DAAM style attribution maps.

Attention probabilities only exist while the sampler runs, so the ``attn2``
(cross) and ``attn1`` (self) blocks of the UNet are replaced for the duration of
one sampling pass. ComfyUI calls a replacement with the already projected q, k,
v and with ``extra_options`` - the transformer options of the current UNet call.
Those carry

* ``sigmas``        - the sigma of the call being made,
* ``sample_sigmas`` - the whole schedule of the sampling pass,
* ``cond_or_uncond``- which entries of the batch are conditional,
* ``block``         - ("input" | "middle" | "output", block index).

Comparing the current sigma with the schedule gives the step index without
counting calls, so a multi-evaluation sampler (dpmpp_2s, ...) cannot drift and
the capture window can be limited to, for example, the last step only. This is
the same trick ComfyUI itself uses for context windows.

What is kept, and why:

* cross attention maps, per token, at every grid size the UNet uses, summed in
  place on the device and merged to the finest grid on the way out;
* the same maps per sampling step as well, when asked for: 77 tokens on a 64x64
  grid is 1.3 MB per step, which is what allows SNR weighting and inspecting a
  single step without a re-run;
* self attention affinity, at one or two chosen resolutions, heavily reduced
  (heads aggregated, blocks averaged) because the raw matrix is (hw, hw) and
  costs 67 MB per block at 64x64. This is the part that lets a mask be refined
  along the model's own notion of "same object", past the 16 pixel cell of the
  cross attention grid.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

from .heads import aggregate_heads, apply_value_norm

try:  # the fused kernels; only used to pass a step through untouched
    from comfy.ldm.modules.attention import optimized_attention
except Exception:  # pragma: no cover - the node still works without it
    optimized_attention = None

log = logging.getLogger("daam_mixed")

# Self attention is captured chunk by chunk over the queries: a whole
# (heads, hw, hw) probability tensor at 64x64 is 268 MB, a chunk is 1/16 of it.
SELF_CHUNK = 256
# Cells that a per step self attention map may take before it is refused:
# 32x32 fits (4 MB), 64x64 does not (67 MB).
MAX_SELF_CELLS_PER_STEP = 2**22
# Steps beyond this are accumulated, but no longer kept individually.
MAX_STORED_STEPS = 128

SELF_LEVELS = ("off", "secondary", "primary", "combined")
CROSS_AGGREGATIONS = ("mean", "median", "max", "min_entropy")


def raw_tensor(value):
    """The plain tensor inside a ComfyUI attention container, if there is one.

    Newer ComfyUI wraps q/k/v in ``AttentionTensorContainer`` and treats them
    as single-owner: reading ``.tensor`` leaves the ownership untouched, while
    ``.take()`` would hand the tensor over and break the backend call.
    """
    inner = getattr(value, "tensor", None)
    return inner if isinstance(inner, torch.Tensor) else value


def split_heads(tensor, heads):
    batch, length, inner = tensor.shape
    return tensor.view(batch, length, heads, inner // heads).transpose(1, 2)


def attention_with_probs(q, k, v, heads, mask=None):
    """Plain scaled dot product attention, returning (output, probabilities)."""
    batch, q_len, inner = q.shape
    dim_head = inner // heads
    qh, kh, vh = split_heads(q, heads), split_heads(k, heads), split_heads(v, heads)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) / math.sqrt(dim_head)
    if mask is not None:
        scores = scores + mask
    probs = scores.softmax(dim=-1)
    out = torch.matmul(probs, vh).transpose(1, 2).reshape(batch, q_len, inner)
    return out, probs


def self_attention_with_probs(q, k, v, heads, mask=None, chunk=SELF_CHUNK, on_probs=None):
    """Self attention computed query chunk by query chunk.

    Self attention is (hw, hw) per head, so the whole probability tensor is
    hundreds of megabytes at the finest grid. Chunking keeps the peak at a
    fraction of that and lets ``on_probs`` fold the chunk away immediately
    instead of keeping it.
    """
    batch, q_len, inner = q.shape
    heads_dim = inner // heads
    qh, kh, vh = split_heads(q, heads), split_heads(k, heads), split_heads(v, heads)
    scale = 1.0 / math.sqrt(heads_dim)
    out = torch.empty(batch, q_len, inner, dtype=q.dtype, device=q.device)

    for start in range(0, q_len, chunk):
        stop = min(start + chunk, q_len)
        scores = torch.matmul(qh[:, :, start:stop], kh.transpose(-2, -1)) * scale
        if mask is not None:
            scores = scores + mask
        probs = scores.softmax(dim=-1)
        out[:, start:stop] = torch.matmul(probs, vh).transpose(1, 2).reshape(
            batch, stop - start, inner
        )
        if on_probs is not None:
            on_probs(probs, start)
        del probs, scores
    return out


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


def locate_attention_blocks(diffusion_model):
    """Every (group, block index, transformer index) that has an attn2.

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


# kept for the older name used by the docs and by callers of the first version
locate_attn2_blocks = locate_attention_blocks


def install_attention_capture(model_patcher, collector):
    """Patch the attention blocks of one model patcher. Returns the keys.

    Cross attention is patched on every block that has one; self attention only
    when the collector wants it, so an unused capture costs nothing.
    """
    diffusion_model = model_patcher.model.diffusion_model
    keys = []
    for name, index, transformer_index in locate_attention_blocks(diffusion_model):
        model_patcher.set_model_patch_replace(collector.attn2, "attn2", name, index, transformer_index)
        keys.append((name, index, transformer_index, "attn2"))
        if collector.capture_self:
            model_patcher.set_model_patch_replace(
                collector.attn1, "attn1", name, index, transformer_index
            )
            keys.append((name, index, transformer_index, "attn1"))
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

    if not isinstance(patches_replace, dict):
        for key in keys:
            name, index, transformer_index = key[:3]
            kind = key[3] if len(key) > 3 else "attn2"
            model_patcher.set_model_patch_replace(None, kind, name, index, transformer_index)
        return

    new_options = dict(options)
    new_options["transformer_options"] = dict(transformer_options)
    new_replace = {kind: dict(patches) for kind, patches in patches_replace.items()}
    for key in keys:
        name, index, transformer_index = key[:3]
        kind = key[3] if len(key) > 3 else "attn2"
        for block in ((name, index, transformer_index), (name, index)):
            new_replace.get(kind, {}).pop(block, None)
    new_options["transformer_options"]["patches_replace"] = new_replace
    model_patcher.model_options = new_options


class AttentionCollector:
    """Collects attention maps during a sampling pass.

    ``capture_last_steps``/``capture_end_offset`` describe the window in
    sampling steps: with 20 steps, ``capture_last_steps=1`` records the final
    step and 0 records all of them.

    ``head_aggregation`` is mean/median/max/min_entropy (see core/heads.py),
    ``value_norm`` scales each token's row by the norm of its value vector so
    padding tokens stop polluting the map.

    ``capture_self`` switches on the self attention affinity capture and
    ``self_level`` picks the resolution: ``secondary`` is latent/4, ``primary``
    latent/2 (the finest grid a UNet attends at), ``combined`` blends the two.
    """

    def __init__(
        self,
        latent_height,
        latent_width,
        collect_pos=True,
        collect_neg=True,
        capture_last_steps=1,
        capture_end_offset=0,
        head_aggregation="mean",
        value_norm=True,
        keep_per_step=True,
        capture_self=False,
        self_level="secondary",
        self_blend=0.3,
        entropy_fraction=0.25,
    ):
        self.latent_height = max(1, int(latent_height))
        self.latent_width = max(1, int(latent_width))
        self.enabled = {"pos": bool(collect_pos), "neg": bool(collect_neg)}
        self.capture_last_steps = max(0, int(capture_last_steps))
        self.capture_end_offset = max(0, int(capture_end_offset))
        self.head_aggregation = (
            head_aggregation if head_aggregation in CROSS_AGGREGATIONS else "mean"
        )
        self.value_norm = bool(value_norm)
        self.keep_per_step = bool(keep_per_step)
        self.self_level = self_level if self_level in SELF_LEVELS else "off"
        self.capture_self = bool(capture_self) and self.self_level != "off"
        self.self_blend = float(self_blend)
        self.entropy_fraction = float(entropy_fraction)

        # cross attention: side -> batch -> (h, w) -> tokens -> running sum
        self.sums = {"pos": {}, "neg": {}}
        self.counts = {"pos": {}, "neg": {}}
        # the same per sampling step: side -> (side, step) -> batch -> grid
        self.step_sums = {"pos": {}, "neg": {}}
        self.step_counts = {"pos": {}, "neg": {}}

        # self attention: (level, grid) -> batch -> running sum of (hw, hw)
        self.self_sums = {}
        self.self_counts = {}
        self.step_self_sums = {}  # step -> (level, grid) -> batch -> sum
        self.step_self_counts = {}

        self.sigmas = None
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
            return int((sigmas - value).abs().argmin().item())
        except Exception:  # non numeric sigma, e.g. a list of timesteps
            return None

    def window(self, steps_total):
        """(first, last) step index to record, inclusive."""
        last = steps_total - 1 - self.capture_end_offset
        if self.capture_last_steps:
            first = steps_total - self.capture_last_steps - self.capture_end_offset
        else:
            first = 0
        return max(0, first), min(steps_total - 1, last)

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
        if self.sigmas is None:
            self.sigmas = [float(value) for value in sigmas.detach().float().cpu().flatten()]
        first, last = self.window(steps_total)
        if first <= step <= last:
            self.captured_steps.add(step)
            return True
        return False

    # -------------------------------------------------------- cross attention
    def attn2(self, q, k, v, extra_options=None, mask=None):
        """The attn2 replacement ComfyUI calls for every cross attention block."""
        extra_options = extra_options or {}
        self.calls += 1

        try:
            capturing = (self.enabled["pos"] or self.enabled["neg"]) and self.is_capturing(extra_options)
        except Exception as error:  # never break sampling over bookkeeping
            log.warning("[DAAM] step detection failed: %s", error)
            capturing = False

        if not capturing:
            return default_attention(q, k, v, extra_options.get("n_heads", 8), extra_options)

        heads = int(extra_options.get("n_heads", 8))
        raw_q, raw_k, raw_v = raw_tensor(q), raw_tensor(k), raw_tensor(v)
        out, probs = attention_with_probs(raw_q, raw_k, raw_v, heads, mask)

        try:
            with torch.no_grad():
                merged = aggregate_heads(probs.detach(), self.head_aggregation, self.entropy_fraction)
                if self.value_norm:
                    merged = apply_value_norm(merged, raw_v.detach())
                self._accumulate(merged, extra_options)
        except Exception as error:  # a broken map must not break the render
            log.warning("[DAAM] attention map skipped: %s", error)

        return out

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

    def _accumulate(self, merged, extra_options):
        """merged: (batch, queries, tokens), already aggregated over heads."""
        batch, q_len, tokens = merged.shape
        grid = self._grid_for(q_len)
        if grid is None:
            return

        cond_or_uncond = extra_options.get("cond_or_uncond") or [0]
        groups = max(1, len(cond_or_uncond))
        group_size = max(1, batch // groups)
        step = self.step

        for index in range(batch):
            side = "neg" if cond_or_uncond[min(index // group_size, groups - 1)] == 1 else "pos"
            if not self.enabled[side]:
                continue
            latent_index = index % group_size
            maps = merged[index].permute(1, 0).reshape(tokens, *grid)

            self._accumulate_into(
                self.sums, self.counts, side, latent_index, grid, tokens, maps
            )
            if self.keep_per_step and step is not None and step < MAX_STORED_STEPS:
                self._accumulate_into(
                    self.step_sums[side],
                    self.step_counts[side],
                    step,
                    latent_index,
                    grid,
                    tokens,
                    maps,
                )

    @staticmethod
    def _accumulate_into(sums, counts, key, latent_index, grid, tokens, maps):
        by_grid = sums.setdefault(key, {}).setdefault(latent_index, {})
        store = by_grid.setdefault(grid, {})
        if tokens in store:
            store[tokens] += maps
        else:
            store[tokens] = maps.detach().clone()
        counter = counts.setdefault(key, {}).setdefault(latent_index, {}).setdefault(grid, {})
        counter[tokens] = counter.get(tokens, 0) + 1

    # --------------------------------------------------------- self attention
    def _self_targets(self):
        """(primary grid, secondary grid) for the latent of this run."""
        primary = (max(1, self.latent_height // 2), max(1, self.latent_width // 2))
        secondary = (max(1, self.latent_height // 4), max(1, self.latent_width // 4))
        return primary, secondary

    def _wanted_self_grids(self):
        primary, secondary = self._self_targets()
        if self.self_level == "primary":
            return {primary: "primary"}
        if self.self_level == "combined":
            return {primary: "primary", secondary: "secondary"}
        return {secondary: "secondary"}

    def attn1(self, q, k, v, extra_options=None, mask=None):
        """The attn1 replacement: the usual attention, plus the affinity matrix."""
        extra_options = extra_options or {}
        heads = int(extra_options.get("n_heads", 8))
        raw_q, raw_k, raw_v = raw_tensor(q), raw_tensor(k), raw_tensor(v)

        grid = self._grid_for(raw_q.shape[1]) if raw_q.shape[1] == raw_k.shape[1] else None
        wanted = self._wanted_self_grids() if self.capture_self else {}
        if grid not in wanted or not self.is_capturing(extra_options):
            return default_attention(q, k, v, heads, extra_options)

        level = wanted[grid]

        def collect(probs, start):
            merged = aggregate_heads(
                probs.detach(), self._self_aggregation(), self.entropy_fraction
            )
            self._accumulate_self(merged, extra_options, level, grid)

        with torch.no_grad():
            return self_attention_with_probs(
                raw_q, raw_k, raw_v, heads, mask, on_probs=collect
            )

    def _self_aggregation(self):
        """max keeps the sharpest head; min_entropy picks the focused quarter."""
        if self.head_aggregation in ("max", "min_entropy"):
            return self.head_aggregation
        return "mean"

    def _accumulate_self(self, merged, extra_options, level, grid):
        batch = merged.shape[0]
        cells = merged.shape[1] * merged.shape[2]
        cond_or_uncond = extra_options.get("cond_or_uncond") or [0]
        groups = max(1, len(cond_or_uncond))
        group_size = max(1, batch // groups)
        step = self.step

        # Only the conditional branch has a meaningful self attention map, and
        # per step matrices are only kept where they are affordable.
        per_step = (
            self.keep_per_step
            and step is not None
            and cells <= MAX_SELF_CELLS_PER_STEP
            and level == "secondary"
        )

        for index in range(batch):
            if cond_or_uncond[min(index // group_size, groups - 1)] == 1:
                continue
            block = merged[index]
            key = (level, grid)
            self._running_add(self.self_sums, self.self_counts, key, index, block)
            if per_step:
                self._running_add(
                    self.step_self_sums.setdefault(step, {}),
                    self.step_self_counts.setdefault(step, {}),
                    key,
                    index,
                    block,
                )

    @staticmethod
    def _running_add(sums, counts, key, index, block):
        by_batch = sums.setdefault(key, {})
        if index in by_batch:
            by_batch[index] += block
        else:
            by_batch[index] = block.detach().clone()
        counter = counts.setdefault(key, {})
        counter[index] = counter.get(index, 0) + 1

    # ----------------------------------------------------------------- result
    @staticmethod
    def _merge_grids(by_grid, counts, target):
        merged = {}
        for grid, by_tokens in by_grid.items():
            for tokens, total in by_tokens.items():
                averaged = (total / counts[grid][tokens]).float()
                if grid != target:
                    averaged = F.interpolate(
                        averaged[None], size=target, mode="bicubic", align_corners=False
                    )[0]
                merged.setdefault(tokens, []).append(averaged)
        return {tokens: torch.stack(maps).mean(0).cpu() for tokens, maps in merged.items()}

    def heatmaps(self, side):
        """Attention maps for one side, as a dict ready to hand to a node.

        ``{side: {batch: {tokens: (tokens, h, w)}}}`` at the finest grid seen,
        plus ``steps``, ``sigmas`` and, when self attention was captured,
        ``self``.
        """
        result = {"side": side}
        if self.enabled[side] and self.sums.get(side):
            result[side] = {
                latent_index: self._merge_grids(
                    by_grid, self.counts[side][latent_index], self._finest(by_grid)
                )
                for latent_index, by_grid in self.sums[side].items()
            }
        else:
            result[side] = {}

        if self.keep_per_step and self.step_sums.get(side):
            result["steps"] = {
                step: {
                    latent_index: self._merge_grids(
                        by_grid, self.step_counts[side][step][latent_index], self._finest(by_grid)
                    )
                    for latent_index, by_grid in by_batch.items()
                }
                for step, by_batch in sorted(self.step_sums[side].items())
            }
        if self.sigmas:
            result["sigmas"] = {"schedule": list(self.sigmas)}
        if self.self_sums:
            result["self"] = self.self_heatmaps()
        return result

    def self_heatmaps(self):
        """{level: {batch: (hw, hw)}}, the grids, and the per step matrices."""
        if not self.self_sums:
            return {}

        result = {"grids": {}, "steps": {}}
        for (level, grid), by_batch in self.self_sums.items():
            counts = self.self_counts.get((level, grid), {})
            result.setdefault(level, {}).update(
                {index: (total / max(1, counts.get(index, 1))).float().cpu()
                 for index, total in by_batch.items()}
            )
            result["grids"][level] = grid

        primary, secondary = self._self_targets()
        if result.get("primary") and result.get("secondary"):
            blended = {}
            for index, primary_map in result["primary"].items():
                secondary_map = result["secondary"].get(index)
                if secondary_map is None:
                    blended[index] = primary_map
                    continue
                upsampled = F.interpolate(
                    secondary_map[None, None], size=primary, mode="bicubic", align_corners=False
                )[0, 0]
                blended[index] = self.self_blend * primary_map + (1 - self.self_blend) * upsampled
            result["combined"] = blended
            result["grids"]["combined"] = primary
            result["blend"] = self.self_blend

        for step, by_key in self.step_self_sums.items():
            counts_by_key = self.step_self_counts.get(step, {})
            for (level, grid), by_batch in by_key.items():
                counts = counts_by_key.get((level, grid), {})
                target = result["steps"].setdefault(step, {})
                target.setdefault(level, {}).update(
                    {index: (total / max(1, counts.get(index, 1))).float().cpu()
                     for index, total in by_batch.items()}
                )
                target.setdefault("grids", {})[level] = grid
        result["steps"] = dict(sorted(result["steps"].items()))
        return result

    @staticmethod
    def _finest(by_grid):
        return max(by_grid, key=lambda grid: grid[0] * grid[1])

    def summary(self):
        """A short line for the console, so a run says what it recorded."""
        if not self.captured_steps:
            return "[DAAM] no step matched the capture window"
        steps = sorted(self.captured_steps)
        parts = [
            f"[DAAM] captured {len(steps)} step(s) of {self.steps_total or '?'}: "
            f"{', '.join(str(s) for s in steps)}",
            f"heads={self.head_aggregation}",
        ]
        if self.value_norm:
            parts.append("value_norm")
        if self.self_sums:
            levels = sorted({level for level, _ in self.self_sums})
            parts.append(f"self-attention={','.join(levels)}")
        return " | ".join(parts)
