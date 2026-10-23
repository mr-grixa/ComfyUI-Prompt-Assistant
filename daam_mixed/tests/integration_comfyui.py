"""Integration check against a real ComfyUI checkout - no model, no GPU.

This one is optional and not part of the normal suite: it imports ComfyUI's own
``comfy.ldm.modules.attention``, builds a real ``BasicTransformerBlock``, patches
it through ComfyUI's real ``set_model_options_patch_replace`` and runs a forward
pass, so the whole contract our replacement depends on is the real one:

* the ``set_model_patch_replace`` keys reach the block,
* the block calls the replacement as ``patch(q, k, v, extra_options)``,
* ``extra_options`` carries ``sigmas``, ``sample_sigmas`` and ``cond_or_uncond``,
* the maps come out aligned with the token axis.

Usage, with a ComfyUI checkout and its dependencies importable:

    COMFYUI=/path/to/ComfyUI python daam_mixed/tests/integration_comfyui.py

On a machine without a GPU, force CPU mode first:
    COMFYUI_CPU=1 COMFYUI=/path/to/ComfyUI python daam_mixed/tests/integration_comfyui.py
"""

from __future__ import annotations

import os
import sys

import torch

COMFY = os.environ.get("COMFYUI")
if not COMFY:
    print("set COMFYUI=/path/to/ComfyUI to run this check")
    raise SystemExit(2)

sys.path.insert(0, COMFY)

if os.environ.get("COMFYUI_CPU"):
    import comfy.options

    comfy.options.enable_args_parsing()
    import comfy.cli_args

    comfy.cli_args.args.cpu = True

from comfy.ldm.modules.attention import BasicTransformerBlock  # noqa: E402
from comfy.model_patcher import set_model_options_patch_replace  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from daam_mixed.core.capture import (  # noqa: E402
    AttentionCollector,
    install_attention_capture,
    locate_attn2_blocks,
    remove_attention_capture,
)

results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(f"[{'ok  ' if condition else 'FAIL'}] {name}{' - ' + detail if detail and not condition else ''}")


class FakeBlock(torch.nn.Module):
    """What a UNet block looks like to the locator: something with transformers."""

    def __init__(self, transformer):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([transformer])


class FakePatcher:
    """The two things the pack uses of a model patcher, on a real model_options."""

    def __init__(self, diffusion_model):
        self.model = type("Model", (), {"diffusion_model": diffusion_model})()
        self.model_options = {"transformer_options": {}}

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        self.model_options = set_model_options_patch_replace(
            self.model_options, patch, name, block_name, number, transformer_index
        )

    def clone(self):
        return self

    def patches(self):
        return self.patches_for("attn2")

    def patches_for(self, kind):
        replace = self.model_options["transformer_options"].setdefault("patches_replace", {})
        return replace.setdefault(kind, {})


def main():
    torch.manual_seed(0)
    block = BasicTransformerBlock(dim=64, n_heads=2, d_head=32, dropout=0.0, context_dim=64)
    # ComfyUI's Linear leaves its weights uninitialised (they come from a
    # checkpoint), and uninitialised memory reads back as NaN, so give the block
    # real numbers before running it.
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.normal_(0.0, 0.2)
    block.eval()

    diffusion = type(
        "Diffusion",
        (),
        {
            "input_blocks": torch.nn.ModuleList([FakeBlock(block)]),
            "middle_block": None,
            "output_blocks": torch.nn.ModuleList(),
        },
    )()
    patcher = FakePatcher(diffusion)

    located = locate_attn2_blocks(diffusion)
    check("real block is located", located == [("input", 0, 0)], str(located))

    collector = AttentionCollector(latent_height=8, latent_width=8, capture_last_steps=1)
    install_attention_capture(patcher, collector)
    check("real patch keys installed", ("input", 0, 0) in patcher.patches(), str(list(patcher.patches())))

    sigmas = torch.linspace(14.0, 0.0, 5)
    x = torch.randn(1, 16, 64)  # 16 = 4x4 grid, latent / 2
    context = torch.randn(1, 7, 64)  # 7 prompt tokens

    def run_step(index):
        options = {
            "patches_replace": {"attn2": dict(patcher.patches())},
            "block": ("input", 0),
            "block_index": 0,
            "sigmas": sigmas[index : index + 1],
            "sample_sigmas": sigmas,
            "cond_or_uncond": [0],
        }
        with torch.no_grad():
            return block(x, context=context, transformer_options=options)

    outputs = [run_step(index) for index in range(4)]

    check("forward pass still returns the same shape", outputs[0].shape == x.shape, str(outputs[0].shape))
    check("only the last step was recorded", collector.captured_steps == {3}, str(collector.captured_steps))

    maps = collector.heatmaps("pos")["pos"]
    check("map for batch entry 0", 0 in maps, str(list(maps)))
    heat = maps[0][7]
    check("map is (tokens, 4, 4)", tuple(heat.shape) == (7, 4, 4), str(tuple(heat.shape)))
    check(
        "attention sums to one over tokens",
        abs(float(heat.sum(0).mean()) - 1.0) < 1e-4,
        str(float(heat.sum(0).mean())),
    )

    # the same forward with and without the patch must agree on the output
    without = {}
    for key in list(patcher.patches()):
        without = set_model_options_patch_replace(patcher.model_options, None, "attn2", key[0], key[1], key[2])
    original = block(
        x,
        context=context,
        transformer_options={
            "patches_replace": {"attn2": {} if without else {}},
            "block": ("input", 0),
            "block_index": 0,
        },
    )
    difference = float((original.detach() - outputs[-1].detach()).abs().max())
    check("recorded step gives the same output as no patch", difference < 1e-4, f"max diff {difference:.2e}")

    # ---- self attention: the attn1 path, on the real block
    self_collector = AttentionCollector(
        8, 8, capture_self=True, self_level="primary", keep_per_step=True
    )
    patcher2 = FakePatcher(diffusion)
    self_keys = install_attention_capture(patcher2, self_collector)
    check(
        "attn1 keys are installed next to attn2",
        len(patcher2.patches_for("attn1")) == 1 and len(patcher2.patches_for("attn2")) == 1,
        str({kind: len(patcher2.patches_for(kind)) for kind in ("attn1", "attn2")}),
    )

    def run_attn1_step(index):
        options = {
            "patches_replace": {
                "attn2": dict(patcher2.patches_for("attn2")),
                "attn1": dict(patcher2.patches_for("attn1")),
            },
            "block": ("input", 0),
            "block_index": 0,
            "sigmas": sigmas[index : index + 1],
            "sample_sigmas": sigmas,
            "cond_or_uncond": [0],
        }
        with torch.no_grad():
            return block(x, context=context, transformer_options=options)

    outputs_self = [run_attn1_step(index) for index in range(4)]
    self_payload = self_collector.heatmaps("pos")
    affinity = (self_payload.get("self") or {}).get("primary", {}).get(0)
    check("self attention affinity was captured", affinity is not None)
    if affinity is not None:
        # x is 16 tokens on an 8x8 latent, so the finest grid is 4x4 = 16 cells
        check("affinity is (cells, cells)", tuple(affinity.shape) == (16, 16), str(tuple(affinity.shape)))
        check(
            "affinity rows are distributions",
            abs(float(affinity.sum(-1).mean()) - 1.0) < 1e-4,
            str(float(affinity.sum(-1).mean())),
        )
    no_patch = block(
        x,
        context=context,
        transformer_options={"patches_replace": {}, "block": ("input", 0), "block_index": 0},
    )
    diff_self = float((no_patch.detach() - outputs_self[-1].detach()).abs().max())
    check("the attn1 replacement computes the same output", diff_self < 1e-4, f"max diff {diff_self:.2e}")
    remove_attention_capture(patcher2, self_keys)
    check(
        "both attn1 and attn2 keys are gone",
        not patcher2.patches_for("attn1") and not patcher2.patches_for("attn2"),
        str({kind: len(patcher2.patches_for(kind)) for kind in ("attn1", "attn2")}),
    )

    remove_attention_capture(patcher, located)
    check("unpatch clears the keys", patcher.patches() == {}, str(patcher.patches()))
    # a patch left behind as None would still be called, and crash
    check(
        "no key is left behind",
        all(patch is not None for patch in patcher.patches().values()),
        str(patcher.patches()),
    )

    print()
    print(f"{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
