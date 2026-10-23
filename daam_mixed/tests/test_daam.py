"""Headless tests for the DAAM-Mixed pack.

Run with the repo's venv:  .venv/bin/python daam_mixed/tests/test_daam.py

Nothing here needs a GPU, a checkpoint or a ComfyUI checkout: the few ComfyUI
APIs the pack touches are stubbed in ``tests/stubs``, and the sampling schedule
is driven by hand so the step window can be checked exactly.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "stubs"))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from comfy.ldm.modules import attention as stub_attention  # noqa: E402

from daam_mixed.core import refine  # noqa: E402
from daam_mixed.core.capture import (  # noqa: E402
    AttentionCollector,
    install_attention_capture,
    remove_attention_capture,
)
from daam_mixed.core.tags import split_tags, tokenize_prompt, flatten, stream_key  # noqa: E402
from daam_mixed.nodes.mask import DAAMMasks  # noqa: E402

OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

HEAD_RED, HEAD_BLUE = None, None
results = []


def check(name, condition, detail=""):
    results.append((name, bool(condition), detail))
    status = "ok  " if condition else "FAIL"
    print(f"[{status}] {name}{(' - ' + detail) if detail and not condition else ''}")
    return bool(condition)


# --------------------------------------------------------------------- fakes


class FakeAttention:
    pass


class FakeTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn2 = FakeAttention()


class FakeBlock(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([FakeTransformer() for _ in range(count)])


class FakeDiffusionModel(torch.nn.Module):
    def __init__(self, depth=2):
        super().__init__()
        self.input_blocks = torch.nn.ModuleList([FakeBlock(depth) for _ in range(2)])
        self.middle_block = FakeBlock(depth)
        self.output_blocks = torch.nn.ModuleList([FakeBlock(depth) for _ in range(2)])


class FakePatcher:
    """model_options with the same layout and semantics as the real patcher.

    set_model_patch_replace stores None under the key instead of deleting it,
    which is what ComfyUI does, so the tests exercise our own removal.
    """

    def __init__(self, diffusion_model=None):
        self.model = types.SimpleNamespace(diffusion_model=diffusion_model or FakeDiffusionModel())
        self.model_options = {"transformer_options": {"patches_replace": {"attn2": {}}}}

    @property
    def patches(self):
        return self.patches_for("attn2")

    def patches_for(self, kind):
        replace = self.model_options["transformer_options"]["patches_replace"]
        return replace.setdefault(kind, {})

    def set_model_patch_replace(self, patch, name, block_name, number, transformer_index=None):
        if patch is not None:
            self.patches_for(name)[(block_name, number, transformer_index)] = patch

    def clone(self):
        return FakePatcher(self.model.diffusion_model)


class FakeSpatialTransformer(torch.nn.Module):
    """The nesting real SDXL uses: block -> SpatialTransformer -> transformers."""

    def __init__(self, count):
        super().__init__()
        self.transformer_blocks = torch.nn.ModuleList([FakeTransformer() for _ in range(count)])


class FakeNestedBlock(torch.nn.Module):
    def __init__(self, count):
        super().__init__()
        self.attn = FakeSpatialTransformer(count)


def one_hot_vectors(dim=8, count=5):
    matrix = torch.zeros(count, dim)
    for index in range(count):
        matrix[index, index] = 1.0
    return matrix


def drive_step(patcher, sigmas, step, latent=(32, 32), batch=1, heads=1, dim=8):
    """Run one UNet call the way ComfyUI would: every attn2 patch, once."""
    grid_side = latent[0] // 2  # level 1 cross attention is latent / 2
    q_len = grid_side * grid_side
    seeds = one_hot_vectors(dim, 5)

    q = torch.zeros(batch, q_len, heads * dim)
    k = torch.zeros(batch, 5, heads * dim)
    k[:, :, :dim] = seeds * 8.0
    for position in range(q_len):
        token = position % 5
        q[:, position, token * 0] = 0.0
        q[:, position, :dim] += seeds[token] * 8.0

    extra = {
        "n_heads": heads,
        "sigmas": sigmas[step : step + 1],
        "sample_sigmas": sigmas,
        "cond_or_uncond": [0] * batch,
        "block": ("input", 0),
    }
    for patch in list(patcher.patches.values()):
        patch(q, k, k, extra)


class FakeTokenizer:
    def __init__(self, words):
        self.inv_vocab = {index: word for index, word in enumerate(words)}

    def untokenize(self, ids):
        return [(token_id, self.inv_vocab[token_id]) for token_id in ids]


class FakeClip:
    """Enough of a CLIP to tokenize and decode a prompt."""

    def __init__(self):
        words = ["<|startoftext|>", "a", "cat", "red", "hat", ",", "girl", "park", "<|endoftext|>"]
        self.vocab = {word: index for index, word in enumerate(words)}
        self.tokenizer = FakeTokenizer(words)

    def tokenize(self, text):
        start = self.vocab["<|startoftext|>"]
        end = self.vocab["<|endoftext|>"]
        ids = [start]
        for word in text.replace(",", " , ").split():
            ids.append(self.vocab[word])
        ids.append(end)
        ids.extend([end] * (77 - len(ids)))
        return {"l": [ids]}


# ------------------------------------------------------------------ capture


def test_capture_window():
    patcher = FakePatcher()  # 5 blocks (2+1+2), 2 transformers each = 10 attn2
    collector = AttentionCollector(32, 32, capture_last_steps=1)
    keys = install_attention_capture(patcher, collector)
    check("attn2 blocks located", len(keys) == 10, f"found {len(keys)}")

    sigmas = torch.linspace(14.0, 0.0, 21)
    stub_attention.CALLS["optimized"] = 0
    for step in range(20):
        drive_step(patcher, sigmas, step)

    check("only the last step captured", collector.captured_steps == {19}, str(collector.captured_steps))
    check(
        "other steps go through the fused kernel",
        stub_attention.CALLS["optimized"] == 19 * 10,
        f"{stub_attention.CALLS['optimized']} pass-through calls",
    )

    payload = collector.heatmaps("pos")
    maps = payload["pos"]
    check("heat map for the batch entry", 0 in maps)
    check("one token count", list(maps[0]) == [5], str(list(maps[0])))
    heat = maps[0][5]
    check("map grid is latent / 2", tuple(heat.shape) == (5, 16, 16), str(tuple(heat.shape)))
    check("attention over tokens sums to one at every pixel", abs(float(heat.sum(0).mean()) - 1.0) < 1e-4,
          str(float(heat.sum(0).mean())))

    # token 0 sits on every fifth query position of the 16x16 grid
    token0 = heat[0].numpy()
    expected = (np.arange(256) % 5 == 0).reshape(16, 16)
    check("token lands on its own pixels", np.abs(token0 - expected).max() < 1e-4, f"max diff {np.abs(token0 - expected).max():.4f}")

    check("neg side was not collected", collector.heatmaps("neg")["neg"] == {})
    remove_attention_capture(patcher, keys)
    check("unpatch removes every block", patcher.patches == {}, f"{len(patcher.patches)} left")


def test_nested_blocks():
    """The keys must be the ones ComfyUI looks up, also with deeper nesting."""

    class Nested(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_blocks = torch.nn.ModuleList([FakeNestedBlock(2), torch.nn.Identity()])
            self.middle_block = FakeNestedBlock(2)
            self.output_blocks = torch.nn.ModuleList([FakeNestedBlock(2)])

    patcher = FakePatcher(Nested())
    keys = install_attention_capture(patcher, AttentionCollector(32, 32))
    expected = [
        ("input", 0, 0), ("input", 0, 1),
        ("middle", 0, 0), ("middle", 0, 1),
        ("output", 0, 0), ("output", 0, 1),
    ]
    found = [key[:3] for key in keys]
    check("nested blocks give the lookup keys", found == expected, str(found))
    check("blocks without attention are skipped", ("input", 1, 0) not in found)
    check("the patch keys land in model options",
          set(expected) == set(patcher.patches),
          str(sorted(patcher.patches)))


def test_capture_all_and_offset():
    sigmas = torch.linspace(14.0, 0.0, 21)

    collector = AttentionCollector(32, 32, capture_last_steps=0)
    patcher = FakePatcher()
    install_attention_capture(patcher, collector)
    for step in range(20):
        drive_step(patcher, sigmas, step)
    check("0 steps means every step", collector.captured_steps == set(range(20)), str(sorted(collector.captured_steps)))

    collector = AttentionCollector(32, 32, capture_last_steps=1, capture_end_offset=1)
    patcher = FakePatcher()
    install_attention_capture(patcher, collector)
    for step in range(20):
        drive_step(patcher, sigmas, step)
    check("end offset walks back from the last step", collector.captured_steps == {18}, str(collector.captured_steps))

    collector = AttentionCollector(32, 32, capture_last_steps=3, capture_end_offset=1)
    patcher = FakePatcher()
    install_attention_capture(patcher, collector)
    for step in range(20):
        drive_step(patcher, sigmas, step)
    check("window is the last three before the offset", collector.captured_steps == {16, 17, 18}, str(sorted(collector.captured_steps)))


def test_cond_uncond_and_grids():
    collector = AttentionCollector(32, 32, capture_last_steps=1)
    patcher = FakePatcher()
    install_attention_capture(patcher, collector)
    sigmas = torch.linspace(14.0, 0.0, 21)

    q_len = 256
    seeds = one_hot_vectors(8, 5)
    q = torch.zeros(2, q_len, 8)
    q[0, :, 0] = 8.0  # conditional row looks at token 0
    q[1, :, 1] = 8.0  # unconditional row looks at token 1
    k = torch.zeros(2, 5, 8)
    k[:, :, :8] = seeds * 8.0

    # one level 1 (16x16) and one level 2 (8x8) call, to check the merge
    for grid_side in (16, 8):
        extra = {
            "n_heads": 1,
            "sigmas": sigmas[19:20],
            "sample_sigmas": sigmas,
            "cond_or_uncond": [0, 1],
        }
        small_q = torch.zeros(2, grid_side * grid_side, 8)
        small_q[0, :, 0] = 8.0
        small_q[1, :, 1] = 8.0
        for patch in list(patcher.patches.values()):
            patch(small_q, k, k, extra)

    pos = collector.heatmaps("pos")["pos"][0][5]
    neg = collector.heatmaps("neg")["neg"][0][5]
    check("pos and neg are separated", pos is not None and neg is not None)
    check("merged grid is the finest", tuple(pos.shape) == (5, 16, 16), str(tuple(pos.shape)))
    check("positive row is the pos map", float(pos[0].mean()) > 0.5 and float(pos[1].mean()) < 0.1,
          f"{float(pos[0].mean()):.2f} / {float(pos[1].mean()):.2f}")
    check("negative row is the neg map", float(neg[1].mean()) > 0.5 and float(neg[0].mean()) < 0.1,
          f"{float(neg[1].mean()):.2f} / {float(neg[0].mean()):.2f}")


# --------------------------------------------------------------------- tags


def test_tags():
    clip = FakeClip()
    tags = split_tags(clip, "a cat, red hat")
    labels = [tag["label"] for tag in tags]
    check("tags split on the comma", labels == ["a cat", "red hat"], str(labels))
    check("tag indices are the token rows", tags[0]["indices"] == [1, 2] and tags[1]["indices"] == [4, 5],
          f"{tags[0]['indices']} {tags[1]['indices']}")

    tokens = tokenize_prompt(clip, "a cat, red hat")
    check("token count matches the padded chunk", len(flatten(tokens, stream_key(tokens))) == 77)

    broken = split_tags(clip, "a cat BREAK red hat")
    check("BREAK starts a new tag", [tag["label"] for tag in broken] == ["a cat", "red hat"],
          str([tag["label"] for tag in broken]))
    check(
        "second chunk is offset by 77 tokens",
        broken[1]["indices"] == [78, 79],
        str(broken[1]["indices"]),
    )


# ------------------------------------------------------------------- refine


def synthetic_scene(size=1024, grid=64):
    """A photo-like image with one sharply outlined object and its attention blob.

    The outline is a star, not a disc: a disc is what bicubic upsampling of an
    area averaged blob already reproduces almost perfectly, so it would hide
    the difference the guided filter makes.
    """
    ys, xs = np.mgrid[0:size, 0:size]
    centre = size / 2
    angle = np.arctan2(ys - centre, xs - centre)
    distance = np.sqrt((ys - centre) ** 2 + (xs - centre) ** 2)
    radius = size * 0.28 * (1 + 0.28 * np.sin(5 * angle) + 0.09 * np.sin(11 * angle))
    truth = distance <= radius

    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = 0.15
    image[..., 1] = 0.2
    image[..., 2] = 0.25
    image[truth] = np.array([0.95, 0.55, 0.25], dtype=np.float32)
    image[truth] *= (0.8 + 0.2 * np.abs(np.sin(xs[truth] / 2.5)))[:, None]

    cell = size // grid
    blob = np.zeros((grid, grid), dtype=np.float32)
    for y in range(grid):
        for x in range(grid):
            blob[y, x] = truth[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell].mean()
    return image, truth, refine.robust_normalize(blob)


def test_refine_precision():
    from scipy import ndimage

    image, truth, blob = synthetic_scene()

    guided = refine.attention_to_mask(
        blob, image=image, pixel_precision=True, radius=1, mode="auto"
    )
    bicubic = refine.attention_to_mask(
        blob, image=image, pixel_precision=False, mode="auto"
    )

    def iou(mask):
        predicted, real = mask > 0.5, truth
        return float((predicted & real).sum() / max(1, (predicted | real).sum()))

    def boundary_distance(mask):
        """Mean distance from the mask's pixels to the true outline, in pixels."""
        return float(ndimage.distance_transform_edt(~truth)[mask > 0.5].mean())

    print(
        f"       guided: IoU {iou(guided):.4f}, {boundary_distance(guided):.2f}px off; "
        f"bicubic: IoU {iou(bicubic):.4f}, {boundary_distance(bicubic):.2f}px off"
    )
    check(
        "guided filter mask beats bicubic",
        iou(guided) > iou(bicubic) + 0.01,
        f"{iou(guided):.4f} vs {iou(bicubic):.4f}",
    )
    check(
        "guided filter edge sits closer to the silhouette",
        boundary_distance(guided) < boundary_distance(bicubic),
        f"{boundary_distance(guided):.2f}px vs {boundary_distance(bicubic):.2f}px",
    )
    check("guided filter mask is close to the truth", iou(guided) > 0.97, f"IoU {iou(guided):.4f}")

    panel = np.concatenate(
        [
            image,
            np.repeat(guided[..., None], 3, axis=-1),
            np.repeat(bicubic[..., None], 3, axis=-1),
            refine.overlay(image, guided) * 0.5 + refine.overlay(image, bicubic) * 0.5,
        ],
        axis=1,
    )
    save_png(os.path.join(OUT, "refine_compare.png"), panel)
    return panel


def ndimage_edge(mask):
    from scipy import ndimage

    return mask & ~ndimage.binary_erosion(mask, iterations=2)


def save_png(path, array):
    from PIL import Image

    Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8)).save(path)
    print(f"       wrote {path}")


def test_mask_node():
    clip = FakeClip()
    tokens = split_tags(clip, "a cat, red hat")
    rows = 77

    heat = torch.zeros(rows, 16, 16)
    # 'a cat' owns the left half, 'red hat' the right half
    heat[tokens[0]["indices"], :, :8] = 1.0
    heat[tokens[1]["indices"], :, 8:] = 1.0
    heatmaps = {"pos": {0: {rows: heat}}}

    size = 512
    ys, xs = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = 0.9
    image[..., 1] = 0.9
    image[..., 2] = 0.9
    image[:, : size // 2] = np.array([0.9, 0.2, 0.2])
    image[:, size // 2 :] = np.array([0.2, 0.2, 0.9])
    images = torch.from_numpy(image)[None]

    node = DAAMMasks()
    masks, union, overlay, labels, names = node.make(
        heatmaps=heatmaps,
        clip=clip,
        text="a cat, red hat",
        images=images,
        tags="",
        side="positive",
        threshold_mode="percentile",
        threshold=0.5,
        percentile=50.0,
        pixel_precision=True,
        smooth=1,
        hysteresis=True,
        open_px=1,
        close_px=0,
        grow_px=0,
        min_area=64,
        feather_px=1,
        batch_index=0,
        soft=False,
        overlay_strength=0.6,
    )

    check("one mask per tag", tuple(masks.shape) == (2, size, size), str(tuple(masks.shape)))
    check("union mask shape", tuple(union.shape) == (size, size), str(tuple(union.shape)))
    check("overlay per tag", tuple(overlay.shape) == (2, size, size, 3), str(tuple(overlay.shape)))
    check("label map shape", tuple(labels.shape) == (size, size, 3), str(tuple(labels.shape)))
    check("tag names", names == "a cat|red hat", names)

    left = masks[0].numpy()[:, : size // 2].mean()
    right = masks[0].numpy()[:, size // 2 :].mean()
    check("first tag owns the left half", left > 0.9 and right < 0.1, f"{left:.2f} / {right:.2f}")

    panel = np.concatenate([overlay[0].numpy(), labels.numpy()], axis=1)
    save_png(os.path.join(OUT, "mask_node.png"), panel)

    selected = node.make(
        heatmaps=heatmaps,
        clip=clip,
        text="a cat, red hat",
        images=images,
        tags="red hat",
        side="positive",
        threshold_mode="auto",
        threshold=0.5,
        percentile=85.0,
        pixel_precision=True,
        smooth=1,
        hysteresis=True,
        open_px=1,
        close_px=0,
        grow_px=0,
        min_area=64,
        feather_px=0,
        batch_index=0,
        soft=False,
        overlay_strength=0.6,
    )
    check("tag filter keeps one mask", tuple(selected[0].shape)[0] == 1 and selected[4] == "red hat", selected[4])


def test_soft_and_cleanup():
    image, truth, blob = synthetic_scene(512, 32)
    soft = refine.attention_to_mask(blob, image=image, pixel_precision=True, mode="auto", soft=True)
    check("soft output keeps values", soft.max() > 0.9 and soft.min() < 0.1, f"{soft.min():.2f}..{soft.max():.2f}")

    noisy = (truth > 0).astype(np.float32)
    noisy[::37, ::41] = 0.0
    noisy[5, 5] = 1.0
    cleaned = refine.clean(noisy > 0.5, open_px=2, close_px=3, grow_px=2)
    check("cleanup keeps the object", cleaned.mean() > 0.05, f"{cleaned.mean():.3f}")
    check("cleanup removes the speck", not cleaned[5, 5])


def test_explorer_payload():
    from daam_mixed.nodes.explorer import DAAMExplorer, EXPLORER_RESULTS

    clip = FakeClip()
    tags = split_tags(clip, "a cat, red hat")
    rows = 77
    heat = torch.zeros(rows, 16, 16)
    heat[tags[0]["indices"]] = 1.0
    heat[tags[1]["indices"]] = 0.4
    heatmaps = {"pos": {0: {rows: heat}}}

    size = 128
    image = torch.from_numpy(np.random.RandomState(0).rand(size, size, 3).astype(np.float32))[None]

    payload = DAAMExplorer().explore(
        heatmaps=heatmaps,
        clip=clip,
        text="a cat, red hat",
        images=image,
        tags="",
        side="positive",
        batch_index=0,
        unique_id="42",
    )
    entry = payload["ui"]["daam"][0]
    check("explorer reports the tags", entry["tags"] == ["a cat", "red hat"], str(entry["tags"]))
    check("explorer reports the grid", entry["shape"] == [16, 16], str(entry["shape"]))

    stored = EXPLORER_RESULTS["42"]
    path = os.path.join(OUT, "explorer_maps.npy")
    with open(path, "wb") as handle:
        handle.write(stored["maps"])
    print(f"       wrote {path}")

    from PIL import Image
    image_path = os.path.join(OUT, "explorer_image.png")
    with open(image_path, "wb") as handle:
        handle.write(stored["image"])
    check("explorer image is a png", Image.open(image_path).size == (size, size))


def test_node_schemas():
    """What ComfyUI checks when the folder is loaded, minus the loading itself."""
    import importlib

    package = importlib.import_module("daam_mixed")
    mapping = package.NODE_CLASS_MAPPINGS
    check("node classes registered", len(mapping) == 7, str(sorted(mapping)))
    check("every node has a display name", set(mapping) == set(package.NODE_DISPLAY_NAME_MAPPINGS))

    for name, node in mapping.items():
        schema = node.INPUT_TYPES()
        required = schema.get("required", {})
        bad = [
            key
            for key, value in required.items()
            if not isinstance(value, (tuple, list)) or not isinstance(value[0], (str, list))
        ]
        check(f"{name}: input types are well formed", not bad, str(bad))
        check(f"{name}: FUNCTION exists", callable(getattr(node, name and node.FUNCTION, None)))

        returns = getattr(node, "RETURN_TYPES", ())
        names = getattr(node, "RETURN_NAMES", returns)
        check(f"{name}: return names match return types", len(returns) == len(names),
              f"{returns} / {names}")

        for key, value in required.items():
            default = value[1].get("default") if len(value) > 1 and isinstance(value[1], dict) else None
            if isinstance(value[0], str) and value[0] == "INT" and default is not None:
                options = value[1]
                check(
                    f"{name}.{key}: default inside range",
                    options.get("min", default) <= default <= options.get("max", default),
                    str(default),
                )

    check("web directory is declared", package.WEB_DIRECTORY == "./js")
    for relative in ("js/daamCore.js", "js/daamExplorer.js"):
        path = os.path.join(os.path.dirname(HERE), relative)
        check(f"{relative} ships", os.path.isfile(path), path)



# ------------------------------------------------------- the diploma features


def test_head_aggregation():
    from daam_mixed.core.heads import aggregate_heads, apply_value_norm, head_entropy

    heads, queries, keys = 4, 16, 5
    probs = torch.full((1, heads, queries, keys), 1.0 / keys)
    probs[0, 0, :, :] = 0.0
    probs[0, 0, :, 2] = 1.0  # a focused head
    probs[0, 3, :, :4] = 0.1 / 4
    probs[0, 3, :, 4] = 0.9  # a head that spills onto another token

    entropy = head_entropy(probs)
    check("entropy is lowest for the focused head", int(entropy.argmin()) == 0, str(entropy.tolist()))

    mean = aggregate_heads(probs, "mean")
    focused = aggregate_heads(probs, "min_entropy")
    check(
        "min_entropy keeps the focused head",
        float(focused[0, :, 2].mean()) > 0.6,
        f"{float(focused[0, :, 2].mean()):.2f}",
    )
    check(
        "plain mean dilutes it",
        float(mean[0, :, 2].mean()) < 0.45,
        f"{float(mean[0, :, 2].mean()):.2f}",
    )
    for method in ("mean", "median", "max", "min_entropy"):
        merged = aggregate_heads(probs, method)
        check(
            f"{method}: rows stay normalised",
            abs(float(merged.sum(-1).mean()) - 1.0) < 1e-5,
            str(float(merged.sum(-1).mean())),
        )

    value = torch.ones(1, keys, 8)
    value[0, 4, :] = 0.01  # an empty token, like padding
    scaled = apply_value_norm(mean, value)
    check(
        "value norm pushes the empty token down",
        float(scaled[0, :, 4].mean()) < float(mean[0, :, 4].mean()) * 0.2,
        f"{float(scaled[0, :, 4].mean()):.4f} vs {float(mean[0, :, 4].mean()):.3f}",
    )
    check(
        "value norm keeps the map normalised",
        abs(float(scaled.sum(-1).mean()) - 1.0) < 1e-5,
        str(float(scaled.sum(-1).mean())),
    )


def drive_token(patcher, sigmas, step, token, latent=(32, 32), heads=1, dim=8, batch=1):
    """One UNet call whose queries all look at a single prompt token."""
    grid_side = latent[0] // 2
    q_len = grid_side * grid_side
    seeds = one_hot_vectors(dim, 5)
    q = torch.zeros(batch, q_len, heads * dim)
    q[:, :, :dim] = seeds[token] * 8.0
    k = torch.zeros(batch, 5, heads * dim)
    k[:, :, :dim] = seeds * 8.0
    extra = {
        "n_heads": heads,
        "sigmas": sigmas[step : step + 1],
        "sample_sigmas": sigmas,
        "cond_or_uncond": [0] * batch,
    }
    for patch in list(patcher.patches.values()):
        patch(q, k, k, extra)


def test_per_step_capture_and_snr():
    from daam_mixed.core.aggregate import side_maps, step_weights

    sigmas = torch.linspace(14.0, 0.0, 21)
    late_steps = [16, 17, 18, 19]
    collector = AttentionCollector(32, 32, capture_last_steps=0, keep_per_step=True)
    patcher = FakePatcher()
    install_attention_capture(patcher, collector)
    for order, step in enumerate(late_steps):
        drive_token(patcher, sigmas, step, order)  # token 0 at step 16 ... token 3 at step 19

    payload = collector.heatmaps("pos")
    check("per step maps are kept", len(payload.get("steps") or {}) == 4, str(sorted(payload.get("steps") or {})))
    check("the schedule is shipped with the maps", len((payload.get("sigmas") or {}).get("schedule") or []) == 21)

    default = side_maps(payload, "pos", "none")
    means = [float(default[0][5][token].mean()) for token in range(4)]
    check("without weighting every step counts the same", all(abs(value - 0.25) < 0.02 for value in means),
          " ".join(f"{value:.2f}" for value in means))
    check(
        "each step keeps its own map",
        abs(float(payload["steps"][16][0][5][0].mean()) - 1.0) < 1e-5,
        str(float(payload["steps"][16][0][5][0].mean())),
    )

    weights = step_weights(payload["sigmas"]["schedule"], payload["steps"].keys(), "inverse_sigma_squared")
    check("SNR weights sum to one", abs(sum(weights.values()) - 1.0) < 1e-6, str(weights))
    ordered = [weights[step] for step in sorted(weights)]
    check("SNR weights grow with the step", ordered == sorted(ordered), str([round(v, 3) for v in ordered]))
    check("the last step dominates the weighting", weights[19] > 0.5, f"{weights[19]:.3f}")

    weighted = side_maps(payload, "pos", "inverse_sigma")
    check("weighted maps keep the token rows", set(weighted[0]) == {5}, str(set(weighted[0])))
    late = float(weighted[0][5][3].mean())
    early = float(weighted[0][5][0].mean())
    check("weighted maps lean on the late steps", late > early * 2, f"token3 {late:.3f} vs token0 {early:.3f}")


def test_self_attention_capture():
    collector = AttentionCollector(
        32, 32, capture_self=True, self_level="secondary", keep_per_step=True
    )
    patcher = FakePatcher()
    install_attention_capture(patcher, collector)
    check(
        "attn1 is patched too",
        len(patcher.patches_for("attn1")) == len(patcher.patches_for("attn2")) > 0,
        str(len(patcher.patches_for("attn1"))),
    )
    check(
        "a self attention patch is bound to this collector",
        all(getattr(patch, "__self__", None) is collector for patch in patcher.patches_for("attn1").values()),
    )

    # 8x8 = 64 cells; the first half and the second half share a direction
    cells, dim = 64, 8
    direction = torch.zeros(cells, dim)
    direction[:32, 0] = 1.0
    direction[32:, 1] = 1.0
    q = (direction * 6.0)[None]
    v = direction[None]

    out = collector.attn1(q, q, v, {"n_heads": 1, "cond_or_uncond": [0]})
    check("self attention still returns the output", tuple(out.shape) == (1, cells, dim), str(tuple(out.shape)))

    payload = collector.heatmaps("pos")
    affinity = (payload.get("self") or {}).get("secondary", {}).get(0)
    check("affinity was captured", affinity is not None)
    if affinity is None:
        return

    check("affinity is square on the secondary grid", tuple(affinity.shape) == (cells, cells), str(tuple(affinity.shape)))
    check(
        "affinity rows are distributions",
        abs(float(affinity.sum(-1).mean()) - 1.0) < 1e-4,
        str(float(affinity.sum(-1).mean())),
    )
    inside = float(affinity[:32, :32].sum(-1).mean())
    across = float(affinity[:32, 32:].sum(-1).mean())
    check("affinity stays inside the region", inside > 0.99 and across < 0.01, f"{inside:.3f} / {across:.3f}")


def test_guided_diffusion():
    from scipy import ndimage

    grid = 16
    centre = (grid - 1) / 2
    ys, xs = np.mgrid[0:grid, 0:grid]
    truth = (((ys - centre) ** 2 + (xs - centre) ** 2) <= 5.5**2).astype(np.float32)

    rng = np.random.RandomState(3)
    noisy = ndimage.gaussian_filter(truth, 1.2) + rng.normal(0.0, 0.18, (grid, grid))
    noisy = np.clip(noisy, 0.0, 1.0).ravel()

    coordinates = np.stack(np.mgrid[0:grid, 0:grid], -1).reshape(-1, 2).astype(np.float32)
    distance = ((coordinates[:, None, :] - coordinates[None, :, :]) ** 2).sum(-1)
    same = (truth.ravel()[:, None] == truth.ravel()[None, :]).astype(np.float32)
    affinity = np.exp(-distance / (2 * 3.0**2)) * (1 + 1.5 * same)
    affinity /= affinity.sum(1, keepdims=True)

    def iou(values):
        level = refine.threshold_map(values, "gmm")
        predicted = values >= level
        real = truth.ravel() > 0.5
        return float((predicted & real).sum() / max(1, (predicted | real).sum()))

    before = iou(noisy)
    after = iou(refine.guided_diffuse(noisy[None], affinity, iterations=3, mode="gmm")[0])
    print(f"       guided diffusion IoU {before:.3f} -> {after:.3f}")
    check("guided diffusion sharpens a noisy map", after > before + 0.15, f"{before:.3f} -> {after:.3f}")
    check("guided diffusion ends near the truth", after > 0.85, f"{after:.3f}")

    blended = refine.guided_diffuse(noisy[None], affinity, iterations=3, mode="gmm", blend=0.0)[0]
    check(
        "blend 0 leaves the map alone",
        np.allclose(blended, refine.normalize_rows(noisy[None], "minmax")[0], atol=1e-5),
    )
    check(
        "iterations 0 changes nothing",
        np.allclose(refine.guided_diffuse(noisy[None], affinity, iterations=0)[0], noisy[None][0], atol=1e-6),
    )

    for mode in ("otsu", "gmm", "fixed"):
        level = refine.threshold_map(np.clip(noisy, 0, 1), mode, threshold=0.4)
        check(f"threshold mode {mode} is in range", 0.0 <= level <= 1.0, str(level))
    bimodal = np.concatenate([rng.rand(2000) * 0.12, 0.7 + rng.rand(1000) * 0.25])
    level = refine.gmm(bimodal)
    check("gmm splits a bimodal map", 0.2 < level < 0.6, f"{level:.3f}")


def _guide_payload(grid=8, tokens=5, seed=5):
    """A payload shaped like the sampler's, with a self attention affinity."""
    rng = np.random.RandomState(seed)
    truth = np.zeros((grid, grid), dtype=np.float32)
    truth[:, : grid // 2] = 1.0
    maps = np.stack([truth * (0.2 + 0.6 * rng.rand()) for _ in range(tokens)])
    maps += rng.normal(0.0, 0.05, maps.shape).astype(np.float32)
    maps = np.clip(maps, 0.0, 1.0)

    flat = truth.ravel()
    same = (flat[:, None] == flat[None, :]).astype(np.float32)
    affinity = 0.7 * same + 0.3 * (1.0 / flat.size)
    affinity /= affinity.sum(1, keepdims=True)

    return {
        "pos": {0: {tokens: torch.from_numpy(maps)}},
        "steps": {
            0: {0: {tokens: torch.from_numpy(maps * 0.5)}},
            1: {0: {tokens: torch.from_numpy(np.clip(maps * 1.2, 0, 1))}},
        },
        "sigmas": {"schedule": [14.0, 1.0, 0.0]},
        "self": {"secondary": {0: torch.from_numpy(affinity)}, "grids": {"secondary": (grid, grid)}},
    }


def test_guide_node():
    from daam_mixed.nodes.guide import DAAMGuide

    payload = _guide_payload()
    node = DAAMGuide()
    refined, report = node.guide(
        heatmaps=payload,
        step_weighting="inverse_sigma_squared",
        self_level="secondary",
        iterations=3,
        threshold_mode="gmm",
        threshold=0.02,
        percentile=85.0,
        normalize="minmax",
        blend=1.0,
        batch_index=0,
    )
    check("guide keeps the payload shape", set(refined) == {"pos"}, str(set(refined)))
    maps = refined["pos"][0][5]
    check("guide keeps the token axis", tuple(maps.shape) == (5, 8, 8), str(tuple(maps.shape)))
    check("guide returns values in range", float(maps.min()) >= 0.0 and float(maps.max()) <= 1.0)
    print(f"       guide report: {report}")

    left = float(maps[0][:, :4].mean())
    right = float(maps[0][:, 4:].mean())
    check("guide sharpened the split", left > right * 3, f"{left:.3f} vs {right:.3f}")

    off, report_off = node.guide(
        heatmaps=payload,
        step_weighting="none",
        self_level="off",
        iterations=5,
        threshold_mode="gmm",
        threshold=0.02,
        percentile=85.0,
        normalize="minmax",
        blend=1.0,
        batch_index=0,
    )
    original = payload["pos"][0][5]
    check("without self attention the maps pass through", torch.allclose(off["pos"][0][5], original, atol=1e-6))
    check("the report says why", "no self-attention" in report_off, report_off)


def test_merge_node():
    from daam_mixed.nodes.guide import DAAMMerge

    first = _guide_payload(seed=1)
    second = _guide_payload(seed=2)
    node = DAAMMerge()

    mean_maps, report = node.merge(first, second, "mean")
    expected = (first["pos"][0][5] + second["pos"][0][5]) / 2.0
    check("merge averages the runs", torch.allclose(mean_maps["pos"][0][5], expected, atol=1e-6), report)

    third = _guide_payload(seed=3)
    median_maps, _ = node.merge(first, second, "median", third)
    stacked = torch.stack([first["pos"][0][5], second["pos"][0][5], third["pos"][0][5]])
    check("merge takes the median", torch.allclose(median_maps["pos"][0][5], stacked.median(0).values, atol=1e-6))


def main():
    test_node_schemas()
    test_capture_window()
    test_nested_blocks()
    test_explorer_payload()
    test_capture_all_and_offset()
    test_cond_uncond_and_grids()
    test_tags()
    test_refine_precision()
    test_mask_node()
    test_soft_and_cleanup()
    test_head_aggregation()
    test_per_step_capture_and_snr()
    test_self_attention_capture()
    test_guided_diffusion()
    test_guide_node()
    test_merge_node()

    failed = [name for name, ok, _ in results if not ok]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print("failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
