"""Builds the figure in tests/out/demo.png from synthetic attention maps.

The scene is drawn, not generated, so the demo needs no model: three shapes
stand in for three prompt tags, their attention maps are computed the way the
sampler would see them (a coarse grid, blurred, with a little noise), and the
mask pipeline in core/refine.py then turns those maps back into pixel level
masks. The point of the figure is to show what the pack does to a map, with a
ground truth to compare against.

Run:  python daam_mixed/tests/demo_figure.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(HERE, "stubs"))  # the pack's ComfyUI imports

from daam_mixed.core import refine  # noqa: E402

SIZE = 512
GRID = 32  # attention is image/16 at best; 512 / 16 = 32
OUT = os.path.join(HERE, "out")


def scene():
    """A simple picture plus one boolean truth mask per tag."""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]

    sky = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    sky[..., 0] = 0.20 + 0.35 * (1 - ys / SIZE)
    sky[..., 1] = 0.26 + 0.32 * (1 - ys / SIZE)
    sky[..., 2] = 0.38 + 0.30 * (1 - ys / SIZE)

    ground = ys > SIZE * 0.78
    sky[ground] = np.array([0.22, 0.30, 0.20], dtype=np.float32)

    body = ((xs - SIZE * 0.50) / (SIZE * 0.15)) ** 2 + ((ys - SIZE * 0.66) / (SIZE * 0.22)) ** 2 <= 1
    head = ((xs - SIZE * 0.50) / (SIZE * 0.085)) ** 2 + ((ys - SIZE * 0.40) / (SIZE * 0.085)) ** 2 <= 1
    hat = (
        (((xs - SIZE * 0.50) / (SIZE * 0.13)) ** 2 + ((ys - SIZE * 0.31) / (SIZE * 0.045)) ** 2 <= 1)
        | ((ys < SIZE * 0.31) & (ys > SIZE * 0.245) & (np.abs(xs - SIZE * 0.50) < SIZE * 0.085))
    )

    person = body | head
    sky[person] = np.array([0.72, 0.46, 0.36], dtype=np.float32)
    sky[person] *= (0.85 + 0.15 * np.sin(xs[person] / 4.0))[:, None]
    sky[hat] = np.array([0.86, 0.28, 0.30], dtype=np.float32)

    # a little grain, so the guide is not perfectly flat
    rng = np.random.RandomState(7)
    sky += rng.normal(0.0, 0.012, sky.shape).astype(np.float32)

    truth = {"a woman": person, "red hat": hat, "a park": ground}
    return np.clip(sky, 0, 1), truth


def attention_blob(truth):
    """The map a sampler would record: coarse, edge-softened, slightly noisy."""
    cell = SIZE // GRID
    blob = np.zeros((GRID, GRID), dtype=np.float32)
    for y in range(GRID):
        for x in range(GRID):
            blob[y, x] = truth[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell].mean()
    blob = ndimage.gaussian_filter(blob, 0.6)
    rng = np.random.RandomState(11)
    blob += rng.normal(0.0, 0.02, blob.shape).astype(np.float32)
    return np.clip(blob, 0.0, None)


def caption(draw, xy, text, width=None):
    draw.text(xy, text, fill=(235, 235, 235))


def main():
    image, truth = scene()
    labels = list(truth)

    rows = []
    for label in labels:
        blob = attention_blob(truth[label])
        mask = refine.attention_to_mask(
            blob, image=image, radius=1, mode="auto", open_px=2, min_area=48, feather_px=1
        )
        heat = refine.overlay(image, refine.upsample_map(refine.robust_normalize(blob), (SIZE, SIZE)))
        guided = refine.overlay(image, mask, 0.75)
        rows.append((label, heat, mask, guided))

    masks = [mask for _, _, mask, _ in rows]
    labels_image = refine.label_image(masks, threshold=0.5)
    labels_image = np.clip(labels_image + image * 0.12, 0, 1)

    columns = len(rows) + 2  # render, one per tag, label map
    pad = 8
    header = 26
    canvas = np.zeros((header + SIZE, columns * (SIZE + pad) + pad, 3), dtype=np.float32)
    canvas[header:] = 0.09

    def put(column, picture, title):
        x0 = pad + column * (SIZE + pad)
        canvas[header:, x0 : x0 + SIZE] = picture

    put(0, image, "render")
    for column, (label, heat, mask, guided) in enumerate(rows, start=1):
        put(column, guided, label)
    put(columns - 1, labels_image, "who owns what")

    panel = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    result = Image.fromarray(panel)
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    draw.text((pad, 7), "render", fill=(235, 235, 235), font=font)
    for column, (label, _, _, _) in enumerate(rows, start=1):
        draw.text((pad + column * (SIZE + pad), 7), label, fill=(235, 235, 235), font=font)
    draw.text((pad + (columns - 1) * (SIZE + pad), 7), "who owns what", fill=(235, 235, 235), font=font)

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "demo.png")
    result.save(path)

    # a number per tag, so the figure is also a test of the mask quality
    for label in labels:
        blob = attention_blob(truth[label])
        mask = refine.attention_to_mask(
            blob, image=image, radius=1, mode="auto", open_px=2, min_area=48, feather_px=1
        ) > 0.5
        real = truth[label]
        iou = float((mask & real).sum() / max(1, (mask | real).sum()))
        print(f"{label:10s} IoU {iou:.3f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
