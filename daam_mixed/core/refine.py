"""Turning a low resolution attention map into a pixel level mask.

A cross attention map is at best image/16 (64x64 for a 1024px SDXL render),
which is far too coarse to cut anything out: bicubic upsampling turns every
attention cell into a 16px blob with a soft border. The maps also carry noise
from the layers that were averaged and from the tokens of words that were only
partly attended to.

So the map is first stretched to the pixel grid with a guided filter that uses
the decoded image as its edge guide (He, Sun & Tang, "Guided Image Filtering",
the fast variant: the linear coefficients are solved on the coarse grid and
applied at full resolution). Flat areas then follow the attention value, while
object borders come from the image itself, which is what makes the mask land on
the silhouette instead of on the 16px grid. Everything after that - threshold,
hysteresis, morphology, feather - works on the full resolution map.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

# --------------------------------------------------------------------- colour


def jet(values):
    """Matplotlib's jet for a 0..1 array, without the matplotlib dependency."""
    x = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return np.stack([red, green, blue], axis=-1)


# ------------------------------------------------------------------ the map


def robust_normalize(values, low=2.0, high=99.0):
    """Stretch the map from its low percentile to its high one, then clamp.

    A single hot cell - the token sink that sits in a corner - would otherwise
    own the top of the scale and flatten everything else.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    floor, top = np.percentile(array, low), np.percentile(array, high)
    if top - floor < 1e-6:
        span = float(array.max() - array.min())
        if span < 1e-6:
            return np.zeros_like(array)
        return (array - array.min()) / span
    return np.clip((array - floor) / (top - floor), 0.0, 1.0)


def _box_blur(tensor, radius):
    if radius <= 0:
        return tensor
    kernel = 2 * radius + 1
    padded = F.pad(tensor, (radius, radius, radius, radius), mode="replicate")
    return F.avg_pool2d(padded, kernel, stride=1)


def guided_upsample(low, guide, radius=2, eps=1e-3):
    """Edge aware upsampling of ``low`` (h, w) guided by ``guide`` (H, W).

    The linear model out = a * guide + b is fitted on the coarse grid, where the
    attention value lives, and evaluated at pixel resolution. Both a and b are
    smooth, so they upsample without ringing.
    """
    low_t = torch.as_tensor(low, dtype=torch.float32)[None, None]
    height, width = guide.shape
    coarse_h, coarse_w = low.shape

    guide_t = torch.as_tensor(guide, dtype=torch.float32)[None, None]
    guide_low = F.interpolate(guide_t, size=(coarse_h, coarse_w), mode="area")

    mean_i = _box_blur(guide_low, radius)
    mean_p = _box_blur(low_t, radius)
    corr_i = _box_blur(guide_low * guide_low, radius)
    corr_ip = _box_blur(guide_low * low_t, radius)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i

    a_full = F.interpolate(a, size=(height, width), mode="bilinear", align_corners=False)
    b_full = F.interpolate(b, size=(height, width), mode="bilinear", align_corners=False)
    out = a_full * guide_t + b_full
    return out[0, 0].numpy()


def luminance(image):
    """(H, W) grey of a (H, W, 3) float image."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        return array
    return array[..., 0] * 0.299 + array[..., 1] * 0.587 + array[..., 2] * 0.114


def upsample_map(low, size, mode="bicubic"):
    tensor = torch.as_tensor(low, dtype=torch.float32)[None, None]
    out = F.interpolate(tensor, size=size, mode=mode, align_corners=False)
    return out[0, 0].clamp(0.0, 1.0).numpy()


# --------------------------------------------------------------- thresholds


def otsu(values):
    """Otsu's threshold of a 0..1 array."""
    histogram, edges = np.histogram(np.clip(values, 0.0, 1.0), bins=256, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total == 0:
        return 0.5
    probabilities = histogram / total
    centres = (edges[:-1] + edges[1:]) * 0.5
    omega = np.cumsum(probabilities)
    mu = np.cumsum(probabilities * centres)
    mu_total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_total * omega - mu) ** 2 / (omega * (1.0 - omega))
    between = np.nan_to_num(between, nan=0.0, posinf=0.0)
    return float(centres[int(between.argmax())])


def gmm(values, iterations=30, bins=96):
    """Two component 1D Gaussian mixture by EM; threshold where they cross.

    A DAAM map is close to two populations - the object and everything else -
    and a single fixed threshold cuts one of them. Otsu answers the same
    question with a histogram, this one with two gaussians, which behaves
    better on a map with a long low tail (a word that is barely attended to).
    """
    sample = np.asarray(values, dtype=np.float64).ravel()
    if sample.size == 0:
        return 0.5
    if sample.size > 20000:  # EM on a histogram is enough and much faster
        histogram, edges = np.histogram(np.clip(sample, 0, 1), bins=bins, range=(0, 1))
        sample = (edges[:-1] + edges[1:]) * 0.5
        sample = np.repeat(sample, histogram)

    low, high = np.percentile(sample, 25), np.percentile(sample, 75)
    means = np.array([low, high], dtype=np.float64)
    deviations = np.array([sample.std(), sample.std()], dtype=np.float64) + 1e-6
    weights = np.array([0.5, 0.5], dtype=np.float64)

    for _ in range(max(1, int(iterations))):
        densities = weights / (deviations * np.sqrt(2 * np.pi)) * np.exp(
            -0.5 * ((sample[:, None] - means) / deviations) ** 2
        )
        total = densities.sum(1, keepdims=True)
        responsibilities = densities / np.maximum(total, 1e-12)
        mass = responsibilities.sum(0) + 1e-12
        weights = mass / sample.size
        means = (responsibilities * sample[:, None]).sum(0) / mass
        deviations = np.sqrt(
            (responsibilities * (sample[:, None] - means) ** 2).sum(0) / mass
        ) + 1e-3

    if means[0] > means[1]:
        means, deviations, weights = means[::-1], deviations[::-1], weights[::-1]

    # where the two gaussians are equally likely: A x^2 + B x + C = 0
    sigma1, sigma2 = deviations[0], deviations[1]
    mu1, mu2 = means[0], means[1]
    a = 1.0 / sigma2**2 - 1.0 / sigma1**2
    b = 2.0 * (mu1 / sigma1**2 - mu2 / sigma2**2)
    c = mu2**2 / sigma2**2 - mu1**2 / sigma1**2 - 2.0 * np.log(sigma1 / sigma2)

    candidates = []
    if abs(a) < 1e-12:
        candidates = [-c / b] if abs(b) > 1e-12 else []
    else:
        discriminant = b * b - 4 * a * c
        if discriminant >= 0:
            root = np.sqrt(discriminant)
            candidates = [(-b + root) / (2 * a), (-b - root) / (2 * a)]

    inside = [x for x in candidates if mu1 - 1e-6 <= x <= mu2 + 1e-6]
    if inside:
        return float(np.clip(min(inside), 0.0, 1.0))
    if candidates:
        return float(np.clip(np.clip(min(candidates, key=lambda x: abs(x - (mu1 + mu2) / 2)), 0.0, 1.0), 0.0, 1.0))
    return float(np.clip((mu1 + mu2) / 2.0, 0.0, 1.0))


def threshold_map(values, mode="auto", threshold=0.5, percentile=85.0):
    """The level a pixel has to reach to count as part of the tag."""
    if mode == "fixed":
        return float(np.clip(threshold, 0.0, 1.0))
    if mode == "gmm":
        return gmm(values)
    if mode == "otsu":
        return float(np.clip(otsu(values), 0.0, 1.0))
    if mode == "percentile":
        return float(np.percentile(values, np.clip(percentile, 1.0, 99.9)))
    # auto: Otsu, kept inside a sane band so a flat or a very peaky map cannot
    # push it to 0 or to the maximum.
    level = otsu(values)
    return float(np.clip(level, np.percentile(values, 50.0), np.percentile(values, 97.0)))


def hysteresis(values, low, high):
    """Keep the regions that reach ``high``, grown down to ``low``."""
    strong = values >= high
    if not strong.any():
        return np.zeros_like(values, dtype=bool)
    weak = values >= low
    labels, count = ndimage.label(weak)
    if count == 0:
        return strong
    keep = np.unique(labels[strong])
    keep = keep[keep != 0]
    return np.isin(labels, keep)


# -------------------------------------------------------------- morphology


def _disk(radius):
    radius = int(radius)
    if radius <= 0:
        return None
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def clean(mask, open_px=0, close_px=0, grow_px=0):
    """Speck removal, hole filling, and growing or shrinking by a pixel count."""
    result = np.asarray(mask, dtype=bool)
    if open_px > 0:
        result = ndimage.binary_opening(result, structure=_disk(open_px))
    if close_px > 0:
        result = ndimage.binary_closing(result, structure=_disk(close_px))
    if grow_px > 0:
        result = ndimage.binary_dilation(result, structure=_disk(grow_px))
    elif grow_px < 0:
        result = ndimage.binary_erosion(result, structure=_disk(-grow_px))
    return result


def drop_small(mask, min_area):
    """Remove islands smaller than ``min_area`` pixels; keeps the rest intact."""
    mask = np.asarray(mask, dtype=bool)
    if min_area <= 0:
        return mask
    labels, count = ndimage.label(mask)
    if count == 0:
        return mask
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    keep = np.nonzero(sizes >= min_area)[0] + 1
    return np.isin(labels, keep)


def keep_largest(mask):
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if count <= 1:
        return np.asarray(mask, dtype=bool)
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    largest = int(np.argmax(sizes)) + 1
    return labels == largest


def feather(mask, sigma):
    values = np.asarray(mask, dtype=np.float32)
    if sigma and sigma > 0:
        values = ndimage.gaussian_filter(values, sigma=float(sigma))
    return np.clip(values, 0.0, 1.0)


# ------------------------------------------------------------------ pipeline


def attention_to_mask(
    low_map,
    image=None,
    radius=2,
    eps=1e-3,
    pixel_precision=True,
    mode="auto",
    use_hysteresis=True,
    threshold=0.5,
    percentile=85.0,
    open_px=0,
    close_px=0,
    grow_px=0,
    keep_main=False,
    min_area=0,
    feather_px=0,
    soft=False,
):
    """A full resolution mask for one tag.

    ``image`` is (H, W, 3) in 0..1 and only used as the edge guide; without it
    the map is upsampled bicubically.
    """
    low_map = np.asarray(low_map, dtype=np.float32)
    if image is None:
        size = (low_map.shape[0] * 16, low_map.shape[1] * 16)
        full = upsample_map(robust_normalize(low_map), size)
    else:
        size = (image.shape[0], image.shape[1])
        guide = robust_normalize(luminance(image), low=1.0, high=99.0)
        coarse = robust_normalize(low_map)
        if pixel_precision:
            full = guided_upsample(coarse, guide, radius=radius, eps=eps)
        else:
            full = upsample_map(coarse, size)
        full = robust_normalize(full, low=1.0, high=99.0)

    level = threshold_map(full, mode=mode, threshold=threshold, percentile=percentile)
    if soft:
        return full.astype(np.float32)

    if use_hysteresis:
        binary = hysteresis(full, level * 0.5, level)
    else:
        binary = full >= level
    binary = clean(binary, open_px=open_px, close_px=close_px, grow_px=grow_px)
    binary = drop_small(binary, min_area)
    if keep_main:
        binary = keep_largest(binary)
    return feather(binary, feather_px)


# ----------------------------------------------------- guided diffusion


def normalize_rows(array, mode="minmax"):
    """Normalise every row (one token's map) of a (rows, n) array."""
    array = np.asarray(array, dtype=np.float32)
    if mode == "none":
        return array
    if mode == "zscore":
        mean = array.mean(1, keepdims=True)
        spread = array.std(1, keepdims=True)
        shifted = (array - mean) / np.maximum(spread, 1e-6)
        return 1.0 / (1.0 + np.exp(-shifted))  # squash to 0..1, monotonic
    low = array.min(1, keepdims=True)
    high = array.max(1, keepdims=True)
    return (array - low) / np.maximum(high - low, 1e-8)


def threshold_rows(array, mode="otsu", threshold=0.02, percentile=85.0):
    """Threshold every row separately, returning the mask of what survives."""
    array = np.asarray(array, dtype=np.float32)
    keep = np.zeros_like(array, dtype=bool)
    for index in range(array.shape[0]):
        row = array[index]
        level = threshold_map(row, mode=mode, threshold=threshold, percentile=percentile)
        keep[index] = row >= level
    return keep


def guided_diffuse(
    maps,
    affinity,
    iterations=3,
    mode="otsu",
    threshold=0.02,
    percentile=85.0,
    normalize="minmax",
    blend=1.0,
):
    """Refine token maps by letting them spread along the self attention graph.

    ``maps`` is (tokens, hw) on the same grid as ``affinity``, which is
    (hw, hw) and row stochastic: row i holds how much pixel i looks at every
    other pixel. Multiplying a map by it makes every pixel take the average of
    the map over what it attends to - inside the object that average is the
    object's own value, across an edge it is not, so the region tightens with
    every iteration. After each pass the row is renormalised and everything
    below the threshold is dropped, which is what stops the map from creeping
    over the whole image.

    ``blend`` mixes the result with the untouched map: 1 is fully refined.
    """
    original = np.asarray(maps, dtype=np.float32)
    if affinity is None or iterations <= 0:
        return original

    affinity = np.asarray(affinity, dtype=np.float32)
    if affinity.shape[0] != original.shape[1]:
        raise ValueError(
            f"[DAAM] the self attention map is {affinity.shape[0]} cells but the "
            f"token maps have {original.shape[1]}; capture both at the same level"
        )

    current = original.copy()
    for _ in range(int(iterations)):
        current = current @ affinity
        current = normalize_rows(current, normalize)
        if mode == "none":
            continue
        keep = threshold_rows(current, mode=mode, threshold=threshold, percentile=percentile)
        current = current * keep

    if blend >= 1.0:
        return np.clip(current, 0.0, 1.0)
    mixed = float(blend) * current + (1.0 - float(blend)) * normalize_rows(original, normalize)
    return np.clip(mixed, 0.0, 1.0)


def overlay(image, mask, strength=0.6):
    """The image with the map painted over it in jet colours."""
    base = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    colour = jet(np.clip(mask, 0.0, 1.0))
    alpha = (np.clip(mask, 0.0, 1.0)[..., None] * float(strength)).clip(0.0, 1.0)
    return np.clip(base * (1.0 - alpha) + colour * alpha, 0.0, 1.0)


PALETTE = np.array(
    [
        [0.95, 0.26, 0.21],
        [0.13, 0.59, 0.95],
        [0.30, 0.80, 0.35],
        [0.98, 0.74, 0.02],
        [0.61, 0.35, 0.81],
        [0.00, 0.74, 0.76],
        [0.96, 0.51, 0.19],
        [0.83, 0.30, 0.61],
        [0.55, 0.63, 0.20],
        [0.35, 0.34, 0.85],
        [0.80, 0.55, 0.42],
        [0.11, 0.48, 0.55],
    ],
    dtype=np.float32,
)


def label_image(masks, threshold=0.5):
    """Every pixel painted with the tag that owns it, background black."""
    if not masks:
        return np.zeros((1, 1, 3), dtype=np.float32)
    stack = np.stack([np.asarray(mask, dtype=np.float32) for mask in masks])
    height, width = stack.shape[1:]
    owner = stack.argmax(0)
    best = stack.max(0)
    out = np.zeros((height, width, 3), dtype=np.float32)
    for index in range(len(masks)):
        selected = (owner == index) & (best >= threshold)
        out[selected] = PALETTE[index % len(PALETTE)]
    return out
