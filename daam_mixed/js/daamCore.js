/**
 * Pure helpers for the DAAM attention panel.
 *
 * Kept free of DOM and ComfyUI imports so they can be exercised with node,
 * which is where the maths that turns a small .npy into pixels is tested.
 */

/** Parse a .npy buffer holding a C-ordered float array. */
export function parseNpy(buffer) {
  const bytes = new Uint8Array(buffer);
  const magic = String.fromCharCode(...bytes.slice(1, 6));
  if (magic !== "NUMPY") throw new Error("not a .npy file");

  const major = bytes[6];
  const headerLength = major === 1
    ? bytes[8] | (bytes[9] << 8)
    : bytes[8] | (bytes[9] << 8) | (bytes[10] << 16) | (bytes[11] << 24);
  const headerStart = major === 1 ? 10 : 12;
  const header = new TextDecoder("latin1").decode(
    bytes.slice(headerStart, headerStart + headerLength)
  );

  const descr = /'descr'\s*:\s*'([^']+)'/.exec(header);
  const shape = /'shape'\s*:\s*\(([^)]*)\)/.exec(header);
  const fortran = /'fortran_order'\s*:\s*(True|False)/.exec(header);
  if (!descr || !shape) throw new Error("unsupported .npy header");

  const dims = shape[1].split(",").map((part) => part.trim()).filter(Boolean).map(Number);
  const little = descr[1][0] !== ">";
  const kind = descr[1].slice(1);
  const width = kind === "f8" ? 8 : kind === "f4" ? 4 : 0;
  if (!width) throw new Error(`unsupported .npy dtype ${descr[1]}`);
  if (fortran && fortran[1] === "True") throw new Error("fortran order is not supported");

  const offset = headerStart + headerLength;
  const count = dims.reduce((total, size) => total * size, 1);
  const view = new DataView(buffer, offset, count * width);
  const data = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    data[i] = width === 8 ? view.getFloat64(i * 8, little) : view.getFloat32(i * 4, little);
  }
  return { data, shape: dims };
}

/** Bilinear lookup of tag `index` at normalized coordinates (0..1). */
export function sample(maps, index, u, v) {
  const [, height, width] = maps.shape;
  const x = Math.min(Math.max(u, 0), 1) * (width - 1);
  const y = Math.min(Math.max(v, 0), 1) * (height - 1);
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const x1 = Math.min(x0 + 1, width - 1);
  const y1 = Math.min(y0 + 1, height - 1);
  const fx = x - x0;
  const fy = y - y0;
  const at = (xx, yy) => maps.data[index * height * width + yy * width + xx];
  const top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
  const bottom = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
  return top * (1 - fy) + bottom * fy;
}

/** Box blur of every tag's map by `radius` grid cells; keeps the cells. */
export function blurMaps(maps, radius) {
  if (!radius || radius < 1) return maps;
  const [tags, height, width] = maps.shape;
  const out = new Float32Array(maps.data.length);
  for (let t = 0; t < tags; t += 1) {
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        let sum = 0;
        let count = 0;
        for (let dy = -radius; dy <= radius; dy += 1) {
          const yy = y + dy;
          if (yy < 0 || yy >= height) continue;
          for (let dx = -radius; dx <= radius; dx += 1) {
            const xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            sum += maps.data[t * height * width + yy * width + xx];
            count += 1;
          }
        }
        out[t * height * width + y * width + x] = sum / count;
      }
    }
  }
  return { data: out, shape: maps.shape };
}

/** Tag indices that claim a pixel, strongest first, above `floor`. */
export function topTags(maps, u, v, count = 3, floor = 0) {
  const total = maps.shape[0];
  const values = [];
  for (let index = 0; index < total; index += 1) {
    const value = sample(maps, index, u, v);
    if (value > floor) values.push({ index, value });
  }
  values.sort((a, b) => b.value - a.value);
  return values.slice(0, count);
}

/** One tag's contribution at a pixel: its own map, or the mean of pinned ones. */
export function combinedAt(maps, pinned, u, v) {
  if (!pinned.length) return 0;
  let sum = 0;
  for (const index of pinned) sum += sample(maps, index, u, v);
  return sum / pinned.length;
}

/** Matplotlib's jet, for a 0..1 value. */
export function jet(value) {
  const t = Math.min(Math.max(value, 0), 1);
  const channel = (centre) => Math.min(Math.max(1.5 - Math.abs(4 * t - centre), 0), 1);
  return [channel(3) * 255, channel(2) * 255, channel(1) * 255];
}

/**
 * Paint the image and the overlay into `imageData`.
 *
 * `view` is "heatmap" (jet over the image), "mask" (the tag lights the image
 * and the rest goes dark) or "labels" (each pinned tag in its own colour).
 */
export function renderFrame(imageData, maps, options) {
  const { pinned, strength = 0.6, view = "heatmap", palette = [] } = options;
  const { data, width, height } = imageData;
  const active = pinned.length ? pinned : maps.shape[0] ? [0] : [];

  if (!active.length) return imageData;

  for (let y = 0; y < height; y += 1) {
    const v = height > 1 ? y / (height - 1) : 0;
    for (let x = 0; x < width; x += 1) {
      const u = width > 1 ? x / (width - 1) : 0;
      const offset = (y * width + x) * 4;
      if (view === "labels") {
        let best = -1;
        let bestValue = 0;
        for (let i = 0; i < active.length; i += 1) {
          const value = sample(maps, active[i], u, v);
          if (value > bestValue) {
            bestValue = value;
            best = i;
          }
        }
        if (best >= 0 && bestValue > 0.15) {
          const colour = palette[active[best] % palette.length] || [255, 255, 255];
          data[offset] = colour[0];
          data[offset + 1] = colour[1];
          data[offset + 2] = colour[2];
        } else {
          data[offset] *= 0.25;
          data[offset + 1] *= 0.25;
          data[offset + 2] *= 0.25;
        }
        continue;
      }

      const value = combinedAt(maps, active, u, v);
      if (view === "mask") {
        const keep = 0.35 + 0.65 * Math.min(Math.max(value, 0), 1);
        data[offset] *= keep;
        data[offset + 1] *= keep;
        data[offset + 2] *= keep;
        continue;
      }

      const colour = jet(value);
      const alpha = Math.min(Math.max(value * strength, 0), 1);
      data[offset] = data[offset] * (1 - alpha) + colour[0] * alpha;
      data[offset + 1] = data[offset + 1] * (1 - alpha) + colour[1] * alpha;
      data[offset + 2] = data[offset + 2] * (1 - alpha) + colour[2] * alpha;
    }
  }
  return imageData;
}
