/**
 * Interactive attention panel for the DAAM Explorer node.
 *
 * The node ships one PNG of the render and one small .npy of the per tag maps
 * at their native image/16 grid. Everything below happens in the browser, so
 * hovering, pinning tags and moving the sliders never touches the server.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { blurMaps, jet, parseNpy, renderFrame, sample, topTags } from "./daamCore.js";

const PANEL_WIDTH = 384;
const PALETTE = [
  [242, 66, 54], [33, 150, 243], [76, 204, 89], [250, 189, 5],
  [156, 89, 207], [0, 189, 194], [245, 130, 48], [212, 76, 156],
];

const styles = `
.daam-panel { display: flex; flex-direction: column; gap: 6px; width: 100%; font-size: 11px; }
.daam-panel canvas { width: 100%; display: block; border-radius: 6px; background: #111; cursor: crosshair; }
.daam-tags { display: flex; flex-wrap: wrap; gap: 4px; max-height: 62px; overflow-y: auto; }
.daam-tag { padding: 2px 6px; border-radius: 10px; border: 1px solid #555; background: #2a2a2a;
  color: #ddd; cursor: pointer; user-select: none; max-width: 100%; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap; }
.daam-tag.on { background: #3a6ea5; border-color: #7fb2e5; color: #fff; }
.daam-row { display: flex; align-items: center; gap: 6px; color: #aaa; }
.daam-row input[type=range] { flex: 1; min-width: 0; }
.daam-hover { color: #ccc; min-height: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.daam-hover b { color: #fff; font-weight: 600; }
`;

let styleInjected = false;
function injectStyle() {
  if (styleInjected) return;
  const element = document.createElement("style");
  element.textContent = styles;
  document.head.appendChild(element);
  styleInjected = true;
}

class DaamPanel {
  constructor(node) {
    this.node = node;
    this.container = document.createElement("div");
    this.container.className = "daam-panel";

    this.canvas = document.createElement("canvas");
    this.canvas.width = PANEL_WIDTH;
    this.canvas.height = PANEL_WIDTH;
    this.container.appendChild(this.canvas);

    this.hoverLine = document.createElement("div");
    this.hoverLine.className = "daam-hover";
    this.hoverLine.textContent = "hover the image";
    this.container.appendChild(this.hoverLine);

    this.tagRow = document.createElement("div");
    this.tagRow.className = "daam-tags";
    this.container.appendChild(this.tagRow);

    this.strength = this.addSlider("strength", 0, 1, 0.05, 0.6);
    this.smooth = this.addSlider("smooth", 0, 4, 1, 0);
    this.view = this.addViewToggle();

    this.state = null;
    this.pinned = new Set();
    this.pointer = null;

    this.canvas.addEventListener("pointermove", (event) => this.onPointer(event));
    this.canvas.addEventListener("pointerleave", () => {
      this.pointer = null;
      this.draw();
    });
    this.canvas.addEventListener("pointerdown", (event) => this.onClick(event));
  }

  addSlider(label, min, max, step, value) {
    const row = document.createElement("div");
    row.className = "daam-row";
    const name = document.createElement("span");
    name.textContent = label;
    name.style.minWidth = "46px";
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(value);
    input.addEventListener("input", () => this.draw());
    row.appendChild(name);
    row.appendChild(input);
    this.container.appendChild(row);
    return input;
  }

  addViewToggle() {
    const row = document.createElement("div");
    row.className = "daam-row";
    const select = document.createElement("select");
    select.style.flex = "1";
    for (const [value, label] of [["heatmap", "heatmap"], ["mask", "mask"], ["labels", "labels"]]) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    }
    select.addEventListener("change", () => this.draw());
    row.appendChild(select);
    this.container.appendChild(row);
    return select;
  }

  /** Called from onExecuted with the node's ui payload. */
  async load(payload) {
    const id = payload.node;
    const [imageResponse, mapResponse] = await Promise.all([
      fetch(api.apiURL(`/daam_mixed/${id}/image.png`)),
      fetch(api.apiURL(`/daam_mixed/${id}/maps.npy`)),
    ]);
    if (!imageResponse.ok || !mapResponse.ok) {
      this.hoverLine.textContent = "maps expired - run the node again";
      return;
    }

    const blob = await imageResponse.blob();
    const bitmap = await createImageBitmap(blob);
    const maps = parseNpy(await mapResponse.arrayBuffer());

    this.state = { bitmap, maps, tags: payload.tags || [], smoothed: maps };
    this.canvas.height = Math.max(
      96,
      Math.round((PANEL_WIDTH * bitmap.height) / Math.max(1, bitmap.width))
    );
    this.pinned.clear();
    this.buildTags();
    this.draw();
  }

  buildTags() {
    this.tagRow.textContent = "";
    if (!this.state) return;
    this.state.tags.forEach((label, index) => {
      const chip = document.createElement("span");
      chip.className = "daam-tag";
      chip.textContent = label;
      chip.title = label;
      chip.addEventListener("click", (event) => {
        if (this.pinned.has(index)) {
          this.pinned.delete(index);
        } else {
          if (!event.ctrlKey && !event.shiftKey && !event.metaKey) this.pinned.clear();
          this.pinned.add(index);
        }
        this.syncChips();
        this.draw();
      });
      this.tagRow.appendChild(chip);
    });
    this.syncChips();
  }

  syncChips() {
    [...this.tagRow.children].forEach((chip, index) => {
      chip.classList.toggle("on", this.pinned.has(index));
    });
  }

  pointerToUv(event) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      u: (event.clientX - rect.left) / Math.max(1, rect.width),
      v: (event.clientY - rect.top) / Math.max(1, rect.height),
    };
  }

  onPointer(event) {
    if (!this.state) return;
    this.pointer = this.pointerToUv(event);
    const ranked = topTags(this.state.smoothed, this.pointer.u, this.pointer.v, 3, 0.02);
    this.hoverLine.textContent = "";
    if (!ranked.length) {
      this.hoverLine.textContent = "no tag claims this pixel";
    } else {
      ranked.forEach((entry, rank) => {
        if (rank) this.hoverLine.append(" · ");
        const strong = document.createElement("b");
        strong.textContent = this.state.tags[entry.index] || `tag ${entry.index}`;
        this.hoverLine.append(strong, ` ${(entry.value * 100).toFixed(0)}%`);
      });
    }
    this.draw();
  }

  onClick(event) {
    if (!this.state) return;
    const { u, v } = this.pointerToUv(event);
    const ranked = topTags(this.state.smoothed, u, v, 1, 0.02);
    if (!ranked.length) {
      this.pinned.clear();
      this.syncChips();
      this.draw();
      return;
    }
    const index = ranked[0].index;
    if (!event.ctrlKey && !event.shiftKey && !event.metaKey) {
      const already = this.pinned.has(index) && this.pinned.size === 1;
      this.pinned.clear();
      if (!already) this.pinned.add(index);
    } else if (this.pinned.has(index)) {
      this.pinned.delete(index);
    } else {
      this.pinned.add(index);
    }
    this.syncChips();
    this.draw();
  }

  draw() {
    if (!this.state) return;
    const { bitmap, maps } = this.state;
    const radius = Number(this.smooth.value) || 0;
    if (this._radius !== radius) {
      this.state.smoothed = blurMaps(maps, radius);
      this._radius = radius;
    }

    const context = this.canvas.getContext("2d");
    const { width, height } = this.canvas;
    context.clearRect(0, 0, width, height);
    context.drawImage(bitmap, 0, 0, width, height);

    const frame = context.getImageData(0, 0, width, height);
    renderFrame(frame, this.state.smoothed, {
      pinned: [...this.pinned],
      strength: Number(this.strength.value),
      view: this.view.value,
      palette: PALETTE,
    });
    context.putImageData(frame, 0, 0);

    if (this.view.value === "heatmap") {
      // colour bar, top right
      const barWidth = 8;
      const barHeight = Math.min(96, height * 0.4);
      const x0 = width - barWidth - 6;
      const y0 = 6;
      for (let y = 0; y < barHeight; y += 1) {
        const [r, g, b] = jet(1 - y / barHeight);
        context.fillStyle = `rgb(${r},${g},${b})`;
        context.fillRect(x0, y0 + y, barWidth, 1);
      }
    }

    if (this.pointer) {
      const px = Math.round(this.pointer.u * width);
      const py = Math.round(this.pointer.v * height);
      context.strokeStyle = "rgba(255,255,255,0.7)";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(px - 6, py);
      context.lineTo(px + 6, py);
      context.moveTo(px, py - 6);
      context.lineTo(px, py + 6);
      context.stroke();

      if (this.pinned.size === 1) {
        const [index] = [...this.pinned];
        const value = sample(this.state.smoothed, index, this.pointer.u, this.pointer.v);
        context.fillStyle = "rgba(0,0,0,0.55)";
        context.fillRect(4, height - 20, 118, 16);
        context.fillStyle = "#fff";
        context.font = "11px monospace";
        context.fillText(`tag ${index}: ${(value * 100).toFixed(0)}%`, 8, height - 8);
      }
    }
  }
}

app.registerExtension({
  name: "DAAM.Mixed.Explorer",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "DAAMExplorer") return;
    injectStyle();

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function (...args) {
      const result = onNodeCreated?.apply(this, args);
      this.daamPanel = new DaamPanel(this);
      this.addDOMWidget("daam_panel", "daam_panel", this.daamPanel.container, { serialize: false });
      this.setSize([PANEL_WIDTH + 24, 560]);
      return result;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message, ...rest) {
      onExecuted?.apply(this, [message, ...rest]);
      const payload = message?.daam?.[0];
      if (payload) this.daamPanel?.load(payload);
    };
  },
});
