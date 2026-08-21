"""
Green bowtie visualizer for SENTRY V3.

WHY this exists:
  Anomaly detectors get accused of being black boxes. Showing neurons
  light up in real time as flows pass through makes it obvious that
  "this is what the model is looking at", which helps both debugging
  and demo-time explaining.

Two modes:
  - draw_architecture(model): static layout (like the reference pic).
  - live_activation_viz(model, X): matplotlib animation, cycles frames,
    color = activation magnitude (dim green -> bright neon green).

We downsample display nodes to <= MAX_DISPLAY per layer, because rendering
78 or 128 circles per column turns into visual mush.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from model import SentryAutoencoderV3

# Green-on-dark palette (matches your reference, swapped yellow -> green)
BG = "#0a0a0a"
EDGE = "#00cc66"
NODE_EDGE = "#00ff88"
TEXT = "#00ff88"
MAX_DISPLAY = 20


def _positions(sizes):
    """Return [(x, y), ...] per layer with equal vertical spacing."""
    layers = []
    for i, s in enumerate(sizes):
        d = min(s, MAX_DISPLAY)
        ys = np.linspace(-1.0, 1.0, d)
        layers.append([(i * 2.0, y) for y in ys])
    return layers


def _draw_edges(ax, layers):
    for i in range(len(layers) - 1):
        for (x1, y1) in layers[i]:
            for (x2, y2) in layers[i + 1]:
                ax.plot([x1, x2], [y1, y2], color=EDGE, alpha=0.12, linewidth=0.4)


def _draw_labels(ax, sizes):
    for i, s in enumerate(sizes):
        ax.text(i * 2.0, -1.45, f"{s}\nnodes", ha="center", va="top",
                color=TEXT, fontsize=9)


def draw_architecture(model: SentryAutoencoderV3, ax=None):
    """Static bowtie diagram."""
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.set_facecolor(BG)

    sizes = model.get_layer_sizes()
    layers = _positions(sizes)
    _draw_edges(ax, layers)
    for layer in layers:
        for (x, y) in layer:
            c = plt.Circle((x, y), 0.08, color="#1a3a1a",
                           ec=NODE_EDGE, lw=1.4, zorder=5)
            ax.add_patch(c)
    _draw_labels(ax, sizes)

    ax.set_xlim(-0.6, len(sizes) * 2 - 1.4)
    ax.set_ylim(-1.9, 1.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("SENTRY V3 Autoencoder", color=TEXT, fontsize=13)
    return ax


def live_activation_viz(model: SentryAutoencoderV3,
                        X_batch: torch.Tensor,
                        save_gif: Optional[str] = None,
                        interval_ms: int = 400):
    """
    Animate the model reacting to each sample in X_batch.
    Set save_gif='out.gif' to export instead of show().
    """
    device = next(model.parameters()).device
    X_batch = X_batch.to(device)
    n_frames = min(len(X_batch), 40)

    # Cache activations up front so animation loop stays cheap
    cached = []
    model.eval()
    with torch.no_grad():
        for i in range(n_frames):
            model(X_batch[i:i + 1])
            cached.append([a.cpu().numpy().flatten() for a in model.last_activations])

    sizes = model.get_layer_sizes()
    layers = _positions(sizes)

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)

    def _normalize(a):
        if a.size == 0 or a.max() == a.min():
            return np.zeros_like(a)
        return (a - a.min()) / (a.max() - a.min())

    def _downsample(a, target):
        if len(a) <= target:
            return a
        idx = np.linspace(0, len(a) - 1, target).astype(int)
        return a[idx]

    def update(frame_idx):
        ax.clear()
        ax.set_facecolor(BG)
        _draw_edges(ax, layers)

        acts = cached[frame_idx]
        for i, layer in enumerate(layers):
            act = _normalize(_downsample(acts[i], len(layer)))
            for j, (x, y) in enumerate(layer):
                v = float(act[j]) if j < len(act) else 0.0
                # Interpolate dark green -> neon green by activation
                color = (v * 0.1, 0.2 + v * 0.8, v * 0.55)
                c = plt.Circle((x, y), 0.08, color=color,
                               ec=NODE_EDGE, lw=1.4, zorder=5)
                ax.add_patch(c)
        _draw_labels(ax, sizes)

        ax.set_xlim(-0.6, len(sizes) * 2 - 1.4)
        ax.set_ylim(-1.9, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"SENTRY V3  —  sample {frame_idx + 1}/{n_frames}",
                     color=TEXT, fontsize=13)

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=interval_ms, repeat=True)
    if save_gif:
        anim.save(save_gif, writer="pillow", fps=1000 // interval_ms)
        print(f"[viz] saved -> {save_gif}")
    else:
        plt.show()
    return anim


if __name__ == "__main__":
    m = SentryAutoencoderV3(input_dim=78)
    draw_architecture(m)
    plt.show()
