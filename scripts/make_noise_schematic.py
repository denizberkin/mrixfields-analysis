"""
Draw schematic of white noise masking a power-law spectrum.

Redrawn after Dieleman, "Diffusion is spectral autoregression", as vector output sized for slides. 
Carries no data, only a diagram, so nothing here reads from the spectral data. Output is a tracked
report asset, so it is written straight into reports/spectral/.

Usage:
    python scripts/make_noise_schematic.py --out-dir reports/spectral
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

SIGNAL_COLOUR = "#CF2E2E"
NOISE_COLOUR = "#2B3FD4"
SUM_COLOUR = "#2F6B3A"

# Straight lines in log-log space: the signal falls, the noise floor is flat.
SIGNAL_START, SIGNAL_END = 0.93, 0.06
NOISE_LOW, NOISE_HIGH = 0.30, 0.58
PANELS = ("natural image", "Gaussian noise", "noisy image", "very noisy image")


def signal(x: np.ndarray) -> np.ndarray:
    return SIGNAL_START + (SIGNAL_END - SIGNAL_START) * x


def frame(axis) -> None:
    """Bare axes with arrowheads -- no ticks, because the diagram carries no numbers."""
    for side in ("top", "right", "left", "bottom"):
        axis.spines[side].set_visible(False)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(-0.02, 1.12)
    axis.set_ylim(-0.05, 1.12)
    arrow = dict(arrowstyle="-|>", color="0.25", linewidth=1.1, shrinkA=0, shrinkB=0)
    axis.annotate("", xy=(1.10, 0.0), xytext=(0.0, 0.0), arrowprops=arrow)
    axis.annotate("", xy=(0.0, 1.10), xytext=(0.0, 0.0), arrowprops=arrow)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/spectral"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 1.0, 400)
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 2.5))

    for axis, title in zip(axes, PANELS, strict=True):
        frame(axis)
        axis.set_title(title, color={"natural image": SIGNAL_COLOUR,
                                     "Gaussian noise": NOISE_COLOUR}.get(title, SUM_COLOUR),
                       pad=8)
        axis.set_xlabel("spatial frequency")

    # Panel 1: the signal alone.
    axes[0].plot(x, signal(x), color=SIGNAL_COLOUR, linewidth=2.2)
    axes[0].set_ylabel("power\n(log scale)")

    # Panel 2: white noise alone - flat.
    axes[1].plot(x, np.full_like(x, NOISE_LOW), color=NOISE_COLOUR, linewidth=2.2)

    # Panels 3 and 4: signal plus noise. Adding in linear power space looks, on a log axis, like
    # taking the larger of the two - the knee sits where they cross.
    for axis, floor in ((axes[2], NOISE_LOW), (axes[3], NOISE_HIGH)):
        axis.plot(x, signal(x), color=SIGNAL_COLOUR, linewidth=1.6, alpha=0.22)
        axis.plot(x, np.full_like(x, floor), color=NOISE_COLOUR, linewidth=1.6, alpha=0.22)
        axis.plot(x, np.maximum(signal(x), floor), color=SUM_COLOUR, linewidth=2.4)

    fig.tight_layout()
    for suffix in ("pdf", "png"):
        path = args.out_dir / f"fig_noise_schematic.{suffix}"
        fig.savefig(path)
        print(f"wrote {path}", flush=True)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
