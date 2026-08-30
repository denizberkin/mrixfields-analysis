"""
Render the bar charts used by reports/progress_slides_2.tex

The numbers are constants of the reports, not recomputed here: 
- control case and Task 3 rows:     Baseline Experiment Logs excel, 
- spectral exponents:               outputs/spectral/alpha_summary.csv
- diversity ratios from the synthetic generator validation reported
reports/progress_report_2.tex.

Usage:
    python scripts/make_slide_figures.py --out-dir reports/figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

BASE = "#8A8F98"      # trained baselines
HIGHLIGHT = "#9C2B2B" # the row that carries the message
ACCENT = "#1F4E79"    # our models
GOOD = "#2F6B3A"

MODALITY_COLOURS = {"T1W": ACCENT, "T2W": "#4E88BF", "T2FLAIR": "#A8C6E0"}


def control_case(out_dir: Path) -> None:
    """Task 1, modality-averaged. The control case returns the input volume unchanged."""
    metrics = [
        ("nRMSE $\\downarrow$", [0.3041, 0.3002, 0.4085]),
        ("SSIM $\\uparrow$", [0.9053, 0.9052, 0.8935]),
        ("LPIPS $\\downarrow$", [0.0750, 0.0764, 0.0897]),
        ("Dice $\\uparrow$", [0.8542, 0.8498, 0.8549]),
        ("Volume $\\uparrow$", [0.8601, 0.8485, 0.8586]),
    ]
    labels = ["CUT", "CycleGAN", "control case"]
    colours = [BASE, BASE, HIGHLIGHT]

    fig, axes = plt.subplots(1, 5, figsize=(12.4, 3.0))
    for ax, (title, values) in zip(axes, metrics, strict=True):
        ax.bar(range(3), values, color=colours, width=0.66)
        ax.set_title(title)
        ax.set_xticks(range(3))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.0)
        for x, v in enumerate(values):
            ax.text(x, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    fig.savefig(out_dir / "slide_control_case.pdf")
    plt.close(fig)


def delta_alpha(out_dir: Path) -> None:
    """alpha_source - alpha_7T, retrospective split. Positive means detail must be added."""
    fields = ["0.1 T", "1.5 T", "3 T", "5 T"]
    values = {
        "T1W":     [1.009, 0.158, -0.171, -0.505],
        "T2W":     [1.502, 0.100, -0.060, -0.214],
        "T2FLAIR": [1.294, 0.350, 0.344, 0.182],
    }

    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    x = np.arange(len(fields))
    width = 0.26
    for i, (modality, vals) in enumerate(values.items()):
        ax.bar(x + (i - 1) * width, vals, width, label=modality, color=MODALITY_COLOURS[modality])
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(fields)
    ax.set_ylabel(r"$\Delta\alpha$ vs 7 T")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_ylim(-0.85, 1.85)
    ax.annotate("detail must be added", xy=(0, 1.70), ha="center", fontsize=9.5, color=HIGHLIGHT)
    ax.annotate("source already sharper", xy=(2.5, -0.72), ha="center", fontsize=9.5, color=BASE)
    fig.savefig(out_dir / "slide_delta_alpha.pdf")
    plt.close(fig)


def diversity(out_dir: Path) -> None:
    """Synthetic / real between-volume sd of tissue means. 1.0 = matches real variability."""
    fields = ["0.1 T", "1.5 T", "3 T", "5 T"]
    before = {"0.1 T": [0.294, 0.273, 0.232], "1.5 T": [0.223, 0.197, 0.157],
              "3 T": [0.409, 0.286, 0.283], "5 T": [0.273, 0.138, 0.108]}
    after = {"0.1 T": [0.983, 0.940, 0.916], "1.5 T": [0.664, 0.638, 0.661],
             "3 T": [0.855, 0.884, 0.898], "5 T": [0.942, 0.916, 0.940]}

    def stats(source: dict[str, list[float]]) -> tuple[np.ndarray, np.ndarray]:
        means = np.array([np.mean(source[f]) for f in fields])
        spans = np.array([[means[i] - min(source[f]), max(source[f]) - means[i]]
                          for i, f in enumerate(fields)]).T
        return means, spans

    before_mean, before_span = stats(before)
    after_mean, after_span = stats(after)

    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    x = np.arange(len(fields))
    width = 0.34
    ax.bar(x - width / 2, before_mean, width, yerr=before_span, capsize=3,
           color=BASE, label="initial generator", error_kw={"lw": 0.9})
    ax.bar(x + width / 2, after_mean, width, yerr=after_span, capsize=3,
           color=GOOD, label="corrected generator", error_kw={"lw": 0.9})
    ax.axhline(1.0, color=HIGHLIGHT, linewidth=1.0, linestyle="--")
    ax.text(3.52, 1.02, "real spread", color=HIGHLIGHT, fontsize=9.5, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(fields)
    ax.set_ylabel("synthetic / real spread")
    ax.set_ylim(0, 1.22)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.savefig(out_dir / "slide_diversity.pdf")
    plt.close(fig)


def task3(out_dir: Path) -> None:
    """Task 3, modality-averaged. FPS-Former at 30 epochs, conditional U-Net at 10.

    The control case is the unmodified input, scored on the platform like any other submission.
    """
    labels = ["control\ncase", "StarGAN v2", "FPS-Former\n(30 ep)", "cond. U-Net"]
    colours = [HIGHLIGHT, BASE, ACCENT, GOOD]
    metrics = [
        ("nRMSE $\\downarrow$", [0.5216, 0.3436, 0.2839, 0.2364], 0.63),
        ("SSIM $\\uparrow$", [0.8365, 0.7397, 0.8649, 0.9014], 1.05),
        ("LPIPS $\\downarrow$", [0.1573, 0.1545, 0.1562, 0.0890], 0.20),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.2))
    for ax, (title, values, top) in zip(axes, metrics, strict=True):
        ax.bar(range(len(values)), np.nan_to_num(values), color=colours, width=0.66)
        ax.set_title(title)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, top)
        for x, v in enumerate(values):
            if np.isnan(v):
                ax.text(x, top * 0.18, "not recorded", ha="center", va="bottom", fontsize=9,
                        color="#555555", rotation=90)
            else:
                ax.text(x, v + top * 0.03, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(out_dir / "slide_task3.pdf")
    plt.close(fig)


def lejepa_loss(out_dir: Path) -> None:
    """Crop panel A, the optimisation objective, out of the four-panel LeJEPA diagnostics figure.

    The source PNG is the training run's own output and is not regenerated here; the box is in its
    pixel coordinates (3094x1984) and covers the panel title, both axis labels and the legend.
    """
    from PIL import Image

    source = Path("loss_curve_through_42kstep.png")
    if not source.is_file():
        print(f"skip lejepa_loss: {source} not found", flush=True)
        return

    with Image.open(source) as image:
        image.crop((31, 85, 1562, 1022)).save(out_dir / "slide_lejepa_loss.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="reports/figures", help="where to write the PDFs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for render in (control_case, delta_alpha, diversity, task3, lejepa_loss):
        render(out_dir)
        print(f"wrote {render.__name__}", flush=True)


if __name__ == "__main__":
    main()
