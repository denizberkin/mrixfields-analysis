"""Figures for the spectral analysis of the MRIxFields dataset.

Reads the per-volume spectra produced by ``scripts/spectral_analysis.py`` and renders vector PDFs for
the progress report. Also renders the noise/diffusion figures, which are derived from the measured
mean spectrum rather than from the raw volumes.

Usage:
    python scripts/spectral_figures.py --in-dir reports/spectral --out-dir reports/spectral
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.env import get_data_dir
from spectral_analysis import (
    FIELD_STRENGTHS,
    FIT_HIGH,
    FIT_LOW,
    MODALITIES,
    SPLIT_DIRS,
    fit_alpha,
    rapsd,
    square_crop,
)

# Computer Modern so figure text matches the pdflatex document body.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})

FIELD_COLOURS = {
    "0.1T": "#4C3BCF",
    "1.5T": "#2E86AB",
    "3T": "#3EA65A",
    "5T": "#E8A33D",
    "7T": "#D1495B",
}


def load(in_dir: Path) -> dict:
    data = np.load(in_dir / "volume_spectra.npz", allow_pickle=False)
    return {key: data[key] for key in data.files}


def cell_mask(data: dict, split: str, modality: str, field: str) -> np.ndarray:
    return ((data["split"] == split) & (data["modality"] == modality) & (data["field"] == field))


def cell_mean(data: dict, split: str, modality: str, field: str) -> np.ndarray | None:
    mask = cell_mask(data, split, modality, field)
    if not mask.any():
        return None
    return data["mean_log_rapsd"][mask].mean(axis=0)


def save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"  wrote {name}.pdf", flush=True)


def figure_spectrum_panels(out_dir: Path, split: str = "val") -> None:
    """Sample slices with their log-magnitude and phase spectra, at the two extreme field strengths."""
    import nibabel

    data_dir = get_data_dir()
    fig, axes = plt.subplots(3, 2, figsize=(5.2, 7.4))

    for column, field in enumerate(["0.1T", "7T"]):
        pattern = f"{data_dir}/{SPLIT_DIRS[split]}/T1W/{field}/*.nii.gz"
        path = sorted(glob.glob(pattern))[0]
        volume = np.asarray(nibabel.load(path).dataobj, dtype=np.float32)
        fraction = np.array([(volume[:, :, k] > 0).mean() for k in range(volume.shape[2])])
        candidates = np.where(fraction > 0.15)[0]
        slice_2d = square_crop(volume[:, :, candidates[len(candidates) // 2]]).astype(np.float64)

        spectrum = np.fft.fftshift(np.fft.fft2((slice_2d - slice_2d.mean()) / (slice_2d.std() + 1e-12)))

        axes[0, column].imshow(slice_2d.T, cmap="gray", origin="lower")
        axes[0, column].set_title(f"T1W {field}")
        axes[1, column].imshow(np.log(np.abs(spectrum) + 1e-8).T, cmap="viridis", origin="lower")
        axes[1, column].set_title("log magnitude")
        axes[2, column].imshow(np.angle(spectrum).T, cmap="RdBu", vmin=-np.pi, vmax=np.pi, origin="lower")
        axes[2, column].set_title("phase")

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)

    fig.tight_layout()
    save(fig, out_dir, "fig_spectrum_panels")


def figure_rapsd_by_field(data: dict, out_dir: Path, split: str) -> None:
    """Mean RAPSD per field strength, one panel per modality. The central dataset figure."""
    frequencies = data["frequencies"]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5), sharey=True)

    for axis, modality in zip(axes, MODALITIES, strict=True):
        for field in FIELD_STRENGTHS:
            mean_log = cell_mean(data, split, modality, field)
            if mean_log is None:
                continue
            axis.plot(frequencies[1:], np.exp(mean_log[1:]), color=FIELD_COLOURS[field],
                      label=field, linewidth=1.3)
        axis.axvspan(FIT_LOW, FIT_HIGH, color="0.85", zorder=0, linewidth=0)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("spatial frequency (cycles/pixel)")
        axis.set_title(modality)
    axes[0].set_ylabel("power")
    axes[0].legend(title="field", frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, out_dir, f"fig_rapsd_by_field_{split}")


def figure_powerlaw_fit(data: dict, out_dir: Path, split: str) -> None:
    """Mean RAPSD with its fitted power law, for the two extreme field strengths (T1W)."""
    frequencies = data["frequencies"]
    fig, axis = plt.subplots(figsize=(5.0, 3.4))

    for field in ["0.1T", "7T"]:
        mean_log = cell_mean(data, split, "T1W", field)
        if mean_log is None:
            continue
        alpha, intercept, _ = fit_alpha(frequencies, mean_log)
        axis.plot(frequencies[1:], np.exp(mean_log[1:]), color=FIELD_COLOURS[field],
                  linewidth=1.3, label=f"T1W {field}")
        band = np.linspace(np.log(FIT_LOW), np.log(FIT_HIGH), 50)
        axis.plot(np.exp(band), np.exp(-alpha * band + intercept), linestyle="--", color="black",
                  linewidth=1.0, label=rf"$\hat{{\alpha}} = {alpha:.2f}$")

    axis.axvspan(FIT_LOW, FIT_HIGH, color="0.85", zorder=0, linewidth=0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("spatial frequency (cycles/pixel)")
    axis.set_ylabel("power")
    axis.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dir, f"fig_powerlaw_fit_{split}")


def figure_alpha_by_field(data: dict, out_dir: Path, split: str) -> None:
    """Fitted alpha against field strength, with bootstrap intervals over volumes."""
    from spectral_analysis import bootstrap_alpha_ci

    frequencies = data["frequencies"]
    fig, axis = plt.subplots(figsize=(5.0, 3.4))
    offsets = {"T1W": -0.12, "T2W": 0.0, "T2FLAIR": 0.12}
    markers = {"T1W": "o", "T2W": "s", "T2FLAIR": "^"}

    for modality in MODALITIES:
        xs, ys, los, his = [], [], [], []
        for index, field in enumerate(FIELD_STRENGTHS):
            mask = cell_mask(data, split, modality, field)
            if not mask.any():
                continue
            spectra = list(data["mean_log_rapsd"][mask])
            alpha, low, high = bootstrap_alpha_ci(spectra, frequencies)
            xs.append(index + offsets[modality])
            ys.append(alpha)
            los.append(alpha - low)
            his.append(high - alpha)
        axis.errorbar(xs, ys, yerr=[los, his], marker=markers[modality], markersize=4,
                      linewidth=1.2, capsize=2.5, label=modality)

    axis.axhline(2.0, color="0.5", linestyle=":", linewidth=1.0)
    axis.text(0.05, 2.06, "natural images ($\\alpha \\approx 2$)", fontsize=7, color="0.4")
    axis.set_xticks(range(len(FIELD_STRENGTHS)))
    axis.set_xticklabels(FIELD_STRENGTHS)
    axis.set_xlabel("field strength")
    axis.set_ylabel(r"power law exponent $\alpha$")
    axis.legend(frameon=False)
    fig.tight_layout()
    save(fig, out_dir, f"fig_alpha_by_field_{split}")


def figure_local_slope(data: dict, out_dir: Path, split: str) -> None:
    """Local slope of the log-log spectrum, showing that a single exponent is an approximation."""
    frequencies = data["frequencies"]
    fig, axis = plt.subplots(figsize=(5.0, 3.4))

    window = 9
    kernel = np.ones(window) / window
    for field in FIELD_STRENGTHS:
        mean_log = cell_mean(data, split, "T1W", field)
        if mean_log is None:
            continue
        log_frequencies = np.log(frequencies[1:])
        smoothed = np.convolve(mean_log[1:], kernel, mode="valid")
        smoothed_frequencies = np.convolve(log_frequencies, kernel, mode="valid")
        axis.plot(np.exp(smoothed_frequencies), -np.gradient(smoothed, smoothed_frequencies),
                  color=FIELD_COLOURS[field], linewidth=1.3, label=field)

    axis.axvspan(FIT_LOW, FIT_HIGH, color="0.85", zorder=0, linewidth=0)
    axis.set_xscale("log")
    # Beyond ~0.3 the power is small enough that the numerical derivative is dominated by noise.
    axis.set_xlim(0.005, 0.25)
    axis.set_ylim(0, 9)
    axis.set_xlabel("spatial frequency (cycles/pixel)")
    axis.set_ylabel(r"local slope $-\mathrm{d}\log P / \mathrm{d}\log f$")
    axis.legend(title="field", frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, out_dir, f"fig_local_slope_{split}")


def figure_noise_hinge(data: dict, out_dir: Path, split: str) -> None:
    """Adding white noise to the measured spectrum produces the hinge that underlies diffusion."""
    frequencies = data["frequencies"]
    mean_log = cell_mean(data, split, "T1W", "7T")
    power = np.exp(mean_log)

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.8))

    # Slices were standardised, so white noise of standard deviation sigma has flat power sigma^2.
    for sigma, colour in zip([0.05, 0.2, 1.0], ["#2E86AB", "#3EA65A", "#D1495B"], strict=True):
        noise_power = np.full_like(power, sigma ** 2)
        axes[0].plot(frequencies[1:], (power + noise_power)[1:], color=colour, linewidth=1.3,
                     label=rf"$\sigma = {sigma}$")
        axes[0].plot(frequencies[1:], noise_power[1:], color=colour, linewidth=0.7, linestyle=":")
    axes[0].plot(frequencies[1:], power[1:], color="black", linewidth=1.5, label="clean (T1W 7T)")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("spatial frequency (cycles/pixel)")
    axes[0].set_ylabel("power")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("signal plus white noise")

    # Maximal detectable frequency against diffusion time, using sigma(t) = t / (1 - t).
    timesteps = np.linspace(0.001, 0.999, 400)
    for threshold, colour in zip([1.0, 4.0], ["#4C3BCF", "#E8A33D"], strict=True):
        minimum_power = threshold * (timesteps / (1 - timesteps)) ** 2
        detectable = power[:, None] > minimum_power[None, :]
        reverse = np.argmax(detectable[::-1], axis=0)
        maximal = frequencies[::-1][reverse] * np.any(detectable, axis=0)
        axes[1].plot(timesteps, maximal, color=colour, linewidth=1.4, label=rf"SNR $> {threshold:.0f}$")
    axes[1].set_xlabel("diffusion time $t$")
    axes[1].set_ylabel(r"maximal detectable frequency $f_{\max}$")
    axes[1].legend(frameon=False)
    axes[1].set_title("frequencies resolvable at each noise level")

    fig.tight_layout()
    save(fig, out_dir, f"fig_noise_hinge_{split}")


def figure_field_gap(data: dict, out_dir: Path, split: str) -> None:
    """Power ratio of each field strength to 7T: what a translation model must actually synthesise."""
    frequencies = data["frequencies"]
    fig, axis = plt.subplots(figsize=(5.0, 3.4))

    reference = cell_mean(data, split, "T1W", "7T")
    for field in FIELD_STRENGTHS:
        if field == "7T":
            continue
        mean_log = cell_mean(data, split, "T1W", field)
        if mean_log is None:
            continue
        axis.plot(frequencies[1:], np.exp(reference[1:] - mean_log[1:]), color=FIELD_COLOURS[field],
                  linewidth=1.3, label=f"7T / {field}")

    axis.axhline(1.0, color="0.5", linestyle=":", linewidth=1.0)
    axis.axvspan(FIT_LOW, FIT_HIGH, color="0.85", zorder=0, linewidth=0)
    # Past the fit band both spectra are in instrumental roll-off and their ratio is not interpretable.
    axis.axvspan(FIT_HIGH, 0.5, color="#D1495B", alpha=0.07, zorder=0, linewidth=0)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("spatial frequency (cycles/pixel)")
    axis.set_ylabel("power ratio to 7T")
    axis.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, out_dir, f"fig_field_gap_{split}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("reports/spectral"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/spectral"))
    parser.add_argument("--split", default="retro", help="split to draw the aggregate figures from")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = load(args.in_dir)
    available = set(np.unique(data["split"]))
    split = args.split if args.split in available else sorted(available)[0]
    print(f"drawing figures from split '{split}' (available: {sorted(available)})", flush=True)

    try:
        figure_spectrum_panels(args.out_dir)
    except (RuntimeError, OSError, IndexError) as error:
        print(f"  skip fig_spectrum_panels, needs the raw dataset: {error}", flush=True)
    figure_rapsd_by_field(data, args.out_dir, split)
    figure_powerlaw_fit(data, args.out_dir, split)
    figure_alpha_by_field(data, args.out_dir, split)
    figure_local_slope(data, args.out_dir, split)
    figure_noise_hinge(data, args.out_dir, split)
    figure_field_gap(data, args.out_dir, split)
    if "val" in available and split != "val":
        figure_rapsd_by_field(data, args.out_dir, "val")
        figure_alpha_by_field(data, args.out_dir, "val")

    print(f"\nfigures in {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
