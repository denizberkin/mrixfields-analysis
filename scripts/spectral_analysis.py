"""Spectral analysis of the MRIxFields dataset, following Dieleman's spectral view of diffusion.

Computes radially averaged power spectral densities (RAPSDs) of axial slices, aggregates them per
(split, modality, field strength) cell, and fits the power law P(f) proportional to f^-alpha by linear
regression in log-log space.

Reference: https://sander.ai/2024/09/02/spectral-autoregression.html

Usage:
    python scripts/spectral_analysis.py --splits val retro --out-dir outputs/spectral
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.env import get_data_dir

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELD_STRENGTHS = ("0.1T", "1.5T", "3T", "5T", "7T")

SPLIT_DIRS = {
    "val": "Validating_prospective",
    "pro": "training_prospective",
    "retro": "Training_retrospective",
}

# Fit band. Below FIT_LOW the spectrum is dominated by the head-versus-background envelope; above
# FIT_HIGH it is dominated by the acquisition/reconstruction roll-off rather than anatomical content.
FIT_LOW = 0.02
FIT_HIGH = 0.20

# A slice must have at least this fraction of foreground voxels to count as containing brain.
MIN_BRAIN_FRACTION = 0.15

# Foreground is thresholded relative to each volume's own intensity scale rather than at zero. Some
# volumes (notably T2W at 1.5T) carry a small constant non-zero floor over the whole field of view, so
# a naive "greater than zero" test marks every slice as brain, including empty end slices whose
# standardisation is numerically meaningless.
FOREGROUND_FRACTION_OF_P995 = 0.05

# Slices flatter than this (after the foreground test) carry no anatomy and are skipped.
MIN_SLICE_STD = 1e-6


def rapsd(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectral density of a 2D field.

    Follows the same convention as ``pysteps.utils.spectral.rapsd``: the squared FFT magnitude is
    divided by the number of samples, then averaged over integer-radius annuli in the shifted
    spectrum. Frequencies are returned in cycles per pixel, so 0.5 is Nyquist.
    """
    if field.ndim != 2:
        raise ValueError(f"expected a 2D field, got shape {field.shape}")
    height, width = field.shape
    side = min(height, width)

    y, x = np.ogrid[:height, :width]
    radius = np.hypot(x - width // 2, y - height // 2)

    power = np.abs(np.fft.fftshift(np.fft.fft2(field))) ** 2 / field.size

    edges = np.arange(0, side // 2 + 1)
    binned = np.empty(len(edges) - 1)
    for index in range(len(edges) - 1):
        annulus = (radius >= edges[index]) & (radius < edges[index + 1])
        binned[index] = power[annulus].mean()

    return binned, edges[:-1] / float(side)


def square_crop(slice_2d: np.ndarray) -> np.ndarray:
    """Centre-crop to a square so the RAPSD radial binning is isotropic."""
    height, width = slice_2d.shape
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return slice_2d[top:top + side, left:left + side]


def fit_alpha(frequencies: np.ndarray, mean_log_power: np.ndarray,
              low: float = FIT_LOW, high: float = FIT_HIGH) -> tuple[float, float, float]:
    """Fit P(f) proportional to f^-alpha over [low, high]. Returns (alpha, intercept, rms_residual).

    The mean log power is resampled onto equally spaced log-frequencies before fitting so the fit is
    not dominated by high frequencies, where the radial bins are far more numerous.
    """
    band = (frequencies >= low) & (frequencies <= high)
    if band.sum() < 5:
        raise ValueError(f"fit band [{low}, {high}] contains only {band.sum()} bins")

    log_frequencies = np.log(frequencies[band])
    resampled_log_frequencies = np.linspace(log_frequencies[0], log_frequencies[-1], 1000)
    resampled_log_power = np.interp(resampled_log_frequencies, log_frequencies, mean_log_power[band])

    slope, intercept = np.polyfit(resampled_log_frequencies, resampled_log_power, 1)
    residual = resampled_log_power - (slope * resampled_log_frequencies + intercept)
    return -slope, intercept, float(np.sqrt((residual ** 2).mean()))


@dataclass
class VolumeSpectrum:
    """Per-volume aggregate: the mean over slices of the log RAPSD."""

    split: str
    modality: str
    field: str
    subject: str
    mean_log_rapsd: np.ndarray
    num_slices: int


def volume_spectrum(path: str, slices_per_volume: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Mean log RAPSD over evenly spaced brain-containing axial slices of one volume.

    Each slice is standardised to zero mean and unit variance. The power law exponent is invariant to
    this rescaling (it shifts log power by a constant), but it makes the curves directly comparable
    across field strengths, whose raw intensity scales differ.
    """
    import nibabel

    volume = np.asarray(nibabel.load(path).dataobj, dtype=np.float32)
    threshold = FOREGROUND_FRACTION_OF_P995 * np.percentile(volume, 99.5)
    brain_fraction = np.array([(volume[:, :, k] > threshold).mean() for k in range(volume.shape[2])])
    candidates = np.where(brain_fraction > MIN_BRAIN_FRACTION)[0]
    if len(candidates) == 0:
        raise ValueError(f"no brain-containing slices in {path}")

    picks = candidates[np.linspace(0, len(candidates) - 1, min(slices_per_volume, len(candidates))).astype(int)]

    log_spectra = []
    frequencies = None
    for k in picks:
        slice_2d = square_crop(volume[:, :, k]).astype(np.float64)
        deviation = slice_2d.std()
        if deviation < MIN_SLICE_STD:
            continue
        power, frequencies = rapsd((slice_2d - slice_2d.mean()) / deviation)
        log_spectra.append(np.log(power + 1e-30))

    if not log_spectra:
        raise ValueError(f"no usable slices in {path}")

    return np.mean(log_spectra, axis=0), frequencies, len(log_spectra)


def subject_id(path: str) -> str:
    match = re.search(r"_(\d+)\.nii", os.path.basename(path))
    return match.group(1) if match else os.path.basename(path)


def collect(split: str, data_dir: str, max_volumes: int, slices_per_volume: int,
            seed: int = 0) -> tuple[list[VolumeSpectrum], np.ndarray]:
    """Compute per-volume spectra for every (modality, field) cell of one split."""
    rng = np.random.default_rng(seed)
    results: list[VolumeSpectrum] = []
    frequencies = None

    for modality in MODALITIES:
        for field in FIELD_STRENGTHS:
            pattern = os.path.join(data_dir, SPLIT_DIRS[split], modality, field, "*.nii.gz")
            paths = sorted(glob.glob(pattern))
            if not paths:
                print(f"  [{split}] {modality}/{field}: no volumes found", flush=True)
                continue
            if len(paths) > max_volumes:
                paths = [paths[i] for i in sorted(rng.choice(len(paths), max_volumes, replace=False))]

            for path in paths:
                mean_log, frequencies, num_slices = volume_spectrum(path, slices_per_volume)
                results.append(VolumeSpectrum(split, modality, field, subject_id(path), mean_log, num_slices))
            print(f"  [{split}] {modality}/{field}: {len(paths)} volumes", flush=True)

    return results, frequencies


def bootstrap_alpha_ci(spectra: list[np.ndarray], frequencies: np.ndarray,
                       num_resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and 95% bootstrap percentile interval for alpha, resampling over volumes."""
    rng = np.random.default_rng(seed)
    stacked = np.array(spectra)
    point, _, _ = fit_alpha(frequencies, stacked.mean(axis=0))

    draws = np.empty(num_resamples)
    for i in range(num_resamples):
        sample = stacked[rng.integers(0, len(stacked), len(stacked))]
        draws[i], _, _ = fit_alpha(frequencies, sample.mean(axis=0))
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["val"], choices=sorted(SPLIT_DIRS))
    parser.add_argument("--max-volumes", type=int, default=30, help="max volumes sampled per cell")
    parser.add_argument("--slices-per-volume", type=int, default=12)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/spectral"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data_dir = get_data_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    everything: list[VolumeSpectrum] = []
    frequencies = None
    for split in args.splits:
        print(f"collecting {split} ...", flush=True)
        results, frequencies = collect(split, data_dir, args.max_volumes, args.slices_per_volume, args.seed)
        everything.extend(results)

    if not everything:
        raise SystemExit("no volumes processed")

    # Persist raw per-volume spectra so figures can be redrawn without recomputing.
    np.savez_compressed(
        args.out_dir / "volume_spectra.npz",
        frequencies=frequencies,
        mean_log_rapsd=np.array([v.mean_log_rapsd for v in everything]),
        split=np.array([v.split for v in everything]),
        modality=np.array([v.modality for v in everything]),
        field=np.array([v.field for v in everything]),
        subject=np.array([v.subject for v in everything]),
    )

    rows = []
    for split in args.splits:
        for modality in MODALITIES:
            for field in FIELD_STRENGTHS:
                cell = [v.mean_log_rapsd for v in everything
                        if v.split == split and v.modality == modality and v.field == field]
                if not cell:
                    continue
                alpha, low, high = bootstrap_alpha_ci(cell, frequencies, seed=args.seed)
                _, _, rms = fit_alpha(frequencies, np.array(cell).mean(axis=0))
                rows.append({
                    "split": split, "modality": modality, "field": field, "num_volumes": len(cell),
                    "alpha": round(alpha, 4), "ci_low": round(low, 4), "ci_high": round(high, 4),
                    "rms_residual": round(rms, 4),
                })
                print(f"{split:6} {modality:8} {field:5}  alpha={alpha:.3f} "
                      f"[{low:.3f}, {high:.3f}]  n={len(cell)}", flush=True)

    with open(args.out_dir / "alpha_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"fit_band": [FIT_LOW, FIT_HIGH], "rows": rows}, handle, indent=2)

    with open(args.out_dir / "alpha_summary.csv", "w", encoding="utf-8") as handle:
        handle.write("split,modality,field,num_volumes,alpha,ci_low,ci_high,rms_residual\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in
                                  ["split", "modality", "field", "num_volumes",
                                   "alpha", "ci_low", "ci_high", "rms_residual"]) + "\n")

    print(f"\nwrote {args.out_dir}/alpha_summary.csv and volume_spectra.npz", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
