#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter


FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
MODALITIES = ("T1W", "T2W", "T2FLAIR")
CASES = (("0.1T", "0001"), ("7T", "0016"))
SLAB_START = 150


def robust_limit(images: list[np.ndarray], percentile: float, mask: np.ndarray) -> float:
    values = np.concatenate([np.abs(image[mask]) for image in images])
    return max(float(np.percentile(values, percentile)), 1e-6)


def brain_crop(mask: np.ndarray, margin: int = 8) -> tuple[slice, slice]:
    rows, columns = np.where(mask)
    if not len(rows):
        return slice(None), slice(None)
    return (
        slice(max(0, rows.min() - margin), min(mask.shape[0], rows.max() + margin + 1)),
        slice(max(0, columns.min() - margin), min(mask.shape[1], columns.max() + margin + 1)),
    )


def make_panel(
    data_root: Path,
    submission_root: Path,
    output_dir: Path,
    modality: str,
    source: str,
    subject_id: str,
    z_index: int,
    model_label: str,
) -> Path:
    source_path = data_root / modality / source / f"P_{modality}_{source}_{subject_id}.nii.gz"
    source_slice = np.asarray(nib.load(str(source_path)).dataobj[:, :, z_index], dtype=np.float32)
    targets = [field for field in FIELDS if field != source]
    predictions = []
    for target in targets:
        path = (
            submission_root
            / modality
            / f"{source}_to_{target}"
            / "pred"
            / f"P_{modality}_{target}_{subject_id}.nii.gz"
        )
        predictions.append(
            np.asarray(nib.load(str(path)).dataobj[:, :, z_index - SLAB_START], dtype=np.float32)
        )

    images = [source_slice, *predictions]
    mask = source_slice > 1e-6
    crop = brain_crop(mask)
    positive = np.concatenate([image[mask] for image in images])
    intensity_min, intensity_max = np.percentile(positive, (0.5, 99.5))
    differences = [(prediction - source_slice) * mask for prediction in predictions]
    difference_limit = robust_limit(differences, 99.0, mask)
    details = [(image - gaussian_filter(image, sigma=2)) * mask for image in images]
    detail_limit = robust_limit(details, 99.0, mask)

    fig, axes = plt.subplots(3, 5, figsize=(18, 11), constrained_layout=True)
    labels = [f"Source {source}", *[f"Predicted {target}" for target in targets]]
    for column, (image, label) in enumerate(zip(images, labels, strict=True)):
        axes[0, column].imshow(
            image[crop].T,
            cmap="gray",
            origin="lower",
            vmin=intensity_min,
            vmax=intensity_max,
        )
        axes[0, column].set_title(label, fontweight="bold")
        detail_plot = axes[2, column].imshow(
            details[column][crop].T,
            cmap="coolwarm",
            origin="lower",
            vmin=-detail_limit,
            vmax=detail_limit,
        )

    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5,
        0.5,
        "Signed change\nPrediction - Source\n\nRed: increased intensity\nBlue: decreased intensity",
        ha="center",
        va="center",
        fontsize=13,
        transform=axes[1, 0].transAxes,
    )
    for column, difference in enumerate(differences, start=1):
        difference_plot = axes[1, column].imshow(
            difference[crop].T,
            cmap="coolwarm",
            origin="lower",
            vmin=-difference_limit,
            vmax=difference_limit,
        )
        changed = np.abs(difference[mask])
        axes[1, column].set_title(
            f"MAE {changed.mean():.4f} | p99 {np.percentile(changed, 99):.4f}", fontsize=10
        )

    for row, row_label in enumerate(("Shared intensity window", "Amplified signed change", "High-pass detail")):
        axes[row, 0].set_ylabel(row_label, fontsize=12, fontweight="bold")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.colorbar(difference_plot, ax=axes[1, 1:], shrink=0.75, label="Intensity change")
    fig.colorbar(detail_plot, ax=axes[2, :], shrink=0.75, label="High-pass response")
    fig.suptitle(
        f"Task 3 {model_label} | {modality} | subject {subject_id} | axial z={z_index}",
        fontsize=16,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{modality}_{source}_subject-{subject_id}_z{z_index}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-index", type=int, default=165)
    parser.add_argument("--model-label", default="conditional U-Net")
    args = parser.parse_args()
    if not SLAB_START <= args.z_index < 180:
        raise ValueError("z-index must be within the submitted [150,180) slab")

    for modality in MODALITIES:
        for source, subject_id in CASES:
            path = make_panel(
                args.data_root,
                args.submission_root,
                args.output_dir,
                modality,
                source,
                subject_id,
                args.z_index,
                args.model_label,
            )
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
