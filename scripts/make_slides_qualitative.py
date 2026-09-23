#!/usr/bin/env python3
"""Qualitative figures for the MICCAI slide deck, from the shipped Task 3 weights.

Renders the deck's one qualitative panel: the two regimes the spectral analysis
predicts. Each row is a translation to 7 T, shown as

    source (which is also the copy baseline) | our prediction | real 7 T

with the slab SSIM of the copy and of the prediction printed underneath, so the
picture and the leaderboard baseline (0.836497 for the whole identity submission)
are the same story. The source column doubles as the copy baseline because the
identity control *is* the source volume, background-masked by itself.

Inference is not reimplemented here: build_model and predict_slab are imported from
scripts/make_task3_submission.py, so the pixels on the slide come out of the exact
path that produced the 0.913652 upload, TTA included.

Usage:
    ~/anaconda3/envs/mri/bin/python scripts/make_slides_qualitative.py \
        --checkpoint docker/task3/weights/task3.pt --out-dir reports/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline")]

from mrixfields.data.utils import get_joint_domain, load_nifti  # noqa: E402
from mrixfields.env import get_data_dir  # noqa: E402
from mrixfields.zclip_constants import Z_CLIP_RANGE  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from make_task3_submission import build_model, predict_slab  # noqa: E402

MODALITIES = ("T1W", "T2W", "T2FLAIR")
# The two rows of the figure: (source field, its alpha gap to the 7 T target for this
# modality, taken from reports/spectral/alpha_summary.csv, retrospective split).
REGIMES = (("0.1T", "+1.01"), ("3T", "-0.17"))
# Symmetric limit for the residual colour scale, in the volumes' own 0..1 intensity
# units. Measured on this subject the residual runs -0.107..+0.064 at the 0.5/99.5
# percentiles, so 0.25 left the whole map pale; 0.15 spends the colour range on the
# body of the distribution and clips only the brain-edge extremes (max |.| ~0.5).
RESIDUAL_LIMIT = 0.15
TARGET = "7T"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
})


def volume_path(split: Path, modality: str, field: str, subject: str) -> Path:
    path = split / modality / field / f"P_{modality}_{field}_{subject}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def slab_ssim(prediction: np.ndarray, target: np.ndarray, source: np.ndarray) -> float:
    """Challenge-shaped SSIM: submission slab only, background zeroed by the source.

    Same computation as scripts/eval_holdout.py::score with slab_source set, transposed
    to axial-first here because these arrays are still in the NIfTI's (X, Y, Z) order.
    """
    start, stop = Z_CLIP_RANGE
    pred = prediction[:, :, start:stop] * (source[:, :, start:stop] > 1e-3)
    truth = target[:, :, start:stop]
    mask = truth > 1e-6
    return float(np.mean([
        structural_similarity(truth[:, :, z], pred[:, :, z], data_range=1.0)
        for z in range(truth.shape[2]) if mask[:, :, z].any()
    ]))


def brain_bbox(images: list[np.ndarray], margin: int = 4, threshold: float = 0.05):
    """Tightest shared box over every panel, so the rows stay pixel-comparable."""
    mask = np.zeros_like(images[0], dtype=bool)
    for image in images:
        mask |= image > threshold
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return (slice(max(rows[0] - margin, 0), min(rows[-1] + 1 + margin, mask.shape[0])),
            slice(max(columns[0] - margin, 0), min(columns[-1] + 1 + margin, mask.shape[1])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "docker/task3/weights/task3.pt")
    parser.add_argument("--subject", default="0006")
    parser.add_argument("--modality", default="T1W", choices=MODALITIES)
    parser.add_argument("--z-index", type=int, default=sum(Z_CLIP_RANGE) // 2)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/figures")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    start, stop = Z_CLIP_RANGE
    if not start <= args.z_index < stop:
        raise ValueError(f"z-index must lie in the submission slab {Z_CLIP_RANGE}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model("conditional", args.checkpoint, device, "")
    print(f"device={device}", flush=True)

    split = Path(get_data_dir()) / "training_prospective"
    target_volume, _ = load_nifti(volume_path(split, args.modality, TARGET, args.subject))
    print(f"target range [{target_volume.min():.4f}, {target_volume.max():.4f}]", flush=True)

    rows = []
    for field, alpha_gap in REGIMES:
        source_volume, _ = load_nifti(volume_path(split, args.modality, field, args.subject))
        # The other two contrasts at the *source* field, in the fixed (T1W, T2W, T2FLAIR)
        # order the multi-contrast dataset stacks them in; channel 0 repeats the primary.
        auxiliary = [load_nifti(volume_path(split, other, field, args.subject))[0]
                     for other in MODALITIES]
        prediction = predict_slab(
            model, source_volume,
            get_joint_domain(args.modality, field), get_joint_domain(args.modality, TARGET),
            device, args.batch_size, tta=True, auxiliary=auxiliary,
        )
        prediction = prediction * (source_volume > 1e-3)
        copy = source_volume * (source_volume > 1e-3)
        rows.append({
            "field": field,
            "alpha_gap": alpha_gap,
            "source": source_volume[:, :, args.z_index],
            "copy": copy[:, :, args.z_index],
            "prediction": prediction[:, :, args.z_index],
            "target": target_volume[:, :, args.z_index],
            # Signed, and masked to the source's foreground so the air around the head
            # stays at exactly 0 instead of painting a colour the eye reads as error.
            "residual": ((prediction - target_volume) * (source_volume > 1e-3))[:, :, args.z_index],
            "ssim_copy": slab_ssim(copy, target_volume, source_volume),
            "ssim_pred": slab_ssim(prediction, target_volume, source_volume),
        })
        residual = rows[-1]["residual"]
        print(f"{field} -> {TARGET}: copy {rows[-1]['ssim_copy']:.4f} "
              f"ours {rows[-1]['ssim_pred']:.4f} | residual "
              f"p0.5 {np.percentile(residual, 0.5):+.3f} "
              f"p99.5 {np.percentile(residual, 99.5):+.3f} "
              f"max|.| {np.abs(residual).max():.3f}", flush=True)

    every = [image for row in rows for image in
             (row["source"], row["copy"], row["prediction"], row["target"])]
    box_rows, box_columns = brain_bbox(every)
    for row in rows:
        for key in ("source", "copy", "prediction", "target", "residual"):
            row[key] = row[key][box_rows, box_columns]

    # The copy panel is dropped: it is the source volume, so drawing it twice spends a
    # column on a duplicate. Its score still appears under the source.
    # The fourth column is the SIGNED residual, not |error|: the sign is the whole point
    # here. Red means the model put intensity where the target has none, blue means it
    # left detail out, and the two regimes fail in visibly different ways.
    keys = ("source", "prediction", "target", "residual")
    headings = ("source = copy baseline", "our model", "real 7\u2009T",
                "our model $-$ real 7\u2009T")
    figure, axes = plt.subplots(len(rows), 4, figsize=(9.2, 2.55 * len(rows) + 0.7))

    for r, row in enumerate(rows):
        for c, key in enumerate(keys):
            axis = axes[r, c]
            if key == "residual":
                drawn = axis.imshow(np.rot90(row[key]), cmap="RdBu_r",
                                    vmin=-RESIDUAL_LIMIT, vmax=RESIDUAL_LIMIT,
                                    interpolation="nearest")
            else:
                axis.imshow(np.rot90(row[key]), cmap="gray", vmin=0.0, vmax=1.0,
                            interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if r == 0:
                axis.set_title(headings[c], fontsize=10.5, pad=6)
            if c == 0:
                axis.set_xlabel(f"SSIM {row['ssim_copy']:.3f}", fontsize=10, labelpad=4,
                                color="#4A5A6A")
            if c == 1:
                axis.set_xlabel(f"SSIM {row['ssim_pred']:.3f}", fontsize=10, labelpad=4,
                                color="#B03030", fontweight="bold")
        axes[r, 0].set_ylabel(f"{row['field']} $\\rightarrow$ 7\u2009T\n"
                              f"$\\Delta\\alpha={row['alpha_gap']}$",
                              fontsize=11, labelpad=8)
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])

    # One shared bar under the residual column; the two rows share a scale on purpose,
    # so their error magnitudes can be compared by eye.
    bar = figure.colorbar(drawn, ax=axes[:, 3], orientation="horizontal",
                          fraction=0.055, pad=0.04, aspect=26,
                          ticks=[-RESIDUAL_LIMIT, 0, RESIDUAL_LIMIT])
    bar.ax.set_xticklabels([f"$-${RESIDUAL_LIMIT:g}", "0", f"$+${RESIDUAL_LIMIT:g}"],
                           fontsize=9)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2)

    figure.subplots_adjust(wspace=0.05, hspace=0.22)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / "slide_two_regimes.pdf"
    figure.savefig(destination)
    plt.close(figure)
    print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
