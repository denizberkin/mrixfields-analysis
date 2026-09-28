#!/usr/bin/env python3
"""The task slide's figure: one subject, one slice, all five field strengths.

No model is involved, these are the acquisitions themselves. The point of the panel
is what the deck argues later: as the field drops the image loses high frequency
content, and 0.1 T is a different kind of problem from 3 T or 5 T.

Usage:
    ~/anaconda3/envs/mri/bin/python scripts/make_slide_task_figure.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline")]

from mrixfields.data.utils import load_nifti  # noqa: E402
from mrixfields.env import get_data_dir  # noqa: E402
from mrixfields.zclip_constants import Z_CLIP_RANGE  # noqa: E402

FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 200,
})


def brain_bbox(images: list[np.ndarray], margin: int = 4, threshold: float = 0.05):
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
    parser.add_argument("--subject", default="0006")
    parser.add_argument("--modality", default="T1W")
    parser.add_argument("--z-index", type=int, default=sum(Z_CLIP_RANGE) // 2)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()

    split = Path(get_data_dir()) / "training_prospective"
    slices = []
    for field in FIELDS:
        path = (split / args.modality / field
                / f"P_{args.modality}_{field}_{args.subject}.nii.gz")
        volume, _ = load_nifti(path)
        slices.append(volume[:, :, args.z_index])

    box_rows, box_columns = brain_bbox(slices)
    slices = [image[box_rows, box_columns] for image in slices]

    figure, axes = plt.subplots(1, len(FIELDS), figsize=(9.6, 2.35))
    for axis, image, field in zip(axes, slices, FIELDS):
        axis.imshow(np.rot90(image), cmap="gray", vmin=0.0, vmax=1.0,
                    interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_xlabel(field.replace("T", " T"), fontsize=13, labelpad=4)
    figure.subplots_adjust(wspace=0.04, left=0.005, right=0.995, top=0.99, bottom=0.10)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    destination = args.out_dir / "slide_task_fields.pdf"
    figure.savefig(destination)
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
