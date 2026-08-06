#!/usr/bin/env python3
"""Run SynthSeg on one MRI and save an MRI/mask/overlay figure.

Usage:
    /home/denizberkin/miniconda3/envs/mri/bin/python \
        scripts/visualize_synthseg.py
"""

import argparse
import csv
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from nibabel.processing import resample_from_to
from skimage.measure import marching_cubes


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = Path(
    "/mnt/c/Users/berki/work/large_datasets/mri-fields/"
    "Validating_prospective/T1W/7T/P_T1W_7T_0016.nii.gz"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/denizberkin/projects/mri-fields/outputs/visuals"
)
SYNTHSEG_PYTHON = ROOT / ".conda" / "synthseg" / "bin" / "python"
SYNTHSEG_DIR = ROOT / "SynthSeg"
SEGMENT_SCRIPT = ROOT / "official" / "MRIxFields2026" / "Evaluation" / "segment.py"
CHALLENGE_LABELS = {
    10: "L Thalamus", 49: "R Thalamus",
    11: "L Caudate", 50: "R Caudate",
    12: "L Putamen", 51: "R Putamen",
    13: "L Pallidum", 52: "R Pallidum",
    17: "L Hippocampus", 53: "R Hippocampus",
    18: "L Amygdala", 54: "R Amygdala",
    26: "L Accumbens", 58: "R Accumbens",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--slice", type=int, default=None, help="Axial slice index; default selects the largest segmented area")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"MRI not found: {args.image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.name.removesuffix(".nii.gz")
    seg_path = args.output_dir / f"{stem}_seg.nii.gz"
    figure_path = args.output_dir / f"{stem}_synthseg.png"
    figure_3d_path = args.output_dir / f"{stem}_synthseg_3d.png"
    volume_figure_path = args.output_dir / f"{stem}_synthseg_volumes.png"
    volume_csv_path = args.output_dir / f"{stem}_synthseg_volumes.csv"

    if args.overwrite and seg_path.exists():
        seg_path.unlink()
    if not seg_path.exists():
        with tempfile.TemporaryDirectory() as temporary_input:
            Path(temporary_input, args.image.name).symlink_to(args.image.resolve())
            env = os.environ.copy()
            env["SYNTHSEG_DIR"] = str(SYNTHSEG_DIR)
            subprocess.run(
                [
                    str(SYNTHSEG_PYTHON),
                    str(SEGMENT_SCRIPT),
                    "--input_dir",
                    temporary_input,
                    "--output_dir",
                    str(args.output_dir),
                ],
                check=True,
                env=env,
            )

    image = nib.as_closest_canonical(nib.load(str(args.image)))
    native_seg = nib.as_closest_canonical(nib.load(str(seg_path)))
    native_seg_data = np.rint(native_seg.get_fdata()).astype(np.int16)
    seg = np.rint(resample_from_to(native_seg, image, order=0).get_fdata()).astype(np.int16)
    mri = image.get_fdata(dtype=np.float32)

    slice_index = args.slice
    if slice_index is None:
        slice_index = int(np.count_nonzero(seg, axis=(0, 1)).argmax())
    if not 0 <= slice_index < mri.shape[2]:
        parser.error(f"--slice must be between 0 and {mri.shape[2] - 1}")

    mri_slice = mri[:, :, slice_index].T
    seg_slice = seg[:, :, slice_index].T
    visible_seg = np.ma.masked_equal(seg_slice, 0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(mri_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[0].set_title(f"MRI (z={slice_index})")
    mask_plot = axes[1].imshow(visible_seg, cmap="nipy_spectral", origin="lower", interpolation="nearest")
    axes[1].set_title("SynthSeg labels")
    axes[2].imshow(mri_slice, cmap="gray", origin="lower", vmin=0, vmax=1)
    axes[2].imshow(visible_seg, cmap="nipy_spectral", origin="lower", interpolation="nearest", alpha=0.45)
    axes[2].set_title("Overlay")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(mask_plot, ax=axes[1], fraction=0.046, pad=0.04, label="SynthSeg label ID")
    fig.suptitle(args.image.name)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    label_dir = SYNTHSEG_DIR / "data" / "labels_classes_priors"
    label_ids = np.load(label_dir / "synthseg_segmentation_labels_2.0.npy")
    names = np.load(label_dir / "synthseg_segmentation_names_2.0.npy")
    label_names = {int(label): str(name) for label, name in zip(label_ids, names)}
    voxel_volume_mm3 = float(np.prod(native_seg.header.get_zooms()[:3]))
    volume_rows = []
    for label in np.unique(native_seg_data):
        if label == 0:
            continue
        voxel_count = int(np.count_nonzero(native_seg_data == label))
        volume_rows.append((int(label), label_names[int(label)], voxel_count, voxel_count * voxel_volume_mm3))

    with volume_csv_path.open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["label_id", "structure", "voxel_count", "volume_mm3", "volume_ml", "challenge_scored"])
        for label, name, count, volume_mm3 in volume_rows:
            writer.writerow([label, name, count, f"{volume_mm3:.3f}", f"{volume_mm3 / 1000:.3f}", label in CHALLENGE_LABELS])

    fig, axis = plt.subplots(figsize=(10, max(7, len(volume_rows) * 0.32)))
    row_labels = [f"{label}: {name}" for label, name, _, _ in volume_rows]
    volumes_ml = [volume / 1000 for _, _, _, volume in volume_rows]
    colors = ["tab:red" if label in CHALLENGE_LABELS else "tab:blue" for label, *_ in volume_rows]
    axis.barh(row_labels, volumes_ml, color=colors)
    axis.invert_yaxis()
    axis.set_xlabel("Volume (mL)")
    axis.set_title("SynthSeg structure volumes (red = challenge-scored)")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(volume_figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11, 9))
    axis = fig.add_subplot(111, projection="3d")
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(CHALLENGE_LABELS)))
    spacing = native_seg.header.get_zooms()[:3]
    vertices, faces, _, _ = marching_cubes(native_seg_data != 0, level=0.5, spacing=spacing, step_size=2)
    brain_surface = Poly3DCollection(vertices[faces], alpha=0.08, linewidth=0)
    brain_surface.set_facecolor("gray")
    axis.add_collection3d(brain_surface)
    legend = [Patch(facecolor="gray", alpha=0.15, label="Whole brain")]
    for color, (label, name) in zip(colors, CHALLENGE_LABELS.items()):
        mask = native_seg_data == label
        if not mask.any():
            continue
        vertices, faces, _, _ = marching_cubes(mask, level=0.5, spacing=spacing, step_size=2)
        surface = Poly3DCollection(vertices[faces], alpha=0.85, linewidth=0)
        surface.set_facecolor(color)
        axis.add_collection3d(surface)
        legend.append(Patch(facecolor=color, label=f"{label}: {name}"))
    extent = np.asarray(native_seg_data.shape) * np.asarray(spacing)
    axis.set_xlim(0, extent[0])
    axis.set_ylim(0, extent[1])
    axis.set_zlim(0, extent[2])
    axis.set_box_aspect(extent)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_zlabel("z (mm)")
    axis.view_init(elev=22, azim=-65)
    axis.set_title("3D SynthSeg challenge structures within whole brain")
    axis.legend(handles=legend, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.savefig(figure_3d_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"MRI:         {args.image} {image.shape}")
    print(f"Segmentation: {seg_path} {native_seg.shape}")
    print(f"2D figure:    {figure_path}")
    print(f"3D figure:    {figure_3d_path}")
    print(f"Volumes:      {volume_figure_path}")
    print(f"Volume CSV:   {volume_csv_path}")


if __name__ == "__main__":
    main()
