"""
Qualitative Task 3 figures from the 10 epoch conditional U-Net checkpoint.

Renders two figures:
    - slide_task3_sweep.pdf    one row per modality, the four source fields translated to 7 T,
                            with the real 7 T volume in the last column
    - slide_task3_gt.pdf       one row per modality, 0.1 T source / prediction / real 7 T / |error|

Usage (need torch in the env):
    C:/Users/berki/.conda/envs/mri/python.exe scripts/make_task3_qualitative.py \
        --checkpoint task3_unet_finetune_10.pt --out-dir reports/figures
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

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline")]

from components.models.conditional_unet import ConditionalUNet  # noqa: E402
from mrixfields.data.transforms import CenterCropOrPad  # noqa: E402
from mrixfields.data.utils import get_joint_domain, load_nifti  # noqa: E402
from mrixfields.env import get_data_dir  # noqa: E402
from mrixfields.zclip_constants import Z_CLIP_RANGE  # noqa: E402

CROP_SIZE = (368, 448)
MODALITIES = ("T1W", "T2W", "T2FLAIR")
SOURCE_FIELDS = ("0.1T", "1.5T", "3T", "5T")
TARGET_FIELD = "7T"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 10,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def find_volume(data_dir: Path, modality: str, field: str, subject: str) -> Path:
    """Locate one volume, tolerating the split directory's inconsistent casing."""
    name = f"P_{modality}_{field}_{subject}.nii.gz"
    for split in ("training_prospective", "Training_prospective"):
        candidate = data_dir / split / modality / field / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{name} not found under {data_dir}")


def load_slice(path: Path, z_index: int) -> np.ndarray:
    volume, _ = load_nifti(path)
    return np.asarray(volume[:, :, z_index], dtype=np.float32)


def translate(
    model: ConditionalUNet,
    source_slice: np.ndarray,
    modality: str,
    source_field: str,
    target_field: str,
    device: torch.device,
) -> np.ndarray:
    """Single-slice forward pass, mirroring scripts/inference_task3_conditional_unet.py."""
    crop = CenterCropOrPad(CROP_SIZE)
    uncrop = CenterCropOrPad(source_slice.shape)

    image = torch.from_numpy(crop(source_slice)[None, None]).to(device=device, dtype=torch.float32)
    image = image.mul(2).sub(1)
    source = torch.full((1,), get_joint_domain(modality, source_field), device=device, dtype=torch.long)
    target = torch.full((1,), get_joint_domain(modality, target_field), device=device, dtype=torch.long)

    with torch.inference_mode(), torch.autocast(device.type, enabled=device.type == "cuda"):
        prediction = model(image, target, source)

    prediction = prediction.float().cpu().numpy()[0, 0]
    prediction = uncrop(np.clip(prediction, -1, 1) * 0.5 + 0.5)
    return prediction * (source_slice > 1e-6)


def brain_bbox(images: list[np.ndarray], margin: int = 6, threshold: float = 0.05) -> tuple[slice, slice]:
    """Tightest box containing the brain, shared so panels stay comparable.

    The threshold is well above zero on purpose: these volumes are min-max normalised but their
    background is not exactly zero, so a near-zero cutoff selects the whole array.
    """
    mask = np.zeros_like(images[0], dtype=bool)
    for image in images:
        mask |= image > threshold
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    return (
        slice(max(rows[0] - margin, 0), min(rows[-1] + 1 + margin, mask.shape[0])),
        slice(max(columns[0] - margin, 0), min(columns[-1] + 1 + margin, mask.shape[1])),
    )


def crop_all(data: dict) -> None:
    """Crop every stored slice to one shared brain box, in place."""
    every = []
    for entry in data.values():
        every.append(entry["target"])
        every.extend(entry["source"].values())
        every.extend(entry["prediction"].values())
    rows, columns = brain_bbox(every)
    print(f"crop {every[0].shape} -> "
          f"({rows.stop - rows.start}, {columns.stop - columns.start})", flush=True)
    for entry in data.values():
        entry["target"] = entry["target"][rows, columns]
        for field in list(entry["source"]):
            entry["source"][field] = entry["source"][field][rows, columns]
            entry["prediction"][field] = entry["prediction"][field][rows, columns]


def show(ax: plt.Axes, image: np.ndarray, title: str | None = None) -> None:
    ax.imshow(np.rot90(image), cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, pad=4)


def sweep_figure(data: dict, out_dir: Path) -> None:
    """One row per modality: each source field translated to 7 T, plus the real 7 T volume."""
    columns = len(SOURCE_FIELDS) + 1
    fig, axes = plt.subplots(len(MODALITIES), columns, figsize=(2.05 * columns, 2.35 * len(MODALITIES)))

    for row, modality in enumerate(MODALITIES):
        for column, field in enumerate(SOURCE_FIELDS):
            title = f"{field} $\\rightarrow$ 7 T" if row == 0 else None
            show(axes[row, column], data[modality]["prediction"][field], title)
        show(axes[row, -1], data[modality]["target"], "real 7 T" if row == 0 else None)
        axes[row, 0].set_ylabel(modality, fontsize=11)
        axes[row, 0].yaxis.set_visible(True)
        axes[row, 0].set_yticks([])

    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    fig.savefig(out_dir / "slide_task3_sweep.pdf")
    plt.close(fig)


def ground_truth_figure(data: dict, out_dir: Path, source_field: str) -> None:
    """One row per modality: source, prediction, real 7 T, and the absolute error."""
    fig, axes = plt.subplots(len(MODALITIES), 4, figsize=(8.6, 2.35 * len(MODALITIES)))
    titles = (f"{source_field} source", "prediction", "real 7 T", "$|$error$|$")

    for row, modality in enumerate(MODALITIES):
        source = data[modality]["source"][source_field]
        prediction = data[modality]["prediction"][source_field]
        target = data[modality]["target"]
        error = np.abs(prediction - target) * (source > 1e-6)

        for column, image in enumerate((source, prediction, target)):
            show(axes[row, column], image, titles[column] if row == 0 else None)
        axes[row, 3].imshow(np.rot90(error), cmap="magma", vmin=0.0, vmax=0.5,
                            interpolation="nearest")
        axes[row, 3].set_xticks([])
        axes[row, 3].set_yticks([])
        for spine in axes[row, 3].spines.values():
            spine.set_visible(False)
        if row == 0:
            axes[row, 3].set_title(titles[3], pad=4)
        axes[row, 0].set_ylabel(modality, fontsize=11)

    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    fig.savefig(out_dir / "slide_task3_gt.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=Path("task3_unet_finetune_10.pt"))
    parser.add_argument("--subject", default="0006")
    parser.add_argument("--z-index", type=int, default=sum(Z_CLIP_RANGE) // 2)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    start, stop = Z_CLIP_RANGE
    if not start <= args.z_index < stop:
        raise ValueError(f"z-index must lie in the submission slab {Z_CLIP_RANGE}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable: {device}")

    model = ConditionalUNet(base_channels=32, max_channels=512, levels=4)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()

    data_dir = Path(get_data_dir())
    data: dict = {}
    for modality in MODALITIES:
        target = load_slice(find_volume(data_dir, modality, TARGET_FIELD, args.subject), args.z_index)
        entry = {"target": target, "source": {}, "prediction": {}}
        for field in SOURCE_FIELDS:
            source = load_slice(find_volume(data_dir, modality, field, args.subject), args.z_index)
            entry["source"][field] = source
            entry["prediction"][field] = translate(model, source, modality, field, TARGET_FIELD, device)
            print(f"{modality}  {field} -> {TARGET_FIELD}", flush=True)
        data[modality] = entry

    crop_all(data)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sweep_figure(data, args.out_dir)
    ground_truth_figure(data, args.out_dir, SOURCE_FIELDS[0])
    print(f"wrote slide_task3_sweep.pdf and slide_task3_gt.pdf to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
