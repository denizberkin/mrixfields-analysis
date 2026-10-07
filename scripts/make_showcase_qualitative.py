#!/usr/bin/env python3
"""Residual-map grids for the showcase deck, from the shipped Task 3 weights.

The MICCAI deck's qualitative panel (`make_slides_qualitative.py`) shows two transitions,
both to 7 T and both T1W. This extends the same layout to every modality and to both
directions of travel:

    source | our model | target GT | our model - target GT  (signed, shared scale)

Two figures, three rows each, one modality per row:
    toward  the target is sharper than the source (mapping up in field)
    away    the target is smoother than the source (mapping down in field)

Each figure is written once per reveal step (`qual_<set>_<k>.pdf` shows rows 1..k, the rest
blank but laid out), so a beamer `\\only<k>` swap keeps every row pixel-aligned.

`--figures grids` (the default) instead draws one 3 x 5 grid per direction: rows are the three
contrasts, column 1 the source, columns 2-5 our prediction at each other field, with the slab
SSIM under each. `grid_<dir>_<k>.pdf` shows rows 1..k; `grid_<dir>_res.pdf` the same grid as
prediction minus target. Directions: `up` from 0.1 T, `down` from 7 T, so the two slides span
the whole field range.

Inference and scoring come from the MICCAI script, which imports them from
`make_task3_submission.py`: the pixels are the ones the 0.913652 submission produced,
TTA on. Subject 0006 is a training subject (all three paired subjects are).

    ~/anaconda3/envs/mri/bin/python scripts/make_showcase_qualitative.py [--figures rows|grids|all]
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
sys.path.insert(0, str(ROOT / "scripts"))
from make_slides_qualitative import (  # noqa: E402
    MODALITIES, RESIDUAL_LIMIT, Z_CLIP_RANGE, brain_bbox, build_model, get_data_dir,
    get_joint_domain, load_nifti, predict_slab, slab_ssim, volume_path)

# (modality, source, target): one modality per row, field pairs spread over the grid
SETS = {
    "toward": (("T1W", "3T", "5T"), ("T2W", "0.1T", "3T"), ("T2FLAIR", "1.5T", "7T")),
    "away": (("T1W", "7T", "0.1T"), ("T2W", "5T", "1.5T"), ("T2FLAIR", "3T", "0.1T")),
}
HEADINGS = ("source", "our model", "target GT", "our model $-$ target GT")
# (source field, targets in column order): to every higher field, and to every lower one
GRIDS = {
    "up": ("0.1T", ("1.5T", "3T", "5T", "7T")),
    "down": ("7T", ("5T", "3T", "1.5T", "0.1T")),
}


def field_label(field: str) -> str:
    return field.replace("T", " T")


def compute_rows(model, device, subject: str, z: int, batch_size: int, transitions):
    split = Path(get_data_dir()) / "training_prospective"
    rows = []
    for modality, source_field, target_field in transitions:
        source, _ = load_nifti(volume_path(split, modality, source_field, subject))
        target, _ = load_nifti(volume_path(split, modality, target_field, subject))
        auxiliary = [load_nifti(volume_path(split, other, source_field, subject))[0]
                     for other in MODALITIES]
        mask = source > 1e-3
        prediction = predict_slab(
            model, source, get_joint_domain(modality, source_field),
            get_joint_domain(modality, target_field), device, batch_size,
            tta=True, auxiliary=auxiliary) * mask
        rows.append({
            "label": f"{modality}\n{field_label(source_field)} $\\rightarrow$ "
                     f"{field_label(target_field)}",
            "source": (source * mask)[:, :, z],
            "prediction": prediction[:, :, z],
            "target": target[:, :, z],
            "residual": ((prediction - target) * mask)[:, :, z],
            "ssim_copy": slab_ssim(source * mask, target, source),
            "ssim_pred": slab_ssim(prediction, target, source),
        })
        print(f"{modality} {source_field}->{target_field}: copy {rows[-1]['ssim_copy']:.4f} "
              f"ours {rows[-1]['ssim_pred']:.4f}", flush=True)
    box_rows, box_columns = brain_bbox(
        [row[k] for row in rows for k in ("source", "prediction", "target")])
    for row in rows:
        for key in ("source", "prediction", "target", "residual"):
            row[key] = row[key][box_rows, box_columns]
    return rows


def draw(rows, shown: int, destination: Path) -> None:
    figure, axes = plt.subplots(len(rows), 4, figsize=(9.2, 2.45 * len(rows) + 0.7))
    keys = ("source", "prediction", "target", "residual")
    drawn = None
    for r, row in enumerate(rows):
        for c, key in enumerate(keys):
            axis = axes[r, c]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if r == 0:
                axis.set_title(HEADINGS[c], fontsize=10.5, pad=6)
            if r >= shown:
                continue                      # laid out, left blank: keeps rows aligned
            if key == "residual":
                drawn = axis.imshow(np.rot90(row[key]), cmap="RdBu_r", vmin=-RESIDUAL_LIMIT,
                                    vmax=RESIDUAL_LIMIT, interpolation="nearest")
            else:
                axis.imshow(np.rot90(row[key]), cmap="gray", vmin=0.0, vmax=1.0,
                            interpolation="nearest")
            if c == 0:
                axis.set_xlabel(f"copy SSIM {row['ssim_copy']:.3f}", fontsize=9.5,
                                labelpad=3, color="#4A5A6A")
                axis.set_ylabel(row["label"], fontsize=10.5, labelpad=8)
            if c == 1:
                axis.set_xlabel(f"SSIM {row['ssim_pred']:.3f}", fontsize=9.5, labelpad=3,
                                color="#B03030", fontweight="bold")
    figure.subplots_adjust(wspace=0.05, hspace=0.24)
    column = axes[-1, 3].get_position()
    bar_axes = figure.add_axes([column.x0 + 0.1 * column.width, column.y0 - 0.045,
                                0.8 * column.width, 0.014])
    if drawn is None:                         # nothing drawn yet: still reserve the bar
        drawn = plt.cm.ScalarMappable(cmap="RdBu_r",
                                      norm=plt.Normalize(-RESIDUAL_LIMIT, RESIDUAL_LIMIT))
    bar = figure.colorbar(drawn, cax=bar_axes, orientation="horizontal",
                          ticks=[-RESIDUAL_LIMIT, 0, RESIDUAL_LIMIT])
    bar.ax.set_xticklabels([f"$-${RESIDUAL_LIMIT:g} too dark", "0",
                            f"$+${RESIDUAL_LIMIT:g} too bright"], fontsize=8.5)
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, pad=2)
    figure.savefig(destination)
    plt.close(figure)


def compute_grid(model, device, subject: str, z: int, batch_size: int, source_field: str,
                 targets) -> list[dict]:
    split = Path(get_data_dir()) / "training_prospective"
    rows = []
    for modality in MODALITIES:
        volumes = {f: load_nifti(volume_path(split, modality, f, subject))[0]
                   for f in (source_field, *targets)}
        auxiliary = [load_nifti(volume_path(split, other, source_field, subject))[0]
                     for other in MODALITIES]
        source = volumes[source_field]
        mask = source > 1e-3
        cells = []
        for target_field in targets:
            prediction = predict_slab(
                model, source, get_joint_domain(modality, source_field),
                get_joint_domain(modality, target_field), device, batch_size,
                tta=True, auxiliary=auxiliary) * mask
            cells.append({
                "target": target_field,
                "prediction": prediction[:, :, z],
                "residual": ((prediction - volumes[target_field]) * mask)[:, :, z],
                "ssim": slab_ssim(prediction, volumes[target_field], source),
            })
            print(f"{modality} {source_field}->{target_field}: ours {cells[-1]['ssim']:.4f}",
                  flush=True)
        rows.append({"modality": modality, "source": (source * mask)[:, :, z], "cells": cells})
    box_rows, box_columns = brain_bbox(
        [row["source"] for row in rows] + [c["prediction"] for row in rows for c in row["cells"]])
    for row in rows:
        row["source"] = row["source"][box_rows, box_columns]
        for cell in row["cells"]:
            cell["prediction"] = cell["prediction"][box_rows, box_columns]
            cell["residual"] = cell["residual"][box_rows, box_columns]
    return rows


def draw_grid(rows, source_field: str, shown: int, residual: bool, destination: Path) -> None:
    columns = 1 + len(rows[0]["cells"])
    figure, axes = plt.subplots(len(rows), columns, figsize=(7.4, 4.9))
    drawn = None
    for r, row in enumerate(rows):
        for c in range(columns):
            axis = axes[r, c]
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            if r == 0:
                if c == 0:
                    title = f"source {field_label(source_field)}"
                else:
                    title = f"\u2192 {field_label(row['cells'][c - 1]['target'])}"
                axis.set_title(title, fontsize=12, pad=4)
            if c == 0:                        # label appears with its row; alpha 0 keeps layout
                axis.set_ylabel(row["modality"], fontsize=12, labelpad=6,
                                alpha=1.0 if r < shown else 0.0)
            if r >= shown:
                continue                      # laid out, left blank: keeps rows aligned
            if c == 0:
                axis.imshow(np.rot90(row["source"]), cmap="gray", vmin=0.0, vmax=1.0,
                            interpolation="nearest")
                continue
            cell = row["cells"][c - 1]
            if residual:
                drawn = axis.imshow(np.rot90(cell["residual"]), cmap="RdBu_r",
                                    vmin=-RESIDUAL_LIMIT, vmax=RESIDUAL_LIMIT,
                                    interpolation="nearest")
            else:
                axis.imshow(np.rot90(cell["prediction"]), cmap="gray", vmin=0.0, vmax=1.0,
                            interpolation="nearest")
            axis.set_xlabel(f"SSIM {cell['ssim']:.3f}", fontsize=10.5, labelpad=2,
                            color="#B03030")
    figure.subplots_adjust(wspace=0.04, hspace=0.20, left=0.06, right=0.995, top=0.93,
                           bottom=0.10)
    # colour bar under the prediction columns, drawn only in the residual version but laid
    # out in every version so the image grid does not move between build steps
    first, last = axes[-1, 1].get_position(), axes[-1, -1].get_position()
    bar_axes = figure.add_axes([first.x0 + 0.25 * (last.x1 - first.x0), 0.025,
                                0.5 * (last.x1 - first.x0), 0.018])
    if residual and drawn is not None:
        bar = figure.colorbar(drawn, cax=bar_axes, orientation="horizontal",
                              ticks=[-RESIDUAL_LIMIT, 0, RESIDUAL_LIMIT])
        bar.ax.set_xticklabels([f"$-${RESIDUAL_LIMIT:g} too dark", "0",
                                f"$+${RESIDUAL_LIMIT:g} too bright"], fontsize=9.5)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, pad=2)
    else:
        bar_axes.set_axis_off()
    figure.savefig(destination)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "docker/task3/weights/task3.pt")
    parser.add_argument("--subject", default="0006")
    parser.add_argument("--z-index", type=int, default=sum(Z_CLIP_RANGE) // 2)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/figures/showcase")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--figures", choices=("rows", "grids", "all"), default="grids",
                        help="rows: the one-transition-per-row figures; grids: the 3 x 5 grids")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model("conditional", args.checkpoint, device, "")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.figures in ("grids", "all"):
        for name, (source_field, targets) in GRIDS.items():
            rows = compute_grid(model, device, args.subject, args.z_index, args.batch_size,
                                source_field, targets)
            for shown in range(1, len(rows) + 1):
                draw_grid(rows, source_field, shown, False, args.out_dir / f"grid_{name}_{shown}.pdf")
            draw_grid(rows, source_field, len(rows), True, args.out_dir / f"grid_{name}_res.pdf")
            print(f"wrote grid_{name}_1..{len(rows)}.pdf and grid_{name}_res.pdf", flush=True)
    if args.figures == "grids":
        return
    for name, transitions in SETS.items():
        rows = compute_rows(model, device, args.subject, args.z_index, args.batch_size,
                            transitions)
        for shown in range(1, len(rows) + 1):
            draw(rows, shown, args.out_dir / f"qual_{name}_{shown}.pdf")
        print(f"wrote qual_{name}_1..{len(rows)}.pdf", flush=True)


if __name__ == "__main__":
    main()
