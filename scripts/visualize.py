"""Visualize predictions: input vs prediction vs target for middle slices.

Usage:
    python scripts/visualize.py \
        --input_dir $DATA_DIR/Validating_prospective/T1W/0.1T/ \
        --pred_dir $OUTPUT_DIR/task1_0.1T_to_7T_T1W/cyclegan/predictions/ \
        --target_dir $DATA_DIR/Validating_prospective/T1W/7T/ \
        --output_dir $OUTPUT_DIR/task1_0.1T_to_7T_T1W/cyclegan/visualizations/ \
        --n_slices 5
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.data.utils import load_nifti


def match_files(input_dir: Path, pred_dir: Path, target_dir: Path):
    """Match files across input/pred/target by subject ID prefix."""
    pred_files = sorted(pred_dir.glob("*.nii.gz"))

    # Build lookups by subject ID prefix
    target_lookup = {f.name.split("_")[0]: f for f in target_dir.glob("*.nii.gz")}
    input_lookup = {f.name.split("_")[0]: f for f in input_dir.glob("*.nii.gz")}

    matched = []
    for pred_path in pred_files:
        prefix = pred_path.name.split("_")[0]
        if prefix in target_lookup and prefix in input_lookup:
            matched.append({
                "subject": prefix,
                "input": input_lookup[prefix],
                "pred": pred_path,
                "target": target_lookup[prefix],
            })
    return matched


def visualize_case(input_vol, pred_vol, target_vol, subject_id, output_path, n_slices=5):
    """Generate a comparison figure for one case: n_slices x 3 grid."""
    total_slices = input_vol.shape[2]
    center = total_slices // 2
    half = n_slices // 2
    slice_indices = list(range(center - half, center + half + (1 if n_slices % 2 else 0)))[:n_slices]

    fig, axes = plt.subplots(n_slices, 3, figsize=(12, 4 * n_slices))
    if n_slices == 1:
        axes = axes[np.newaxis, :]

    titles = ["Input (0.1T)", "Prediction", "Target (7T)"]
    volumes = [input_vol, pred_vol, target_vol]

    for row, s_idx in enumerate(slice_indices):
        for col, (vol, title) in enumerate(zip(volumes, titles)):
            ax = axes[row, col]
            slc = vol[:, :, s_idx].T
            ax.imshow(slc, cmap="gray", origin="lower", vmin=0, vmax=1)
            if row == 0:
                ax.set_title(title, fontsize=14, fontweight="bold")
            ax.set_ylabel(f"slice {s_idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"Subject {subject_id}", fontsize=16, fontweight="bold", y=1.01)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize MRI translation results")
    parser.add_argument("--input_dir", type=str, required=True, help="Input nii.gz directory")
    parser.add_argument("--pred_dir", type=str, required=True, help="Prediction nii.gz directory")
    parser.add_argument("--target_dir", type=str, required=True, help="Target nii.gz directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for PNG images")
    parser.add_argument("--n_slices", type=int, default=5, help="Number of middle slices to display")
    parser.add_argument("--max_cases", type=int, default=None, help="Max number of cases to visualize")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    pred_dir = Path(args.pred_dir)
    target_dir = Path(args.target_dir)
    output_dir = Path(args.output_dir)

    matched = match_files(input_dir, pred_dir, target_dir)
    if not matched:
        print("No matched files found.")
        return

    if args.max_cases:
        matched = matched[:args.max_cases]

    print(f"Visualizing {len(matched)} cases, {args.n_slices} slices each")

    for m in matched:
        input_vol, _ = load_nifti(m["input"])
        pred_vol, _ = load_nifti(m["pred"])
        target_vol, _ = load_nifti(m["target"])

        out_path = output_dir / f"{m['subject']}_comparison.png"
        visualize_case(input_vol, pred_vol, target_vol, m["subject"], out_path, args.n_slices)
        print(f"  Saved: {out_path}")

    print(f"Done. {len(matched)} visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
