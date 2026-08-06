"""Preprocessing script for MRIxFields2026 data.

Two modes:
    resample       — Resample full NIfTI volumes to isotropic spacing (nii.gz → nii.gz).
                     Pass --normalize for optional per-volume p99.5 clip + min-max rescale.
    extract-slices — Extract 2D axial slices as npz for fast training.
                     Input volumes are in [0, 1]; slices are saved as float32.

Slice naming: {split_abbr}_{modality}_{field}_{subject_id}_s{NNN}.npz
    e.g. retro_T1W_0.1T_R_0001_s128.npz

Usage:
    # Extract all slices (scans directory directly, no metadata needed)
    python scripts/preprocess.py extract-slices --splits retro_train pro_train

    # Single field strength
    python scripts/preprocess.py extract-slices \
        --data_dir ... --output_dir ... --splits retro_train --fields 0.1T 7T

    # Debug: 1 case per split/modality/field, save png
    python scripts/preprocess.py extract-slices \
        --data_dir ... --output_dir ... --debug

    # Resample volumes (optional)
    python scripts/preprocess.py resample \
        --input_dir $DATA_DIR/raw --output_dir $DATA_DIR/resampled
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mrixfields.data.utils import (
    load_nifti, save_nifti,
    FIELD_STRENGTHS, MODALITIES,
    SPLIT_ABBR, ABBR_TO_SPLIT,
)
from mrixfields.env import load_env, get_data_dir, get_preprocessed_dir

# Fixed slice range: middle 220 of 364 → indices 72..291
SLICE_START = 72
SLICE_END = 292  # exclusive, so 72..291 = 220 slices


# --------------------------------------------------------------------------- #
#  Mode 1: Resample (nii.gz → nii.gz)
# --------------------------------------------------------------------------- #

def resample_to_isotropic(image, target_spacing=(1.0, 1.0, 1.0)):
    import SimpleITK as sitk

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / tspc))
                for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)


def run_resample(args):
    import SimpleITK as sitk

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    nifti_files = sorted(input_dir.rglob("*.nii.gz"))
    print(f"Found {len(nifti_files)} NIfTI files")
    for nifti_path in tqdm(nifti_files, desc="Resampling"):
        rel_path = nifti_path.relative_to(input_dir)
        out_path = output_dir / rel_path
        image = sitk.ReadImage(str(nifti_path), sitk.sitkFloat32)
        image = resample_to_isotropic(image, tuple(args.spacing))
        if args.normalize:
            data = sitk.GetArrayFromImage(image)
            upper = np.percentile(data, 99.5)
            data = np.clip(data, 0, upper)
            if upper > 1e-8:
                data = data / upper
            result = sitk.GetImageFromArray(data)
            result.CopyInformation(image)
            image = result
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(out_path))
    print(f"Done. Output: {output_dir}")


# --------------------------------------------------------------------------- #
#  Mode 2: Extract Slices (nii.gz → npz), directory-based
# --------------------------------------------------------------------------- #

def _extract_subject_id(filename: str) -> str:
    """Extract subject ID from standardized filename.

    Filename format: {R,P}_{modality}_{field}_{ID}.nii.gz
    Examples:
        R_T1W_0.1T_0001.nii.gz -> R_0001
        P_T1W_0.1T_0001.nii.gz -> P_0001
    """
    base = filename.replace(".nii.gz", "")
    parts = base.split("_")
    # parts = [R/P, modality, field, ID]
    if len(parts) >= 4:
        prefix = parts[0]  # R or P
        subject_id = parts[-1]  # 0001
        return f"{prefix}_{subject_id}"
    return base


def _list_nifti_files(data_dir: Path, split: str, modality: str, field: str) -> list:
    """List all nii.gz files for a given split/modality/field."""
    split_dir = data_dir / split
    if not split_dir.exists() and data_dir.is_dir():
        split_dir = next(
            (path for path in data_dir.iterdir() if path.is_dir() and path.name.casefold() == split.casefold()),
            split_dir,
        )
    d = split_dir / modality / field
    if not d.exists():
        return []
    return sorted(d.glob("*.nii.gz"))


def extract_slices_from_volume(
    nifti_path: Path,
    subject_id: str,
    split_abbr: str,
    modality: str,
    field: str,
    output_dir: Path,
    dtype: str = "float32",
    save_debug_png: bool = False,
) -> dict:
    """Extract fixed axial slices from a 3D volume, save as npz.

    Input volumes are in [0, 1]; output slices are float32 in [0, 1].
    """
    data, _ = load_nifti(nifti_path)  # (364, 436, 364) float32
    data = data.astype(np.float32)

    output_dir.mkdir(parents=True, exist_ok=True)
    kept = 0

    for i in range(SLICE_START, SLICE_END):
        slc = data[:, :, i]  # (364, 436) axial

        fname = f"{split_abbr}_{modality}_{field}_{subject_id}_s{i:03d}.npz"
        np.savez_compressed(
            output_dir / fname,
            image=slc.astype(dtype),
            subject_id=subject_id,
            split=split_abbr,
            modality=modality,
            field_strength=field,
            slice_idx=np.int32(i),
        )
        kept += 1

    # Debug: save a PNG of the middle slice
    if save_debug_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            mid = data[:, :, data.shape[2] // 2]
            debug_dir = output_dir.parent / "debug_png"
            debug_dir.mkdir(parents=True, exist_ok=True)
            png_path = debug_dir / f"{split_abbr}_{modality}_{field}_{subject_id}.png"

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for ax, (s_idx, title) in zip(axes, [
                (SLICE_START, f"First (z={SLICE_START})"),
                (data.shape[2] // 2, f"Middle (z={data.shape[2] // 2})"),
                (SLICE_END - 1, f"Last (z={SLICE_END - 1})"),
            ]):
                ax.imshow(data[:, :, s_idx].T, cmap="gray", origin="lower", vmin=0, vmax=1)
                ax.set_title(title)
                ax.axis("off")
            fig.suptitle(f"{subject_id} | {modality} | {field} | norm [0,1]", fontsize=12)
            plt.tight_layout()
            plt.savefig(png_path, dpi=100, bbox_inches="tight")
            plt.close()
            print(f"    [DEBUG PNG] {png_path}")
        except ImportError:
            pass

    return {
        "subject_id": subject_id,
        "source_file": nifti_path.name,
        "n_slices": kept,
        "shape": list(data.shape),
    }


def run_extract_slices(args):
    if args.data_dir is None:
        args.data_dir = get_data_dir()
    if args.output_dir is None:
        val = get_preprocessed_dir()
        if not val:
            raise RuntimeError(
                "PREPROCESSED_DIR is not set. Configure it in .env (repo root) "
                "or pass --output_dir. See .env.example."
            )
        args.output_dir = val
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Resolve split abbreviations
    requested_splits = []
    for s in (args.splits or ["retro_train"]):
        if s in ABBR_TO_SPLIT:
            requested_splits.append(ABBR_TO_SPLIT[s])
        elif s in SPLIT_ABBR:
            requested_splits.append(s)
        else:
            raise ValueError(f"Unknown split: {s}. Use: {list(ABBR_TO_SPLIT.keys())}")

    modalities = args.modalities or MODALITIES
    fields = args.fields or FIELD_STRENGTHS

    # Scan directories directly (no metadata needed)
    print(f"Data root: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Splits: {[SPLIT_ABBR[s] for s in requested_splits]}")
    print(f"Modalities: {modalities}")
    print(f"Fields: {fields}")
    print(f"Slice range: {SLICE_START}~{SLICE_END - 1} ({SLICE_END - SLICE_START} slices)")

    # Count total files
    total_files = 0
    for split in requested_splits:
        for mod in modalities:
            for field in fields:
                total_files += len(_list_nifti_files(data_dir, split, mod, field))

    print(f"Found {total_files} NIfTI files")
    if args.debug:
        print("[DEBUG MODE] Processing 1 case per split/modality/field\n")
    else:
        print()

    if total_files == 0:
        print("No matching files found.")
        return

    total_volumes = 0
    total_slices = 0
    all_meta = []

    for split in requested_splits:
        abbr = SPLIT_ABBR[split]
        for mod in modalities:
            for field in fields:
                nifti_files = _list_nifti_files(data_dir, split, mod, field)
                if not nifti_files:
                    continue

                out_subdir = output_dir / abbr / mod / field

                if args.debug:
                    nifti_files = nifti_files[:1]  # 1 case only

                desc = f"{abbr}/{mod}/{field}"
                n_expected = SLICE_END - SLICE_START
                for nifti_path in tqdm(nifti_files, desc=desc, leave=False):
                    subject_id = _extract_subject_id(nifti_path.name)

                    if args.skip_existing:
                        existing = list(out_subdir.glob(f"{abbr}_{mod}_{field}_{subject_id}_s*.npz"))
                        if len(existing) == n_expected:
                            all_meta.append({
                                "subject_id": subject_id,
                                "source_file": nifti_path.name,
                                "n_slices": n_expected,
                                "shape": None,
                                "skipped": True,
                            })
                            total_volumes += 1
                            total_slices += n_expected
                            continue

                    meta = extract_slices_from_volume(
                        nifti_path=nifti_path,
                        subject_id=subject_id,
                        split_abbr=abbr,
                        modality=mod,
                        field=field,
                        output_dir=out_subdir,
                        dtype=args.dtype,
                        save_debug_png=args.debug,
                    )
                    all_meta.append(meta)
                    total_volumes += 1
                    total_slices += meta["n_slices"]

                print(f"  {desc}: {len(nifti_files)} volumes → {len(nifti_files) * (SLICE_END - SLICE_START)} slices")

    # Save manifest
    manifest_out = {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "slice_range": [SLICE_START, SLICE_END],
        "n_slices_per_volume": SLICE_END - SLICE_START,
        "normalization": "identity",
        "dtype": args.dtype,
        "total_volumes": total_volumes,
        "total_slices": total_slices,
        "volumes": all_meta,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_out, f, indent=2)

    print(f"\nDone: {total_volumes} volumes → {total_slices} slices")
    print(f"Manifest: {manifest_path}")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="MRIxFields2026 Preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode", help="Preprocessing mode")

    # --- resample ---
    p_resample = subparsers.add_parser("resample", help="Resample NIfTI volumes")
    p_resample.add_argument("--input_dir", type=str, required=True)
    p_resample.add_argument("--output_dir", type=str, required=True)
    p_resample.add_argument("--spacing", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    p_resample.add_argument("--normalize", action="store_true",
                            help="Apply per-volume p99.5 clip + min-max normalize.")

    # --- extract-slices ---
    p_extract = subparsers.add_parser("extract-slices", help="Extract 2D axial slices as npz")
    load_env()
    p_extract.add_argument("--data_dir", type=str, default=None,
                           help="Root data dir (default: DATA_DIR from .env)")
    p_extract.add_argument("--output_dir", type=str, default=None,
                           help="Output dir (default: PREPROCESSED_DIR from .env)")
    p_extract.add_argument("--splits", nargs="+", default=None,
                           help="Split abbreviations: retro_train, pro_train, pro_val, pro_test (default: retro_train)")
    p_extract.add_argument("--modalities", nargs="+", default=None,
                           help="Modalities (default: all 3)")
    p_extract.add_argument("--fields", nargs="+", default=None,
                           help="Field strengths (default: all 5)")
    p_extract.add_argument("--debug", action="store_true",
                           help="Debug: 1 case per group, save PNG visualization")
    p_extract.add_argument("--skip_existing", action="store_true",
                           help="Skip subjects whose full set of 220 slices already exists in output_dir.")
    p_extract.add_argument("--dtype", choices=("float16", "float32"), default="float32",
                           help="Cached image dtype (default: float32). float16 roughly halves disk usage.")

    args = parser.parse_args()
    if args.mode == "resample":
        run_resample(args)
    elif args.mode == "extract-slices":
        run_extract_slices(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
