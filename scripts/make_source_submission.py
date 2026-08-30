"""
Build a Task 3 submission ZIP with unmodified source, control group for metrics.

Written directly, no intermediary scripts/generate_challenge_submission.py 
Task 3 accepts no segmentations.

Usage:
    python scripts/make_source_submission.py                     # writes to $SUBMISSION_DIR
    python scripts/make_source_submission.py --out-dir some/dir --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import nibabel as nib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mrixfields.zclip_constants import Z_CLIP_RANGE

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")

#: Validation subject ids per source field. They are disjoint across fields by design, so the ids in
#: an output filename always come from the source folder even though the name carries the target.
SOURCE_IDS = {
    "0.1T": ("0001", "0002", "0003"),
    "1.5T": ("0004", "0005", "0008"),
    "3T": ("0010", "0011", "0012"),
    "5T": ("0013", "0014", "0015"),
    "7T": ("0016", "0017", "0018"),
}

EXPECTED_SHAPE = (364, 436, Z_CLIP_RANGE[1] - Z_CLIP_RANGE[0])


def find_split(data_dir: Path) -> Path:
    """Validation split directory, tolerating the dataset's inconsistent casing."""
    for name in ("Validating_prospective", "validating_prospective"):
        candidate = data_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No validation split under {data_dir}")


def clip_to_slab(source: Path, destination: Path) -> tuple[int, ...]:
    """Copy one volume, keeping only the submission slab. Values are not touched."""
    start, stop = Z_CLIP_RANGE
    image = nib.load(str(source))
    clipped = image.slicer[:, :, start:stop]
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(clipped, str(destination))
    return clipped.shape


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None, help="defaults to DATA_DIR from .env")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="defaults to SUBMISSION_DIR from .env, under an identity_task3/ subdir")
    parser.add_argument("--dry-run", action="store_true", help="list the work without writing")
    args = parser.parse_args()

    from mrixfields.env import get_data_dir, get_submission_dir

    data_dir = args.data_dir or Path(get_data_dir())
    out_dir = args.out_dir or (Path(get_submission_dir()) / "identity_task3")
    split = find_split(data_dir)

    pairs = [(source, target) for source in FIELDS for target in FIELDS if source != target]
    total = len(MODALITIES) * len(pairs) * 3
    print(f"source: {split}", flush=True)
    print(f"output: {out_dir}", flush=True)
    print(f"{len(pairs)} field pairs x {len(MODALITIES)} modalities x 3 subjects = {total} files",
          flush=True)

    if args.dry_run:
        for modality in MODALITIES[:1]:
            for source, target in pairs[:2]:
                for subject in SOURCE_IDS[source]:
                    print(f"  {split.name}/{modality}/{source}/P_{modality}_{source}_{subject}.nii.gz"
                          f"  ->  task3/{modality}/{source}_to_{target}/pred/"
                          f"P_{modality}_{target}_{subject}.nii.gz", flush=True)
        print("... (nothing written)", flush=True)
        return

    task_dir = out_dir / "task3"
    if task_dir.exists():
        shutil.rmtree(task_dir)

    written: list[Path] = []
    for modality in MODALITIES:
        for source, target in pairs:
            for subject in SOURCE_IDS[source]:
                origin = split / modality / source / f"P_{modality}_{source}_{subject}.nii.gz"
                if not origin.is_file():
                    raise FileNotFoundError(origin)
                destination = (task_dir / modality / f"{source}_to_{target}" / "pred"
                               / f"P_{modality}_{target}_{subject}.nii.gz")
                shape = clip_to_slab(origin, destination)
                if shape != EXPECTED_SHAPE:
                    raise RuntimeError(f"{destination} has shape {shape}, expected {EXPECTED_SHAPE}")
                written.append(destination)
        print(f"{modality}: {len(written)} files so far", flush=True)

    if len(written) != total:
        raise RuntimeError(f"wrote {len(written)} files, expected {total}")

    zip_path = out_dir / "task3.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(written):
            archive.write(path, path.relative_to(task_dir.parent))
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
        members = [name for name in archive.namelist() if name.endswith(".nii.gz")]
        if len(members) != total:
            raise RuntimeError(f"ZIP holds {len(members)} volumes, expected {total}")

    size_gb = zip_path.stat().st_size / 1024**3
    print(f"wrote {zip_path}  ({total} volumes, {size_gb:.2f} GiB)", flush=True)


if __name__ == "__main__":
    main()
