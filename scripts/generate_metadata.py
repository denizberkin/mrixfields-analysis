"""Generate metadata for MRIxFields2026 dataset.

Output layout (1 json + 1 csv per split):
    $DATA_DIR/metadata/
        dataset_meta.json
        Training_retrospective.csv
        Training_prospective.csv
        Validating_prospective.csv
        Testing_prospective.csv

Supports incremental updates: re-running with --splits only refreshes the
specified splits' csv and their fields in dataset_meta.json. Other splits'
entries are preserved verbatim from the existing dataset_meta.json.

Usage:
    # Full generation (all 4 splits, release_date = 'TBD' if not previously set)
    python scripts/generate_metadata.py --data_dir $DATA_DIR

    # Incremental: refresh one split with a release date
    python scripts/generate_metadata.py \\
        --data_dir $DATA_DIR \\
        --splits Validating_prospective \\
        --release_date 2026-04-13
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Dataset constants (kept in sync with mrixfields/data/utils.py;
#  duplicated here so this tool can run without the training stack / torch)
# ---------------------------------------------------------------------------

FIELD_STRENGTHS = ["0.1T", "1.5T", "3T", "5T", "7T"]
MODALITIES = ["T1W", "T2W", "T2FLAIR"]
SPLITS = [
    "Training_retrospective",
    "Training_prospective",
    "Validating_prospective",
    "Testing_prospective",
]


# ---------------------------------------------------------------------------
#  Static metadata (hand-curated; lives in the json, not in csv)
# ---------------------------------------------------------------------------

SCANNERS = [
    {"field_strength": "0.1T", "field_tesla": 0.1, "scanner": "piMR-820H",
     "manufacturer": "Point Imaging", "institution": "China"},
    {"field_strength": "1.5T", "field_tesla": 1.5, "scanner": "uMR 670",
     "manufacturer": "United Imaging Healthcare", "institution": "China"},
    {"field_strength": "3T", "field_tesla": 3.0, "scanner": "MAGNETOM Prisma",
     "manufacturer": "Siemens Healthineers", "institution": "Germany"},
    {"field_strength": "5T", "field_tesla": 5.0, "scanner": "uMR Jupiter",
     "manufacturer": "United Imaging Healthcare", "institution": "China"},
    {"field_strength": "7T", "field_tesla": 7.0, "scanner": "MAGNETOM Terra",
     "manufacturer": "Siemens Healthineers", "institution": "Germany"},
]

TASKS = [
    {"task": 1, "pair_id": "1a", "source_field": "0.1T", "target_field": "7T",
     "description": "Ultra-low to ultra-high"},
    {"task": 1, "pair_id": "1b", "source_field": "1.5T", "target_field": "7T",
     "description": "Clinical to ultra-high"},
    {"task": 1, "pair_id": "1c", "source_field": "3T", "target_field": "7T",
     "description": "Clinical 3T to ultra-high"},
    {"task": 1, "pair_id": "1d", "source_field": "5T", "target_field": "7T",
     "description": "High-field to ultra-high"},
    {"task": 2, "pair_id": "2a", "source_field": "0.1T", "target_field": "1.5T",
     "description": "Ultra-low enhancement"},
    {"task": 2, "pair_id": "2b", "source_field": "0.1T", "target_field": "3T",
     "description": "Ultra-low to clinical 3T"},
    {"task": 2, "pair_id": "2c", "source_field": "0.1T", "target_field": "5T",
     "description": "Ultra-low to high-field"},
    {"task": 2, "pair_id": "2d", "source_field": "0.1T", "target_field": "7T",
     "description": "Ultra-low to ultra-high"},
    {"task": 3, "pair_id": "3", "source_field": "any", "target_field": "any",
     "description": "Universal field-to-field (30 directions)"},
]

SPLIT_DEFAULTS = {
    "Training_retrospective": {
        "type": "train", "paired": False,
        "description": "Different subjects per scanner; unpaired; main pretraining data",
    },
    "Training_prospective": {
        "type": "train", "paired": True,
        "description": "Travelling volunteers; paired across all 5 field strengths",
    },
    "Validating_prospective": {
        "type": "val", "paired": True,
        "description": "Travelling volunteers; paired validation set",
    },
    "Testing_prospective": {
        "type": "test", "paired": True,
        "description": "Travelling volunteers; paired held-out test set",
    },
}


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _extract_subject_id(filename: str) -> str:
    """{R,P}_{mod}_{field}_{ID}.nii.gz -> ID (e.g. '0001')."""
    base = filename.replace(".nii.gz", "")
    parts = base.split("_")
    if len(parts) >= 4:
        return parts[-1]
    return base


def collect_split_files(data_dir: Path, split: str) -> Tuple[int, List[Tuple[str, str, str, str, str]]]:
    """Scan one split directory; return (subject_count, file_rows).

    file_rows columns: (subject_id, modality, field_strength, filename, relative_path)

    Subject ID convention:
      - Prospective splits:    'P{NNNN}'        (same volunteer across all 5 fields)
      - Training_retrospective: 'R_{field}_{NNNN}' (different patients per field)
    """
    file_rows: List[Tuple[str, str, str, str, str]] = []
    subject_ids = set()
    is_retro = (split == "Training_retrospective")

    for mod in MODALITIES:
        for field in FIELD_STRENGTHS:
            d = data_dir / split / mod / field
            if not d.exists():
                continue
            for f in sorted(d.glob("*.nii.gz")):
                num_id = _extract_subject_id(f.name)
                sid = f"R_{field}_{num_id}" if is_retro else f"P{num_id}"
                rel = f"{split}/{mod}/{field}/{f.name}"
                file_rows.append((sid, mod, field, f.name, rel))
                subject_ids.add(sid)

    return len(subject_ids), file_rows


def write_split_csv(output_dir: Path, split: str, file_rows: List[Tuple]) -> None:
    """Write {split}.csv to output_dir root."""
    path = output_dir / f"{split}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "modality", "field_strength", "filename", "relative_path"])
        w.writerows(file_rows)


def load_existing_meta(meta_path: Path) -> dict:
    """Load existing dataset_meta.json; return {} if missing or unreadable."""
    if not meta_path.exists():
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def build_dataset_meta(
    splits_info: Dict[str, Dict[str, int]],
    refreshed_splits: List[str],
    release_date: Optional[str],
    existing_meta: dict,
) -> dict:
    """Merge new scan results with previously written metadata.

    splits_info:      {split: {"subject_count": int, "file_count": int}} for refreshed splits
    refreshed_splits: which splits were re-scanned this run
    release_date:     if given, applied to refreshed splits; else preserves existing or 'TBD'
    existing_meta:    previous dataset_meta.json contents ({} on first run)
    """
    existing_splits = existing_meta.get("splits", {})
    splits_block: Dict[str, dict] = {}

    for split in SPLITS:
        defaults = SPLIT_DEFAULTS[split]
        prev = existing_splits.get(split, {})

        if split in refreshed_splits:
            counts = splits_info[split]
            rd = release_date if release_date else prev.get("release_date", "TBD")
            splits_block[split] = {
                "release_date": rd,
                "type": defaults["type"],
                "paired": defaults["paired"],
                "subject_count": counts["subject_count"],
                "file_count": counts["file_count"],
                "files_csv": f"{split}.csv",
                "description": defaults["description"],
            }
        elif prev:
            # Not refreshed but exists from a previous run: preserve verbatim
            splits_block[split] = prev
        # else: skip entirely (never scanned, no record)

    return {
        "dataset_name": "MRIxFields2026",
        "field_strengths": FIELD_STRENGTHS,
        "modalities": MODALITIES,
        "scanners": SCANNERS,
        "tasks": TASKS,
        "splits": splits_block,
    }


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate MRIxFields2026 metadata (1 json + 4 csv)")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root data directory ($DATA_DIR)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for metadata files (default: {data_dir}/metadata)")
    parser.add_argument("--splits", nargs="+", default=None, choices=SPLITS,
                        help="Only refresh specified splits (default: all 4)")
    parser.add_argument("--release_date", type=str, default=None,
                        help="YYYY-MM-DD; sets release_date for the refreshed splits")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)

    refreshed = args.splits if args.splits else list(SPLITS)
    print(f"Data dir: {data_dir}")
    print(f"Output:   {output_dir}")
    print(f"Refresh:  {refreshed}")
    if args.release_date:
        print(f"Release:  {args.release_date}")
    print()

    # Scan refreshed splits, write per-split csv
    splits_info: Dict[str, Dict[str, int]] = {}
    for split in refreshed:
        n_subj, file_rows = collect_split_files(data_dir, split)
        write_split_csv(output_dir, split, file_rows)
        splits_info[split] = {"subject_count": n_subj, "file_count": len(file_rows)}
        print(f"  {split}.csv: {n_subj} subjects, {len(file_rows)} files")

    # Merge with any existing metadata, write dataset_meta.json
    meta_path = output_dir / "dataset_meta.json"
    existing = load_existing_meta(meta_path)
    meta = build_dataset_meta(splits_info, refreshed, args.release_date, existing)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  dataset_meta.json: {len(meta['splits'])} splits recorded")


if __name__ == "__main__":
    main()
