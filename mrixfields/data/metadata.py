"""Metadata CSV reader utilities for MRIxFields2026.

Loads the structured metadata tables (files.csv, subjects.csv, etc.)
and provides lookup functions for file pairing and discovery.

The metadata CSVs are the authoritative source for file lookups —
code should query these tables rather than relying on filename patterns.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_files_csv(metadata_dir: str | Path) -> List[Dict[str, str]]:
    """Load files.csv as a list of dicts.

    Each dict has keys: subject_id, split, modality, field_strength, filename, relative_path
    """
    path = Path(metadata_dir) / "files.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_subjects_csv(metadata_dir: str | Path) -> List[Dict[str, str]]:
    """Load subjects.csv as a list of dicts."""
    path = Path(metadata_dir) / "subjects.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def get_files(
    files: List[Dict[str, str]],
    split: Optional[str] = None,
    modality: Optional[str] = None,
    field_strength: Optional[str] = None,
    subject_id: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Filter files by any combination of split, modality, field_strength, subject_id."""
    result = files
    if split:
        result = [r for r in result if r["split"] == split]
    if modality:
        result = [r for r in result if r["modality"] == modality]
    if field_strength:
        result = [r for r in result if r["field_strength"] == field_strength]
    if subject_id:
        result = [r for r in result if r["subject_id"] == subject_id]
    return result


def get_paired_paths(
    files: List[Dict[str, str]],
    data_root: str | Path,
    split: str,
    modality: str,
    source_field: str,
    target_field: str,
) -> List[Tuple[Path, Path]]:
    """Get matched (source_path, target_path) pairs by subject_id.

    Returns list of (source_path, target_path) tuples for subjects that
    have files at both the source and target field strengths.
    """
    data_root = Path(data_root)

    # Build subject -> path lookup for source and target
    source_lookup: Dict[str, Path] = {}
    target_lookup: Dict[str, Path] = {}

    for r in files:
        if r["split"] != split or r["modality"] != modality:
            continue
        full_path = data_root / r["relative_path"]
        if r["field_strength"] == source_field:
            source_lookup[r["subject_id"]] = full_path
        elif r["field_strength"] == target_field:
            target_lookup[r["subject_id"]] = full_path

    # Match by subject_id
    pairs = []
    for sid in sorted(source_lookup.keys()):
        if sid in target_lookup:
            pairs.append((source_lookup[sid], target_lookup[sid]))

    return pairs


def build_subject_file_map(
    files: List[Dict[str, str]],
    split: str,
    modality: str,
) -> Dict[str, Dict[str, str]]:
    """Build {subject_id: {field_strength: relative_path}} map.

    Useful for looking up all files for a given subject across field strengths.
    """
    result: Dict[str, Dict[str, str]] = {}
    for r in files:
        if r["split"] != split or r["modality"] != modality:
            continue
        sid = r["subject_id"]
        if sid not in result:
            result[sid] = {}
        result[sid][r["field_strength"]] = r["relative_path"]
    return result
