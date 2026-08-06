#!/usr/bin/env python3
"""Run SynthSeg on Baseline inference predictions for Task 1 / Task 2.

Mirrors the source predictions tree into a sibling tree of SynthSeg outputs
(one ``{stem}_seg.nii.gz`` per source NIfTI). The output tree is what
``build_submission`` reads as its ``seg/`` source for Task 1/2.

Source layout:
    $INFERENCE_DIR/{task}_{pair}_{MOD}/{method}/{mode}/{epoch}/
        P_{MOD}_{src_field}_{ID}.nii.gz

Output layout (mirror, with ``_seg`` suffix per file):
    $PREDICTIONS_SEG_DIR/{task}_{pair}_{MOD}/{method}/{mode}/{epoch}/
        P_{MOD}_{src_field}_{ID}_seg.nii.gz

Task 3 has no seg requirement; this script skips it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Reuse SynthSeg plumbing from Evaluation/segment.py.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO / "Evaluation"))
from segment import _get_synthseg_dir, run_synthseg  # noqa: E402


MODS = ("T1W", "T2W", "T2FLAIR")

PAIRS: dict[str, list[str]] = {
    "task1": [
        "0.1T_to_7T",
        "1.5T_to_7T",
        "3T_to_7T",
        "5T_to_7T",
    ],
    "task2": [
        "0.1T_to_1.5T",
        "0.1T_to_3T",
        "0.1T_to_5T",
        "0.1T_to_7T",
    ],
    # task3 deliberately omitted: voxel-level metrics only, no seg.
}

# Match build_submission DEFAULTS so the packer finds segs without
# explicit overrides on either side.
DEFAULTS = {
    "task1": ("cut", "pro_pretrained", "epoch100"),
    "task2": ("cut", "pro_pretrained", "epoch100"),
}


def segment_one_task(
    task: str,
    method: str,
    mode: str,
    epoch_tag: str,
    predictions_dir: Path,
    predictions_seg_dir: Path,
    synthseg_dir: Path,
    overwrite: bool,
    dry_run: bool,
    file_counter: list[int],
    started_at: float,
) -> tuple[int, int, list[str]]:
    """Segment one task. Returns (segmented, skipped, warnings)."""
    pairs = PAIRS[task]
    segmented = 0
    skipped = 0
    warnings: list[str] = []

    print(f"\n=== {task}: {method}/{mode}/{epoch_tag} ===")

    for mod in MODS:
        for pair in pairs:
            src_field = pair.split("_to_")[0]
            src_dir = (
                predictions_dir / f"{task}_{pair}_{mod}" / method / mode / epoch_tag
            )
            dst_dir = (
                predictions_seg_dir / f"{task}_{pair}_{mod}" / method / mode / epoch_tag
            )

            if not src_dir.is_dir():
                warnings.append(f"missing src dir: {src_dir}")
                continue

            src_files = sorted(src_dir.glob(f"P_{mod}_{src_field}_*.nii.gz"))
            # Defensive: drop any *_seg.nii.gz that snuck in.
            src_files = [f for f in src_files if not f.name.endswith("_seg.nii.gz")]
            if not src_files:
                warnings.append(f"no input files in {src_dir}")
                continue

            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)

            for src in src_files:
                file_counter[0] += 1
                stem = src.name.replace(".nii.gz", "")
                dst = dst_dir / f"{stem}_seg.nii.gz"

                if dst.exists() and not overwrite:
                    skipped += 1
                    continue

                if dry_run:
                    print(f"  [dry-run] {src} -> {dst}")
                    segmented += 1
                    continue

                t0 = time.time()
                try:
                    run_synthseg(src, dst, synthseg_dir)
                    # SynthSeg writes a 1 mm volume (~(182, 218, 182)).
                    # NN-resample back to the pred's 0.5 mm grid so seg and
                    # pred share the same shape downstream (build_submission's
                    # axial slice then drops both to (364, 436, 30)).
                    import nibabel as nib
                    from nibabel.processing import resample_from_to
                    seg_1mm = nib.load(str(dst))
                    pred_img = nib.load(str(src))
                    seg_05mm = resample_from_to(seg_1mm, pred_img, order=0)
                    nib.save(seg_05mm, str(dst))
                    elapsed = time.time() - t0
                    total = time.time() - started_at
                    print(
                        f"  [{file_counter[0]}] {src.relative_to(predictions_dir)}  "
                        f"{elapsed:.1f}s  (total {total:.0f}s)"
                    )
                    segmented += 1
                except Exception as e:
                    warnings.append(f"FAIL {src}: {e}")

    return segmented, skipped, warnings


def main() -> int:
    """Run SynthSeg on every Task 1/2 prediction, mirroring INFERENCE_DIR to PREDICTIONS_SEG_DIR.

    SynthSeg model is loaded once on the first call and reused for the full sweep.
    Skips files whose output already exists (use ``--overwrite`` to force).
    """
    p = argparse.ArgumentParser(
        description="Run SynthSeg on Baseline predictions (Task 1/2 only)."
    )
    p.add_argument(
        "--predictions-dir", type=Path, default=None,
        help="Source of Baseline predictions. Default: $INFERENCE_DIR from .env",
    )
    p.add_argument(
        "--predictions-seg-dir", type=Path, default=None,
        help="Output root for SynthSeg results (mirror tree). Default: $PREDICTIONS_SEG_DIR from .env",
    )
    p.add_argument(
        "--tasks", default="task1,task2",
        help="Comma-separated subset (task1,task2). Default: task1,task2. Task 3 is not supported.",
    )
    for task, (m_def, mo_def, e_def) in DEFAULTS.items():
        p.add_argument(f"--{task}-method", default=m_def,
                       help=f"Method to segment for {task} (default: {m_def}).")
        p.add_argument(f"--{task}-mode", default=mo_def,
                       help=f"Training mode to segment for {task} (default: {mo_def}).")
        p.add_argument(f"--{task}-epoch", default=e_def,
                       help=f"Epoch tag to segment for {task} (default: {e_def}).")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-segment files whose seg output already exists (default: skip).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print src -> dst mapping without running SynthSeg.")
    args = p.parse_args()

    if args.predictions_dir is None or args.predictions_seg_dir is None:
        from mrixfields.env import get_inference_dir, get_predictions_seg_dir
        if args.predictions_dir is None:
            args.predictions_dir = Path(get_inference_dir())
        if args.predictions_seg_dir is None:
            args.predictions_seg_dir = Path(get_predictions_seg_dir())

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in PAIRS:
            print(f"ERROR: unknown task {t!r}; supported: {list(PAIRS)}", file=sys.stderr)
            return 2

    if not args.predictions_dir.is_dir():
        print(f"ERROR: --predictions-dir not found: {args.predictions_dir}", file=sys.stderr)
        return 2

    # Resolve SynthSeg + load model lazily (skip on dry-run to keep it fast).
    if args.dry_run:
        synthseg_dir = None
    else:
        print("Loading SynthSeg model (this can take ~30 s)...")
        synthseg_dir = _get_synthseg_dir()
        print(f"SynthSeg dir: {synthseg_dir}")

    print(f"Predictions:     {args.predictions_dir}")
    print(f"Predictions seg: {args.predictions_seg_dir}")
    print(f"Tasks:           {tasks}")
    print(f"Overwrite:       {args.overwrite}")

    file_counter = [0]
    started_at = time.time()
    totals = {"segmented": 0, "skipped": 0}
    all_warnings: list[str] = []

    for t in tasks:
        method = getattr(args, f"{t}_method")
        mode = getattr(args, f"{t}_mode")
        epoch_tag = getattr(args, f"{t}_epoch")
        seg, skip, ws = segment_one_task(
            t, method, mode, epoch_tag,
            args.predictions_dir, args.predictions_seg_dir, synthseg_dir,
            args.overwrite, args.dry_run, file_counter, started_at,
        )
        totals["segmented"] += seg
        totals["skipped"] += skip
        all_warnings.extend(f"[{t}] {w}" for w in ws)

    elapsed = time.time() - started_at
    print("\n=== Summary ===")
    print(f"  segmented: {totals['segmented']}")
    print(f"  skipped (already existed): {totals['skipped']}")
    print(f"  elapsed: {elapsed:.0f}s")
    if all_warnings:
        print(f"\n=== Warnings ({len(all_warnings)}) ===")
        for w in all_warnings:
            print(f"  {w}")
    else:
        print("  (no warnings)")

    if not args.dry_run:
        print(
            f"\nNext step: run build_submission to repack "
            f"$INFERENCE_DIR + $PREDICTIONS_SEG_DIR into $SUBMISSION_DIR."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
