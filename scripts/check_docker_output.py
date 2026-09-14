#!/usr/bin/env python3
"""Check a container's /output against the manifest and the platform's stated rules.

The published rejection criteria are: "missing or extra prediction files, unreadable
NIfTI files, incorrect paths or filenames, incorrect shapes, NaN/infinity values, or
values outside the required range". This checks all six, so a rejected run is something
we find here rather than something the organizers tell us about after the deadline.

With --ground-truth it also scores, per-slice SSIM averaged over z exactly as
scripts/eval_holdout.py does, which only works on the testbed built from training
subjects -- the real test set has no ground truth on our side.

    ~/anaconda3/envs/mri/bin/python scripts/check_docker_output.py \
        --manifest <input>/manifest.json --output <output> [--ground-truth]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXPECTED_SHAPE = (364, 436, 364)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ground-truth", action="store_true",
                        help="score against the training split (testbed only)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    samples = manifest["samples"]
    expected = {sample["output"] for sample in samples}
    found = {str(p.relative_to(args.output)) for p in args.output.rglob("*.nii.gz")}

    failures: list[str] = []
    for name in sorted(expected - found):
        failures.append(f"MISSING  {name}")
    for name in sorted(found - expected):
        failures.append(f"EXTRA    {name}")

    scores: dict[str, list[float]] = {}
    if args.ground_truth:
        from mrixfields.env import get_data_dir
        from skimage.metrics import structural_similarity

        data_dir = Path(get_data_dir())
        split = next(d for d in (data_dir / "Training_prospective", data_dir / "training_prospective")
                     if d.is_dir())

    for index, sample in enumerate(samples, 1):
        path = args.output / sample["output"]
        if not path.is_file():
            continue
        try:
            image = nib.load(str(path))
            data = np.asarray(image.dataobj, dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"UNREADABLE {sample['output']}: {exc}")
            continue
        if data.shape != EXPECTED_SHAPE:
            failures.append(f"SHAPE    {sample['output']}: {data.shape} != {EXPECTED_SHAPE}")
        if not np.isfinite(data).all():
            failures.append(f"NONFINITE {sample['output']}: "
                            f"{int((~np.isfinite(data)).sum())} voxels")
        low, high = float(np.nanmin(data)), float(np.nanmax(data))
        if low < 0.0 or high > 1.0:
            failures.append(f"RANGE    {sample['output']}: [{low:.4f}, {high:.4f}] outside [0, 1]")

        if args.ground_truth:
            reference = (split / sample["modality"] / sample["target_field"]
                         / f"P_{sample['modality']}_{sample['target_field']}_{sample['case_id']}.nii.gz")
            if reference.is_file():
                # Read the reference exactly as this reads the prediction -- straight off
                # disk, no reorientation. Both files are written in the dataset's own
                # ('L', 'A', 'S') orientation, and the scorer compares the arrays as
                # stored; canonicalising only one side mirrors it and cost 0.11 SSIM
                # when this checker first did it.
                truth = np.asarray(nib.load(str(reference)).dataobj, dtype=np.float32)
                value = float(np.mean([
                    structural_similarity(truth[:, :, z], data[:, :, z], data_range=1.0)
                    for z in range(data.shape[2])]))
                scores.setdefault(sample["modality"], []).append(value)
                print(f"[{index}/{len(samples)}] {sample['output']}  SSIM {value:.4f}", flush=True)

    print()
    print(f"manifest samples : {len(samples)}")
    print(f"files found      : {len(found)}")
    if scores:
        every = [v for values in scores.values() for v in values]
        for modality, values in sorted(scores.items()):
            print(f"  SSIM {modality:8s} {np.mean(values):.4f}  (n={len(values)})")
        print(f"  SSIM {'all':8s} {np.mean(every):.4f}  (n={len(every)})")
    if failures:
        print(f"\n{len(failures)} FAILURES")
        for line in failures[:40]:
            print(f"  {line}")
        raise SystemExit(1)
    print("\nOK: every manifest sample present, correct shape, finite, within [0, 1].")


if __name__ == "__main__":
    main()
