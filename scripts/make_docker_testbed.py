#!/usr/bin/env python3
"""Build a fake /input tree in the test-phase layout, so the container can be run locally.

The real test input is hidden, so the only way to check the container end to end is to
reconstruct the published layout from data we have. Training subjects are used, not the
validation ones, because they are the only subjects with ground truth at every field --
which means the container's own output can then be scored, not merely counted.

The tree is the one in the participant guide:

    <root>/manifest.json
    <root>/{MOD}/{SRC}_to_{TGT}/src/P_{MOD}_{SRC}_{ID}.nii.gz

Files are hard-linked where the filesystem allows it (the same volume appears under
every mapping that uses it, so copying would write the same ~40 MB a dozen times).

    ~/anaconda3/envs/mri/bin/python scripts/make_docker_testbed.py --root <dir>
    ~/anaconda3/envs/mri/bin/python scripts/make_docker_testbed.py --root <dir> --mappings 2
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")

#: The three paired prospective training subjects. The test phase gives two cases per
#: mapping; two of these stand in for them, and the third is left out so the count matches.
CASES = ("0006", "0009")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True, help="directory to build the tree in")
    parser.add_argument("--mappings", type=int, default=20,
                        help="how many of the 20 field mappings to include (smaller = faster smoke test)")
    parser.add_argument("--modalities", nargs="*", default=list(MODALITIES))
    parser.add_argument("--cases", nargs="*", default=list(CASES))
    args = parser.parse_args()

    from mrixfields.env import get_data_dir

    data_dir = Path(get_data_dir())
    split = next((d for d in (data_dir / "Training_prospective", data_dir / "training_prospective")
                  if d.is_dir()), None)
    if split is None:
        raise SystemExit(f"no training split under {data_dir}")

    pairs = [(s, t) for s in FIELDS for t in FIELDS if s != t][: args.mappings]
    root = args.root
    if root.exists():
        shutil.rmtree(root)

    samples = []
    linked = copied = 0
    for modality in args.modalities:
        for source, target in pairs:
            for case in args.cases:
                name = f"P_{modality}_{source}_{case}.nii.gz"
                origin = split / modality / source / name
                if not origin.is_file():
                    raise SystemExit(f"missing source volume: {origin}")
                relative = Path(modality) / f"{source}_to_{target}" / "src" / name
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    try:
                        destination.hardlink_to(origin)
                        linked += 1
                    except OSError:
                        shutil.copy2(origin, destination)
                        copied += 1
                samples.append({
                    "task": "task3",
                    "case_id": case,
                    "modality": modality,
                    "source_field": source,
                    "target_field": target,
                    "pair": f"{source}_to_{target}",
                    "input": str(relative),
                    "output": str(Path(modality) / f"{source}_to_{target}" / "pred"
                                   / f"P_{modality}_{target}_{case}.nii.gz"),
                })

    manifest = {
        "challenge": "MRIxFields2026",
        "phase": "test",
        "task": "task3",
        "input_root": "/input",
        "output_root": "/output",
        "volume_shape": [364, 436, 364],
        "samples": samples,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"{root}: {len(samples)} samples, {linked} hard-linked, {copied} copied")
    print(f"  {len(args.modalities)} modalities x {len(pairs)} mappings x {len(args.cases)} cases")


if __name__ == "__main__":
    main()
