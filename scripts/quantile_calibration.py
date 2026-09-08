#!/usr/bin/env python3
"""Parameter-free monotone quantile calibration for Task 3 (TODO A2).

One map per (modality, source field, target field): the monotone function carrying the
source intensity distribution onto the target's, fitted on the paired prospective
subjects. No network, no training, nothing to overfit beyond 512 quantile knots.

Why it is worth a slot. The identity transform already scores SSIM 0.8365 on the
leaderboard against the best model's 0.9026, so most of the image is already correct and
what a translation model mainly has to fix is the intensity mapping. Report finding F3
measured that the conditional U-Net's target conditioning is 12-96% explained by a single
global affine, and F4 that a monotone map removes 46% of the identity's error. If a
fitted histogram match lands near the trained models, then the architecture comparisons
in TODO section 0 have largely been resolving calibration quality, and that is worth
knowing before spending more GPU time on architectures.

Fitting uses foreground voxels only. Air is ~80% of a volume and sits at a single value,
so including it would put most of the quantile mass on background and flatten the map
exactly where the anatomy lives.

Usage:
    # fit and score locally, leave-one-subject-out so no subject calibrates itself
    python scripts/quantile_calibration.py --score

    # fit on all paired subjects and write a submission
    python scripts/quantile_calibration.py --submit --name task3_quantile
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
PAIRED_SUBJECTS = ("0006", "0007", "0009")
KNOTS = 512
# The T2W 1.5T volumes carry a flat non-zero pedestal in air (1.2e-4 and below), so a
# 1e-6 foreground test would select the whole volume for those and the anatomy only for
# every other one -- inconsistent masking across the very transitions being compared.
FOREGROUND = 1e-3


def cached_volume(cache_dir: Path, split: str, modality: str, field: str, subject: str):
    directory = cache_dir / split / modality / field
    if not directory.is_dir():
        return None
    matches = sorted(p for p in directory.glob("*.npy") if subject in p.name)
    return np.load(matches[0], mmap_mode="r") if matches else None


def fit_map(sources: list[np.ndarray], targets: list[np.ndarray]) -> np.ndarray:
    """Return the target quantiles at KNOTS evenly spaced probabilities.

    Pooling the subjects' foreground voxels before taking quantiles (rather than
    averaging per-subject curves) weights each subject by how much brain it actually
    contributes, and keeps the result monotone by construction.
    """
    probabilities = np.linspace(0.0, 1.0, KNOTS)
    source_pool = np.concatenate([v[v > FOREGROUND] for v in sources])
    target_pool = np.concatenate([v[v > FOREGROUND] for v in targets])
    return np.stack([
        np.quantile(source_pool, probabilities),
        np.quantile(target_pool, probabilities),
    ])


def apply_map(volume: np.ndarray, mapping: np.ndarray) -> np.ndarray:
    """Interpolate through the fitted map, leaving background at zero.

    np.interp clamps outside the fitted range, which is the wanted behaviour: an
    intensity brighter than anything seen in fitting maps to the brightest target value
    rather than extrapolating off the end of the curve.
    """
    source_knots, target_knots = mapping
    out = np.zeros_like(volume, dtype=np.float32)
    mask = volume > FOREGROUND
    out[mask] = np.interp(volume[mask], source_knots, target_knots).astype(np.float32)
    return out


def build_maps(cache_dir: Path, split: str, subjects: list[str]) -> dict[str, np.ndarray]:
    maps: dict[str, np.ndarray] = {}
    for modality in MODALITIES:
        for source in FIELDS:
            for target in FIELDS:
                if source == target:
                    continue
                pairs = [
                    (cached_volume(cache_dir, split, modality, source, s),
                     cached_volume(cache_dir, split, modality, target, s))
                    for s in subjects
                ]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if not pairs:
                    continue
                maps[f"{modality}|{source}|{target}"] = fit_map(
                    [np.asarray(a, np.float32) for a, _ in pairs],
                    [np.asarray(b, np.float32) for _, b in pairs],
                )
    return maps


def score(cache_dir: Path, split: str) -> None:
    """Leave-one-subject-out: fit on two subjects, score on the third, three times.

    Fitting and scoring on the same subject would let each volume calibrate itself and
    report a number no submission could reproduce.
    """
    import torch
    from skimage.metrics import structural_similarity
    from mrixfields.losses.perceptual import PerceptualLoss

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lpips_fn = PerceptualLoss(net="alex").to(device).eval()
    rows = []
    for held in PAIRED_SUBJECTS:
        fit_subjects = [s for s in PAIRED_SUBJECTS if s != held]
        maps = build_maps(cache_dir, split, fit_subjects)
        print(f"fold: fit on {fit_subjects}, score on {held} ({len(maps)} maps)", flush=True)
        for modality in MODALITIES:
            volumes = {f: cached_volume(cache_dir, split, modality, f, held) for f in FIELDS}
            for source in FIELDS:
                for target in FIELDS:
                    key = f"{modality}|{source}|{target}"
                    if source == target or key not in maps:
                        continue
                    if volumes[source] is None or volumes[target] is None:
                        continue
                    src = np.asarray(volumes[source], np.float32)
                    tgt = np.asarray(volumes[target], np.float32)
                    pred = apply_map(src, maps[key])
                    # score on the axial plane, matching scripts/eval_holdout.py
                    pred, tgt = pred.transpose(2, 0, 1), tgt.transpose(2, 0, 1)
                    mask = tgt > 1e-6
                    if not mask.any():
                        continue
                    diff = pred[mask] - tgt[mask]
                    nrmse = float(np.linalg.norm(diff) / (np.linalg.norm(tgt[mask]) + 1e-12))
                    ssim = float(np.mean([
                        structural_similarity(tgt[z], pred[z], data_range=1.0)
                        for z in range(tgt.shape[0]) if mask[z].any()
                    ]))
                    values = []
                    with torch.no_grad():
                        for start in range(0, pred.shape[0], 16):
                            p = torch.from_numpy(pred[start:start + 16]).unsqueeze(1).to(device)
                            t = torch.from_numpy(tgt[start:start + 16]).unsqueeze(1).to(device)
                            values.append(float(lpips_fn(p.mul(2).sub(1), t.mul(2).sub(1))))
                    rows.append({"subject": held, "modality": modality, "source": source,
                                 "target": target, "SSIM": ssim, "nRMSE": nrmse,
                                 "LPIPS": float(np.mean(values))})
                    print(f"  {held} {modality:8s} {source:5s}->{target:5s}  SSIM {ssim:.4f}  "
                          f"nRMSE {nrmse:.4f}  LPIPS {rows[-1]['LPIPS']:.4f}", flush=True)

    import csv
    out_dir = REPO_ROOT / "reports" / "quantile_calibration"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_transition.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for name in ("SSIM", "nRMSE", "LPIPS"):
        print(f"{name}: {np.mean([r[name] for r in rows]):.4f}", flush=True)
    print(f"({len(rows)} transitions across 3 LOSO folds)  wrote {out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("/home/ruru/Documents/deniz/mri_fields/data_preprocessed/volumes_3d"))
    parser.add_argument("--split", default="Training_prospective")
    parser.add_argument("--score", action="store_true", help="leave-one-subject-out scoring")
    parser.add_argument("--dump-maps", type=Path, help="write the fitted maps as .npz")
    args = parser.parse_args()

    if args.score:
        score(args.cache_dir, args.split)
    if args.dump_maps:
        maps = build_maps(args.cache_dir, args.split, list(PAIRED_SUBJECTS))
        np.savez_compressed(args.dump_maps, **maps)
        print(f"wrote {len(maps)} maps to {args.dump_maps}")
    if not args.score and not args.dump_maps:
        parser.error("pass --score or --dump-maps")


if __name__ == "__main__":
    main()
