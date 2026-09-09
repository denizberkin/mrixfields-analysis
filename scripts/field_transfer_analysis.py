#!/usr/bin/env python3
"""Is the field-strength axis a smooth intensity transform, or is it spatial?

Task 3 maps the same registered anatomy between field strengths, so the question a
generative model of the field axis would rest on is whether the mapping is, to first order,
a *pointwise intensity transfer* -- I_target = T(I_source) -- rather than something that
moves structure around. If it is, three things follow: the transfer curves should explain
most of the target's variance, they should compose (T_{3->7} = T_{5->7} o T_{3->5}), and the
residual left over is the part any model actually has to learn.

Three tests, in increasing strength:

1. ``affine``  -- I_target = a*I_source + b, one (a, b) per (modality, field pair). The
   crudest possible transfer; its R^2 is the floor.
2. ``curve``   -- a monotone lookup built from per-quantile medians of the source intensity.
   This is the best possible *pointwise* transfer, so its R^2 is the ceiling for anything
   that ignores spatial context, and 1 - R^2 is what genuinely requires a model.
3. ``compose`` -- does T_{i->k} equal T_{j->k} o T_{i->j}? A composable family means the
   field axis is a one-parameter flow and intermediate or extrapolated fields could be
   synthesised. A non-composable one means each pair is its own problem.

Also reports corr(D(7,5), D(5,3)) on the difference images: if adjacent-field differences are
proportional, the axis is close to a single direction scaled by field.

Everything is measured on held-out-capable data: pass --subjects to restrict, and the curves
are fitted per subject so a curve fitted on one subject can be applied to another
(--transfer-subject) to see whether the relation is anatomy-independent at all.

Usage:
    python scripts/field_transfer_analysis.py --subjects 0006 0007 0009
    python scripts/field_transfer_analysis.py --modalities T1W --transfer-subject 0009
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
MODALITIES = ("T1W", "T2W", "T2FLAIR")


def load(data_dir: Path, split: str, modality: str, field: str, subject: str):
    import nibabel as nib
    path = data_dir / split / modality / field / f"P_{modality}_{field}_{subject}.nii.gz"
    if not path.is_file():
        return None
    return np.asarray(nib.load(str(path)).dataobj, np.float32)


def r_squared(pred: np.ndarray, truth: np.ndarray) -> float:
    ss_res = float(np.sum((truth - pred) ** 2))
    ss_tot = float(np.sum((truth - truth.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_curve(source: np.ndarray, target: np.ndarray, bins: int):
    """Monotone pointwise transfer: per-quantile median of the target, as a lookup."""
    edges = np.unique(np.quantile(source, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    centres = 0.5 * (edges[:-1] + edges[1:])
    values = np.empty(len(centres), np.float32)
    index = np.clip(np.searchsorted(edges, source, side="right") - 1, 0, len(centres) - 1)
    for b in range(len(centres)):
        sel = index == b
        values[b] = np.median(target[sel]) if sel.sum() >= 32 else np.nan
    good = ~np.isnan(values)
    if good.sum() < 3:
        return None
    return centres[good], values[good]


def apply_curve(curve, source: np.ndarray) -> np.ndarray:
    x, y = curve
    return np.interp(source, x, y).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="training_prospective")
    parser.add_argument("--subjects", nargs="+", default=["0006", "0007", "0009"])
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--voxels", type=int, default=400_000,
                        help="random masked voxels sampled per subject/modality")
    parser.add_argument("--transfer-subject", default=None,
                        help="apply curves fitted on the other subjects to this one")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from mrixfields.env import get_data_dir, load_env
    load_env()
    data_dir = args.data_dir or Path(get_data_dir())
    rng = np.random.default_rng(args.seed)

    for modality in args.modalities:
        print(f"\n{'='*78}\n{modality}\n{'='*78}")
        samples = {}
        for subject in args.subjects:
            volumes = {f: load(data_dir, args.split, modality, f, subject) for f in FIELDS}
            if any(v is None for v in volumes.values()):
                print(f"  {subject}: missing a field, skipped")
                continue
            mask = np.ones_like(volumes[FIELDS[0]], bool)
            for v in volumes.values():
                mask &= v > 1e-3
            idx = np.flatnonzero(mask.ravel())
            if len(idx) > args.voxels:
                idx = rng.choice(idx, args.voxels, replace=False)
            samples[subject] = {f: volumes[f].ravel()[idx] for f in FIELDS}
            print(f"  {subject}: {mask.sum():,} masked voxels, sampled {len(idx):,}")

        if not samples:
            continue

        # ---- 1 & 2: how much of the target is a pointwise function of the source
        print(f"\n  {'pair':<14}{'affine R2':>11}{'curve R2':>11}{'resid sd':>11}"
              f"{'gain a':>9}{'offset b':>10}")
        for i in range(len(FIELDS) - 1):
            src_f, tgt_f = FIELDS[i], FIELDS[i + 1]
            rows = []
            for subject, per_field in samples.items():
                s, t = per_field[src_f], per_field[tgt_f]
                a, b = np.polyfit(s, t, 1)
                curve = fit_curve(s, t, args.bins)
                r_curve = r_squared(apply_curve(curve, s), t) if curve else float("nan")
                rows.append((r_squared(a * s + b, t), r_curve,
                             float(np.std(t - apply_curve(curve, s))) if curve else np.nan,
                             float(a), float(b)))
            m = np.nanmean(np.array(rows), axis=0)
            print(f"  {src_f+'->'+tgt_f:<14}{m[0]:>11.4f}{m[1]:>11.4f}{m[2]:>11.4f}"
                  f"{m[3]:>9.3f}{m[4]:>10.4f}")

        # ---- 3: composition. T_{i->k} vs T_{j->k} o T_{i->j}
        print(f"\n  composition  (does T(i->j) then T(j->k) equal T(i->k)?)")
        print(f"  {'path':<22}{'direct R2':>11}{'composed R2':>13}{'loss':>9}")
        for i in range(len(FIELDS) - 2):
            a_f, b_f, c_f = FIELDS[i], FIELDS[i + 1], FIELDS[i + 2]
            direct, composed = [], []
            for per_field in samples.values():
                x, y, z = per_field[a_f], per_field[b_f], per_field[c_f]
                c_ac = fit_curve(x, z, args.bins)
                c_ab = fit_curve(x, y, args.bins)
                c_bc = fit_curve(y, z, args.bins)
                if not (c_ac and c_ab and c_bc):
                    continue
                direct.append(r_squared(apply_curve(c_ac, x), z))
                composed.append(r_squared(apply_curve(c_bc, apply_curve(c_ab, x)), z))
            if direct:
                d, c = float(np.mean(direct)), float(np.mean(composed))
                print(f"  {f'{a_f}->{b_f}->{c_f}':<22}{d:>11.4f}{c:>13.4f}{d-c:>9.4f}")

        # ---- difference images: is D(7,5) proportional to D(5,3)?
        print(f"\n  difference correlations")
        pairs = [("7T", "5T"), ("5T", "3T"), ("3T", "1.5T"), ("1.5T", "0.1T")]
        diffs = {}
        for hi, lo in pairs:
            diffs[f"{hi}-{lo}"] = np.mean(
                [per[hi] - per[lo] for per in samples.values()], axis=0)
        keys = list(diffs)
        for j in range(len(keys) - 1):
            u, v = diffs[keys[j]], diffs[keys[j + 1]]
            r = float(np.corrcoef(u, v)[0, 1])
            slope = float(np.polyfit(v, u, 1)[0])
            print(f"    corr({keys[j]:<9}, {keys[j+1]:<9}) = {r:+.4f}   "
                  f"slope {slope:+.3f}")

        # ---- does a curve fitted on one subject transfer to another?
        if args.transfer_subject and args.transfer_subject in samples and len(samples) > 1:
            held = args.transfer_subject
            donors = [s for s in samples if s != held]
            print(f"\n  cross-subject transfer: curves fitted on {'+'.join(donors)} "
                  f"applied to {held}")
            print(f"  {'pair':<14}{'own R2':>10}{'donor R2':>10}{'drop':>9}")
            for i in range(len(FIELDS) - 1):
                src_f, tgt_f = FIELDS[i], FIELDS[i + 1]
                s_h, t_h = samples[held][src_f], samples[held][tgt_f]
                own = fit_curve(s_h, t_h, args.bins)
                s_d = np.concatenate([samples[d][src_f] for d in donors])
                t_d = np.concatenate([samples[d][tgt_f] for d in donors])
                donor = fit_curve(s_d, t_d, args.bins)
                if not (own and donor):
                    continue
                r_own = r_squared(apply_curve(own, s_h), t_h)
                r_donor = r_squared(apply_curve(donor, s_h), t_h)
                print(f"  {src_f+'->'+tgt_f:<14}{r_own:>10.4f}{r_donor:>10.4f}"
                      f"{r_own-r_donor:>9.4f}")


if __name__ == "__main__":
    main()
