"""E1: post-hoc detail gain on the shipped Task 3 predictions (no training).

reports/error_spectrum_report.tex (section 4, second pass) found the shipped prediction smoother
than the target in every band, and 0.0149 of the 0.0474 per-pixel SSIM deficit in the contrast
factor. Scaling the prediction's local deviation leaves SSIM's structure factor unchanged and
moves its contrast factor toward 1, so an unsharp mask with the right gain should raise SSIM
without new information:

    y' = m * clip(y + (g - 1) * (y - blur_sigma(y)), 0, 1),    m = source > 1e-3

blur_sigma is a normalised convolution inside m (G*(y m) / G*m), not a plain Gaussian: a plain
blur reaches across the brain boundary into the zeroed background and the unsharp mask then
overshoots the edge, where the luminance factor already dominates.

Stages (run in order; each reads what the previous wrote):
    cache   GPU. Shipped weights + 4-flip TTA through make_task3_submission.predict_slab,
            slab only; also the source/target slabs. ~2.5 GB under runs/detail_gain_cache/.
    sweep   CPU, parallel. SSIM and nRMSE per transition per setting (SETTINGS), with the formulas of
            scripts/eval_holdout.py::score (slab, source-zeroed, target > 1e-6 for nRMSE).
    select  Leave-one-subject-out: pick on two subjects, score on the third, all three rotations;
            one global setting, then one per target field. LPIPS for the baseline and the pick.

All three subjects were in training (holdout_subjects = []), so leave-one-subject-out guards
against tuning on the scored subject, not against the train/test gap.

    ~/anaconda3/envs/mri/bin/python scripts/detail_gain_sweep.py --stage cache
    ~/anaconda3/envs/mri/bin/python scripts/detail_gain_sweep.py --stage sweep
    ~/anaconda3/envs/mri/bin/python scripts/detail_gain_sweep.py --stage select
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline"), str(ROOT / "scripts")]

from error_spectrum_analysis import CORRUPT, FIELDS, MODALITIES, SUBJECTS, volume_path  # noqa: E402
from mrixfields.zclip_constants import Z_CLIP_RANGE  # noqa: E402

# (mode, gain, parameter). "unsharp": y + (g-1)(y - blur_sigma y), parameter = sigma in px.
# "window": scale the deviation from the 7x7 local mean (the SSIM window) by g, only where the
# prediction's local std exceeds tau, parameter = tau. Spot checks on five transitions showed
# unsharp g >= 1.75 and sigma = 3 strictly worse, so the grid stops there; "window" is the
# variant that lost least. ("none", 1, 0) is the baseline.
SETTINGS = ([("none", 1.0, 0.0)]
            + [("unsharp", g, s) for g in (1.1, 1.2, 1.3, 1.5) for s in (1.0, 2.0)]
            + [("window", g, t) for g in (1.05, 1.1, 1.2) for t in (0.0, 0.03, 0.06)])
SETTING_KEYS = ["mode", "gain", "param"]
MASK_THRESHOLD = 1e-3


def transitions():
    for subject, modality, source, target in itertools.product(SUBJECTS, MODALITIES, FIELDS, FIELDS):
        if source != target:
            yield subject, modality, source, target


def volume_file(cache: Path, subject: str, modality: str, field: str) -> Path:
    return cache / f"vol_{subject}_{modality}_{field}.npy"


def prediction_file(cache: Path, subject: str, modality: str, source: str, target: str) -> Path:
    return cache / f"pred_{subject}_{modality}_{source}_{target}.npy"


# ------------------------------------------------------------------------------ cache
def cache_predictions(args) -> None:
    import torch

    from make_task3_submission import build_model, predict_slab
    from mrixfields.data.utils import get_joint_domain, load_nifti
    from mrixfields.env import get_data_dir

    start, stop = Z_CLIP_RANGE
    args.cache.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model("conditional", args.checkpoint, device, "")
    split = Path(get_data_dir()) / "training_prospective"
    for subject, modality in itertools.product(SUBJECTS, MODALITIES):
        volumes = {f: load_nifti(volume_path(split, modality, f, subject))[0] for f in FIELDS}
        for field, volume in volumes.items():
            # (Z, X, Y): axial slice first, as eval_holdout.cached_volume returns it
            np.save(volume_file(args.cache, subject, modality, field),
                    np.ascontiguousarray(volume[:, :, start:stop].transpose(2, 0, 1), np.float32))
        for source in FIELDS:
            auxiliary = [load_nifti(volume_path(split, other, source, subject))[0]
                         for other in MODALITIES]
            for target in FIELDS:
                destination = prediction_file(args.cache, subject, modality, source, target)
                if target == source or destination.exists():
                    continue
                prediction = predict_slab(
                    model, volumes[source], get_joint_domain(modality, source),
                    get_joint_domain(modality, target), device, args.batch_size,
                    tta=True, auxiliary=auxiliary)
                np.save(destination, np.ascontiguousarray(
                    prediction[:, :, start:stop].transpose(2, 0, 1), np.float32))
            print(f"cached {subject} {modality} {source} -> *", flush=True)


# ------------------------------------------------------------------------------ sweep
def masked_filter(image: np.ndarray, weight: np.ndarray, filter_fn) -> np.ndarray:
    """Normalised convolution: the filter only averages pixels inside the mask."""
    numerator = filter_fn(image * weight)
    denominator = filter_fn(weight)
    return np.where(denominator > 1e-6, numerator / np.maximum(denominator, 1e-6), 0.0)


def sharpen(prediction: np.ndarray, mask: np.ndarray, mode: str, gain: float,
            param: float) -> np.ndarray:
    """Apply one setting per axial slice; see SETTINGS for the two modes."""
    if mode == "none":
        return prediction * mask
    out = np.empty_like(prediction)
    for z in range(prediction.shape[0]):
        image, weight = prediction[z], mask[z].astype(np.float32)
        if mode == "unsharp":
            blur = masked_filter(image, weight, lambda a: ndimage.gaussian_filter(a, param))
            out[z] = image + (gain - 1.0) * (image - blur)
        else:
            box = lambda a: ndimage.uniform_filter(a, 7)  # noqa: E731
            mean = masked_filter(image, weight, box)
            std = np.sqrt(np.maximum(masked_filter(image ** 2, weight, box) - mean ** 2, 0.0))
            # smooth the gate so the gain has no hard seam at the threshold
            local_gain = ndimage.uniform_filter(np.where(std > param, gain, 1.0), 7)
            out[z] = mean + local_gain * (image - mean)
    return np.clip(out, 0.0, 1.0) * mask


def metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """eval_holdout.score's SSIM and nRMSE; ``prediction`` is already source-zeroed."""
    from skimage.metrics import structural_similarity

    valid = target > 1e-6
    diff = prediction[valid] - target[valid]
    nrmse = float(np.linalg.norm(diff) / (np.linalg.norm(target[valid]) + 1e-12))
    ssim = float(np.mean([structural_similarity(target[z], prediction[z], data_range=1.0)
                          for z in range(target.shape[0]) if valid[z].any()]))
    return ssim, nrmse


def sweep_one(job):
    cache, (subject, modality, source, target) = job
    prediction = np.load(prediction_file(cache, subject, modality, source, target))
    mask = np.load(volume_file(cache, subject, modality, source)) > MASK_THRESHOLD
    truth = np.load(volume_file(cache, subject, modality, target))
    rows = []
    for mode, gain, param in SETTINGS:
        ssim, nrmse = metrics(sharpen(prediction, mask, mode, gain, param), truth)
        rows.append({"subject": subject, "modality": modality, "source": source,
                     "target": target, "corrupt": (modality, target) in CORRUPT,
                     "mode": mode, "gain": gain, "param": param, "ssim": ssim, "nrmse": nrmse})
    return rows


def run_sweep(args) -> None:
    jobs = [(args.cache, t) for t in transitions()]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k, result in enumerate(pool.map(sweep_one, jobs, chunksize=2), 1):
            rows.extend(result)
            if k % 20 == 0:
                print(f"swept {k}/{len(jobs)}", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "sweep.csv", index=False)


# ----------------------------------------------------------------------------- select
def best_setting(frame: pd.DataFrame) -> tuple[str, float, float]:
    means = frame.groupby(SETTING_KEYS)["ssim"].mean()
    return means.idxmax()


def pick(frame: pd.DataFrame, setting: tuple[str, float, float]) -> pd.DataFrame:
    mode, gain, param = setting
    return frame[(frame["mode"] == mode) & (frame["gain"] == gain) & (frame["param"] == param)]


def run_select(args) -> None:
    # subject IDs are zero-padded strings; without dtype, "0006" parses to the integer 6
    table = pd.read_csv(args.out_dir / "sweep.csv", dtype={"subject": str})
    ranked = table[~table["corrupt"]]
    base = table[table["mode"] == "none"].set_index(["subject", "modality", "source", "target"])

    grid = ranked.groupby(SETTING_KEYS)[["ssim", "nrmse"]].mean()
    grid["d_ssim"] = grid["ssim"] - grid.loc[("none", 1.0, 0.0), "ssim"]
    grid["d_nrmse"] = grid["nrmse"] - grid.loc[("none", 1.0, 0.0), "nrmse"]
    grid["improved"] = ranked.assign(d=ranked["ssim"] - ranked.join(
        base["ssim"].rename("b"), on=["subject", "modality", "source", "target"])["b"]
    ).groupby(SETTING_KEYS)["d"].apply(lambda d: (d > 0).mean())
    grid.to_csv(args.out_dir / "grid.csv")
    print("in-sample grid, 168 transitions (corrupt cell excluded):")
    print(grid.to_string(float_format="{:.5f}".format))

    picks = []        # one row per (rule, held-out subject[, target field])
    chosen = []       # the setting applied to each held-out transition, per rule
    for held in SUBJECTS:
        fit, test = ranked[ranked["subject"] != held], table[table["subject"] == held]
        setting = best_setting(fit)
        picks.append({"rule": "global", "held_out": held, "target": "all",
                      **dict(zip(SETTING_KEYS, setting))})
        chosen.append(pick(test, setting).assign(rule="global"))
        for field in FIELDS:
            setting = best_setting(fit[fit["target"] == field])
            picks.append({"rule": "per_target", "held_out": held, "target": field,
                          **dict(zip(SETTING_KEYS, setting))})
            chosen.append(pick(test[test["target"] == field], setting).assign(rule="per_target"))
    oracle = (ranked.loc[ranked.groupby(["subject", "modality", "source", "target"])["ssim"].idxmax()]
              .assign(rule="oracle_per_transition"))
    chosen = pd.concat(chosen + [oracle], ignore_index=True)
    keys = ["subject", "modality", "source", "target"]
    chosen = chosen.join(base[["ssim", "nrmse"]].rename(columns={"ssim": "ssim_base",
                                                                 "nrmse": "nrmse_base"}), on=keys)
    chosen["d_ssim"] = chosen["ssim"] - chosen["ssim_base"]
    chosen["d_nrmse"] = chosen["nrmse"] - chosen["nrmse_base"]
    pd.DataFrame(picks).to_csv(args.out_dir / "picks.csv", index=False)
    chosen.to_csv(args.out_dir / "loso.csv", index=False)

    print("\npicks (fit on the other two subjects):")
    print(pd.DataFrame(picks).to_string(index=False))
    summary = []
    for rule, part in chosen.groupby("rule"):
        for scope, sel in (("all 180", np.ones(len(part), bool)), ("168 ranked", ~part["corrupt"])):
            p = part[sel]
            summary.append({"rule": rule, "scope": scope, "n": len(p),
                            "ssim_base": p["ssim_base"].mean(), "ssim": p["ssim"].mean(),
                            "d_ssim": p["d_ssim"].mean(), "d_nrmse": p["d_nrmse"].mean(),
                            "improved": f"{(p['d_ssim'] > 0).mean():.0%}"})
        for held in SUBJECTS:
            p = part[(part["subject"] == held) & ~part["corrupt"]]
            summary.append({"rule": rule, "scope": f"held {held}", "n": len(p),
                            "ssim_base": p["ssim_base"].mean(), "ssim": p["ssim"].mean(),
                            "d_ssim": p["d_ssim"].mean(), "d_nrmse": p["d_nrmse"].mean(),
                            "improved": f"{(p['d_ssim'] > 0).mean():.0%}"})
    summary = pd.DataFrame(summary)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    print("\nleave-one-subject-out:")
    print(summary.to_string(index=False, float_format="{:.5f}".format))

    if args.lpips:
        lpips_for_global(args, table)


def lpips_for_global(args, table: pd.DataFrame) -> None:
    """LPIPS through eval_holdout.score for the baseline and the in-sample global pick."""
    import torch

    from eval_holdout import score
    from mrixfields.losses.perceptual import PerceptualLoss

    setting = best_setting(table[~table["corrupt"]])
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    lpips_fn = PerceptualLoss(net="alex").to(device).eval()
    rows = []
    for subject, modality, source, target in transitions():
        prediction = np.load(prediction_file(args.cache, subject, modality, source, target))
        source_slab = np.load(volume_file(args.cache, subject, modality, source))
        truth = np.load(volume_file(args.cache, subject, modality, target))
        mask = source_slab > MASK_THRESHOLD
        for label, chosen in (("base", ("none", 1.0, 0.0)), ("gain", setting)):
            # score() expects the full depth when slab_source is set; the cache already is the
            # slab, so zero it here and call score() without slab_source
            values = score(sharpen(prediction, mask, *chosen), truth, lpips_fn, device)
            rows.append({"subject": subject, "modality": modality, "source": source,
                         "target": target, "corrupt": (modality, target) in CORRUPT,
                         "setting": label, **values})
    frame = pd.DataFrame(rows)
    frame.to_csv(args.out_dir / "lpips.csv", index=False)
    print(f"\nLPIPS, global pick {setting}:")
    print(frame.groupby("setting")[["SSIM", "nRMSE", "LPIPS"]].mean().to_string(
        float_format="{:.5f}".format))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("cache", "sweep", "select"), required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "docker/task3/weights/task3.pt")
    parser.add_argument("--cache", type=Path, default=ROOT / "runs/detail_gain_cache")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/detail_gain")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--lpips", action="store_true", help="select: also score LPIPS")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    {"cache": cache_predictions, "sweep": run_sweep, "select": run_select}[args.stage](args)


if __name__ == "__main__":
    main()
