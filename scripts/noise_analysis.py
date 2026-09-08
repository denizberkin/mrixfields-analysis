#!/usr/bin/env python3
"""Test whether this dataset's noise is Rician or effectively Gaussian, per field strength.

Why it matters: 40% of our Task 3 score comes from transitions touching 0.1T, and those run
0.018 below the rest. 0.1T is the low-SNR regime, so if the noise there is Rician we can both
augment for it and generate synthetic low-field data from high-field volumes by resampling the
correct distribution -- and a diffusion model over this data would be denoising the wrong noise
if it assumes Gaussian.

Rayleigh is not an alternative hypothesis here: MRI magnitude images come from complex data with
i.i.d. Gaussian noise in the real and imaginary channels, so the magnitude is Rician(nu, sigma)
where nu is the true signal. Rayleigh is that distribution at nu = 0 (air) and Gaussian is its
nu >> sigma limit. The question is therefore *where on the Rician family* this data sits.

The background is masked to zero in this dataset, so the classic "fit Rayleigh to air" route is
unavailable. Two signatures work inside tissue instead:

1. ``sigma(signal)`` --- the noise level as a function of local intensity. Additive Gaussian
   noise gives a flat line. Rician gives a rise from 0.655*sigma at zero signal to sigma at high
   signal, because the magnitude operation compresses variance where the signal is weak.

2. ``skew(residual | signal)`` --- the skewness of the local residual. Gaussian noise is
   symmetric everywhere. Rician noise is right-skewed at low signal (the magnitude cannot go
   below zero) and becomes symmetric as the signal grows.

Both are estimated from a *lower envelope*: in a homogeneous patch the local standard deviation
is the noise level, while anatomy only ever adds variance, so the low percentile of local sd
within an intensity bin estimates sigma without needing a tissue segmentation.

Caveat this script cannot remove: registration and resampling average neighbouring voxels, which
pushes any marginal distribution toward Gaussian by the central limit theorem and correlates the
noise spatially. A Gaussian-looking result is therefore evidence about the *preprocessed* data we
train on --- which is the thing that matters for augmentation --- not about the scanner.

Usage:
    python scripts/noise_analysis.py --subjects 0006 0007 --modality T1W
    python scripts/noise_analysis.py --out-dir reports/noise --tex reports/tables/noise.tex
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")


def local_stats(plane: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Local mean and standard deviation over a square window, 'valid' region only."""
    from scipy.ndimage import uniform_filter

    mean = uniform_filter(plane, window, mode="reflect")
    mean_sq = uniform_filter(plane * plane, window, mode="reflect")
    var = np.clip(mean_sq - mean * mean, 0, None)
    return mean, np.sqrt(var)


def sigma_versus_signal(volume: np.ndarray, window: int, bins: int,
                        envelope: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Noise level as a function of local intensity, plus residual skewness per bin.

    Only windows lying entirely inside the brain mask are used: a window straddling the mask
    edge has a huge local sd that has nothing to do with noise.
    """
    from scipy.ndimage import binary_erosion, uniform_filter

    means, sds, skews = [], [], []
    inside_all, mean_all, sd_all, resid_all = [], [], [], []

    for z in range(volume.shape[2]):
        plane = volume[:, :, z]
        if (plane > 0).mean() < 0.15:          # skip slices that are mostly outside the head
            continue
        mask = binary_erosion(plane > 0, np.ones((window + 2, window + 2), bool))
        if mask.sum() < 500:
            continue
        mean, sd = local_stats(plane, window)
        smooth = uniform_filter(plane, window, mode="reflect")
        inside_all.append(mask)
        mean_all.append(mean[mask])
        sd_all.append(sd[mask])
        resid_all.append((plane - smooth)[mask])

    mean_v = np.concatenate(mean_all)
    sd_v = np.concatenate(sd_all)
    res_v = np.concatenate(resid_all)

    # Bin by local intensity; within each bin take a low percentile of the local sd as the
    # noise estimate, because anatomy can only push local sd up, never down.
    edges = np.quantile(mean_v, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    centres, sigmas, skewness = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (mean_v >= lo) & (mean_v < hi)
        if sel.sum() < 2000:
            continue
        centres.append(float(np.median(mean_v[sel])))
        sigmas.append(float(np.percentile(sd_v[sel], envelope)))
        r = res_v[sel]
        # skewness of the quietest residuals in the bin, for the same reason as the envelope
        quiet = r[np.abs(r) <= np.percentile(np.abs(r), 60)]
        s = quiet.std()
        skewness.append(float(((quiet - quiet.mean()) ** 3).mean() / (s ** 3)) if s > 0 else 0.0)
    return np.array(centres), np.array(sigmas), np.array(skewness)


def rician_sigma_curve(signal: np.ndarray, sigma: float) -> np.ndarray:
    """sd of a Rician magnitude as a function of the underlying signal nu, for fixed sigma."""
    from scipy.special import eval_laguerre  # noqa: F401  (documented fallback below)
    from scipy.stats import rice

    return np.array([rice.std(max(nu, 0) / sigma, scale=sigma) for nu in signal])


def fit_sigma(centres: np.ndarray, sigmas: np.ndarray) -> tuple[float, float, float]:
    """Fit the Rician sigma(signal) curve and a flat Gaussian line; return sigma and both RMSEs.

    The Rician curve has one free parameter, the same as the flat line, so the comparison is
    like for like.
    """
    from scipy.optimize import minimize_scalar

    def rician_rmse(s):
        if s <= 0:
            return 1e9
        # nu is recovered from the observed local mean: E[M] ~ sqrt(nu^2 + sigma^2)
        nu = np.sqrt(np.clip(centres ** 2 - s ** 2, 0, None))
        return float(np.sqrt(np.mean((rician_sigma_curve(nu, s) - sigmas) ** 2)))

    best = minimize_scalar(rician_rmse, bounds=(1e-5, max(sigmas.max() * 4, 1e-3)),
                           method="bounded")
    flat = float(np.sqrt(np.mean((sigmas - sigmas.mean()) ** 2)))
    return float(best.x), float(best.fun), flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="training_prospective")
    parser.add_argument("--modality", default="T1W")
    parser.add_argument("--subjects", nargs="+", default=["0006"])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--envelope", type=float, default=10.0,
                        help="percentile of local sd taken as the noise level in each bin")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "noise")
    args = parser.parse_args()

    import nibabel as nib
    from mrixfields.env import get_data_dir, load_env

    load_env()
    data_dir = args.data_dir or Path(get_data_dir())
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for field in FIELDS:
        per_subject = []
        for subject in args.subjects:
            path = (data_dir / args.split / args.modality / field
                    / f"P_{args.modality}_{field}_{subject}.nii.gz")
            if not path.is_file():
                print(f"  missing {path}", flush=True)
                continue
            volume = np.asarray(nib.load(str(path)).dataobj, np.float32)
            centres, sigmas, skews = sigma_versus_signal(volume, args.window, args.bins,
                                                         args.envelope)
            if len(centres) < 4:
                continue
            sigma, rmse_rice, rmse_flat = fit_sigma(centres, sigmas)
            low = skews[: max(1, len(skews) // 3)].mean()
            high = skews[-max(1, len(skews) // 3):].mean()
            per_subject.append((sigma, rmse_rice, rmse_flat, low, high,
                                sigmas.min(), sigmas.max(), np.median(centres)))
            for c, s, k in zip(centres, sigmas, skews):
                rows.append({"field": field, "subject": subject, "signal": c,
                             "sigma": s, "skew": k})
        if not per_subject:
            continue
        a = np.array(per_subject)
        print(f"{field:<5} sigma {a[:,0].mean():.5f}   "
              f"RMSE rician {a[:,1].mean():.2e} vs flat {a[:,2].mean():.2e}   "
              f"ratio {a[:,2].mean()/max(a[:,1].mean(),1e-12):.2f}x   "
              f"skew low-signal {a[:,3].mean():+.3f} high-signal {a[:,4].mean():+.3f}   "
              f"sigma range {a[:,5].mean():.5f}-{a[:,6].mean():.5f}", flush=True)

    out = args.out_dir / f"noise_{args.modality}.csv"
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out}  ({len(rows)} bins)")


if __name__ == "__main__":
    main()
