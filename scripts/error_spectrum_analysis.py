"""Where the shipped Task 3 model's error sits, in frequency and in space.

For every transition (3 paired subjects x 3 modalities x 20 ordered field pairs = 180),
predict the submission slab with the shipped weights and 4-flip TTA, then measure:

* frequency: the radial power spectrum of the error e = y_hat * mask(x) - y, of the target
  y, and of the copy baseline's error x * mask(x) - y. Each slice is zero-padded to a square
  rather than cropped, so the head is never cut and no artificial edge leaks into the bins.
  Power is *summed* per annulus (not averaged), so band shares are energy shares (Parseval).
* space: the per-pixel 1 - SSIM map (the ranked metric; its image mean is exactly the
  slice's SSIM deficit) and e^2, split by deciles of target edge strength |grad y| inside the
  brain, and by region: outer edge (within EDGE_PX of the brain boundary, either side),
  folds (interior, top 20% of |grad y|), flat (the rest of the interior), background.

Inference is not reimplemented: build_model and predict_slab come from
scripts/make_task3_submission.py, the path that produced the 0.913652 upload.

All three subjects were in training (every config has holdout_subjects = []), so absolute
errors are optimistic; the *split* of the error is what this script is for.

Usage (writes reports/error_spectrum/ and reports/spectral/fig_error_*.pdf):
    ~/anaconda3/envs/mri/bin/python scripts/error_spectrum_analysis.py
    ~/anaconda3/envs/mri/bin/python scripts/error_spectrum_analysis.py --figures-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage, stats
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline")]

from mrixfields.data.utils import get_joint_domain, load_nifti  # noqa: E402
from mrixfields.env import get_data_dir  # noqa: E402
from mrixfields.zclip_constants import Z_CLIP_RANGE  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
from make_task3_submission import build_model, predict_slab  # noqa: E402

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELDS = ("0.1T", "1.5T", "3T", "5T", "7T")
SUBJECTS = ("0006", "0007", "0009")
MASK_THRESHOLD = 1e-3          # the challenge's background mask, x > 1e-3
EDGE_PX = 4                    # half-width of the outer-edge band, pixels
FOLD_QUANTILE = 0.80           # interior pixels above this |grad y| quantile are "folds"
BANDS = {"low": (0.0, 0.05), "mid": (0.05, 0.20), "high": (0.20, 0.71)}  # cycles/pixel
FIT_BAND = (0.02, 0.20)        # the alpha fit band of scripts/spectral_analysis.py
FINE_BANDS = ((0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50))
SPEARMAN_SAMPLE = 20000
# memory: two T2W 1.5T ground-truth volumes have no zero air; that cell is excluded from
# every aggregate here and kept in the CSV with a flag.
CORRUPT = {("T2W", "1.5T")}

# ordinal ramp, one hue, validated (dataviz validate_palette.js --ordinal, light): 0.1 T darkest
FIELD_COLOURS = dict(zip(FIELDS, ("#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec")))
AWAY, TOWARD = "#eb6834", "#2a78d6"     # categorical slots 2 and 1
INK, MUTED, GRID = "#102A43", "#627D98", "#D9E2EC"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "legend.frameon": False, "figure.dpi": 200, "savefig.bbox": "tight",
})


def volume_path(split: Path, modality: str, field: str, subject: str) -> Path:
    path = split / modality / field / f"P_{modality}_{field}_{subject}.nii.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_alpha() -> dict[tuple[str, str], float]:
    table = pd.read_csv(ROOT / "reports/spectral/alpha_summary.csv")
    table = table[table["split"] == "retro"]
    return {(row.modality, row.field): row.alpha for row in table.itertuples()}


def annulus_energy(field: np.ndarray, side: int) -> np.ndarray:
    """Spectral energy summed over integer-radius annuli of the zero-padded square.

    Same FFT normalisation as spectral_analysis.rapsd (|F|^2 / N); bin r covers radius
    [r, r+1), i.e. frequency r / side cycles/pixel, up to the corner (r < side/sqrt 2).
    """
    padded = np.zeros((side, side), dtype=np.float64)
    top, left = (side - field.shape[0]) // 2, (side - field.shape[1]) // 2
    padded[top:top + field.shape[0], left:left + field.shape[1]] = field
    power = np.abs(np.fft.fftshift(np.fft.fft2(padded))) ** 2 / padded.size
    y, x = np.indices(power.shape)
    radius = np.hypot(x - side // 2, y - side // 2).astype(int)
    return np.bincount(radius.ravel(), weights=power.ravel())


def ssim_components(target, prediction, check: bool = False):
    """SSIM map split exactly into luminance x contrast x structure.

    Same window and constants as skimage.metrics.structural_similarity's defaults, which the
    scorer uses (7x7 uniform, sample covariance, K1 = 0.01, K2 = 0.03, data_range 1), with
    C3 = C2 / 2 so that contrast * structure equals SSIM's second factor exactly. ``check``
    asserts the product against skimage's own map.
    """
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    window = 7
    cov_norm = window ** 2 / (window ** 2 - 1)
    mean_t = ndimage.uniform_filter(target, window)
    mean_p = ndimage.uniform_filter(prediction, window)
    var_t = cov_norm * (ndimage.uniform_filter(target * target, window) - mean_t ** 2)
    var_p = cov_norm * (ndimage.uniform_filter(prediction * prediction, window) - mean_p ** 2)
    cov = cov_norm * (ndimage.uniform_filter(target * prediction, window) - mean_t * mean_p)
    std_t, std_p = np.sqrt(np.maximum(var_t, 0)), np.sqrt(np.maximum(var_p, 0))
    luminance = (2 * mean_t * mean_p + c1) / (mean_t ** 2 + mean_p ** 2 + c1)
    contrast = (2 * std_t * std_p + c2) / (var_t + var_p + c2)
    structure = (cov + c2 / 2) / (std_t * std_p + c2 / 2)
    ssim_map = luminance * contrast * structure
    if check:
        _, reference = structural_similarity(target, prediction, data_range=1.0, full=True)
        gap = float(np.abs(reference - ssim_map).max())
        assert gap < 1e-6, f"SSIM decomposition disagrees with skimage by {gap:.2e}"
    return ssim_map, luminance, contrast, structure, var_t, var_p


def slice_measures(target, prediction, source_mask, side, check: bool = False):
    """All per-slice quantities for one axial slice, as sums (aggregated later)."""
    error = prediction - target
    ssim_map, luminance, contrast, structure, var_t, var_p = ssim_components(
        target, prediction, check)
    deficit = 1.0 - ssim_map
    squared = error ** 2

    brain = source_mask
    inside = ndimage.distance_transform_edt(brain)
    outside = ndimage.distance_transform_edt(~brain)
    edge = (brain & (inside <= EDGE_PX)) | (~brain & (outside <= EDGE_PX))
    interior = brain & ~edge
    gradient = np.hypot(ndimage.sobel(target, 0), ndimage.sobel(target, 1))
    fold_cut = np.quantile(gradient[interior], FOLD_QUANTILE) if interior.any() else np.inf
    folds = interior & (gradient >= fold_cut)
    regions = {"edge": edge, "folds": folds, "flat": interior & ~folds,
               "background": ~brain & ~edge}

    in_brain = gradient[brain]
    cuts = np.quantile(in_brain, np.linspace(0.1, 0.9, 9))
    labels = np.digitize(in_brain, cuts)

    out = {"ssim": float(ssim_map.mean())}
    for name, region in regions.items():
        out[f"deficit_{name}"] = float(deficit[region].sum())
        out[f"sq_{name}"] = float(squared[region].sum())
        out[f"px_{name}"] = int(region.sum())
        # which SSIM factor carries the deficit, and is the prediction locally smoother
        out[f"lum_{name}"] = float((1 - luminance[region]).sum())
        out[f"con_{name}"] = float((1 - contrast[region]).sum())
        out[f"str_{name}"] = float((1 - structure[region]).sum())
        out[f"vart_{name}"] = float(var_t[region].sum())
        out[f"varp_{name}"] = float(var_p[region].sum())
        out[f"smoother_{name}"] = int((var_p[region] < var_t[region]).sum())
    out["deficit_decile"] = np.bincount(labels, weights=deficit[brain], minlength=10)
    out["sq_decile"] = np.bincount(labels, weights=squared[brain], minlength=10)
    count = int(brain.sum())
    sample = np.random.default_rng(0).choice(count, min(SPEARMAN_SAMPLE // 30, count),
                                             replace=False)
    out["pairs"] = (np.abs(error[brain])[sample], in_brain[sample])
    out["energy_error"] = annulus_energy(error, side)
    out["energy_target"] = annulus_energy(target, side)
    out["energy_prediction"] = annulus_energy(prediction, side)
    return out


def compute(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model("conditional", args.checkpoint, device, "")
    split = Path(get_data_dir()) / "training_prospective"
    alpha = load_alpha()
    start, stop = Z_CLIP_RANGE

    rows, spectra = [], {"error": [], "target": [], "copy": [], "prediction": []}
    for subject in SUBJECTS:
        for modality in MODALITIES:
            volumes = {field: load_nifti(volume_path(split, modality, field, subject))[0]
                       for field in FIELDS}
            for source_field in FIELDS:
                source = volumes[source_field]
                # other contrasts at the source field, fixed (T1W, T2W, T2FLAIR) order
                auxiliary = [load_nifti(volume_path(split, other, source_field, subject))[0]
                             for other in MODALITIES]
                mask = source > MASK_THRESHOLD
                for target_field in FIELDS:
                    if target_field == source_field:
                        continue
                    target = volumes[target_field]
                    prediction = predict_slab(
                        model, source, get_joint_domain(modality, source_field),
                        get_joint_domain(modality, target_field), device, args.batch_size,
                        tta=True, auxiliary=auxiliary) * mask
                    copy = source * mask
                    side = max(source.shape[:2])
                    sums, pairs, copy_ssim = {}, [], []
                    energy = {k: 0.0 for k in spectra}
                    for z in range(start, stop):
                        y = target[:, :, z].astype(np.float64)
                        if not (y > 1e-6).any():
                            continue       # slab_ssim skips empty slices too
                        m = mask[:, :, z]
                        if m.sum() < 100:
                            continue
                        measures = slice_measures(y, prediction[:, :, z].astype(np.float64), m,
                                                  side, check=not sums)
                        copy_ssim.append(structural_similarity(
                            y, copy[:, :, z].astype(np.float64), data_range=1.0))
                        for kind in ("error", "target", "prediction"):
                            energy[kind] = energy[kind] + measures.pop(f"energy_{kind}")
                        energy["copy"] = energy["copy"] + annulus_energy(
                            copy[:, :, z] - y, side)
                        pairs.append(measures.pop("pairs"))
                        for key, value in measures.items():
                            sums.setdefault(key, []).append(value)
                    abs_error = np.concatenate([p[0] for p in pairs])
                    grad = np.concatenate([p[1] for p in pairs])
                    row = {"subject": subject, "modality": modality, "source": source_field,
                           "target": target_field,
                           "delta_alpha": alpha[(modality, target_field)] - alpha[(modality, source_field)],
                           "corrupt": (modality, target_field) in CORRUPT,
                           "ssim": float(np.mean(sums.pop("ssim"))),
                           "ssim_copy": float(np.mean(copy_ssim)),
                           "spearman_abs_error_grad": float(stats.spearmanr(abs_error, grad)[0])}
                    for key in ("deficit_decile", "sq_decile"):
                        total = np.sum(sums.pop(key), axis=0)
                        row.update({f"{key}_{i + 1}": v for i, v in enumerate(total)})
                    row.update({key: float(np.sum(values)) for key, values in sums.items()})
                    rows.append(row)
                    for key in spectra:
                        spectra[key].append(energy[key])
                    print(f"{subject} {modality:7s} {source_field:>4s} -> {target_field:<4s} "
                          f"ssim {row['ssim']:.4f} (copy {row['ssim_copy']:.4f}) "
                          f"da {row['delta_alpha']:+.2f} rho {row['spearman_abs_error_grad']:.2f}",
                          flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "per_transition.csv", index=False)
    length = min(len(s) for s in spectra["error"])
    np.savez_compressed(args.out_dir / "spectra.npz",
                        frequency=np.arange(length) / side,
                        **{k: np.stack([s[:length] for s in v]) for k, v in spectra.items()})


# ------------------------------------------------------------------------- summaries
def band_shares(energy: np.ndarray, frequency: np.ndarray) -> dict[str, float]:
    total = energy.sum()
    return {band: float(energy[(frequency >= lo) & (frequency < hi)].sum() / total)
            for band, (lo, hi) in BANDS.items()}


def summarise(out_dir: Path) -> pd.DataFrame:
    table = pd.read_csv(out_dir / "per_transition.csv")
    data = np.load(out_dir / "spectra.npz")
    frequency = data["frequency"]
    keep = ~table["corrupt"].to_numpy()

    def group_rows(name, selector):
        sel = selector & keep
        part = table[sel]
        row = {"group": name, "n": int(sel.sum()), "ssim": part["ssim"].mean(),
               "ssim_copy": part["ssim_copy"].mean(),
               "rho": part["spearman_abs_error_grad"].mean()}
        for prefix in ("deficit", "sq"):
            total = sum(part[f"{prefix}_{r}"].sum() for r in ("edge", "folds", "flat", "background"))
            for region in ("edge", "folds", "flat", "background"):
                row[f"{prefix}_share_{region}"] = part[f"{prefix}_{region}"].sum() / total
            deciles = np.array([part[f"{prefix}_decile_{i}"].sum() for i in range(1, 11)])
            row[f"{prefix}_top2_deciles"] = deciles[-2:].sum() / deciles.sum()
        brain_px = sum(part[f"px_{r}"].sum() for r in ("edge", "folds", "flat"))
        for region in ("edge", "folds", "flat"):
            row[f"px_share_{region}"] = part[f"px_{region}"].sum() / brain_px
        for kind in ("error", "copy"):
            for band, share in band_shares(data[kind][sel].sum(0), frequency).items():
                row[f"{kind}_{band}"] = share
        # power ratios per fine band: error/target, error/copy-error, prediction/target
        energies = {kind: data[kind][sel].sum(0) for kind in ("error", "target", "copy", "prediction")}
        for lo, hi in FINE_BANDS:
            band = (frequency >= lo) & (frequency < hi)
            tag = f"{lo:.2f}_{hi:.2f}"
            row[f"e_over_y_{tag}"] = energies["error"][band].sum() / energies["target"][band].sum()
            row[f"e_over_copy_{tag}"] = energies["error"][band].sum() / energies["copy"][band].sum()
            row[f"p_over_y_{tag}"] = energies["prediction"][band].sum() / energies["target"][band].sum()
        for region in ("edge", "folds", "flat"):
            pixels = part[f"px_{region}"].sum()
            for factor in ("lum", "con", "str"):
                row[f"{factor}_{region}"] = part[f"{factor}_{region}"].sum() / pixels
            row[f"deficit_mean_{region}"] = part[f"deficit_{region}"].sum() / pixels
            row[f"var_ratio_{region}"] = part[f"varp_{region}"].sum() / part[f"vart_{region}"].sum()
            row[f"smoother_{region}"] = part[f"smoother_{region}"].sum() / pixels
        return row

    groups = [group_rows("all", np.ones(len(table), bool))]
    groups += [group_rows(f"src {f}", (table["source"] == f).to_numpy()) for f in FIELDS]
    groups += [group_rows(f"tgt {f}", (table["target"] == f).to_numpy()) for f in FIELDS]
    groups += [group_rows("da > 0 (away)", (table["delta_alpha"] > 0).to_numpy()),
               group_rows("da < 0 (toward)", (table["delta_alpha"] < 0).to_numpy())]
    groups += [group_rows("no 0.1T", ((table["source"] != "0.1T")
                                      & (table["target"] != "0.1T")).to_numpy())]
    summary = pd.DataFrame(groups)
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


# --------------------------------------------------------------------------- figures
def figures(out_dir: Path, figure_dir: Path) -> None:
    table = pd.read_csv(out_dir / "per_transition.csv")
    data = np.load(out_dir / "spectra.npz")
    frequency = data["frequency"]
    keep = ~table["corrupt"].to_numpy()
    valid = (frequency > 0) & (frequency <= 0.5)

    figure, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    for field in FIELDS:
        sel = keep & (table["source"] == field).to_numpy()
        error, target = data["error"][sel].sum(0), data["target"][sel].sum(0)
        axes[0].plot(frequency[valid], (error / target)[valid], color=FIELD_COLOURS[field],
                     label=f"{field.replace('T', ' T')}")
        axes[1].plot(frequency[valid], np.cumsum(error[valid]) / error[valid].sum(),
                     color=FIELD_COLOURS[field])
    copy = data["copy"][keep].sum(0)
    axes[1].plot(frequency[valid], np.cumsum(copy[valid]) / copy[valid].sum(),
                 color=MUTED, linestyle="--", linewidth=1.2, label="copy baseline")
    for name, colour, sign in (("away from high field ($\\Delta\\alpha>0$)", AWAY, 1),
                               ("toward high field ($\\Delta\\alpha<0$)", TOWARD, -1)):
        sel = keep & (np.sign(table["delta_alpha"]) == sign).to_numpy()
        error, target = data["error"][sel].sum(0), data["target"][sel].sum(0)
        axes[2].plot(frequency[valid], (error / target)[valid], color=colour, label=name)
    for axis in (axes[0], axes[2]):
        axis.set_yscale("log")
        axis.set_ylabel("error power / target power")
    axes[1].set_ylabel("cumulative share of error energy")
    for axis, title in zip(axes, ("a · relative error, by source field",
                                  "b · cumulative error",
                                  "c · relative error, by direction")):
        axis.set_xscale("log")
        axis.set_xlabel("frequency (cycles/pixel)")
        axis.axvspan(*FIT_BAND, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        axis.set_title(title, fontsize=9, loc="left", color=INK)
    axes[0].legend(title="source", fontsize=7.5, title_fontsize=7.5)
    axes[1].legend(fontsize=7.5, loc="upper left")
    axes[2].legend(fontsize=7.5, loc="upper left")
    figure.tight_layout()
    figure.savefig(figure_dir / "fig_error_rapsd_by_field.pdf")
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(8.6, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
    for field in FIELDS:
        part = table[keep & (table["source"] == field).to_numpy()]
        deciles = np.array([part[f"deficit_decile_{i}"].sum() for i in range(1, 11)])
        axes[0].plot(range(1, 11), 100 * deciles / deciles.sum(), color=FIELD_COLOURS[field],
                     marker="o", markersize=3.5, label=field.replace("T", " T"))
    axes[0].axhline(10, color=MUTED, linestyle="--", linewidth=1.0)
    axes[0].text(1.1, 10.6, "uniform (10%)", color=MUTED, fontsize=7.5)
    axes[0].set_xticks(range(1, 11))
    axes[0].set_xlabel("decile of target edge strength $|\\nabla y|$ (inside brain)")
    axes[0].set_ylabel("share of $1-\\mathrm{SSIM}$ (%)")
    axes[0].set_title("a · error by edge strength, by source field", fontsize=9, loc="left")
    axes[0].legend(title="source", fontsize=7.5, title_fontsize=7.5)

    regions = ("edge", "folds", "flat")
    width = 0.16
    for k, field in enumerate(FIELDS):
        part = table[keep & (table["source"] == field).to_numpy()]
        deficit = np.array([part[f"deficit_{r}"].sum() for r in regions])
        pixels = np.array([part[f"px_{r}"].sum() for r in regions], dtype=float)
        enrichment = (deficit / deficit.sum()) / (pixels / pixels.sum())
        axes[1].bar(np.arange(3) + (k - 2) * width, enrichment, width - 0.02,
                    color=FIELD_COLOURS[field], label=field.replace("T", " T"))
    axes[1].axhline(1, color=MUTED, linestyle="--", linewidth=1.0)
    axes[1].set_xticks(range(3), ("outer edge", "folds", "flat interior"))
    axes[1].set_ylabel("error share / pixel share")
    axes[1].set_title("b · enrichment of $1-\\mathrm{SSIM}$ by region", fontsize=9, loc="left")
    axes[1].grid(axis="x", visible=False)
    figure.tight_layout()
    figure.savefig(figure_dir / "fig_error_by_edge.pdf")
    plt.close(figure)

    # Is the prediction too smooth or too rough, and which SSIM factor carries the deficit?
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 2.9),
                                gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for name, colour, sign in (("away ($\\Delta\\alpha>0$)", AWAY, 1),
                               ("toward ($\\Delta\\alpha<0$)", TOWARD, -1)):
        sel = keep & (np.sign(table["delta_alpha"]) == sign).to_numpy()
        ratio = data["prediction"][sel].sum(0) / data["target"][sel].sum(0)
        axes[0].plot(frequency[valid], ratio[valid], color=colour, label=name)
    for field in FIELDS:
        sel = keep & (table["target"] == field).to_numpy()
        ratio = data["prediction"][sel].sum(0) / data["target"][sel].sum(0)
        axes[1].plot(frequency[valid], ratio[valid], color=FIELD_COLOURS[field],
                     label=field.replace("T", " T"))
    for axis, title in zip(axes[:2], ("a · prediction / target power, by direction",
                                      "b · prediction / target power, by target")):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.axhline(1, color=MUTED, linestyle="--", linewidth=1.0)
        axis.axvspan(*FIT_BAND, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        axis.set_xlabel("frequency (cycles/pixel)")
        axis.set_title(title, fontsize=9, loc="left", color=INK)
    axes[0].set_ylabel("$P_{\\hat y} / P_y$  (<1: too smooth)")
    axes[0].legend(fontsize=7.5, loc="lower left")
    axes[1].legend(title="target", fontsize=7.5, title_fontsize=7.5, loc="lower left")

    factors = (("lum", "luminance", "#1baf7a"), ("con", "contrast", "#4a3aa7"),
               ("str", "structure", "#e87ba4"))
    regions = ("edge", "folds", "flat")
    width = 0.26
    for k, (key, label, colour) in enumerate(factors):
        values = [table.loc[keep, f"{key}_{r}"].sum() / table.loc[keep, f"px_{r}"].sum()
                  for r in regions]
        axes[2].bar(np.arange(3) + (k - 1) * width, values, width - 0.03, color=colour,
                    label=label)
    axes[2].set_xticks(range(3), ("outer edge", "folds", "flat interior"))
    axes[2].set_ylabel("mean $1 -$ factor per pixel")
    axes[2].set_title("c · which SSIM factor is lost", fontsize=9, loc="left", color=INK)
    axes[2].grid(axis="x", visible=False)
    axes[2].legend(fontsize=7.5)
    figure.tight_layout()
    figure.savefig(figure_dir / "fig_error_smoothness.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "docker/task3/weights/task3.pt")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/error_spectrum")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "reports/spectral")
    parser.add_argument("--figures-only", action="store_true",
                        help="redraw from the cached per_transition.csv and spectra.npz")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not args.figures_only:
        compute(args)
    summary = summarise(args.out_dir)
    with pd.option_context("display.width", 200, "display.max_columns", 50,
                           "display.float_format", "{:.4f}".format):
        print(summary.to_string(index=False))
    figures(args.out_dir, args.figure_dir)


if __name__ == "__main__":
    main()
