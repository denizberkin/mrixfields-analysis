"""Score a Task 3 checkpoint against the shipped model, paired per transition.

Same path as the shipped model's local number (0.95069 / 0.08869 / 0.07044 over 180
transitions, reports/detail_gain/lpips.csv "base" rows): 4-flip TTA predictions cached by
`detail_gain_sweep.py --stage cache`, source-masked, scored by `eval_holdout.score`.

    ~/anaconda3/envs/mri/bin/python scripts/detail_gain_sweep.py --stage cache \
        --checkpoint <avg.pt> --cache runs/spectral_loss_cache
    ~/anaconda3/envs/mri/bin/python scripts/spectral_loss_eval.py \
        --cache runs/spectral_loss_cache --out reports/spectral_loss/scores.csv

Groups: ranked (T2W -> 1.5T excluded), target field, and the spectral gate
(d = alpha_target - alpha_source > 0, the transitions the term was charged on).

`--compare` instead reads two `error_spectrum_analysis.py` outputs (shipped and new) and
splits each ring's power into what the prediction adds and what it adds *coherently*. The
error spectrum is of e = y_hat - y on the same padded FFT, so per ring
    cross power C = (P_yhat + P_y - P_e) / 2,   coherence rho = C / sqrt(P_yhat P_y)
exactly. Writes reports/spectral_loss/bands.csv and reports/spectral/fig_spectral_loss.pdf.

    ~/anaconda3/envs/mri/bin/python scripts/spectral_loss_eval.py --compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "experiment-pipeline"), str(ROOT / "scripts")]

from detail_gain_sweep import (  # noqa: E402
    CORRUPT, MASK_THRESHOLD, prediction_file, sharpen, transitions, volume_file)

KEYS = ["subject", "modality", "source", "target"]
METRICS = ["SSIM", "nRMSE", "LPIPS"]


def load_alpha() -> dict[tuple[str, str], float]:
    table = pd.read_csv(ROOT / "reports/spectral/alpha_summary.csv")
    table = table[table["split"] == "retro"]
    return {(m, f): a for m, f, a in zip(table["modality"], table["field"], table["alpha"])}


def score_cache(cache: Path, device_name: str) -> pd.DataFrame:
    import torch

    from eval_holdout import score
    from mrixfields.losses.perceptual import PerceptualLoss

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    lpips_fn = PerceptualLoss(net="alex").to(device).eval()
    rows = []
    for subject, modality, source, target in transitions():
        prediction = np.load(prediction_file(cache, subject, modality, source, target))
        mask = np.load(volume_file(cache, subject, modality, source)) > MASK_THRESHOLD
        truth = np.load(volume_file(cache, subject, modality, target))
        values = score(sharpen(prediction, mask, "none", 1.0, 0.0), truth, lpips_fn, device)
        rows.append({"subject": subject, "modality": modality, "source": source,
                     "target": target, **values})
    return pd.DataFrame(rows)


def compare(new: pd.DataFrame, baseline_csv: Path) -> pd.DataFrame:
    base = pd.read_csv(baseline_csv, dtype={"subject": str})
    base = base[base["setting"] == "base"][KEYS + METRICS]
    frame = new.merge(base, on=KEYS, suffixes=("", "_shipped"), validate="one_to_one")
    if len(frame) != len(new):
        raise SystemExit(f"paired {len(frame)} of {len(new)} transitions")
    alpha = load_alpha()
    frame["delta_alpha"] = [alpha[(m, t)] - alpha[(m, s)]
                            for m, s, t in zip(frame["modality"], frame["source"], frame["target"])]
    frame["corrupt"] = [(m, t) in CORRUPT for m, t in zip(frame["modality"], frame["target"])]
    for metric in METRICS:
        frame[f"d_{metric}"] = frame[metric] - frame[f"{metric}_shipped"]
    return frame


def summarise(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame[~frame["corrupt"]]
    groups = {"all 180": frame, "ranked 168": ranked,
              "gate on (d>0)": ranked[ranked["delta_alpha"] > 0],
              "gate off (d<0)": ranked[ranked["delta_alpha"] < 0]}
    for field in ("0.1T", "1.5T", "3T", "5T", "7T"):
        groups[f"target {field}"] = ranked[ranked["target"] == field]
    for modality in ("T1W", "T2W", "T2FLAIR"):
        groups[modality] = ranked[ranked["modality"] == modality]
    rows = []
    for name, group in groups.items():
        row = {"group": name, "n": len(group)}
        for metric in METRICS:
            row[metric] = group[metric].mean()
            row[f"d_{metric}"] = group[f"d_{metric}"].mean()
        row["ssim_improved"] = float((group["d_SSIM"] > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


BANDS = ((0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50))


def ring_sums(spectrum_dir: Path, selector) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Ring energies summed over the ranked transitions ``selector`` keeps."""
    table = pd.read_csv(spectrum_dir / "per_transition.csv", dtype={"subject": str})
    data = np.load(spectrum_dir / "spectra.npz")
    keep = (~table["corrupt"] & selector(table)).to_numpy()
    return data["frequency"], {k: data[k][keep].sum(0) for k in ("prediction", "target", "error")}


def spectral_terms(energy: dict[str, np.ndarray], band) -> dict[str, np.ndarray | float]:
    """P_yhat/P_y, coherence and P_e/P_y, per ring or (with a boolean band) per band."""
    p, y, e = (energy[k] if band is None else energy[k][band].sum()
               for k in ("prediction", "target", "error"))
    cross = (p + y - e) / 2
    return {"p_over_y": p / y, "coherence": cross / np.sqrt(p * y), "e_over_y": e / y}


GROUPS = {
    "ranked": lambda t: t["target"] == t["target"],
    "gate on (d>0)": lambda t: t["delta_alpha"] > 0,
    "gate off (d<0)": lambda t: t["delta_alpha"] < 0,
    "target 0.1T": lambda t: t["target"] == "0.1T",
}


def compare_spectra(shipped: Path, new: Path, out_csv: Path, figure_path: Path) -> None:
    from error_spectrum_analysis import FIT_BAND, GRID, INK, MUTED, plt

    rows = []
    for name, selector in GROUPS.items():
        for label, directory in (("shipped", shipped), ("spectral", new)):
            frequency, energy = ring_sums(directory, selector)
            for lo, hi in BANDS:
                band = (frequency >= lo) & (frequency < hi)
                rows.append({"group": name, "model": label, "band": f"{lo:.2f}-{hi:.2f}",
                             **spectral_terms(energy, band)})
    table = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    wide = table.pivot_table(index=["group", "band"], columns="model",
                             values=["p_over_y", "coherence", "e_over_y"])
    print(wide.round(3).to_string())

    # categorical slots 1 and 2 of the validated palette (same pair as fig_error_*.pdf)
    colours = {"shipped": "#2a78d6", "spectral": "#eb6834"}
    labels = {"shipped": "shipped", "spectral": "+ spectral term"}
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 2.9))
    for model, directory in (("shipped", shipped), ("spectral", new)):
        frequency, energy = ring_sums(directory, GROUPS["gate on (d>0)"])
        valid = (frequency >= 0.01) & (frequency <= 0.5)
        terms = spectral_terms(energy, None)
        for axis, key in zip(axes, ("p_over_y", "coherence", "e_over_y")):
            axis.plot(frequency[valid], terms[key][valid], color=colours[model],
                      label=labels[model])
    titles = ("a · prediction / target power", "b · coherence with the target",
              "c · error / target power")
    ylabels = ("$P_{\\hat y} / P_y$", "$\\rho = C / \\sqrt{P_{\\hat y} P_y}$", "$P_e / P_y$")
    for axis, title, ylabel in zip(axes, titles, ylabels):
        axis.set_xscale("log")
        axis.axvspan(*FIT_BAND, color=GRID, alpha=0.45, linewidth=0, zorder=0)
        axis.set_xlabel("frequency (cycles/pixel)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, fontsize=9, loc="left", color=INK)
    for axis in (axes[0], axes[2]):
        axis.set_yscale("log")
        axis.axhline(1, color=MUTED, linestyle="--", linewidth=1.0)
    axes[0].legend(fontsize=7.5, loc="lower left")
    figure.tight_layout()
    figure.savefig(figure_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--out", type=Path, help="per-transition CSV")
    parser.add_argument("--baseline", type=Path, default=ROOT / "reports/detail_gain/lpips.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--compare", action="store_true",
                        help="compare the shipped and new error_spectrum_analysis outputs")
    parser.add_argument("--shipped-spectra", type=Path, default=ROOT / "reports/error_spectrum")
    parser.add_argument("--new-spectra", type=Path, default=ROOT / "reports/spectral_loss/spectrum")
    parser.add_argument("--figure", type=Path,
                        default=ROOT / "reports/spectral/fig_spectral_loss.pdf")
    args = parser.parse_args()

    if args.compare:
        compare_spectra(args.shipped_spectra, args.new_spectra,
                        args.new_spectra.parent / "bands.csv", args.figure)
        return
    if args.cache is None or args.out is None:
        parser.error("--cache and --out are required unless --compare")
    frame = compare(score_cache(args.cache, args.device), args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    summary = summarise(frame)
    summary.to_csv(args.out.with_name("summary.csv"), index=False)
    print(summary.to_string(index=False, float_format="{:.5f}".format))


if __name__ == "__main__":
    main()
