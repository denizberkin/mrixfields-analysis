#!/usr/bin/env python3
"""Figures for the field-strength transfer analysis (section 35).

Three rows, answering three questions:

1. **Transfer curves.** Is I_target a pointwise function of I_source, and what shape? A
   curve near the dashed identity means the fields differ little; a straight line means the
   relation is affine; curvature means it is not.
2. **Difference maps.** Where in the image do adjacent fields actually differ? A structured
   map means the residual is spatial and a pointwise transfer cannot reach it.
3. **Diagnostics.** How much variance a pointwise transfer explains (affine vs the best
   monotone curve), and whether a curve fitted on two subjects survives on a third.

Note on the difference correlations: corr(B-A, C-B) is negatively biased *by construction*,
because B enters the two terms with opposite signs -- for independent equal-variance fields
the expected correlation is -0.5, not 0. Only pairs sharing no field are interpretable, so
this plots a disjoint pair (7T-5T against 3T-1.5T) beside a sharing one.

Usage:
    python scripts/field_transfer_figures.py --subject 0009 --out reports/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.field_transfer_analysis import (  # noqa: E402
    FIELDS, apply_curve, fit_curve, load, r_squared,
)

MODALITIES = ("T1W", "T2W", "T2FLAIR")
PAIRS = [(FIELDS[i], FIELDS[i + 1]) for i in range(len(FIELDS) - 1)]
# validated categorical palette (six checks pass, light surface); fixed order, never cycled
SERIES = ("#1B6FC4", "#E06C0A", "#159B63", "#B23FC0")
INK, MUTED, GRID = "#102A43", "#627D98", "#D9E2EC"


def diverging():
    from matplotlib.colors import LinearSegmentedColormap
    # two hues + a neutral gray midpoint -- never a rainbow, never a hue at the middle
    return LinearSegmentedColormap.from_list(
        "field", ["#0B4C86", "#1B6FC4", "#B8C4CE", "#E06C0A", "#8A3F05"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", default="training_prospective")
    parser.add_argument("--subject", default="0009")
    parser.add_argument("--donors", nargs="+", default=["0006", "0007"])
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--voxels", type=int, default=300_000)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports" / "figures")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mrixfields.env import get_data_dir, load_env

    load_env()
    data_dir = args.data_dir or Path(get_data_dir())
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    def sample(subject, modality):
        vols = {f: load(data_dir, args.split, modality, f, subject) for f in FIELDS}
        if any(v is None for v in vols.values()):
            return None, None
        mask = np.ones_like(vols[FIELDS[0]], bool)
        for v in vols.values():
            mask &= v > 1e-3
        idx = np.flatnonzero(mask.ravel())
        if len(idx) > args.voxels:
            idx = rng.choice(idx, args.voxels, replace=False)
        return {f: vols[f].ravel()[idx] for f in FIELDS}, vols

    fig = plt.figure(figsize=(13.5, 11.2))
    grid = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.05, 0.95],
                            hspace=0.52, wspace=0.30)

    # ---------------------------------------------------------------- row 1: curves
    held, volumes_held = {}, {}
    for col, modality in enumerate(MODALITIES):
        pts, vols = sample(args.subject, modality)
        held[modality], volumes_held[modality] = pts, vols
        ax = fig.add_subplot(grid[0, col])
        if pts is None:
            continue
        lo = min(pts[f].min() for f in FIELDS)
        hi = max(np.quantile(pts[f], 0.999) for f in FIELDS)
        ax.plot([lo, hi], [lo, hi], ls=(0, (4, 3)), lw=1.0, color=MUTED, zorder=1)
        for k, (src, tgt) in enumerate(PAIRS):
            curve = fit_curve(pts[src], pts[tgt], args.bins)
            if curve is None:
                continue
            ax.plot(curve[0], curve[1], lw=2.0, color=SERIES[k], zorder=3,
                    label=f"{src}→{tgt}", solid_capstyle="round")
        ax.set_title(modality, fontsize=11, color=INK, pad=6)
        ax.set_xlabel("source intensity", fontsize=8.5, color=MUTED)
        if col == 0:
            ax.set_ylabel("target intensity", fontsize=8.5, color=MUTED)
        ax.grid(True, lw=0.5, color=GRID, zorder=0)
        ax.set_axisbelow(True)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7.5)
        if col == 2:
            ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="lower right")
    note = fig.add_subplot(grid[0, 3]); note.axis("off")
    note.text(0, 0.98, "Pointwise transfer", fontsize=11, color=INK, va="top", weight="bold")
    note.text(0, 0.86,
              "Each curve is the best possible\npointwise map from one field to\nthe next: the "
              "median target intensity\nper source-intensity quantile.\n\nDashed line is identity "
              "(copying\nthe source, which already scores\nSSIM 0.836).\n\nCurves sit close to "
              "straight, so the\nintensity relation is near-affine --\nbut see the R² panel: "
              "even the\nbest curve leaves 15-45% of the\nvariance unexplained, and that\n"
              "remainder is spatial.",
              fontsize=8.2, color=MUTED, va="top", linespacing=1.5)

    # ---------------------------------------------------------------- row 2: difference maps
    vols = volumes_held["T1W"]
    if vols is not None:
        z = vols[FIELDS[0]].shape[2] // 2
        planes = {f: np.asarray(vols[f][:, :, z], np.float32) for f in FIELDS}
        diffs = [(f"{t}−{s}", planes[t] - planes[s]) for s, t in PAIRS]
        scale = float(np.percentile(np.abs(np.stack([d for _, d in diffs])), 99.5))
        cmap = diverging()
        for col, (label, image) in enumerate(diffs):
            ax = fig.add_subplot(grid[1, col])
            im = ax.imshow(image.T, cmap=cmap, vmin=-scale, vmax=scale, origin="lower")
            ax.set_title(label, fontsize=9.5, color=INK, pad=4)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if col == 3:
                bar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                bar.ax.tick_params(labelsize=7, colors=MUTED)
                bar.outline.set_edgecolor(GRID)
        fig.text(0.5, 0.612, f"T1W, subject {args.subject}, axial slice {z} — "
                 f"adjacent-field differences on a shared ±{scale:.3f} scale",
                 ha="center", fontsize=8.5, color=MUTED)

    # ---------------------------------------------------------------- row 3: diagnostics
    ax = fig.add_subplot(grid[2, 0])
    width = 0.38
    xs = np.arange(len(PAIRS))
    aff = [r_squared(np.polyval(np.polyfit(held["T1W"][s], held["T1W"][t], 1),
                                held["T1W"][s]), held["T1W"][t]) for s, t in PAIRS]
    cur = []
    for s, t in PAIRS:
        c = fit_curve(held["T1W"][s], held["T1W"][t], args.bins)
        cur.append(r_squared(apply_curve(c, held["T1W"][s]), held["T1W"][t]) if c else np.nan)
    ax.bar(xs - width/2, aff, width, color=SERIES[0], label="affine", zorder=3)
    ax.bar(xs + width/2, cur, width, color=SERIES[1], label="best curve", zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}→{t}" for s, t in PAIRS],
                                          fontsize=7, rotation=20, color=MUTED)
    ax.set_ylim(0, 1.18); ax.set_ylabel("R²  (T1W)", fontsize=8.5, color=MUTED)
    ax.set_title("A curve barely beats a line", fontsize=9.5, color=INK, pad=6)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=INK, loc="upper center", ncol=2,
              columnspacing=1.0, handlelength=1.2)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, axis="y", lw=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=7.5)

    # sharing vs disjoint difference pairs
    p = held["T1W"]
    combos = [(("5T", "3T"), ("3T", "1.5T"), "shares 3T"),
              (("7T", "5T"), ("3T", "1.5T"), "disjoint")]
    for col, ((a1, a0), (b1, b0), tag) in enumerate(combos, start=1):
        ax = fig.add_subplot(grid[2, col])
        u, v = p[a1] - p[a0], p[b1] - p[b0]
        ax.hexbin(v, u, gridsize=46, bins="log", cmap="Blues", linewidths=0)
        r = float(np.corrcoef(u, v)[0, 1])
        ax.set_xlabel(f"{b1}−{b0}", fontsize=8.5, color=MUTED)
        ax.set_ylabel(f"{a1}−{a0}", fontsize=8.5, color=MUTED)
        ax.set_title(f"{tag}:  r = {r:+.3f}", fontsize=9.5, color=INK, pad=6)
        ax.axhline(0, lw=0.6, color=MUTED); ax.axvline(0, lw=0.6, color=MUTED)
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=7.5)

    # cross-subject transfer
    ax = fig.add_subplot(grid[2, 3])
    donor_pts = [sample(d, "T1W")[0] for d in args.donors]
    donor_pts = [d for d in donor_pts if d is not None]
    own_r, donor_r = [], []
    for s, t in PAIRS:
        own = fit_curve(p[s], p[t], args.bins)
        ds = np.concatenate([d[s] for d in donor_pts])
        dt = np.concatenate([d[t] for d in donor_pts])
        don = fit_curve(ds, dt, args.bins)
        own_r.append(r_squared(apply_curve(own, p[s]), p[t]) if own else np.nan)
        donor_r.append(r_squared(apply_curve(don, p[s]), p[t]) if don else np.nan)
    ax.bar(xs - width/2, own_r, width, color=SERIES[2], label=f"fitted on {args.subject}", zorder=3)
    ax.bar(xs + width/2, np.clip(donor_r, -0.4, None), width, color=SERIES[3],
           label="fitted on 0006+0007", zorder=3)
    for k, value in enumerate(donor_r):
        if value < -0.4:
            ax.text(xs[k] - 0.1, -0.40, f"R² = {value:.1f}", ha="center", fontsize=6.8,
                    color=SERIES[3], va="bottom", weight="bold")
    ax.axhline(0, lw=0.8, color=MUTED)
    ax.set_xticks(xs); ax.set_xticklabels([f"{s}→{t}" for s, t in PAIRS],
                                          fontsize=7, rotation=20, color=MUTED)
    ax.set_ylim(-0.45, 1.22); ax.set_ylabel("R²  on " + args.subject, fontsize=8.5, color=MUTED)
    ax.set_title("The curve does not transfer", fontsize=9.5, color=INK, pad=6)
    ax.legend(fontsize=6.8, frameon=False, labelcolor=INK, loc="upper center", ncol=1,
              handlelength=1.2)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(True, axis="y", lw=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=7.5)

    fig.suptitle("Is the field axis a smooth intensity transform?", fontsize=13.5,
                 color=INK, x=0.5, y=0.985)
    out = args.out / "field_transfer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#fcfcfb")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
