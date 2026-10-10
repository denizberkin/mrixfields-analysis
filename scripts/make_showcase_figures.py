#!/usr/bin/env python3
"""Trade-off figures for the showcase deck (reports/showcase_presentation_v0.tex).

1. `trade_ladder_<k>.pdf`: our validation-phase submissions in the (SSIM, LPIPS) plane, one
   step of the ablation ladder at a time. Numbers: the challenge leaderboard, the same table
   as reports/architecture_report.tex.
2. `trade_final.pdf`: the top-9 teams of the final Task 3 test ranking (Synapse wiki
   "Final Ranking", 1 Oct 2026) in the same plane, with the 3rd-placed team's SSIM and
   LPIPS drawn as dashed lines.

LPIPS is drawn with the axis inverted, so up and right are both better.

    ~/anaconda3/envs/mri/bin/python scripts/make_showcase_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ACCENT, COND, INK, MUTED, GRID = "#D64545", "#3E7CB1", "#102A43", "#627D98", "#D9E2EC"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 12,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 1.8,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

# (label, validation SSIM, validation LPIPS), reports/architecture_report.tex panels 1.2-2.8
LADDER = [
    ("+ conditioning", 0.897646, 0.089527),
    ("+ FiLM & residual", 0.892839, 0.092091),
    ("+ MIM pretraining", 0.903110, 0.083882),
    ("+ 4-channel input", 0.906773, 0.080831),
    ("+ SSIM in the loss", 0.908873, 0.088697),
    ("+ axial token", 0.909433, 0.088294),
    ("+ averaging & TTA", 0.913652, 0.090899),
]
SPEND_FROM = 4          # index of the first step that bought SSIM with LPIPS

# (team, test SSIM, test LPIPS, label offset in points), final Task 3 ranking
FINAL = [
    ("Watt", 0.946844, 0.038111, (6, 4)),
    ("Blue Tent", 0.944392, 0.045311, (6, 5)),
    ("Lzu_LastDance", 0.942018, 0.039788, (-8, -13)),
    ("croissant", 0.940502, 0.047228, (6, -2)),
    ("Newton", 0.939135, 0.050002, (-6, -12)),
    ("imispllove", 0.939022, 0.046523, (-8, -10)),
    ("MRIxFields2026-Zhu", 0.942594, 0.061259, (6, -3)),
    ("Tryer", 0.934850, 0.044196, (6, 4)),
]
OURS = ("inzva mri", 0.942551, 0.050566)
THIRD = "Lzu_LastDance"  # 3rd on the final mean rank; its SSIM and LPIPS are drawn as lines

# (label, validation SSIM), ascending: the dropped backbones and formulations,
# architecture_report 4.a, 4.b, 3.f and the teammates' rows. The tubelet row is furkanycy's
# Tubelet-UNETR, StS-T-lj-un-v1 (lejepa_pretraining/repo/docs/conditional_tubelet_translator_
# design.md), not the 0.859000 re-fine-tune (4.c), whose patch grid and slab seams (TODO.md 7)
# came from that fine-tune rather than from the encoder.
BACKBONES = [
    ("Swin UNETR, 3D", 0.858214),
    ("DINOv3 ViT encoder", 0.878918),
    ("FPS-Former", 0.879531),
    ("conditional flow matching, 3 stages", 0.886725),
    ("LeJEPA tubelet encoder + UNETR", 0.889338),
]
# (label, validation SSIM, colour): the U-Net line they are read against
REFERENCES = [
    ("copy the input", 0.836497, MUTED),
    ("conditional U-Net", 0.897646, COND),
    ("final model", 0.913652, ACCENT),
]
# (group, [(label, SSIM change against the parent run, scored on)]), TODO.md sections in comments.
# Only validation-phase scores and fixed-harness local scores: every local number recorded
# before 2026-09-11 ran on the wrong plane (section 41) and is void.
CHANGES = [
    ("Training", [
        ("epoch 12 instead of 8", -0.000228, "challenge"),          # 32
    ]),
    ("Loss", [
        ("spectral-profile loss", -0.00118, "local"),               # 44, fixed harness
    ]),
    ("Input context", [
        ("2.5D input, slices z\u00b12", 0.00048, "challenge"),      # 37
        ("axial token on the 2.5D model", -0.000999, "challenge"),  # 39
    ]),
    ("Ensembling", [
        ("50/50 weight soup", -0.000449, "challenge"),              # 40
    ]),
]


def axes_frame(figure_size=(5.0, 3.7)):
    figure, axis = plt.subplots(figsize=figure_size)
    axis.invert_yaxis()
    axis.set_xlabel("SSIM  (higher is better) →")
    axis.set_ylabel("← LPIPS  (lower is better)")
    return figure, axis


def ladder(out_dir: Path) -> None:
    for step in (1, 2):
        figure, axis = axes_frame()
        axis.set_xlim(0.8905, 0.9160)
        axis.set_ylim(0.0945, 0.0785)
        last = SPEND_FROM if step == 1 else len(LADDER)
        for i in range(1, last):
            spend = i >= SPEND_FROM
            _, x0, y0 = LADDER[i - 1]
            _, x1, y1 = LADDER[i]
            axis.annotate("", xy=(x1, y1), xytext=(x0, y0),
                          arrowprops=dict(arrowstyle="-|>", lw=1.8, mutation_scale=12,
                                          color=ACCENT if spend else COND,
                                          shrinkA=5, shrinkB=5))
        for i, (label, x, y) in enumerate(LADDER[:last]):
            spend = i >= SPEND_FROM
            axis.plot(x, y, "o", ms=8, color=ACCENT if spend else COND,
                      markeredgecolor="white", markeredgewidth=1.5, zorder=3)
            offset = {0: (6, 6), 1: (6, -12), 2: (-6, 8), 3: (-8, -12), 4: (8, 4),
                      5: (-4, 12), 6: (-8, -14)}[i]
            axis.annotate(label, (x, y), xytext=offset, textcoords="offset points",
                          fontsize=10.5, color=INK, ha="right" if offset[0] < 0 else "left")
        if step == 2:
            axis.text(0.9128, 0.0842, "last three steps:\nSSIM +0.0069\nLPIPS +0.0101",
                      color=ACCENT, fontsize=10.5, fontweight="bold", ha="center")
        figure.savefig(out_dir / f"trade_ladder_{step}.pdf")
        plt.close(figure)


def final(out_dir: Path) -> None:
    figure, axis = axes_frame()
    axis.set_xlim(0.9335, 0.9485)
    axis.set_ylim(0.0640, 0.0360)
    third = next((x, y) for team, x, y, _ in FINAL if team == THIRD)
    axis.axvline(third[0], color=INK, linestyle="--", linewidth=1.0, zorder=1)
    axis.axhline(third[1], color=INK, linestyle="--", linewidth=1.0, zorder=1)
    axis.text(third[0] - 0.0001, 0.0636, "3rd place line", rotation=90, ha="right",
              va="bottom", fontsize=9.5, color=INK)
    axis.text(0.9339, third[1] - 0.0003, "3rd place line", ha="left", va="bottom",
              fontsize=9.5, color=INK)
    for team, x, y, offset in FINAL:
        axis.plot(x, y, "o", ms=7, color=MUTED, markeredgecolor="white",
                  markeredgewidth=1.2, zorder=2)
        axis.annotate(team, (x, y), xytext=offset, textcoords="offset points",
                      fontsize=10, color=MUTED, ha="right" if offset[0] < 0 else "left")
    name, x, y = OURS
    axis.plot(x, y, "o", ms=9, color=ACCENT, markeredgecolor="white", markeredgewidth=1.5,
              zorder=3)
    axis.annotate(name, (x, y), xytext=(8, -4), textcoords="offset points", fontsize=11,
                  color=ACCENT, fontweight="bold", va="top")
    figure.savefig(out_dir / "trade_final.pdf")
    plt.close(figure)


def backbones(out_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 3.1))
    axis.grid(axis="y", visible=False)
    axis.set_xlim(0.830, 0.921)
    axis.set_ylim(-0.7, len(BACKBONES) - 0.3)
    axis.spines["left"].set_visible(False)
    axis.tick_params(axis="y", length=0)
    for label, value, colour in REFERENCES:
        axis.axvline(value, color=colour, linewidth=1.4,
                     linestyle="--" if colour == MUTED else "-", zorder=1)
        align = {MUTED: "center", COND: "right", ACCENT: "left"}[colour]
        axis.annotate(f"{label}\n{value:.4f}", (value, len(BACKBONES) - 0.3),
                      xytext={"center": (0, 2), "right": (-3, 2), "left": (3, 2)}[align],
                      textcoords="offset points", color=colour, fontsize=11, ha=align,
                      va="bottom", fontweight="bold", annotation_clip=False)
    rows = range(len(BACKBONES))
    axis.set_yticks(list(rows), [label for label, _ in BACKBONES], fontsize=12, color=INK)
    for row, (_, value) in zip(rows, BACKBONES):
        axis.plot(value, row, "o", ms=9, color=INK, markeredgecolor="white",
                  markeredgewidth=1.5, zorder=3)
        axis.annotate(f"{value:.4f}", (value, row), xytext=(8, 0), textcoords="offset points",
                      va="center", fontsize=11, color=INK)
    axis.set_xlabel("validation-phase challenge SSIM")
    figure.savefig(out_dir / "other_backbones.pdf")
    plt.close(figure)


def changes(out_dir: Path) -> None:
    # one row per change, a blank row above each group for its heading; every step lays out
    # all rows so nothing moves, and draws (labels included) only the groups shown so far
    layout, y = [], 0
    for group, items in CHANGES:
        layout.append((y, group, None))
        y += 1
        for item in items:
            layout.append((y, None, item))
            y += 1
    for step in range(1, len(CHANGES) + 1):
        shown = {group for group, _ in CHANGES[:step]}
        figure, axis = plt.subplots(figsize=(7.2, 3.0))
        axis.grid(axis="y", visible=False)
        axis.set_xlim(-0.0017, 0.0010)
        axis.set_ylim(y - 0.4, -0.6)
        axis.spines["left"].set_visible(False)
        axis.tick_params(axis="y", length=0)
        axis.axvline(0, color=MUTED, linewidth=1.0, zorder=1)
        ticks, labels, headings = [], [], []
        group = None
        for row, heading, item in layout:
            ticks.append(row)
            if heading is not None:
                group = heading
                headings.append(len(labels))
                labels.append(heading if group in shown else "")
                continue
            if group not in shown:
                labels.append("")
                continue
            label, delta, scored = item
            labels.append(label)
            filled = scored == "challenge"
            axis.plot(delta, row, "o", ms=9, zorder=3,
                      color=INK if filled else "white", markeredgecolor=INK,
                      markeredgewidth=1.6)
            axis.annotate(f"{delta:+.4f}", (delta, row),
                          xytext=(9 if delta >= 0 else -9, 0), textcoords="offset points",
                          va="center", ha="left" if delta >= 0 else "right", fontsize=10.5,
                          color=INK)
        axis.set_yticks(ticks, labels, fontsize=11.5, color=INK)
        for index in headings:
            axis.get_yticklabels()[index].set_fontweight("bold")
        axis.set_xlabel("SSIM change against the parent run")
        axis.plot([], [], "o", ms=8, color=INK, label="challenge validation")
        axis.plot([], [], "o", ms=8, color="white", markeredgecolor=INK, markeredgewidth=1.6,
                  label="local, fixed harness")
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False,
                    fontsize=10, handletextpad=0.3, columnspacing=2.0)
        figure.savefig(out_dir / f"other_changes_{step}.pdf")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports/figures/showcase")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ladder(args.out_dir)
    final(args.out_dir)
    backbones(args.out_dir)
    changes(args.out_dir)
    print(f"wrote trade_ladder_1..2.pdf, trade_final.pdf, other_backbones.pdf and "
          f"other_changes_1..{len(CHANGES)}.pdf to {args.out_dir}")


if __name__ == "__main__":
    main()
