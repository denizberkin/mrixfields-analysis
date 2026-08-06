"""Emit LaTeX tables for the progress report from the spectral analysis output.

Keeps the reported numbers tied to ``outputs/spectral/alpha_summary.csv`` rather than transcribed by
hand, so re-running the analysis updates the report.

Usage:
    python scripts/make_report_tables.py --in-dir outputs/spectral --out-dir reports/tables
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

MODALITIES = ("T1W", "T2W", "T2FLAIR")
FIELD_STRENGTHS = ("0.1T", "1.5T", "3T", "5T", "7T")
SPLIT_LABELS = {"val": "Validation (prospective)", "retro": "Training (retrospective)",
                "pro": "Training (prospective)"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def alpha_table(rows: list[dict[str, str]], split: str, placement: str = "t") -> str:
    """One row per modality, one column per field strength, alpha with bootstrap interval.

    ``placement`` is the LaTeX float specifier. Appendix tables use ``H`` (requires the ``float``
    package) so they cannot drift out of their section into the bibliography.
    """
    cells: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row["split"] == split:
            cells[(row["modality"], row["field"])] = row
    if not cells:
        return ""

    volumes = {int(row["num_volumes"]) for key, row in cells.items()}
    count = f"{min(volumes)}" if len(volumes) == 1 else f"{min(volumes)}--{max(volumes)}"

    lines = [
        rf"\begin{{table}}[{placement}]",
        r"  \centering",
        r"  \small",
        rf"  \caption{{Fitted spectral exponent $\alpha$ on the {SPLIT_LABELS[split].lower()} split, "
        rf"estimated over the band $f \in [0.02, 0.20]$ cycles/pixel. "
        rf"Brackets give 95\% bootstrap intervals over volumes ($n = {count}$ per cell). "
        rf"Larger $\alpha$ means power falls off faster, i.e.\ relatively less fine detail.}}",
        rf"  \label{{tab:alpha-{split}}}",
        r"  \begin{tabular}{l" + "c" * len(FIELD_STRENGTHS) + "}",
        r"    \toprule",
        r"    Modality & " + " & ".join(FIELD_STRENGTHS) + r" \\",
        r"    \midrule",
    ]
    for modality in MODALITIES:
        entries = []
        for field in FIELD_STRENGTHS:
            row = cells.get((modality, field))
            if row is None:
                entries.append("--")
                continue
            entries.append(
                rf"\begin{{tabular}}[t]{{@{{}}c@{{}}}}{float(row['alpha']):.2f}\\"
                rf"\scriptsize[{float(row['ci_low']):.2f}, {float(row['ci_high']):.2f}]\end{{tabular}}"
            )
        lines.append(rf"    {modality} & " + " & ".join(entries) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def delta_table(rows: list[dict[str, str]], split: str, placement: str = "t") -> str:
    """Change in alpha relative to 7T, the target domain for Task 1."""
    cells = {(row["modality"], row["field"]): float(row["alpha"]) for row in rows if row["split"] == split}
    if not cells:
        return ""

    lines = [
        rf"\begin{{table}}[{placement}]",
        r"  \centering",
        r"  \small",
        r"  \caption{Spectral distance to the 7\,T target, $\Delta\alpha = \alpha_{\text{source}} - "
        r"\alpha_{7\text{T}}$. A positive value means the source is spectrally \emph{steeper} than the "
        r"target and the model must synthesise high-frequency power that is not present in its input.}",
        rf"  \label{{tab:delta-alpha-{split}}}",
        r"  \begin{tabular}{l" + "c" * (len(FIELD_STRENGTHS) - 1) + "}",
        r"    \toprule",
        r"    Modality & " + " & ".join(f for f in FIELD_STRENGTHS if f != "7T") + r" \\",
        r"    \midrule",
    ]
    for modality in MODALITIES:
        reference = cells.get((modality, "7T"))
        entries = []
        for field in FIELD_STRENGTHS:
            if field == "7T":
                continue
            value = cells.get((modality, field))
            if value is None or reference is None:
                entries.append("--")
            else:
                entries.append(f"${value - reference:+.2f}$")
        lines.append(rf"    {modality} & " + " & ".join(entries) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def summary_macros(rows: list[dict[str, str]], split: str) -> str:
    """Macros for numbers quoted in the prose, so the text cannot drift from the data."""
    cells = {(row["modality"], row["field"]): float(row["alpha"]) for row in rows if row["split"] == split}
    if not cells:
        return ""
    by_field: dict[str, list[float]] = defaultdict(list)
    for (modality, field), value in cells.items():
        by_field[field].append(value)

    lines = []
    for field in FIELD_STRENGTHS:
        if field in by_field:
            key = field.replace(".", "p").replace("T", "")
            mean = sum(by_field[field]) / len(by_field[field])
            lines.append(rf"\newcommand{{\alphaMean{key}}}{{{mean:.2f}}}")
    lowest = min(by_field, key=lambda f: sum(by_field[f]) / len(by_field[f]))
    highest = max(by_field, key=lambda f: sum(by_field[f]) / len(by_field[f]))
    lines.append(rf"\newcommand{{\alphaLowestField}}{{{lowest}}}")
    lines.append(rf"\newcommand{{\alphaHighestField}}{{{highest}}}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=Path("outputs/spectral"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tables"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows(args.in_dir / "alpha_summary.csv")
    splits = sorted({row["split"] for row in rows})
    primary = "retro" if "retro" in splits else splits[0]

    written = []
    for split in splits:
        # The primary split is typeset in the body and may float; the rest appear in the appendix and
        # are pinned so they cannot migrate into a neighbouring section.
        placement = "t" if split == primary else "H"
        for name, builder in [("alpha", alpha_table), ("delta_alpha", delta_table)]:
            content = builder(rows, split, placement)
            if content:
                path = args.out_dir / f"{name}_{split}.tex"
                path.write_text(content, encoding="utf-8")
                written.append(path.name)

    macros = args.out_dir / "spectral_macros.tex"
    macros.write_text(summary_macros(rows, primary), encoding="utf-8")
    written.append(macros.name)

    print("wrote: " + ", ".join(written), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
