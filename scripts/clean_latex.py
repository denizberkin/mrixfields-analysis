"""delete the auxiliary files a LaTeX run leaves behind, keeping .tex and .pdf.

Usage:
    python scripts/clean_latex.py                        # clean reports/
    python scripts/clean_latex.py --dry-run              # list what would go, delete nothing
    python scripts/clean_latex.py reports docs           # clean several directories
    python scripts/clean_latex.py --extra bbl blg        # add suffixes for this run
"""

from __future__ import annotations

import argparse
from pathlib import Path

AUX_SUFFIXES = (
    ".bbl", ".bcf", ".blg", ".brf", ".fdb_latexmk", ".fls", ".idx", ".ilg", ".ind",
    ".lof", ".log", ".lot", ".nav", ".out", ".run.xml", ".snm", ".synctex.gz", ".toc", ".vrb",
)


def collect(roots: list[Path], suffixes: tuple[str, ...], recursive: bool) -> list[Path]:
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"skip (missing): {root}", flush=True)
            continue
        candidates = root.rglob("*") if recursive else root.glob("*")
        matches.extend(
            path for path in candidates
            if path.is_file() and path.name.endswith(suffixes)
        )
    return sorted(set(matches))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["reports"], help="directories to clean (default: reports)")
    parser.add_argument("--dry-run", action="store_true", help="list the files without deleting them")
    parser.add_argument("--extra", nargs="*", default=[], help="additional suffixes, with or without the leading dot")
    parser.add_argument("--no-recursive", action="store_true", help="do not descend into subdirectories")
    args = parser.parse_args()

    suffixes = AUX_SUFFIXES + tuple(
        extra if extra.startswith(".") else f".{extra}" for extra in args.extra
    )
    roots = [Path(path) for path in args.paths]
    matches = collect(roots, suffixes, recursive=not args.no_recursive)

    if not matches:
        print("nothing to clean", flush=True)
        return

    total = 0
    for path in matches:
        size = path.stat().st_size
        total += size
        print(f"{'would remove' if args.dry_run else 'removed'}  {path}  ({size / 1024:.1f} KiB)", flush=True)
        if not args.dry_run:
            path.unlink()

    verb = "would free" if args.dry_run else "freed"
    print(f"{len(matches)} file(s), {verb} {total / 1024:.1f} KiB", flush=True)


if __name__ == "__main__":
    main()
