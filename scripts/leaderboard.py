#!/usr/bin/env python3
"""Pull the MRIxFields Task 3 leaderboard from Synapse and rank teams by score.

The challenge publishes every submission as a Synapse table (syn74915588). Every row in it
belongs to Task 3 -- ``evaluationid`` is 9619636 throughout -- and carries a per-modality
breakdown that is noise for ranking purposes, so this drops it and keeps only the three
``Mean_of_all_subtasks_*`` metrics.

Two views:

    python scripts/leaderboard.py                 # best submission per team, score order
    python scripts/leaderboard.py --submissions   # every submission, score order

Both accept ``--team`` to restrict to one submitter, ``--mine`` for inzva_mri, and ``--csv``
to read a saved snapshot instead of querying Synapse. A live query writes a dated snapshot to
the repo root unless ``--no-save`` is passed, so an offline rerun is always possible.

Authentication uses PERSONAL_ACCESS_TOKEN from the repo-root .env. The token is never printed
and never leaves the Synapse client.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TABLE_ID = "syn74915588"

#: inzva_mri. ``createdBy`` 3584795 is the user's own account; the other ids under this
#: submitter are teammates'.
OUR_SUBMITTER_ID = 3599505
OUR_CREATED_BY = 3584795

METRICS = {
    "Mean_of_all_subtasks_SSIM_adj": "SSIM",
    "Mean_of_all_subtasks_nRMSE_adj": "nRMSE",
    "Mean_of_all_subtasks_LPIPS_adj": "LPIPS",
}

#: A submission scoring exactly 1.0 reproduced the ground truth voxel for voxel, which no
#: model does; the one in the table is even named "task3_broken.zip". Ranking against it
#: would misstate every gap below it, so it is set aside by default rather than deleted.
PERFECT = 1.0


def fetch(save: bool = True) -> pd.DataFrame:
    """Query the live table. Returns the raw dataframe, one row per submission."""
    import synapseclient

    from mrixfields.env import load_env

    load_env()
    token = os.environ.get("PERSONAL_ACCESS_TOKEN")
    if not token:
        raise SystemExit(
            "PERSONAL_ACCESS_TOKEN is not set. Put it in the repo-root .env "
            "(get one from Synapse: Account Settings -> Personal Access Tokens)."
        )
    syn = synapseclient.Synapse(silent=True)
    syn.login(authToken=token)
    frame = syn.tableQuery(f"SELECT * FROM {TABLE_ID}").asDataFrame()
    if save:
        stamp = datetime.now(timezone.utc).strftime("%d_%m_%Y")
        out = REPO_ROOT / f"table_{stamp}.csv"
        frame.to_csv(out, index=False)
        print(f"snapshot: {out}  ({len(frame)} rows)", file=sys.stderr)
    return frame


def clean(frame: pd.DataFrame, keep_perfect: bool = False) -> pd.DataFrame:
    """Keep scored, complete, accepted submissions and the three mean metrics."""
    frame = frame.copy()
    frame["when"] = pd.to_datetime(frame["createdOn"], unit="ms")
    valid = (
        (frame["status"] == "ACCEPTED")
        & frame["primary_score"].notna()
        & (frame["Num_Files"].astype(str) == "180/180")
    )
    frame = frame[valid]
    if not keep_perfect:
        dropped = frame[frame["primary_score"] >= PERFECT]
        if len(dropped):
            names = ", ".join(sorted(dropped["name"].astype(str).unique()))
            print(f"set aside {len(dropped)} submission(s) scoring {PERFECT} "
                  f"(ground truth, not a model): {names}", file=sys.stderr)
        frame = frame[frame["primary_score"] < PERFECT]
    columns = ["id", "name", "submitterid", "createdBy", "when"] + list(METRICS)
    return frame[columns].rename(columns=METRICS)


def by_team(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per submitter: their best submission, in score order."""
    best = frame.loc[frame.groupby("submitterid")["SSIM"].idxmax()].copy()
    best["n"] = frame.groupby("submitterid")["id"].count().reindex(best["submitterid"]).values
    best = best.sort_values("SSIM", ascending=False).reset_index(drop=True)
    best.insert(0, "rank", range(1, len(best) + 1))
    return best


def render(frame: pd.DataFrame, show_rank: bool) -> str:
    frame = frame.copy()
    frame["when"] = frame["when"].dt.strftime("%m-%d")
    frame["us"] = frame["submitterid"].eq(OUR_SUBMITTER_ID).map({True: "*", False: ""})
    frame["name"] = frame["name"].astype(str).str.slice(0, 34)
    columns = (["rank"] if show_rank else []) + ["us", "name", "submitterid", "when",
                                                 "SSIM", "nRMSE", "LPIPS"]
    if "n" in frame:
        columns.append("n")
    return frame[columns].to_string(index=False,
                                    formatters={m: "{:.6f}".format for m in METRICS.values()})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path,
                        help="read this snapshot instead of querying Synapse")
    parser.add_argument("--no-save", action="store_true",
                        help="do not write a snapshot after a live query")
    parser.add_argument("--submissions", action="store_true",
                        help="list every submission rather than each team's best")
    parser.add_argument("--team", type=int, help="restrict to one submitterid")
    parser.add_argument("--mine", action="store_true",
                        help=f"restrict to inzva_mri (submitterid {OUR_SUBMITTER_ID})")
    parser.add_argument("--created-by", type=int,
                        help="restrict to one account, e.g. 3584795 for denizberkin")
    parser.add_argument("--keep-perfect", action="store_true",
                        help="keep submissions scoring exactly 1.0 (the ground truth)")
    parser.add_argument("--top", type=int, default=0, help="show only the first N rows")
    parser.add_argument("--out", type=Path, help="also write the rendered table to this CSV")
    args = parser.parse_args()

    raw = pd.read_csv(args.csv) if args.csv else fetch(save=not args.no_save)
    frame = clean(raw, keep_perfect=args.keep_perfect)

    team = OUR_SUBMITTER_ID if args.mine else args.team
    ranked = by_team(frame)
    if team is not None:
        # Rank against the whole field first, then filter, so a filtered view still says
        # where the team actually stands rather than renumbering from 1.
        standing = ranked[ranked["submitterid"] == team]
        if len(standing):
            row = standing.iloc[0]
            print(f"submitterid {team}: rank {row['rank']} of {len(ranked)} teams, "
                  f"best SSIM {row['SSIM']:.6f}\n")
        frame = frame[frame["submitterid"] == team]
    if args.created_by is not None:
        frame = frame[frame["createdBy"] == args.created_by]

    if args.submissions or team is not None or args.created_by is not None:
        view = frame.sort_values("SSIM", ascending=False)
        show_rank = False
    else:
        view = ranked
        show_rank = True
    if args.top:
        view = view.head(args.top)

    print(render(view, show_rank))
    if args.out:
        view.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
