#!/usr/bin/env python3
"""Pull the MRIxFields Task 3 leaderboard from Synapse and rank teams by score.

The challenge publishes every submission as a Synapse table (syn74915588). Every row in it
belongs to Task 3 -- ``evaluationid`` is 9619636 throughout -- and carries a per-modality
breakdown that is noise for ranking purposes, so this drops it and keeps only the three
``Mean_of_all_subtasks_*`` metrics.

Two views:

    python scripts/leaderboard.py                 # best submission per team, score order
    python scripts/leaderboard.py --submissions   # every submission, score order

Ranking metric is selectable with ``--metric {ssim,nrmse,lpips}``, default SSIM because that
is what the challenge ranks (``primary_metric``). nRMSE and LPIPS are errors, so they sort
ascending and each team's row becomes their *lowest* value -- ranking them descending like
SSIM would invert the table, which is the actual substance of the flag.

Every row carries the team name alongside the submitterid, resolved from Synapse (teams via
/team, individual submitters via /userProfile) and cached in .leaderboard_teams.json so a
``--csv`` rerun needs no network.

Both accept ``--team`` to restrict to one submitter, ``--mine`` for inzva_mri, ``--just-me``
for the user's own uploads within inzva_mri, and ``--csv``
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

#: Which way is better for each metric. SSIM is a similarity and nRMSE/LPIPS are errors, so
#: a single sort direction would silently rank two of the three backwards -- the whole point
#: of --metric is getting this right, not just changing the sort key.
HIGHER_IS_BETTER = {"SSIM": True, "nRMSE": False, "LPIPS": False}

#: Case-insensitive spellings accepted on the command line.
METRIC_ALIASES = {name.lower(): name for name in HIGHER_IS_BETTER}

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


#: Resolved submitterid -> display name, so --csv keeps working with no network.
TEAM_CACHE = REPO_ROOT / ".leaderboard_teams.json"


def team_names(ids) -> dict[int, str]:
    """Resolve submitterids to display names, cached on disk.

    A submitterid is a *team* id for a team submission and a *user* id for an individual
    one, and only trying both tells them apart -- 3584730 on this leaderboard is a person,
    not a team. Unknown ids fall back to their number rather than raising: a name is a
    convenience, and losing the whole table because one lookup 403s is not a trade worth
    making.

    Cached because --csv exists to be usable offline, and because otherwise every run is
    40 round trips for values that never change.
    """
    import json

    cache: dict[str, str] = {}
    if TEAM_CACHE.is_file():
        try:
            cache = json.loads(TEAM_CACHE.read_text())
        except (OSError, ValueError):
            cache = {}

    missing = [int(i) for i in ids if str(int(i)) not in cache]
    if missing:
        try:
            import synapseclient

            from mrixfields.env import load_env

            load_env()
            syn = synapseclient.Synapse(silent=True)
            syn.login(authToken=os.environ["PERSONAL_ACCESS_TOKEN"])
            for identifier in missing:
                name = None
                for path, key in (("team", "name"), ("userProfile", "userName")):
                    try:
                        name = syn.restGET(f"/{path}/{identifier}").get(key)
                        break
                    except Exception:  # noqa: BLE001 - try the other kind, then give up
                        continue
                cache[str(identifier)] = name or str(identifier)
            try:
                TEAM_CACHE.write_text(json.dumps(cache, indent=2, sort_keys=True))
            except OSError:
                pass
        except Exception as exc:  # noqa: BLE001 - offline, or no token
            print(f"team names unavailable ({type(exc).__name__}); showing ids",
                  file=sys.stderr)

    return {int(i): cache.get(str(int(i)), str(int(i))) for i in ids}


def by_team(frame: pd.DataFrame, metric: str = "SSIM") -> pd.DataFrame:
    """One row per submitter: their best submission on ``metric``, in score order.

    "Best" flips with the metric: the highest SSIM but the *lowest* nRMSE or LPIPS. Both
    the per-team pick and the ordering follow HIGHER_IS_BETTER, so a team's row under
    --metric nrmse is the submission with their lowest nRMSE, not their best SSIM.

    Note this re-ranks the field on a metric the challenge does not rank. Task 3's
    primary_metric is SSIM; the others are diagnostic.
    """
    higher = HIGHER_IS_BETTER[metric]
    grouped = frame.groupby("submitterid")[metric]
    best = frame.loc[grouped.idxmax() if higher else grouped.idxmin()].copy()
    best["n"] = frame.groupby("submitterid")["id"].count().reindex(best["submitterid"]).values
    best = best.sort_values(metric, ascending=not higher).reset_index(drop=True)
    best.insert(0, "rank", range(1, len(best) + 1))
    return best


def render(frame: pd.DataFrame, show_rank: bool) -> str:
    frame = frame.copy()
    frame["when"] = frame["when"].dt.strftime("%m-%d")
    frame["us"] = frame["submitterid"].eq(OUR_SUBMITTER_ID).map({True: "*", False: ""})
    frame["name"] = frame["name"].astype(str).str.slice(0, 30)
    identity = ["submitterid"]
    if "team" in frame:
        frame["team"] = frame["team"].astype(str).str.slice(0, 20)
        # Team first, id second: the name is what a reader recognises, but the id stays
        # because it is the only stable key -- team names are free text and repeat.
        identity = ["team", "submitterid"]
    columns = ((["rank"] if show_rank else []) + ["us", "name"] + identity
               + ["when", "SSIM", "nRMSE", "LPIPS"])
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
    parser.add_argument("--just-me", action="store_true",
                        help=f"only submissions uploaded from the user's own account "
                             f"(--mine --created-by {OUR_CREATED_BY})")
    parser.add_argument("--keep-perfect", action="store_true",
                        help="keep submissions scoring exactly 1.0 (the ground truth)")
    parser.add_argument("--metric", default="SSIM",
                        choices=sorted(METRIC_ALIASES) + sorted(HIGHER_IS_BETTER),
                        help="rank and sort by this metric (default SSIM, which is what the "
                             "challenge actually ranks). nRMSE and LPIPS sort ascending, "
                             "since for those lower is better.")
    parser.add_argument("--top", type=int, default=0, help="show only the first N rows")
    parser.add_argument("--out", type=Path, help="also write the rendered table to this CSV")
    args = parser.parse_args()
    metric = METRIC_ALIASES.get(args.metric.lower(), args.metric)

    raw = pd.read_csv(args.csv) if args.csv else fetch(save=not args.no_save)
    frame = clean(raw, keep_perfect=args.keep_perfect)

    if args.just_me:
        args.mine, args.created_by = True, OUR_CREATED_BY
    team = OUR_SUBMITTER_ID if args.mine else args.team
    ranked = by_team(frame, metric)
    if team is not None:
        # Rank against the whole field first, then filter, so a filtered view still says
        # where the team actually stands rather than renumbering from 1.
        standing = ranked[ranked["submitterid"] == team]
        if len(standing):
            row = standing.iloc[0]
            print(f"submitterid {team}: rank {row['rank']} of {len(ranked)} teams, "
                  f"best {metric} {row[metric]:.6f}\n")
        frame = frame[frame["submitterid"] == team]
    if args.created_by is not None:
        frame = frame[frame["createdBy"] == args.created_by]

    if args.submissions or team is not None or args.created_by is not None:
        view = frame.sort_values(metric, ascending=not HIGHER_IS_BETTER[metric])
        show_rank = False
    else:
        view = ranked
        show_rank = True
    if args.top:
        view = view.head(args.top)

    view = view.assign(team=view["submitterid"].map(team_names(view["submitterid"].unique())))

    print(render(view, show_rank))
    if args.out:
        view.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
