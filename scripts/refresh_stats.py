"""Refresh structured NFL data: nflreadpy + Sleeper -> data/{raw,staged,processed}/*.parquet.

Usage:
    python scripts/refresh_stats.py [--season YYYY] [--week N]

Without --season/--week, resolves the most recently completed regular-season week.
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import nflreadpy as nfl
import polars as pl
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
STAGED_DIR = DATA_DIR / "staged"
PROCESSED_DIR = DATA_DIR / "processed"

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"
SLEEPER_PROJECTIONS_URL = "https://api.sleeper.com/projections/nfl/{season}/{week}"

SLEEPER_PLAYER_FIELDS = [
    "player_id", "gsis_id", "full_name", "position", "team", "status",
    "injury_status", "injury_body_part", "injury_notes", "injury_start_date",
    "practice_participation",
]

LAST_N_WEEKS = 3

# nflreadpy.load_ff_rankings(type="week") always returns the latest FantasyPros
# scrape - there's no way to request a historical week's ECR. Used to build an
# empty-but-correctly-typed stand-in when the requested week isn't the current one.
FF_RANKINGS_ECR_SCHEMA = {
    "fantasypros_id": pl.Int64,
    "player_name": pl.String,
    "pos": pl.String,
    "team": pl.String,
    "ecr": pl.Float64,
    "pos_rank": pl.String,
    "player_bye_week": pl.Int64,
}


def log(msg: str) -> None:
    """Print a progress message prefixed with the script name."""
    print(f"[refresh_stats] {msg}")


def _write(df: pl.DataFrame, directory: Path, name: str) -> pl.DataFrame:
    """Write `df` to `directory/name.parquet`, creating the directory if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    df.write_parquet(path)
    log(f"wrote {path} ({df.height} rows)")
    return df


# ---------------------------------------------------------------------------
# Season/week resolution
# ---------------------------------------------------------------------------

def resolve_season_week(season: int | None, week: int | None) -> tuple[int, int]:
    """Default to the most recently completed regular-season week as of today."""
    if season is not None and week is not None:
        return season, week

    # REG only: postseason week numbers don't line up with Sleeper's regular-season
    # projection weeks (1-18), so playoff games would resolve to a week that doesn't exist there.
    schedules = nfl.load_schedules(seasons=True)
    schedules = schedules.filter(
        (pl.col("game_type") == "REG") & pl.col("gameday").is_not_null()
    ).with_columns(pl.col("gameday").str.strptime(pl.Date, strict=False))

    if season is not None:
        schedules = schedules.filter(pl.col("season") == season)

    # A week only counts once every game in it has been played - using the single
    # most recently played game would return a week still in progress (e.g. right
    # after Thursday Night Football but before that week's Sunday/Monday games).
    week_ends = schedules.group_by(["season", "week"]).agg(
        pl.col("gameday").max().alias("week_end")
    )
    completed = week_ends.filter(pl.col("week_end") <= date.today())
    if completed.height == 0:
        raise RuntimeError(
            "No completed regular-season week found; pass --season/--week explicitly"
        )
    # During the off-season this falls back to the last week of the prior season,
    # which is what "current" should mean until the next season's games start.
    latest = completed.sort("week_end", descending=True).row(0, named=True)
    return season or latest["season"], week or latest["week"]


# ---------------------------------------------------------------------------
# Raw layer: fetch each source and persist as-is (only column selection applied)
# ---------------------------------------------------------------------------

def fetch_sleeper_players() -> pl.DataFrame:
    """Fetch Sleeper's full player dump, keeping only players with a gsis_id."""
    data = requests.get(SLEEPER_PLAYERS_URL, timeout=30).json()
    rows = [
        {field: player.get(field) for field in SLEEPER_PLAYER_FIELDS}
        for player in data.values()
        if player and player.get("gsis_id")
    ]
    return pl.DataFrame(rows, infer_schema_length=None)


def fetch_sleeper_projections(season: int, week: int) -> pl.DataFrame:
    """Fetch Sleeper's weekly projections (PPR/half-PPR/standard) for one season/week."""
    url = SLEEPER_PROJECTIONS_URL.format(season=season, week=week)
    resp = requests.get(url, params={"season_type": "regular"}, timeout=30)
    resp.raise_for_status()
    rows = []
    for entry in resp.json():
        stats = entry.get("stats") or {}
        player = entry.get("player") or {}
        first, last = player.get("first_name"), player.get("last_name")
        # Sleeper's projection entries don't include the FantasyPros ID needed to
        # join with ECR rankings, so carry name/position/team here as a fallback.
        rows.append({
            "sleeper_id": entry.get("player_id"),
            "season": season,
            "week": week,
            "player_name": f"{first} {last}".strip() if first or last else None,
            "position": player.get("position"),
            "team": entry.get("team"),
            "pts_ppr": stats.get("pts_ppr"),
            "pts_half_ppr": stats.get("pts_half_ppr"),
            "pts_std": stats.get("pts_std"),
        })
    return pl.DataFrame(rows, infer_schema_length=None)


def fetch_raw(season: int, week: int, fetch_ecr: bool) -> dict[str, pl.DataFrame]:
    """Fetch every nflreadpy + Sleeper source for the given season/week and write raw/*.parquet.

    `fetch_ecr` should be False for any season/week that isn't the current one -
    load_ff_rankings only ever returns the latest scrape, so joining it against a
    backfilled/historical week would silently pair mismatched data.
    """
    log(f"fetching raw sources for season={season} week={week}")
    if fetch_ecr:
        ff_rankings = nfl.load_ff_rankings(type="week")
    else:
        log(f"skipping FantasyPros ECR: season={season} week={week} is not the current week")
        ff_rankings = pl.DataFrame(schema=FF_RANKINGS_ECR_SCHEMA)
    raw = {
        "player_stats": nfl.load_player_stats(seasons=season, summary_level="week"),
        "snap_counts": nfl.load_snap_counts(seasons=season),
        "ff_opportunity": nfl.load_ff_opportunity(seasons=season, stat_type="weekly"),
        "ngs_passing": nfl.load_nextgen_stats(seasons=season, stat_type="passing"),
        "ngs_receiving": nfl.load_nextgen_stats(seasons=season, stat_type="receiving"),
        "ngs_rushing": nfl.load_nextgen_stats(seasons=season, stat_type="rushing"),
        "injuries": nfl.load_injuries(seasons=season),
        "depth_charts": nfl.load_depth_charts(seasons=season),
        "schedules": nfl.load_schedules(seasons=season),
        "ff_playerids": nfl.load_ff_playerids(),
        "ff_rankings": ff_rankings,
        "sleeper_players": fetch_sleeper_players(),
        "sleeper_projections": fetch_sleeper_projections(season, week),
    }
    for name, df in raw.items():
        _write(df, RAW_DIR, name)
    return raw


# ---------------------------------------------------------------------------
# Staged layer: join sources on canonical player IDs (gsis_id)
# ---------------------------------------------------------------------------

def _pfr_to_gsis(ff_playerids: pl.DataFrame) -> pl.DataFrame:
    """snap_counts is keyed by pfr_player_id, not gsis_id, so bridge through ff_playerids."""
    return (
        ff_playerids.select(["pfr_id", "gsis_id"])
        .filter(pl.col("pfr_id").is_not_null() & pl.col("gsis_id").is_not_null())
        .unique(subset=["pfr_id"])
    )


def build_staged_player_stats(raw: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Join weekly player_stats with snap counts, xFP, and Next Gen Stats on (player_id, season, week)."""
    stats = raw["player_stats"].select([
        "player_id", "player_display_name", "position", "team", "opponent_team",
        "season", "week", "season_type", "targets", "receptions", "carries",
        "rushing_epa", "receiving_epa", "passing_epa", "target_share",
        "air_yards_share", "fantasy_points", "fantasy_points_ppr",
    ]).rename({"player_display_name": "player_name"})

    pfr_map = _pfr_to_gsis(raw["ff_playerids"])
    snaps = (
        raw["snap_counts"]
        .rename({"pfr_player_id": "pfr_id"})
        .join(pfr_map, on="pfr_id", how="left")
        .filter(pl.col("gsis_id").is_not_null())
        .select(["gsis_id", "season", "week", "offense_pct"])
        .rename({"gsis_id": "player_id", "offense_pct": "snap_percentage"})
    )

    # ff_opportunity ships season/week as String/Float64 (unlike every other nflverse
    # table here, which uses Int32) - cast to match player_stats before joining on them.
    xfp = raw["ff_opportunity"].select([
        "player_id", "season", "week", "total_fantasy_points_exp",
    ]).with_columns([
        pl.col("season").cast(pl.Int32),
        pl.col("week").cast(pl.Int32),
    ]).rename({"total_fantasy_points_exp": "xfp"})

    ngs_receiving = raw["ngs_receiving"].select([
        "player_gsis_id", "season", "week", "avg_separation", "avg_yac_above_expectation",
    ]).rename({"player_gsis_id": "player_id"})

    ngs_rushing = raw["ngs_rushing"].select([
        "player_gsis_id", "season", "week", "rush_yards_over_expected_per_att",
    ]).rename({"player_gsis_id": "player_id"})

    ngs_passing = raw["ngs_passing"].select([
        "player_gsis_id", "season", "week",
        "completion_percentage_above_expectation", "aggressiveness",
    ]).rename({"player_gsis_id": "player_id"})

    return (
        stats
        .join(snaps, on=["player_id", "season", "week"], how="left")
        .join(xfp, on=["player_id", "season", "week"], how="left")
        .join(ngs_receiving, on=["player_id", "season", "week"], how="left")
        .join(ngs_rushing, on=["player_id", "season", "week"], how="left")
        .join(ngs_passing, on=["player_id", "season", "week"], how="left")
    )


def build_staged_injuries(raw: dict[str, pl.DataFrame], season: int, week: int) -> pl.DataFrame:
    """Overlay Sleeper's real-time injury status on top of the official weekly report."""
    injuries_df = raw["injuries"]
    # date_modified is present for some season vintages but not others upstream;
    # add it as null rather than letting the later select() fail.
    if "date_modified" not in injuries_df.columns:
        injuries_df = injuries_df.with_columns(pl.lit(None, dtype=pl.String).alias("date_modified"))

    official = (
        injuries_df
        .filter((pl.col("season") == season) & (pl.col("week") == week))
        .select([
            "gsis_id", "season", "week", "team", "position", "full_name",
            "report_primary_injury", "report_status", "practice_status", "date_modified",
        ])
        .rename({"gsis_id": "player_id", "full_name": "player_name"})
    )

    sleeper = (
        raw["sleeper_players"]
        .filter(pl.col("gsis_id").is_not_null())
        .select([
            "gsis_id", "full_name", "position", "team", "injury_status",
            "injury_body_part", "injury_notes", "injury_start_date",
            "practice_participation",
        ])
        .rename({"gsis_id": "player_id"})
    )

    # full join: the official weekly report only covers players teams flagged that
    # week, while Sleeper's dump is a live snapshot of every player - neither side
    # is a superset, so an inner/left join would silently drop rows from one side.
    return official.join(
        sleeper, on="player_id", how="full", coalesce=True, suffix="_sleeper"
    )


def build_staged_projections(raw: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Join Sleeper weekly projections with FantasyPros ECR rankings, both mapped to gsis_id."""
    ids = raw["ff_playerids"].select(["sleeper_id", "fantasypros_id", "gsis_id"])

    sleeper_ids = ids.select(["sleeper_id", "gsis_id"]).filter(
        pl.col("sleeper_id").is_not_null()
    ).unique(subset=["sleeper_id"])
    sleeper_proj = (
        raw["sleeper_projections"]
        # Sleeper's own player_id is numeric except for team defenses, which use the
        # team abbreviation (e.g. "SEA") instead - strict=False turns those into
        # null rather than failing the whole load; ff_playerids doesn't map DSTs anyway.
        .with_columns(pl.col("sleeper_id").cast(pl.Int64, strict=False))
        .join(sleeper_ids, on="sleeper_id", how="left")
        .rename({"gsis_id": "player_id"})
        .drop("sleeper_id")
    )

    # ff_playerids.fantasypros_id is String but ff_rankings ships it as Int64 - align types to join.
    fp_ids = ids.select(["fantasypros_id", "gsis_id"]).filter(
        pl.col("fantasypros_id").is_not_null()
    ).with_columns(pl.col("fantasypros_id").cast(pl.Int64)).unique(subset=["fantasypros_id"])
    ecr = (
        raw["ff_rankings"]
        .select(["fantasypros_id", "player_name", "pos", "team", "ecr", "pos_rank", "player_bye_week"])
        .join(fp_ids, on="fantasypros_id", how="left")
        .rename({"gsis_id": "player_id", "ecr": "ecr_rank", "pos_rank": "ecr_position_rank"})
        .drop("fantasypros_id")
    )

    return sleeper_proj.join(
        ecr, on="player_id", how="full", coalesce=True, suffix="_ecr"
    )


def build_staged(raw: dict[str, pl.DataFrame], season: int, week: int) -> dict[str, pl.DataFrame]:
    """Build and persist all staged/*.parquet tables from the raw sources."""
    staged = {
        "player_stats": build_staged_player_stats(raw),
        "injuries": build_staged_injuries(raw, season, week),
        "projections": build_staged_projections(raw),
    }
    for name, df in staged.items():
        _write(df, STAGED_DIR, name)
    return staged


# ---------------------------------------------------------------------------
# Processed layer: one row per player, current-week snapshot with aggregates
# ---------------------------------------------------------------------------

def build_processed_player_stats(staged: pl.DataFrame, season: int, week: int) -> pl.DataFrame:
    """Collapse weekly staged stats into one row per player: last-3-week and season averages."""
    season_to_date = staged.filter((pl.col("season") == season) & (pl.col("week") <= week))
    # "last 3 weeks" = the 3 weeks up to and including the target week (bye weeks
    # simply thin the window since there's no row for that player that week).
    last3 = season_to_date.filter(pl.col("week") > week - LAST_N_WEEKS)

    aggregates = last3.group_by("player_id").agg([
        pl.col("player_name").last(),
        pl.col("position").last(),
        pl.col("team").last(),
        pl.mean("fantasy_points").alias("avg_fantasy_points_last3"),
        pl.mean("fantasy_points_ppr").alias("avg_fantasy_points_ppr_last3"),
        pl.mean("snap_percentage").alias("avg_snap_percentage_last3"),
        pl.mean("targets").alias("avg_targets_last3"),
        pl.mean("carries").alias("avg_carries_last3"),
        pl.mean("xfp").alias("avg_xfp_last3"),
    ])

    season_avg = season_to_date.group_by("player_id").agg([
        pl.mean("fantasy_points").alias("avg_fantasy_points_season"),
        pl.mean("fantasy_points_ppr").alias("avg_fantasy_points_ppr_season"),
    ])

    latest_week = (
        staged.filter((pl.col("season") == season) & (pl.col("week") == week))
        .select(["player_id", "fantasy_points", "fantasy_points_ppr"])
        .rename({
            "fantasy_points": "fantasy_points_last_week",
            "fantasy_points_ppr": "fantasy_points_ppr_last_week",
        })
    )

    return (
        aggregates
        .join(season_avg, on="player_id", how="left")
        .join(latest_week, on="player_id", how="left")
        .with_columns([pl.lit(season).alias("season"), pl.lit(week).alias("week")])
    )


def build_processed_injuries(staged: pl.DataFrame) -> pl.DataFrame:
    """Flatten staged injuries into the final per-player status/practice_level/notes columns."""
    return staged.filter(pl.col("player_id").is_not_null()).select([
        "player_id",
        pl.coalesce(["player_name", "full_name"]).alias("player_name"),
        pl.coalesce(["position", "position_sleeper"]).alias("position"),
        pl.coalesce(["team", "team_sleeper"]).alias("team"),
        # Most players have no injury report at all - absence of a status means healthy,
        # not unknown, so downstream context building doesn't need its own null-handling.
        pl.coalesce(["injury_status", "report_status"]).fill_null("Healthy").alias("status"),
        pl.coalesce(["practice_status", "practice_participation"]).alias("practice_level"),
        pl.coalesce(["date_modified", "injury_start_date"]).alias("report_date"),
        pl.coalesce(["injury_notes", "report_primary_injury"]).alias("notes"),
    ])


def build_processed_projections(staged: pl.DataFrame) -> pl.DataFrame:
    """Flatten staged projections into the final per-player projected_points/ecr_rank columns."""
    return staged.filter(pl.col("player_id").is_not_null()).select([
        "player_id",
        pl.coalesce(["player_name", "player_name_ecr"]).alias("player_name"),
        pl.coalesce(["position", "pos"]).alias("position"),
        pl.coalesce(["team", "team_ecr"]).alias("team"),
        "season", "week",
        # PPR is this app's default scoring format (README §9); half-PPR/standard
        # are kept alongside for a future league-scoring-aware context builder.
        pl.col("pts_ppr").alias("projected_points"),
        pl.lit("sleeper").alias("source"),
        "pts_half_ppr", "pts_std",
        "ecr_rank", "ecr_position_rank",
    ])


def build_processed(staged: dict[str, pl.DataFrame], season: int, week: int) -> None:
    """Build and persist all processed/*.parquet tables from the staged tables."""
    processed = {
        "player_stats": build_processed_player_stats(staged["player_stats"], season, week),
        "injuries": build_processed_injuries(staged["injuries"]),
        "projections": build_processed_projections(staged["projections"]),
    }
    for name, df in processed.items():
        _write(df, PROCESSED_DIR, name)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entrypoint: resolve season/week, then run raw -> staged -> processed end to end."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--week", type=int, default=None)
    args = parser.parse_args()

    season, week = resolve_season_week(args.season, args.week)
    log(f"resolved season={season} week={week}")

    # ECR rankings are only ever available for the current week (see fetch_raw) -
    # if the caller explicitly requested a different season/week, skip them.
    is_current_week = args.season is None and args.week is None
    if not is_current_week:
        is_current_week = (season, week) == resolve_season_week(None, None)

    raw = fetch_raw(season, week, is_current_week)
    staged = build_staged(raw, season, week)
    build_processed(staged, season, week)
    log("refresh complete")


if __name__ == "__main__":
    main()
