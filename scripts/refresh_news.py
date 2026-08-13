"""Refresh fantasy news: RSS feeds -> data/{raw,processed}/news.parquet (README §4).

Usage:
    python -m scripts.refresh_news [--max-age-days N]

(Run as a module, not a script path: this file imports pipeline.entity_extraction,
which needs the repo root on sys.path - same reason tests use `python -m pytest`.)

Fetches ESPN/Yahoo/RotoBaller RSS, tags each item with the known player(s) it
mentions (regex matching against data/processed/player_stats.parquet, same
approach as pipeline/entity_extraction.py), and keeps only items published in
the last `--max-age-days` days (default 10, per README §4.1's "7-10 days").
Untaggable items (no known player name in title/description) are dropped -
there's nothing to attach them to in a per-player context.
"""
from __future__ import annotations

import argparse
import html
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import polars as pl
import requests
from bs4 import BeautifulSoup

from pipeline.entity_extraction import extract_players, known_player_names

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

FEEDS = [
    ("ESPN", "https://www.espn.com/espn/rss/nfl/news"),
    ("Yahoo", "https://sports.yahoo.com/nfl/rss/"),
    ("RotoBaller", "https://www.rotoballer.com/feed"),
]

DEFAULT_MAX_AGE_DAYS = 10

NEWS_SCHEMA = {
    "title": pl.String,
    "description": pl.String,
    "link": pl.String,
    "source": pl.String,
    "published_at": pl.Datetime("us", "UTC"),
}

TAGGED_NEWS_SCHEMA = {
    "player_id": pl.String,
    "player_name": pl.String,
    "title": pl.String,
    "description": pl.String,
    "link": pl.String,
    "source": pl.String,
    "published_at": pl.Datetime("us", "UTC"),
}


def log(msg: str) -> None:
    """Print a progress message prefixed with the script name."""
    print(f"[refresh_news] {msg}")


def _write(df: pl.DataFrame, directory: Path, name: str) -> pl.DataFrame:
    """Write `df` to `directory/name.parquet`, creating the directory if needed."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.parquet"
    df.write_parquet(path)
    log(f"wrote {path} ({df.height} rows)")
    return df


def _clean_text(text: str) -> str:
    """Plain text from an RSS title/description field: strips any HTML markup and
    unescapes entities (some feeds double-encode titles, e.g. literal "&quot;").
    """
    return html.unescape(BeautifulSoup(text, "html.parser").get_text(separator=" ", strip=True))


def _parse_pub_date(raw: str | None) -> datetime | None:
    """RFC-2822 pubDate -> a UTC datetime, or None if missing/unparseable.

    A feed item with no usable date can't be checked against the recency window,
    so it's dropped later rather than assumed recent.
    """
    if not raw:
        return None
    try:
        parsed = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def parse_rss(xml_text: str, source: str) -> list[dict]:
    """Parse an RSS 2.0 document into raw item dicts (title/description/link/source/published_at)."""
    root = ET.fromstring(xml_text)
    items = []
    for item in root.iter("item"):
        title = _clean_text(item.findtext("title") or "")
        description = _clean_text(item.findtext("description") or "")
        link = (item.findtext("link") or "").strip()
        items.append({
            "title": title,
            "description": description,
            "link": link,
            "source": source,
            "published_at": _parse_pub_date(item.findtext("pubDate")),
        })
    return items


def fetch_feed(name: str, url: str) -> list[dict]:
    """Fetch and parse one RSS feed; returns [] (with a log line) if the feed is unreachable.

    One dead feed shouldn't sink the whole refresh - same failure-isolation
    approach as refresh_stats.load_by_season for a missing nflverse season.
    """
    try:
        resp = requests.get(url, timeout=20, headers={"User-Agent": "fantasy-football-wizard/0.1"})
        resp.raise_for_status()
        return parse_rss(resp.text, name)
    except (requests.RequestException, ET.ParseError) as exc:
        log(f"skipping {name} ({url}): {exc}")
        return []


def fetch_raw_news() -> pl.DataFrame:
    """Fetch every configured RSS feed and write the combined raw items to data/raw/news.parquet."""
    items = [item for name, url in FEEDS for item in fetch_feed(name, url)]
    df = pl.DataFrame(items, schema=NEWS_SCHEMA) if items else pl.DataFrame(schema=NEWS_SCHEMA)
    return _write(df, RAW_DIR, "news")


def tag_and_filter(raw: pl.DataFrame, max_age_days: int, known_names: list[str] | None = None) -> pl.DataFrame:
    """Tag each item with the known player(s) it mentions and drop stale/untagged items.

    One row per (article, player) match - an article mentioning two players is
    relevant context for both, so it's duplicated rather than picking one.
    """
    names = known_names if known_names is not None else known_player_names()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

    recent = raw.filter(
        pl.col("published_at").is_not_null() & (pl.col("published_at") >= cutoff)
    )

    rows = []
    for row in recent.to_dicts():
        text = f"{row['title']} {row['description']}"
        for player_name in extract_players(text, known_names=names):
            rows.append({**row, "player_name": player_name})

    if not rows:
        return pl.DataFrame(schema=TAGGED_NEWS_SCHEMA)

    tagged = pl.DataFrame(rows)
    # player_stats can carry two different real players under the same display
    # name (e.g. two "Byron Young"s, a DT and an LB) - .unique(subset=["player_name"])
    # would arbitrarily keep one of their ids, silently misattributing every article
    # about either player to whichever one happened to survive the dedup. There's no
    # fuzzy/context-based way to tell them apart from RSS title/description text
    # (README's "never implement fuzzy player-name matching"), so an ambiguous name
    # is dropped rather than guessed - same "clear failure over a silent guess"
    # stance as PlayerNotFoundError and the hallucinated-name check elsewhere.
    id_map = pl.read_parquet(
        PROCESSED_DIR / "player_stats.parquet", columns=["player_id", "player_name"]
    ).drop_nulls().unique()
    unambiguous_names = (
        id_map.group_by("player_name").agg(pl.len().alias("n")).filter(pl.col("n") == 1)
    )
    id_map = id_map.join(unambiguous_names, on="player_name", how="semi")

    tagged = tagged.join(id_map, on="player_name", how="left")
    return (
        tagged.filter(pl.col("player_id").is_not_null())
        .select(list(TAGGED_NEWS_SCHEMA))
        .unique(subset=["player_id", "link"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS)
    args = parser.parse_args()

    raw = fetch_raw_news()
    tagged = tag_and_filter(raw, args.max_age_days)
    _write(tagged, PROCESSED_DIR, "news")
    distinct_players = tagged.select("player_id").drop_nulls().unique().height
    log(f"tagged {tagged.height} article-player rows covering {distinct_players} players")


if __name__ == "__main__":
    main()
