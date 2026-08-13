from datetime import datetime, timedelta, timezone

import polars as pl
import pytest

from scripts.refresh_news import NEWS_SCHEMA, _clean_text, _parse_pub_date, parse_rss, tag_and_filter

KNOWN_NAMES = ["Jordan Love", "Jared Goff"]


@pytest.fixture(autouse=True)
def id_map(monkeypatch):
    """tag_and_filter joins against data/processed/player_stats.parquet for player_id -
    stub it so these tests don't depend on a real refresh having run.
    """
    fixture = pl.DataFrame({
        "player_id": ["00-1", "00-2"],
        "player_name": ["Jordan Love", "Jared Goff"],
    })
    monkeypatch.setattr("scripts.refresh_news.pl.read_parquet", lambda *_, **__: fixture)


def _rss(items_xml: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><title>Test Feed</title>{items_xml}</channel></rss>"""


def test_parse_rss_extracts_title_description_link_and_date():
    xml = _rss("""
        <item>
            <title><![CDATA[Jordan Love day-to-day]]></title>
            <description><![CDATA[<p>Packers QB <b>Jordan Love</b> is questionable.</p>]]></description>
            <link>https://example.com/love</link>
            <pubDate>Wed, 12 Aug 2026 12:48:28 EST</pubDate>
        </item>
    """)
    items = parse_rss(xml, "ESPN")

    assert len(items) == 1
    item = items[0]
    assert item["title"] == "Jordan Love day-to-day"
    assert item["description"] == "Packers QB Jordan Love is questionable."
    assert item["link"] == "https://example.com/love"
    assert item["source"] == "ESPN"
    assert item["published_at"] is not None


def test_parse_rss_handles_missing_pubdate():
    xml = _rss("<item><title>No date</title><link>https://example.com/x</link></item>")
    items = parse_rss(xml, "Yahoo")
    assert items[0]["published_at"] is None


def test_clean_text_removes_markup_but_keeps_text():
    assert _clean_text("<p>Hello <b>world</b></p>") == "Hello world"


def test_clean_text_unescapes_double_encoded_entities():
    assert _clean_text("Rodgers &quot;bummed&quot; out") == 'Rodgers "bummed" out'


def test_parse_pub_date_returns_none_for_garbage():
    assert _parse_pub_date("not a date") is None
    assert _parse_pub_date(None) is None


def test_parse_pub_date_normalizes_to_utc():
    parsed = _parse_pub_date("Wed, 12 Aug 2026 12:48:28 EST")
    assert parsed.tzinfo is not None
    assert parsed.tzinfo.utcoffset(parsed).total_seconds() == 0


def _raw_row(title, description, published_at, link="https://example.com/1", source="ESPN"):
    return {
        "title": title,
        "description": description,
        "link": link,
        "source": source,
        "published_at": published_at,
    }


def test_tag_and_filter_matches_known_players_in_title_or_description():
    now = datetime.now(timezone.utc)
    raw = pl.DataFrame(
        [_raw_row("Jordan Love update", "Packers QB is fine.", now)], schema=NEWS_SCHEMA
    )
    tagged = tag_and_filter(raw, max_age_days=10, known_names=KNOWN_NAMES)

    assert tagged.height == 1
    assert tagged.row(0, named=True)["player_name"] == "Jordan Love"


def test_tag_and_filter_drops_articles_older_than_the_cutoff():
    stale = datetime.now(timezone.utc) - timedelta(days=30)
    raw = pl.DataFrame([_raw_row("Jordan Love update", "old news", stale)], schema=NEWS_SCHEMA)
    tagged = tag_and_filter(raw, max_age_days=10, known_names=KNOWN_NAMES)
    assert tagged.height == 0


def test_tag_and_filter_drops_articles_with_no_publish_date():
    raw = pl.DataFrame([_raw_row("Jordan Love update", "no date", None)], schema=NEWS_SCHEMA)
    tagged = tag_and_filter(raw, max_age_days=10, known_names=KNOWN_NAMES)
    assert tagged.height == 0


def test_tag_and_filter_drops_articles_mentioning_no_known_player():
    now = datetime.now(timezone.utc)
    raw = pl.DataFrame([_raw_row("Unrelated headline", "nothing fantasy-relevant here", now)],
                        schema=NEWS_SCHEMA)
    tagged = tag_and_filter(raw, max_age_days=10, known_names=KNOWN_NAMES)
    assert tagged.height == 0


def test_tag_and_filter_produces_one_row_per_mentioned_player():
    now = datetime.now(timezone.utc)
    raw = pl.DataFrame(
        [_raw_row("Love vs Goff preview", "Jordan Love and Jared Goff both start.", now)],
        schema=NEWS_SCHEMA,
    )
    tagged = tag_and_filter(raw, max_age_days=10, known_names=KNOWN_NAMES)
    assert set(tagged.select("player_name").to_series().to_list()) == {"Jordan Love", "Jared Goff"}


def test_tag_and_filter_joins_player_id_from_processed_player_stats(monkeypatch):
    now = datetime.now(timezone.utc)
    raw = pl.DataFrame([_raw_row("Jordan Love update", "Packers QB is fine.", now)],
                        schema=NEWS_SCHEMA)
    id_map = pl.DataFrame({"player_id": ["00-1"], "player_name": ["Jordan Love"]})
    monkeypatch.setattr("scripts.refresh_news.pl.read_parquet", lambda *_, **__: id_map)

    tagged = tag_and_filter(raw, max_age_days=10, known_names=KNOWN_NAMES)
    assert tagged.row(0, named=True)["player_id"] == "00-1"


def test_tag_and_filter_drops_articles_about_an_ambiguous_player_name(monkeypatch):
    """Two real players can share a display name (e.g. two "Byron Young"s) - an
    article about either must be dropped, never silently attached to the wrong id.
    """
    now = datetime.now(timezone.utc)
    raw = pl.DataFrame([_raw_row("Jordan Love update", "Packers QB is fine.", now)],
                        schema=NEWS_SCHEMA)
    id_map = pl.DataFrame({
        "player_id": ["00-1", "00-2"],
        "player_name": ["Jordan Love", "Jordan Love"],
    })
    monkeypatch.setattr("scripts.refresh_news.pl.read_parquet", lambda *_, **__: id_map)

    tagged = tag_and_filter(raw, max_age_days=10, known_names=KNOWN_NAMES)
    assert tagged.height == 0
