import pytest
from fastapi.testclient import TestClient

from api.main import (
    MAX_HISTORY_TURNS,
    MAX_MENTIONED_PLAYERS,
    MAX_QUESTION_LENGTH,
    _docs_config,
    _maybe_sync_from_gcs,
    app,
)
from llm.interface import Recommendation
from pipeline.chat_engine import ChatResult, NoPlayersFoundError
from pipeline.context_builder import PlayerNotFoundError
from pipeline.decision_engine import DataUnavailableError, DecisionError
from retrieval.news_retriever import NewsItem

client = TestClient(app)


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The limiter's in-memory counters persist on `app.state` across tests
    since `app` is a module-level singleton - without a reset, whichever test
    runs 31st in the file would spuriously see a 429.
    """
    app.state.limiter.reset()
    yield


def _fixture_chat_result(recommendation=None):
    return ChatResult(
        answer="Start Jordan Love, he has the better matchup.",
        sources=[NewsItem(
            title="Love update", snippet="Love update\nDetails.", link="https://example.com/a",
            source="ESPN", published_at="2026-09-01T00:00:00+00:00",
        )],
        players_discussed=["Jordan Love", "Jared Goff"],
        recommendation=recommendation,
        week=5,
        context="PLAYER COMPARISON\n\nJordan Love:\n- Avg fantasy points (last 3 weeks): 18.4",
    )


@pytest.fixture
def stub_chat(monkeypatch):
    """Replace the pipeline call so the endpoint is tested without an API call."""
    calls = {}

    def fake_chat(message, mentioned_players=None, week=None, history=None, **_):
        calls.update(
            message=message, mentioned_players=mentioned_players, week=week, history=history
        )
        return _fixture_chat_result()

    monkeypatch.setattr("api.main.chat", fake_chat)
    return calls


def test_post_chat_returns_the_answer_as_json(stub_chat):
    response = client.post(
        "/chat", json={"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Start Jordan Love, he has the better matchup."
    assert body["players_discussed"] == ["Jordan Love", "Jared Goff"]
    assert body["sources"][0]["link"] == "https://example.com/a"
    assert body["recommendation"] is None
    assert body["week"] == 5
    assert body["context"].startswith("PLAYER COMPARISON")


def test_post_chat_includes_a_recommendation_when_present(monkeypatch):
    rec = Recommendation(
        start="Jordan Love", bench="Jared Goff", confidence=0.8, key_factors=["x"], risk_factors=[]
    )
    monkeypatch.setattr("api.main.chat", lambda *_a, **_k: _fixture_chat_result(recommendation=rec))
    response = client.post(
        "/chat", json={"message": "Start/sit?", "mentioned_players": ["Jordan Love", "Jared Goff"]}
    )

    body = response.json()
    assert body["recommendation"]["start"] == "Jordan Love"
    assert body["recommendation"]["confidence"] == 0.8


def test_post_chat_passes_mentioned_players_week_and_history(stub_chat):
    client.post("/chat", json={
        "message": "What about now?",
        "mentioned_players": ["Jordan Love"],
        "week": 5,
        "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    })
    assert stub_chat["mentioned_players"] == ["Jordan Love"]
    assert stub_chat["week"] == 5
    assert stub_chat["history"] == [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}
    ]


def test_post_chat_defaults_mentioned_players_and_history_to_empty(stub_chat):
    client.post("/chat", json={"message": "How's the waiver wire?"})
    assert stub_chat["mentioned_players"] == []
    assert stub_chat["history"] == []


@pytest.fixture
def forbid_chat(monkeypatch):
    """Assert the pipeline is never entered - a rejected request costs no LLM call."""
    def fail(*_a, **_k):
        raise AssertionError("chat() should not be called for an invalid request")

    monkeypatch.setattr("api.main.chat", fail)


def test_post_chat_rejects_an_overlong_message(forbid_chat):
    response = client.post("/chat", json={"message": "a" * (MAX_QUESTION_LENGTH + 1)})
    assert response.status_code == 422


def test_post_chat_rejects_too_many_mentioned_players(forbid_chat):
    response = client.post("/chat", json={
        "message": "flex options?",
        "mentioned_players": [f"Player {i}" for i in range(MAX_MENTIONED_PLAYERS + 1)],
    })
    assert response.status_code == 422


def test_post_chat_rejects_too_much_history(forbid_chat):
    history = [{"role": "user", "content": "hi"}] * (MAX_HISTORY_TURNS + 1)
    response = client.post("/chat", json={"message": "hi", "history": history})
    assert response.status_code == 422


def test_post_chat_rejects_an_invalid_history_role(forbid_chat):
    response = client.post("/chat", json={"message": "hi", "history": [{"role": "system", "content": "x"}]})
    assert response.status_code == 422


def test_no_players_found_is_a_400(monkeypatch):
    def raise_no_players(*_a, **_k):
        raise NoPlayersFoundError("No known players mentioned")

    monkeypatch.setattr("api.main.chat", raise_no_players)
    response = client.post("/chat", json={"message": "How's the waiver wire?"})
    assert response.status_code == 400


def test_missing_data_is_a_503_that_hides_server_paths(monkeypatch):
    """A failed refresh is an outage, not the caller's fault - and shouldn't leak paths."""
    def raise_unavailable(*_a, **_k):
        raise DataUnavailableError(
            "data/processed/player_stats.parquet is empty - run python scripts/refresh_stats.py"
        )

    monkeypatch.setattr("api.main.chat", raise_unavailable)
    response = client.post(
        "/chat", json={"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    )
    assert response.status_code == 503
    assert "parquet" not in response.json()["detail"]


def test_unknown_player_is_a_404_not_a_500(monkeypatch):
    def raise_not_found(*_a, **_k):
        raise PlayerNotFoundError('No stats found for "Nobody Here" in week 5')

    monkeypatch.setattr("api.main.chat", raise_not_found)
    response = client.post(
        "/chat", json={"message": "Should I start Nobody Here?", "mentioned_players": ["Nobody Here"]}
    )
    assert response.status_code == 404
    assert "Nobody Here" in response.json()["detail"]


def test_decision_error_is_a_400(monkeypatch):
    def raise_decision_error(*_a, **_k):
        raise DecisionError("LLM answered about someone else")

    monkeypatch.setattr("api.main.chat", raise_decision_error)
    response = client.post(
        "/chat", json={"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "LLM answered about someone else"


def test_post_chat_missing_data_file_is_a_503(monkeypatch):
    """A GCS sync failure (or a fresh container that hasn't synced yet) leaves the
    processed files missing - that must surface as a clean outage, not a raw 500.
    """
    def raise_file_not_found(*_a, **_k):
        raise FileNotFoundError("data/processed/player_stats.parquet")

    monkeypatch.setattr("api.main.chat", raise_file_not_found)
    response = client.post(
        "/chat", json={"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    )
    assert response.status_code == 503


def test_get_players_returns_the_known_name_universe(monkeypatch):
    monkeypatch.setattr("api.main.known_player_names", lambda: ["Jordan Love", "Jared Goff"])
    response = client.get("/players")

    assert response.status_code == 200
    assert response.json() == ["Jordan Love", "Jared Goff"]


def test_get_players_missing_data_file_is_a_503(monkeypatch):
    def raise_file_not_found():
        raise FileNotFoundError("data/processed/player_stats.parquet")

    monkeypatch.setattr("api.main.known_player_names", raise_file_not_found)
    response = client.get("/players")
    assert response.status_code == 503


def test_health_check():
    assert client.get("/health").json() == {"status": "ok"}


def test_maybe_sync_from_gcs_calls_sync_when_bucket_is_set(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "my-bucket")
    calls = []
    monkeypatch.setattr("api.main.sync_from_gcs", lambda bucket: calls.append(bucket))

    _maybe_sync_from_gcs()

    assert calls == ["my-bucket"]


def test_maybe_sync_from_gcs_skips_when_bucket_is_unset(monkeypatch):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    calls = []
    monkeypatch.setattr("api.main.sync_from_gcs", lambda bucket: calls.append(bucket))

    _maybe_sync_from_gcs()

    assert calls == []


def test_post_chat_rate_limits_after_30_requests_per_hour(stub_chat):
    """POST /chat spends an LLM call every time - cap abuse per caller."""
    payload = {"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    for _ in range(30):
        response = client.post("/chat", json=payload)
        assert response.status_code == 200

    response = client.post("/chat", json=payload)
    assert response.status_code == 429


def test_post_chat_rate_limit_is_keyed_per_forwarded_client_ip(stub_chat):
    """Behind Cloud Run's proxy, every request's raw socket peer is identical -
    the limiter must key off X-Forwarded-For or every real caller would share
    one collective bucket instead of getting 30/hour each.
    """
    payload = {"message": "Should I start Jordan Love?", "mentioned_players": ["Jordan Love"]}
    for _ in range(30):
        response = client.post("/chat", json=payload, headers={"X-Forwarded-For": "1.1.1.1"})
        assert response.status_code == 200
    exhausted = client.post("/chat", json=payload, headers={"X-Forwarded-For": "1.1.1.1"})
    assert exhausted.status_code == 429

    other_caller = client.post("/chat", json=payload, headers={"X-Forwarded-For": "2.2.2.2"})
    assert other_caller.status_code == 200


def test_get_players_is_not_rate_limited(monkeypatch):
    """/players costs no LLM call, so it isn't subject to the /chat limit."""
    monkeypatch.setattr("api.main.known_player_names", lambda: ["Jordan Love"])
    for _ in range(35):
        assert client.get("/players").status_code == 200


def test_docs_config_disables_docs_when_gcs_bucket_is_set(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET", "my-bucket")
    assert _docs_config() == {"docs_url": None, "redoc_url": None, "openapi_url": None}


def test_docs_config_keeps_docs_when_gcs_bucket_is_unset(monkeypatch):
    monkeypatch.delenv("GCS_BUCKET", raising=False)
    assert _docs_config() == {}


def test_docs_are_reachable_in_the_local_dev_environment():
    """GCS_BUCKET is unset for the whole test run, so the live `app` singleton
    was built with docs enabled - confirms the wiring, not just the helper.
    """
    assert client.get("/docs").status_code == 200
