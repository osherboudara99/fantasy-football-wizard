import pytest
from fastapi.testclient import TestClient

from api.main import app
from llm.interface import Recommendation
from pipeline.context_builder import PlayerNotFoundError
from pipeline.decision_engine import Decision, DecisionError

client = TestClient(app)


def _fixture_decision():
    return Decision(
        question="Who should I start, Jordan Love or Jared Goff?",
        players=["Jordan Love", "Jared Goff"],
        week=5,
        context="PLAYER COMPARISON\n\nJordan Love:\n- Avg fantasy points (last 3 weeks): 18.4",
        recommendation=Recommendation(
            start="Jordan Love",
            bench="Jared Goff",
            confidence=0.72,
            key_factors=["Higher recent average"],
            risk_factors=["Questionable injury tag"],
        ),
    )


@pytest.fixture
def stub_decide(monkeypatch):
    """Replace the pipeline call so the endpoint is tested without an API call."""
    calls = {}

    def fake_decide(question, players=None, week=None):
        calls.update(question=question, players=players, week=week)
        return _fixture_decision()

    monkeypatch.setattr("api.main.decide", fake_decide)
    return calls


def test_post_recommendation_returns_the_decision_as_json(stub_decide):
    response = client.post(
        "/recommendation",
        json={"players": ["Jordan Love", "Jared Goff"], "week": 5},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["start"] == "Jordan Love"
    assert body["bench"] == "Jared Goff"
    assert body["confidence"] == 0.72
    assert body["week"] == 5
    assert body["players"] == ["Jordan Love", "Jared Goff"]
    assert body["context"].startswith("PLAYER COMPARISON")


def test_post_recommendation_builds_a_question_when_none_is_given(stub_decide):
    client.post("/recommendation", json={"players": ["Jordan Love", "Jared Goff"]})

    assert stub_decide["question"] == "Who should I start, Jordan Love or Jared Goff?"
    assert stub_decide["week"] is None


def test_post_recommendation_passes_a_free_text_question_through(stub_decide):
    client.post(
        "/recommendation",
        json={
            "players": ["Jordan Love", "Jared Goff"],
            "question": "Who has the better matchup?",
        },
    )

    assert stub_decide["question"] == "Who has the better matchup?"


@pytest.mark.parametrize(
    "players", [[], ["Jordan Love"], ["Jordan Love", "Jared Goff", "Bo Nix"]]
)
def test_post_recommendation_rejects_anything_but_two_players(players):
    """Caught by the request schema, so the pipeline is never entered."""
    response = client.post("/recommendation", json={"players": players})
    assert response.status_code == 422


def test_unknown_player_is_a_404_not_a_500(monkeypatch):
    def raise_not_found(*_, **__):
        raise PlayerNotFoundError('No stats found for "Nobody Here" in week 5')

    monkeypatch.setattr("api.main.decide", raise_not_found)
    response = client.post("/recommendation", json={"players": ["Nobody Here", "Jared Goff"]})

    assert response.status_code == 404
    assert "Nobody Here" in response.json()["detail"]


def test_decision_error_is_a_400(monkeypatch):
    def raise_decision_error(*_, **__):
        raise DecisionError("LLM answered about someone else")

    monkeypatch.setattr("api.main.decide", raise_decision_error)
    response = client.post("/recommendation", json={"players": ["Jordan Love", "Jared Goff"]})

    assert response.status_code == 400


def test_get_players_returns_the_known_name_universe(monkeypatch):
    monkeypatch.setattr(
        "api.main.known_player_names", lambda: ["Jordan Love", "Jared Goff"]
    )
    response = client.get("/players")

    assert response.status_code == 200
    assert response.json() == ["Jordan Love", "Jared Goff"]


def test_health_check():
    assert client.get("/health").json() == {"status": "ok"}
