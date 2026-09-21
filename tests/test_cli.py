import json

import httpx
import pytest

from betsim.cli import main
from betsim.context import assert_no_odds_leak, context_hash
from betsim.db import connect
from betsim.forecaster import Stage1Forecaster
from betsim.nhl import NhlClient
from betsim.oddsapi import OddsApiClient

HEADERS = {"x-requests-remaining": "479", "x-requests-used": "21", "x-requests-last": "1"}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Keep the CLI off any real .env the developer may have created locally."""
    monkeypatch.setattr("betsim.cli.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("ODDS_API_KEY", "test-key")


def stub_client(monkeypatch, payload, headers=HEADERS):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, headers=headers)

    def factory(api_key, **kwargs):
        return OddsApiClient(api_key, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("betsim.cli.OddsApiClient", factory)


def test_init_creates_the_database(tmp_path, capsys):
    db = tmp_path / "b.db"
    assert main(["init", "--db", str(db)]) == 0
    assert db.exists()
    assert "initialised" in capsys.readouterr().out


def test_unimplemented_commands_say_which_milestone(tmp_path, capsys):
    assert main(["slate", "--db", str(tmp_path / "b.db")]) == 2
    assert "M3/M4" in capsys.readouterr().err


def test_missing_api_key_gives_a_useful_message(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="ODDS_API_KEY is not set"):
        main(["quota"])


def test_quota_reports_headroom(monkeypatch, capsys):
    stub_client(monkeypatch, [])
    assert main(["quota"]) == 0
    out = capsys.readouterr().out
    assert "credits remaining: 479" in out
    assert "days of headroom" in out


def test_events_stores_the_schedule_for_free(tmp_path, monkeypatch, capsys, events_payload):
    stub_client(monkeypatch, events_payload)
    db = tmp_path / "b.db"
    assert main(["events", "--db", str(db)]) == 0
    assert "3 games stored (0 credits)" in capsys.readouterr().out
    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) c FROM games").fetchone()["c"] == 3
    conn.close()


def test_odds_ingests_prices_and_records_what_it_spent(tmp_path, monkeypatch, capsys, odds_payload):
    stub_client(monkeypatch, odds_payload)
    db = tmp_path / "b.db"
    assert main(["odds", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "3 games, 12 prices from 2 books" in out
    assert "credits remaining: 479" in out

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) c FROM odds_snapshots").fetchone()["c"] == 12
    run = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
    assert run["kind"] == "slate"
    assert run["credits_remaining"] == 479  # the audit trail for quota spend
    assert "cost 1" in run["notes"]
    conn.close()


def test_closing_flag_marks_the_snapshot(tmp_path, monkeypatch, odds_payload):
    stub_client(monkeypatch, odds_payload)
    db = tmp_path / "b.db"
    assert main(["odds", "--db", str(db), "--closing"]) == 0
    conn = connect(db)
    assert (
        conn.execute("SELECT COUNT(*) c FROM odds_snapshots WHERE is_closing=1").fetchone()["c"]
        == 12
    )
    assert conn.execute("SELECT kind FROM runs").fetchone()["kind"] == "close"
    conn.close()


def test_odds_warns_when_the_designated_bookmaker_is_missing(
    tmp_path, monkeypatch, capsys, odds_payload
):
    # Losing Pinnacle would silently break CLV, the primary metric.
    for event in odds_payload:
        event["bookmakers"] = [b for b in event["bookmakers"] if b["key"] != "pinnacle"]
    stub_client(monkeypatch, odds_payload)
    assert main(["odds", "--db", str(tmp_path / "b.db")]) == 0
    assert "WARNING: no prices from pinnacle" in capsys.readouterr().out


def test_scores_settles_only_completed_games(
    tmp_path, monkeypatch, capsys, events_payload, scores_payload
):
    db = tmp_path / "b.db"
    stub_client(monkeypatch, events_payload)
    main(["events", "--db", str(db)])
    stub_client(monkeypatch, scores_payload)
    assert main(["scores", "--db", str(db)]) == 0
    assert "2 of 3 games completed" in capsys.readouterr().out

    conn = connect(db)
    rows = {r["id"]: r for r in conn.execute("SELECT * FROM games").fetchall()}
    assert rows["nhl_car_fla"]["home_score"] == 4
    assert rows["nhl_bos_nyr"]["status"] == "scheduled"
    conn.close()


def test_api_errors_exit_nonzero_rather_than_traceback(tmp_path, monkeypatch, capsys):
    def handler(request):
        return httpx.Response(401, text="unauthorized")

    def factory(api_key, **kwargs):
        return OddsApiClient(api_key, transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("betsim.cli.OddsApiClient", factory)
    assert main(["odds", "--db", str(tmp_path / "b.db")]) == 1
    assert "Odds API error" in capsys.readouterr().err


# --- M2: Elo seeding and blind context building -----------------------------


def stub_nhl(monkeypatch, standings_payload, schedule_payload):
    """Serve the recorded NHL fixtures to whatever the CLI asks for."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "standings" in request.url.path:
            return httpx.Response(200, json=standings_payload)
        return httpx.Response(200, json=schedule_payload)

    def factory(**kwargs):
        kwargs.pop("transport", None)
        return NhlClient(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr("betsim.cli.NhlClient", factory)
    monkeypatch.setattr("betsim.cli.time.sleep", lambda _: None)


def test_seed_elo_stores_ratings(
    tmp_path, monkeypatch, capsys, nhl_standings_payload, nhl_schedule_payload
):
    stub_nhl(monkeypatch, nhl_standings_payload, nhl_schedule_payload)
    db = tmp_path / "b.db"
    assert main(["seed-elo", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "seeded Elo from" in out
    assert "0 credits" in out  # the NHL API has no quota

    conn = connect(db)
    rows = conn.execute("SELECT tricode, rating FROM elo_ratings").fetchall()
    assert len(rows) == 32
    assert conn.execute("SELECT season FROM elo_ratings LIMIT 1").fetchone()["season"] == 20252026
    conn.close()


def test_context_builds_and_stores_blind_contexts(
    tmp_path, monkeypatch, capsys, events_payload, nhl_standings_payload, nhl_schedule_payload
):
    db = tmp_path / "b.db"
    stub_client(monkeypatch, events_payload)
    main(["events", "--db", str(db)])

    stub_nhl(monkeypatch, nhl_standings_payload, nhl_schedule_payload)
    assert main(["context", "--db", str(db)]) == 0
    assert "3 contexts stored" in capsys.readouterr().out

    conn = connect(db)
    rows = conn.execute("SELECT game_id, payload_json, payload_hash FROM contexts").fetchall()
    assert len(rows) == 3
    for row in rows:
        payload = json.loads(row["payload_json"])
        # Stored verbatim, and provably free of anything odds-derived.
        assert_no_odds_leak(payload)
        assert context_hash(payload) == row["payload_hash"]
    conn.close()


def test_context_is_idempotent(
    tmp_path, monkeypatch, capsys, events_payload, nhl_standings_payload, nhl_schedule_payload
):
    db = tmp_path / "b.db"
    stub_client(monkeypatch, events_payload)
    main(["events", "--db", str(db)])
    stub_nhl(monkeypatch, nhl_standings_payload, nhl_schedule_payload)
    main(["context", "--db", str(db)])
    capsys.readouterr()
    assert main(["context", "--db", str(db)]) == 0
    assert "0 contexts stored, 3 unchanged" in capsys.readouterr().out


def test_context_without_games_tells_you_what_to_run(tmp_path, monkeypatch, capsys):
    assert main(["context", "--db", str(tmp_path / "b.db")]) == 0
    assert "run `betsim events` first" in capsys.readouterr().out


def test_context_refuses_an_unmappable_team(
    tmp_path, monkeypatch, capsys, events_payload, nhl_standings_payload, nhl_schedule_payload
):
    # A team that cannot be resolved must stop the run, not silently skip a game.
    events_payload[0]["home_team"] = "Quebec Nordiques"
    stub_client(monkeypatch, events_payload)
    db = tmp_path / "b.db"
    main(["events", "--db", str(db)])
    stub_nhl(monkeypatch, nhl_standings_payload, nhl_schedule_payload)
    assert main(["context", "--db", str(db)]) == 1
    assert "cannot resolve" in capsys.readouterr().err


# --- M3: blind Stage 1 forecasting ------------------------------------------


def _stub_forecaster(monkeypatch, *responses):
    """Patch the CLI's forecaster to use a stub client instead of the real API."""
    from tests.test_forecaster import StubClient

    holder = {}

    def factory(**kwargs):
        client = StubClient(*responses)
        holder["client"] = client
        return Stage1Forecaster(client=client, **kwargs)

    monkeypatch.setattr("betsim.cli.Stage1Forecaster", factory)
    return holder


def _prepare(tmp_path, monkeypatch, events_payload, standings_payload, schedule_payload):
    db = tmp_path / "b.db"
    stub_client(monkeypatch, events_payload)
    main(["events", "--db", str(db)])
    stub_nhl(monkeypatch, standings_payload, schedule_payload)
    main(["context", "--db", str(db)])
    return db


def test_forecast_dry_run_spends_nothing(
    tmp_path, monkeypatch, capsys, events_payload, nhl_standings_payload, nhl_schedule_payload
):
    db = _prepare(
        tmp_path, monkeypatch, events_payload, nhl_standings_payload, nhl_schedule_payload
    )
    holder = _stub_forecaster(monkeypatch)
    capsys.readouterr()
    assert main(["forecast", "--db", str(db), "--k", "5", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "15 calls (3 games x k=5)" in out
    assert "nothing spent" in out
    assert "client" not in holder or not holder["client"].calls


def test_forecast_stores_calls_and_forecasts(
    tmp_path, monkeypatch, capsys, events_payload, nhl_standings_payload, nhl_schedule_payload
):
    from tests.test_forecaster import forecast as make_forecast
    from tests.test_forecaster import response as make_response

    db = _prepare(
        tmp_path, monkeypatch, events_payload, nhl_standings_payload, nhl_schedule_payload
    )
    _stub_forecaster(monkeypatch, make_response(make_forecast(0.58, 0.42)))
    capsys.readouterr()
    assert main(["forecast", "--db", str(db), "--k", "3"]) == 0
    out = capsys.readouterr().out
    assert "9/9 draws succeeded" in out

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) c FROM llm_calls").fetchone()["c"] == 9
    assert conn.execute("SELECT COUNT(*) c FROM forecasts").fetchone()["c"] == 9
    call = conn.execute("SELECT * FROM llm_calls LIMIT 1").fetchone()
    assert call["stage"] == 1
    assert call["ok"] == 1
    assert call["input_tokens"] == 1200
    params = json.loads(call["params_json"])
    assert "temperature" not in params  # no such parameter exists on these models
    # All draws for one game share an input hash: independent draws, identical input.
    hashes = conn.execute("SELECT COUNT(DISTINCT input_hash) c FROM llm_calls").fetchone()["c"]
    assert hashes == 3  # one per game, not one per call
    conn.close()


def test_forecast_skips_games_that_already_have_draws(
    tmp_path, monkeypatch, capsys, events_payload, nhl_standings_payload, nhl_schedule_payload
):
    from tests.test_forecaster import forecast as make_forecast
    from tests.test_forecaster import response as make_response

    db = _prepare(
        tmp_path, monkeypatch, events_payload, nhl_standings_payload, nhl_schedule_payload
    )
    _stub_forecaster(monkeypatch, make_response(make_forecast()))
    main(["forecast", "--db", str(db), "--k", "2"])
    capsys.readouterr()
    main(["forecast", "--db", str(db), "--k", "2"])
    assert "0/0 draws" in capsys.readouterr().out

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) c FROM forecasts").fetchone()["c"] == 6
    conn.close()


def test_a_refusal_is_recorded_without_stopping_the_run(
    tmp_path, monkeypatch, capsys, events_payload, nhl_standings_payload, nhl_schedule_payload
):
    from types import SimpleNamespace

    from tests.test_forecaster import response as make_response

    db = _prepare(
        tmp_path, monkeypatch, events_payload, nhl_standings_payload, nhl_schedule_payload
    )
    details = SimpleNamespace(category="gambling", explanation="declined")
    _stub_forecaster(monkeypatch, make_response(None, stop_reason="refusal", details=details))
    capsys.readouterr()
    assert main(["forecast", "--db", str(db), "--k", "2"]) == 0
    assert "6 refused" in capsys.readouterr().out

    conn = connect(db)
    # Refusals are logged as calls but produce no forecasts -- they are data.
    assert conn.execute("SELECT COUNT(*) c FROM llm_calls").fetchone()["c"] == 6
    assert conn.execute("SELECT COUNT(*) c FROM forecasts").fetchone()["c"] == 0
    row = conn.execute("SELECT stop_reason, error FROM llm_calls LIMIT 1").fetchone()
    assert row["stop_reason"] == "refusal"
    assert "gambling" in row["error"]
    conn.close()


def test_forecast_reports_games_without_a_context(tmp_path, monkeypatch, capsys, events_payload):
    db = tmp_path / "b.db"
    stub_client(monkeypatch, events_payload)
    main(["events", "--db", str(db)])
    _stub_forecaster(monkeypatch)
    capsys.readouterr()
    assert main(["forecast", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "3 games have no context yet" in out
    assert "nothing to forecast" in out
