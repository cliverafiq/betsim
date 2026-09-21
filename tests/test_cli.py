import json
from pathlib import Path

import httpx
import pytest

from betsim.cli import main
from betsim.config import DESIGNATED_BOOKMAKER
from betsim.context import assert_no_odds_leak, context_hash
from betsim.db import connect
from betsim.decider import Stage2Decider
from betsim.forecaster import Stage1Forecaster
from betsim.ledger import balance, verify_ledger
from betsim.models import Stage2Bet, Stage2Decision
from betsim.money import STARTING_BALANCE_MINOR
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


def test_every_advertised_command_is_wired(tmp_path):
    from betsim.cli import build_parser

    parser = build_parser()
    actions = [a for a in parser._actions if a.choices and "init" in (a.choices or {})]
    assert actions, "expected a subparser action"
    commands = sorted(actions[0].choices)
    assert commands == sorted(
        [
            "calibrate",
            "clv",
            "close",
            "context",
            "doctor",
            "events",
            "forecast",
            "init",
            "odds",
            "quota",
            "report",
            "scores",
            "seed-elo",
            "settle",
            "slate",
        ]
    )
    # Nothing is left as a not-implemented stub.
    for name, sub in actions[0].choices.items():
        assert sub.get_default("func") is not None, f"{name} has no handler"


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


def test_closing_marks_nothing_for_games_that_are_not_about_to_start(
    tmp_path, monkeypatch, odds_payload
):
    # A closing line only means anything near the off. Marking a whole slate
    # closing hours early would record the wrong price as the close and make
    # CLV -- the primary metric -- meaningless.
    stub_client(monkeypatch, odds_payload)
    db = tmp_path / "b.db"
    assert main(["odds", "--db", str(db), "--closing"]) == 0
    conn = connect(db)
    assert (
        conn.execute("SELECT COUNT(*) c FROM odds_snapshots WHERE is_closing=1").fetchone()["c"]
        == 0
    )
    assert conn.execute("SELECT COUNT(*) c FROM odds_snapshots").fetchone()["c"] == 12
    assert conn.execute("SELECT kind FROM runs").fetchone()["kind"] == "close"
    conn.close()


def test_closing_marks_only_the_games_inside_the_window(tmp_path, monkeypatch, odds_payload):
    from datetime import UTC, datetime, timedelta

    soon = datetime.now(UTC) + timedelta(minutes=10)  # inside the 15-minute window
    later = datetime.now(UTC) + timedelta(hours=6)  # outside it
    odds_payload[0]["commence_time"] = soon.isoformat().replace("+00:00", "Z")
    for event in odds_payload[1:]:
        event["commence_time"] = later.isoformat().replace("+00:00", "Z")

    stub_client(monkeypatch, odds_payload)
    db = tmp_path / "b.db"
    assert main(["odds", "--db", str(db), "--closing"]) == 0

    conn = connect(db)
    rows = conn.execute("SELECT DISTINCT game_id FROM odds_snapshots WHERE is_closing=1").fetchall()
    assert [r["game_id"] for r in rows] == [odds_payload[0]["id"]]
    conn.close()


def test_odds_warns_when_the_designated_bookmaker_is_missing(
    tmp_path, monkeypatch, capsys, odds_payload
):
    # Losing the designated book would silently break CLV, the primary metric.
    for event in odds_payload:
        event["bookmakers"] = [b for b in event["bookmakers"] if b["key"] != DESIGNATED_BOOKMAKER]
    stub_client(monkeypatch, odds_payload)
    assert main(["odds", "--db", str(tmp_path / "b.db")]) == 0
    assert f"WARNING: no prices from {DESIGNATED_BOOKMAKER}" in capsys.readouterr().out


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


# --- M4: the full daily loop ------------------------------------------------


from tests.conftest import shift_slate as _shift


def _full_pipeline(
    tmp_path,
    monkeypatch,
    events_payload,
    odds_payload,
    standings_payload,
    schedule_payload,
    *,
    k=2,
    starts_in_minutes=180,
):
    """events -> odds -> context -> seed-elo -> forecast, ready for a slate.

    The slate is shifted to start soon, because a real slate is imminent games
    and the 36-hour horizon deliberately excludes anything further out.
    """
    from tests.test_forecaster import forecast as make_forecast
    from tests.test_forecaster import response as make_response

    _shift(events_payload, starts_in_minutes)
    _shift(odds_payload, starts_in_minutes)
    db = tmp_path / "b.db"
    stub_client(monkeypatch, events_payload)
    main(["events", "--db", str(db)])
    stub_client(monkeypatch, odds_payload)
    main(["odds", "--db", str(db)])
    stub_nhl(monkeypatch, standings_payload, schedule_payload)
    main(["context", "--db", str(db)])
    main(["seed-elo", "--db", str(db)])
    _stub_forecaster(monkeypatch, make_response(make_forecast(0.62, 0.38)))
    main(["forecast", "--db", str(db), "--k", str(k)])
    return db


def _stub_decider(monkeypatch, *responses):
    from tests.test_decider import StubClient as DeciderStub

    def factory(**kwargs):
        return Stage2Decider(client=DeciderStub(*responses), **kwargs)

    monkeypatch.setattr("betsim.cli.Stage2Decider", factory)


def test_dry_run_slate_places_the_shadow_arms_without_spending(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
    )
    capsys.readouterr()
    assert main(["slate", "--db", str(db), "--k", "2", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "shadow arms only, no LLM calls" in out
    assert "ledger verified" in out

    conn = connect(db)
    arms = {r["arm"] for r in conn.execute("SELECT DISTINCT arm FROM bets").fetchall()}
    # Every non-LLM arm runs off the same snapshot.
    assert {"kelly_s1", "kelly_s2", "elo", "fav", "random"} <= arms
    assert not any(a.startswith("llm_") for a in arms)
    conn.close()


def test_full_slate_runs_every_arm_and_the_ledger_balances(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    from tests.test_decider import response as decider_response

    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
    )
    decision = Stage2Decision(
        bets=[
            Stage2Bet(
                game_id="nhl_car_fla",
                selection="home",
                stake_units=15.0,
                p_revised=0.60,
                reason="rest edge",
            )
        ],
        day_notes="one bet",
    )
    _stub_decider(monkeypatch, decider_response(decision))
    capsys.readouterr()
    assert main(["slate", "--db", str(db), "--k", "2"]) == 0
    out = capsys.readouterr().out
    assert "ledger verified" in out

    conn = connect(db)
    arms = {r["arm"] for r in conn.execute("SELECT DISTINCT arm FROM bets").fetchall()}
    assert {"llm_s1", "llm_s2", "kelly_s1", "kelly_s2", "elo", "fav", "random"} <= arms
    # p_revised was captured, so the anchoring arm ran too.
    assert {"kelly_revised_s1", "kelly_revised_s2"} <= arms

    # Money conservation, per arm: balance == opening - staked + payouts.
    for arm in sorted(arms):
        verify_ledger(conn, arm)
        staked = conn.execute(
            "SELECT COALESCE(SUM(stake_minor),0) s FROM bets WHERE arm=?", (arm,)
        ).fetchone()["s"]
        assert balance(conn, arm) == STARTING_BALANCE_MINOR - staked

    # Stage 2 calls are logged alongside Stage 1.
    assert conn.execute("SELECT COUNT(*) c FROM llm_calls WHERE stage=2").fetchone()["c"] == 2
    conn.close()


def test_llm_and_kelly_arms_see_the_same_forecast(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    # The whole decomposition depends on this pairing: llm_s1 versus kelly_s1 is
    # a read on sizing alone, holding the forecast constant.
    from tests.test_decider import response as decider_response

    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
        k=1,
    )
    decision = Stage2Decision(
        bets=[
            Stage2Bet(
                game_id="nhl_car_fla",
                selection="home",
                stake_units=15.0,
                p_revised=0.60,
                reason="x",
            )
        ]
    )
    _stub_decider(monkeypatch, decider_response(decision))
    main(["slate", "--db", str(db), "--k", "1"])

    conn = connect(db)
    llm = conn.execute(
        "SELECT p_blind FROM bets WHERE arm='llm_s1' AND game_id='nhl_car_fla'"
    ).fetchone()
    kelly = conn.execute(
        "SELECT p_blind FROM bets WHERE arm='kelly_s1' AND game_id='nhl_car_fla'"
    ).fetchone()
    assert llm["p_blind"] == kelly["p_blind"] == pytest.approx(0.62)
    conn.close()


def test_an_over_cap_bet_is_rejected_and_recorded(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    from tests.test_decider import response as decider_response

    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
        k=1,
    )
    # 500 units is far above any tier cap; it must be rejected, never clipped.
    decision = Stage2Decision(
        bets=[
            Stage2Bet(
                game_id="nhl_car_fla",
                selection="home",
                stake_units=500.0,
                p_revised=0.60,
                reason="oversized",
            )
        ]
    )
    _stub_decider(monkeypatch, decider_response(decision))
    capsys.readouterr()
    main(["slate", "--db", str(db), "--k", "1"])
    assert "stake_exceeds_tier_cap" in capsys.readouterr().out

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) c FROM bets WHERE arm='llm_s1'").fetchone()["c"] == 0
    row = conn.execute("SELECT * FROM rejections WHERE arm='llm_s1'").fetchone()
    assert row["reason"] == "stake_exceeds_tier_cap"
    assert json.loads(row["payload_json"])["stake_units"] == 500.0
    conn.close()


def test_a_stage2_refusal_leaves_the_shadow_arms_running(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    from types import SimpleNamespace

    from tests.test_decider import response as decider_response

    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
        k=1,
    )
    details = SimpleNamespace(category="gambling", explanation="declined")
    _stub_decider(monkeypatch, decider_response(None, stop_reason="refusal", details=details))
    capsys.readouterr()
    assert main(["slate", "--db", str(db), "--k", "1"]) == 0
    assert "refusals" in capsys.readouterr().out

    conn = connect(db)
    assert conn.execute("SELECT COUNT(*) c FROM bets WHERE arm='llm_s1'").fetchone()["c"] == 0
    # The baselines are unaffected -- one refusal must not lose the night's data.
    assert conn.execute("SELECT COUNT(*) c FROM bets WHERE arm='kelly_s1'").fetchone()["c"] > 0
    assert conn.execute("SELECT COUNT(*) c FROM bets WHERE arm='fav'").fetchone()["c"] == 3
    conn.close()


def test_slate_without_prices_says_what_to_run(tmp_path, monkeypatch, capsys, events_payload):
    db = tmp_path / "b.db"
    stub_client(monkeypatch, events_payload)
    main(["events", "--db", str(db)])
    capsys.readouterr()
    assert main(["slate", "--db", str(db)]) == 0
    assert "run `betsim odds` first" in capsys.readouterr().out


# --- M5: closing snapshots, settlement and CLV ------------------------------


def test_a_full_day_runs_end_to_end(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    scores_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    """slate -> close -> settle -> clv, with the ledger verified throughout."""
    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
        k=1,
        starts_in_minutes=12,  # inside the 15-minute closing window
    )

    # 1. Place bets across every shadow arm.
    assert main(["slate", "--db", str(db), "--k", "1", "--dry-run"]) == 0

    # 2. Closing snapshot -- these games are inside the window, so they mark.
    stub_client(monkeypatch, odds_payload)
    assert main(["close", "--db", str(db)]) == 0
    conn = connect(db)
    assert (
        conn.execute(
            "SELECT COUNT(DISTINCT game_id) c FROM odds_snapshots WHERE is_closing=1"
        ).fetchone()["c"]
        == 3
    )
    open_before = conn.execute("SELECT COUNT(*) c FROM bets WHERE status='open'").fetchone()["c"]
    assert open_before > 0
    conn.close()

    # 3. Settle against final scores.
    stub_client(monkeypatch, scores_payload)
    capsys.readouterr()
    assert main(["settle", "--db", str(db)]) == 0
    out = capsys.readouterr().out
    assert "bets settled" in out
    assert "ledger verified" in out

    conn = connect(db)
    for arm in {r["arm"] for r in conn.execute("SELECT DISTINCT arm FROM bets")}:
        verify_ledger(conn, arm)
        # Money conservation across the whole cycle.
        row = conn.execute(
            "SELECT COALESCE(SUM(stake_minor),0) s, COALESCE(SUM(payout_minor),0) p "
            "FROM bets WHERE arm=?",
            (arm,),
        ).fetchone()
        assert balance(conn, arm) == STARTING_BALANCE_MINOR - row["s"] + row["p"]
    # The third game never finished, so its bets stay open.
    assert conn.execute("SELECT COUNT(*) c FROM bets WHERE status='open'").fetchone()["c"] > 0
    conn.close()

    # 4. CLV is derivable now that both prices exist.
    capsys.readouterr()
    assert main(["clv", "--db", str(db)]) == 0
    clv_out = capsys.readouterr().out
    assert "mean CLV" in clv_out
    assert "<- null" in clv_out  # the random arm is the baseline, not zero


def test_settle_reports_per_arm_profit(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    scores_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
        k=1,
        starts_in_minutes=12,  # inside the 15-minute closing window
    )
    main(["slate", "--db", str(db), "--k", "1", "--dry-run"])
    stub_client(monkeypatch, scores_payload)
    capsys.readouterr()
    main(["settle", "--db", str(db)])
    out = capsys.readouterr().out
    assert "fav" in out and "random" in out
    assert " u  balance" in out


def test_clv_says_what_to_run_when_there_are_no_closing_prices(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
        k=1,
    )
    main(["slate", "--db", str(db), "--k", "1", "--dry-run"])
    capsys.readouterr()
    assert main(["clv", "--db", str(db)]) == 0
    assert "run `betsim close` near puck drop" in capsys.readouterr().out


# --- doctor -----------------------------------------------------------------


def test_doctor_reports_a_missing_env_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("betsim.cli.Path", Path)
    assert main(["doctor", "--db", str(tmp_path / "b.db")]) == 0
    out = capsys.readouterr().out
    assert "credentials" in out
    assert "prompts" in out


def test_doctor_masks_keys_instead_of_printing_them(tmp_path, monkeypatch, capsys):
    # The whole point of the command is that it is safe to run and to paste.
    import betsim.cli as cli_module

    odds_key = "a" * 32
    anthropic_key = "sk-ant-" + "A" * 40
    (tmp_path / "src" / "betsim").mkdir(parents=True)
    (tmp_path / ".env").write_text(f"ODDS_API_KEY={odds_key}\nANTHROPIC_API_KEY={anthropic_key}\n")
    monkeypatch.setattr(cli_module, "__file__", str(tmp_path / "src" / "betsim" / "cli.py"))

    main(["doctor", "--db", str(tmp_path / "b.db")])
    out = capsys.readouterr().out

    # It really did read the file (otherwise the assertions below prove nothing).
    assert "ODDS_API_KEY" in out and "looks right" in out
    assert odds_key not in out
    assert anthropic_key not in out
    assert "aaaaaa...aaaa" in out  # masked form only


def test_slate_respects_the_horizon_for_every_arm(
    tmp_path,
    monkeypatch,
    capsys,
    events_payload,
    odds_payload,
    nhl_standings_payload,
    nhl_schedule_payload,
):
    """Regression: run_slate must use the caller's filtered slate.

    It previously re-queried without the horizon, so `fav` and `random` bet
    every upcoming game -- including ones ten days out, priced now. Those arms
    are the CLV floor and null, so betting them that far ahead of the close
    corrupts the benchmark the whole experiment is measured against.
    """
    from tests.conftest import shift_slate

    db = _full_pipeline(
        tmp_path,
        monkeypatch,
        events_payload,
        odds_payload,
        nhl_standings_payload,
        nhl_schedule_payload,
        k=1,
    )
    # Push one game far beyond the horizon and re-ingest.
    shift_slate(odds_payload[2:], 60 * 24 * 10)  # ten days out
    stub_client(monkeypatch, odds_payload)
    main(["odds", "--db", str(db)])

    capsys.readouterr()
    assert main(["slate", "--db", str(db), "--k", "1", "--dry-run"]) == 0
    assert "2 games" in capsys.readouterr().out

    conn = connect(db)
    for arm in ("fav", "random"):
        rows = conn.execute("SELECT game_id FROM bets WHERE arm = ?", (arm,)).fetchall()
        assert len(rows) == 2, f"{arm} bet outside the horizon"
        assert odds_payload[2]["id"] not in {r["game_id"] for r in rows}
    conn.close()
