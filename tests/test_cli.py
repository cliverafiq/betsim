import json

import httpx
import pytest

from betsim.cli import main
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


def test_unimplemented_commands_say_which_milestone(tmp_path, capsys):
    assert main(["report", "--db", str(tmp_path / "b.db")]) == 2
    assert "M6" in capsys.readouterr().err


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


# --- M4: the full daily loop ------------------------------------------------


def _full_pipeline(
    tmp_path, monkeypatch, events_payload, odds_payload, standings_payload, schedule_payload, *, k=2
):
    """events -> odds -> context -> seed-elo -> forecast, ready for a slate."""
    from tests.test_forecaster import forecast as make_forecast
    from tests.test_forecaster import response as make_response

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
