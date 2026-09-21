import httpx
import pytest

from betsim.cli import main
from betsim.db import connect
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
