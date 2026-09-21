from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from betsim.db import connect, init_db
from betsim.ingest import parse_events, parse_odds, parse_scores
from betsim.settlement import GameStatus
from betsim.store import (
    apply_scores,
    closing_prices,
    finish_run,
    insert_snapshots,
    iso,
    latest_prices,
    mark_void,
    open_game_ids,
    start_run,
    upsert_games,
)

CAPTURED = datetime(2026, 9, 29, 18, 5, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_db(c)
    yield c
    c.close()


def test_iso_refuses_naive_datetimes():
    with pytest.raises(ValueError, match="naive datetime"):
        iso(datetime(2026, 9, 29, 18, 5))  # noqa: DTZ001 -- naive input is the point


def test_upsert_then_read_back(conn, events_payload):
    games = parse_events(events_payload)
    assert upsert_games(conn, games) == 3
    conn.commit()
    rows = conn.execute("SELECT id, home, away, status FROM games ORDER BY id").fetchall()
    assert len(rows) == 3
    assert {r["status"] for r in rows} == {"scheduled"}


def test_upsert_is_idempotent(conn, events_payload):
    games = parse_events(events_payload)
    upsert_games(conn, games)
    upsert_games(conn, games)
    conn.commit()
    assert conn.execute("SELECT COUNT(*) c FROM games").fetchone()["c"] == 3


def test_upsert_updates_a_rescheduled_game(conn, events_payload):
    games = parse_events(events_payload)
    upsert_games(conn, games)
    moved = replace(games[0], commence_utc=games[0].commence_utc + timedelta(days=1))
    upsert_games(conn, [moved])
    conn.commit()
    row = conn.execute("SELECT commence_utc FROM games WHERE id = ?", (games[0].id,)).fetchone()
    assert row["commence_utc"] == iso(moved.commence_utc)


def test_store_functions_do_not_commit(conn, events_payload):
    # The caller owns the transaction, so a half-written slate can be rolled back.
    upsert_games(conn, parse_events(events_payload))
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) c FROM games").fetchone()["c"] == 0


def test_snapshots_are_appended_not_replaced(conn, odds_payload):
    games, prices = parse_odds(odds_payload)
    upsert_games(conn, games)
    insert_snapshots(conn, prices, captured_utc=CAPTURED)
    insert_snapshots(
        conn,
        prices,
        captured_utc=CAPTURED + timedelta(hours=2),
        closing_game_ids={p.game_id for p in prices},
    )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) c FROM odds_snapshots").fetchone()["c"] == 24
    closing = conn.execute("SELECT COUNT(*) c FROM odds_snapshots WHERE is_closing=1").fetchone()
    assert closing["c"] == 12


def test_latest_and_closing_prices(conn, odds_payload):
    games, prices = parse_odds(odds_payload)
    upsert_games(conn, games)
    insert_snapshots(conn, prices, captured_utc=CAPTURED)
    conn.commit()

    opening = latest_prices(conn, "nhl_car_fla", bookmaker="pinnacle")
    assert opening == pytest.approx({"home": 2.10, "away": 1.80})
    assert closing_prices(conn, "nhl_car_fla", bookmaker="pinnacle") == {}

    # A closing snapshot at a shorter price is what produces positive CLV.
    shortened = [
        replace(p, price_decimal=1.95)
        for p in prices
        if p.game_id == "nhl_car_fla" and p.bookmaker == "pinnacle" and p.selection == "home"
    ]
    insert_snapshots(
        conn,
        shortened,
        captured_utc=CAPTURED + timedelta(hours=3),
        closing_game_ids={"nhl_car_fla"},
    )
    conn.commit()
    assert closing_prices(conn, "nhl_car_fla", bookmaker="pinnacle")["home"] == pytest.approx(1.95)


def test_apply_scores_writes_only_completed_games(conn, events_payload, scores_payload):
    upsert_games(conn, parse_events(events_payload))
    assert apply_scores(conn, parse_scores(scores_payload)) == 2
    conn.commit()

    rows = {r["id"]: r for r in conn.execute("SELECT * FROM games").fetchall()}
    assert rows["nhl_car_fla"]["status"] == GameStatus.COMPLETED.value
    assert (rows["nhl_car_fla"]["home_score"], rows["nhl_car_fla"]["away_score"]) == (4, 1)
    # The unstarted game must stay scheduled with no scores -- settlement must
    # never see a half-finished game.
    assert rows["nhl_bos_nyr"]["status"] == "scheduled"
    assert rows["nhl_bos_nyr"]["home_score"] is None


def test_mark_void(conn, events_payload):
    upsert_games(conn, parse_events(events_payload))
    mark_void(conn, "nhl_bos_nyr", GameStatus.POSTPONED)
    conn.commit()
    row = conn.execute("SELECT status FROM games WHERE id='nhl_bos_nyr'").fetchone()
    assert row["status"] == "postponed"


def test_mark_void_rejects_a_non_voiding_status(conn):
    with pytest.raises(ValueError, match="not a voiding status"):
        mark_void(conn, "g1", GameStatus.COMPLETED)


def test_runs_record_the_credit_balance(conn):
    run_id = start_run(conn, "slate")
    finish_run(conn, run_id, credits_remaining=479, notes="nhl opening night")
    conn.commit()
    row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    assert row["kind"] == "slate"
    assert row["credits_remaining"] == 479
    assert row["finished_utc"] is not None


def test_open_game_ids(conn, events_payload):
    upsert_games(conn, parse_events(events_payload))
    conn.execute(
        "INSERT INTO bets (arm, game_id, placed_utc, selection, price_decimal, tier,"
        " stake_minor, status) VALUES ('llm_s1','nhl_car_fla','2026-09-29T20:00:00Z','home',"
        "2.10,'medium',2000,'open')"
    )
    conn.execute(
        "INSERT INTO bets (arm, game_id, placed_utc, selection, price_decimal, tier,"
        " stake_minor, status) VALUES ('llm_s1','nhl_tor_mtl','2026-09-29T20:00:00Z','home',"
        "1.65,'safe',3000,'won')"
    )
    conn.commit()
    assert open_game_ids(conn, "llm_s1") == {"nhl_car_fla"}
    assert open_game_ids(conn, "kelly_s1") == set()
