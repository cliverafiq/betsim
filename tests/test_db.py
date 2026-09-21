import sqlite3

import pytest

from betsim.db import connect, init_db

GAME = ("g1", "icehockey_nhl", "BOS", "MTL", "2026-09-29T23:00:00Z", "scheduled", None, None, None)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "test.db")
    init_db(c)
    c.execute("INSERT INTO games VALUES (?,?,?,?,?,?,?,?,?)", GAME)
    c.commit()
    yield c
    c.close()


def _tables(c):
    rows = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return {r["name"] for r in rows}


def test_init_creates_every_table(conn):
    assert {
        "games",
        "odds_snapshots",
        "contexts",
        "llm_calls",
        "forecasts",
        "bets",
        "rejections",
        "ledger",
        "calibrations",
        "runs",
    } <= _tables(conn)


def test_init_is_idempotent(tmp_path):
    c = connect(tmp_path / "x.db")
    init_db(c)
    before = _tables(c)
    init_db(c)
    assert _tables(c) == before
    c.close()


def test_foreign_keys_are_enforced(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO bets (arm, game_id, placed_utc, selection, price_decimal, tier,"
            " stake_minor) VALUES ('llm_s1','ghost','2026-09-29T21:00:00Z','home',1.91,"
            "'medium',1000)"
        )


def _insert_bet(conn, arm="llm_s1", game_id="g1", stake=1000, price=1.91):
    conn.execute(
        "INSERT INTO bets (arm, game_id, placed_utc, selection, price_decimal, tier,"
        " stake_minor) VALUES (?,?,?,?,?,?,?)",
        (arm, game_id, "2026-09-29T21:00:00Z", "home", price, "medium", stake),
    )


def test_one_bet_per_game_per_arm_is_enforced(conn):
    _insert_bet(conn)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_bet(conn)


def test_different_arms_may_bet_the_same_game(conn):
    _insert_bet(conn, arm="llm_s1")
    _insert_bet(conn, arm="kelly_s1")
    conn.commit()
    assert conn.execute("SELECT COUNT(*) c FROM bets").fetchone()["c"] == 2


def test_stake_below_one_unit_is_rejected_by_the_schema(conn):
    with pytest.raises(sqlite3.IntegrityError):
        _insert_bet(conn, stake=99)


def test_price_must_exceed_one(conn):
    with pytest.raises(sqlite3.IntegrityError):
        _insert_bet(conn, price=1.0)


def test_llm_calls_reject_a_stage_other_than_one_or_two(conn):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO llm_calls (stage, seed_idx, model_id, prompt_version, params_json,"
            " input_hash, input_json, created_utc, ok) VALUES (3,0,'claude-opus-5','stage1_v1',"
            "'{}','h','{}','2026-09-29T20:00:00Z',1)"
        )


def test_ledger_accepts_append_only_corrections(conn):
    # Corrections are new rows, never updates. Two rows for the same bet is normal.
    _insert_bet(conn)
    bet_id = conn.execute("SELECT id FROM bets").fetchone()["id"]
    for delta, balance, reason in ((-1000, 99000, "stake"), (1910, 100910, "settle")):
        conn.execute(
            "INSERT INTO ledger (arm, ts_utc, delta_minor, balance_minor, reason, bet_id)"
            " VALUES (?,?,?,?,?,?)",
            ("llm_s1", "2026-09-29T21:00:00Z", delta, balance, reason, bet_id),
        )
    conn.commit()
    assert conn.execute("SELECT COUNT(*) c FROM ledger").fetchone()["c"] == 2
