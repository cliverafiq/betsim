"""SQLite schema and connection helpers.

Conventions:

* All timestamps are ISO-8601 **UTC** strings.
* All money is integer minor units (1 unit = 100).
* The ``ledger`` table is append-only. Corrections are new rows; nothing is ever
  updated or deleted.
* ``llm_calls.params_json`` records ``model_id``, ``effort``, the ``thinking``
  config and ``max_tokens``. It must **not** record a temperature -- no such
  parameter exists on the models under test.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS games (
    id            TEXT PRIMARY KEY,
    sport_key     TEXT NOT NULL,
    home          TEXT NOT NULL,
    away          TEXT NOT NULL,
    commence_utc  TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'scheduled',
    home_score    INTEGER,
    away_score    INTEGER,
    completed_utc TEXT
);
CREATE INDEX IF NOT EXISTS idx_games_commence ON games(sport_key, commence_utc);

CREATE TABLE IF NOT EXISTS odds_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id       TEXT NOT NULL REFERENCES games(id),
    captured_utc  TEXT NOT NULL,
    bookmaker     TEXT NOT NULL,
    market        TEXT NOT NULL,
    outcome       TEXT NOT NULL,
    price_decimal REAL NOT NULL CHECK (price_decimal > 1.0),
    is_closing    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_snap_game ON odds_snapshots(game_id, captured_utc);
CREATE INDEX IF NOT EXISTS idx_snap_closing ON odds_snapshots(game_id, is_closing);

CREATE TABLE IF NOT EXISTS contexts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id      TEXT NOT NULL REFERENCES games(id),
    built_utc    TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    sources      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_contexts_game ON contexts(game_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stage          INTEGER NOT NULL CHECK (stage IN (1, 2)),
    seed_idx       INTEGER NOT NULL,
    model_id       TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    params_json    TEXT NOT NULL,
    input_hash     TEXT NOT NULL,
    input_json     TEXT NOT NULL,
    raw_response   TEXT,
    stop_reason    TEXT,
    created_utc    TEXT NOT NULL,
    ok             INTEGER NOT NULL,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_stage ON llm_calls(stage, created_utc);

CREATE TABLE IF NOT EXISTS forecasts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     TEXT NOT NULL REFERENCES games(id),
    llm_call_id INTEGER NOT NULL REFERENCES llm_calls(id),
    seed_idx    INTEGER NOT NULL,
    p_home      REAL NOT NULL,
    p_away      REAL NOT NULL,
    p_draw      REAL
);
CREATE INDEX IF NOT EXISTS idx_forecasts_game ON forecasts(game_id, seed_idx);

CREATE TABLE IF NOT EXISTS bets (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    arm           TEXT NOT NULL,
    game_id       TEXT NOT NULL REFERENCES games(id),
    forecast_id   INTEGER REFERENCES forecasts(id),
    snapshot_id   INTEGER REFERENCES odds_snapshots(id),
    placed_utc    TEXT NOT NULL,
    selection     TEXT NOT NULL,
    price_decimal REAL NOT NULL CHECK (price_decimal > 1.0),
    tier          TEXT NOT NULL,
    stake_minor   INTEGER NOT NULL CHECK (stake_minor >= 100),
    p_blind       REAL,
    p_revised     REAL,
    reason        TEXT,
    status        TEXT NOT NULL DEFAULT 'open',
    payout_minor  INTEGER,
    settled_utc   TEXT,
    UNIQUE (arm, game_id)
);
CREATE INDEX IF NOT EXISTS idx_bets_arm_status ON bets(arm, status);

CREATE TABLE IF NOT EXISTS rejections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    arm          TEXT NOT NULL,
    llm_call_id  INTEGER REFERENCES llm_calls(id),
    payload_json TEXT NOT NULL,
    reason       TEXT NOT NULL,
    detail       TEXT,
    created_utc  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_rejections_reason ON rejections(arm, reason);

CREATE TABLE IF NOT EXISTS ledger (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    arm           TEXT NOT NULL,
    ts_utc        TEXT NOT NULL,
    delta_minor   INTEGER NOT NULL,
    balance_minor INTEGER NOT NULL,
    reason        TEXT NOT NULL,
    bet_id        INTEGER REFERENCES bets(id)
);
CREATE INDEX IF NOT EXISTS idx_ledger_arm ON ledger(arm, ts_utc);

CREATE TABLE IF NOT EXISTS calibrations (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    fitted_utc     TEXT NOT NULL,
    method         TEXT NOT NULL,
    params_json    TEXT NOT NULL,
    n_games        INTEGER NOT NULL,
    valid_from_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    tricode       TEXT PRIMARY KEY,
    rating        REAL NOT NULL,
    games_seeded  INTEGER NOT NULL,
    season        INTEGER NOT NULL,
    updated_utc   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    kind              TEXT NOT NULL,
    started_utc       TEXT NOT NULL,
    finished_utc      TEXT,
    credits_remaining INTEGER,
    notes             TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_kind ON runs(kind, started_utc);
"""


def connect(path: str | Path) -> sqlite3.Connection:
    """Open a connection with foreign keys on and rows accessible by column name."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Create the schema if it does not already exist. Safe to call repeatedly."""
    conn.executescript(SCHEMA)
    conn.commit()
