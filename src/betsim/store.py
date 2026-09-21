"""Persisting ingested API data.

These functions do **not** commit. The caller owns the transaction, so a whole
slate ingest either lands or does not -- a half-written slate would leave the
experiment betting against prices it never recorded.

Timestamps are stored as ISO-8601 UTC strings throughout.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime

from betsim.ingest import IngestedGame, IngestedPrice, IngestedScore
from betsim.settlement import GameStatus


def iso(dt: datetime) -> str:
    """Render an aware datetime as an ISO-8601 UTC string."""
    if dt.tzinfo is None:
        raise ValueError(f"refusing to store a naive datetime: {dt!r}")
    return dt.astimezone(UTC).isoformat()


def now_utc() -> datetime:
    return datetime.now(UTC)


def upsert_games(conn: sqlite3.Connection, games: Sequence[IngestedGame]) -> int:
    """Insert or refresh game rows. Games can be rescheduled, so commence_utc updates."""
    rows = [(g.id, g.sport_key, g.home_team, g.away_team, iso(g.commence_utc)) for g in games]
    conn.executemany(
        """
        INSERT INTO games (id, sport_key, home, away, commence_utc, status)
        VALUES (?, ?, ?, ?, ?, 'scheduled')
        ON CONFLICT(id) DO UPDATE SET
            sport_key    = excluded.sport_key,
            home         = excluded.home,
            away         = excluded.away,
            commence_utc = excluded.commence_utc
        """,
        rows,
    )
    return len(rows)


def insert_snapshots(
    conn: sqlite3.Connection,
    prices: Sequence[IngestedPrice],
    *,
    captured_utc: datetime,
    is_closing: bool = False,
) -> int:
    """Append price rows. Snapshots are immutable history and are never updated."""
    captured = iso(captured_utc)
    rows = [
        (p.game_id, captured, p.bookmaker, p.market, p.selection, p.price_decimal, int(is_closing))
        for p in prices
    ]
    conn.executemany(
        """
        INSERT INTO odds_snapshots
            (game_id, captured_utc, bookmaker, market, outcome, price_decimal, is_closing)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def apply_scores(
    conn: sqlite3.Connection,
    scores: Sequence[IngestedScore],
    *,
    completed_utc: datetime | None = None,
) -> int:
    """Write final scores for completed games.

    Games still in progress are skipped rather than written with partial scores --
    settlement must never see a half-finished game.
    """
    stamp = iso(completed_utc or now_utc())
    finished = [s for s in scores if s.completed]
    conn.executemany(
        """
        UPDATE games
           SET status = ?, home_score = ?, away_score = ?, completed_utc = ?
         WHERE id = ?
        """,
        [
            (GameStatus.COMPLETED.value, s.home_score, s.away_score, stamp, s.game_id)
            for s in finished
        ],
    )
    return len(finished)


def mark_void(conn: sqlite3.Connection, game_id: str, status: GameStatus) -> None:
    """Mark a game postponed or cancelled. Bets on it void and refund."""
    if status not in (GameStatus.POSTPONED, GameStatus.CANCELLED):
        raise ValueError(f"{status!r} is not a voiding status")
    conn.execute("UPDATE games SET status = ? WHERE id = ?", (status.value, game_id))


def start_run(conn: sqlite3.Connection, kind: str, *, started_utc: datetime | None = None) -> int:
    """Open a run row and return its id."""
    cur = conn.execute(
        "INSERT INTO runs (kind, started_utc) VALUES (?, ?)",
        (kind, iso(started_utc or now_utc())),
    )
    if cur.lastrowid is None:
        raise RuntimeError("failed to create a run row")
    return cur.lastrowid


def finish_run(
    conn: sqlite3.Connection,
    run_id: int,
    *,
    credits_remaining: int | None = None,
    notes: str | None = None,
    finished_utc: datetime | None = None,
) -> None:
    """Close a run, recording the credit balance the API reported.

    Recording the balance per run is what makes the quota auditable after the
    fact -- it is the only record of what the experiment actually spent.
    """
    conn.execute(
        "UPDATE runs SET finished_utc = ?, credits_remaining = ?, notes = ? WHERE id = ?",
        (iso(finished_utc or now_utc()), credits_remaining, notes, run_id),
    )


def open_game_ids(conn: sqlite3.Connection, arm: str) -> set[str]:
    """Games this arm already has an unsettled bet on."""
    rows = conn.execute(
        "SELECT game_id FROM bets WHERE arm = ? AND status = 'open'", (arm,)
    ).fetchall()
    return {r["game_id"] for r in rows}


def latest_prices(
    conn: sqlite3.Connection,
    game_id: str,
    *,
    bookmaker: str,
    market: str = "h2h",
) -> dict[str, float]:
    """The most recent price per outcome from one bookmaker for one game."""
    rows = conn.execute(
        """
        SELECT outcome, price_decimal
          FROM odds_snapshots
         WHERE game_id = ? AND bookmaker = ? AND market = ?
         ORDER BY captured_utc ASC
        """,
        (game_id, bookmaker, market),
    ).fetchall()
    return {r["outcome"]: r["price_decimal"] for r in rows}


def closing_prices(
    conn: sqlite3.Connection,
    game_id: str,
    *,
    bookmaker: str,
    market: str = "h2h",
) -> dict[str, float]:
    """Closing prices for one game, used to compute CLV."""
    rows = conn.execute(
        """
        SELECT outcome, price_decimal
          FROM odds_snapshots
         WHERE game_id = ? AND bookmaker = ? AND market = ? AND is_closing = 1
         ORDER BY captured_utc ASC
        """,
        (game_id, bookmaker, market),
    ).fetchall()
    return {r["outcome"]: r["price_decimal"] for r in rows}
