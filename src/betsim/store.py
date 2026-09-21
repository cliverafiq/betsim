"""Persisting ingested API data.

These functions do **not** commit. The caller owns the transaction, so a whole
slate ingest either lands or does not -- a half-written slate would leave the
experiment betting against prices it never recorded.

Timestamps are stored as ISO-8601 UTC strings throughout.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
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


def insert_context(
    conn: sqlite3.Connection,
    game_id: str,
    payload: Mapping[str, object],
    *,
    payload_hash: str,
    sources: str,
    built_utc: datetime | None = None,
) -> int:
    """Store a Stage 1 context verbatim, with its hash.

    Stored as-is so that what the model saw can be reconstructed exactly; the
    hash makes an accidental change detectable.
    """
    cur = conn.execute(
        """
        INSERT INTO contexts (game_id, built_utc, payload_json, payload_hash, sources)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            game_id,
            iso(built_utc or now_utc()),
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            payload_hash,
            sources,
        ),
    )
    if cur.lastrowid is None:
        raise RuntimeError("failed to insert a context row")
    return cur.lastrowid


def save_elo_ratings(
    conn: sqlite3.Connection,
    ratings: Mapping[str, float],
    *,
    games_seeded: int,
    season: int,
    updated_utc: datetime | None = None,
) -> int:
    """Persist the Elo table so the baseline arm survives a restart."""
    stamp = iso(updated_utc or now_utc())
    conn.executemany(
        """
        INSERT INTO elo_ratings (tricode, rating, games_seeded, season, updated_utc)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(tricode) DO UPDATE SET
            rating       = excluded.rating,
            games_seeded = excluded.games_seeded,
            season       = excluded.season,
            updated_utc  = excluded.updated_utc
        """,
        [(code, float(r), games_seeded, season, stamp) for code, r in ratings.items()],
    )
    return len(ratings)


def load_elo_ratings(conn: sqlite3.Connection) -> dict[str, float]:
    rows = conn.execute("SELECT tricode, rating FROM elo_ratings").fetchall()
    return {r["tricode"]: r["rating"] for r in rows}


def upcoming_games(
    conn: sqlite3.Connection,
    sport_key: str,
    *,
    after: datetime | None = None,
) -> list[IngestedGame]:
    """Scheduled games that have not started, oldest first."""
    cutoff = iso(after or now_utc())
    rows = conn.execute(
        """
        SELECT id, sport_key, home, away, commence_utc
          FROM games
         WHERE sport_key = ? AND status = 'scheduled' AND commence_utc > ?
         ORDER BY commence_utc ASC
        """,
        (sport_key, cutoff),
    ).fetchall()
    return [
        IngestedGame(
            id=r["id"],
            sport_key=r["sport_key"],
            home_team=r["home"],
            away_team=r["away"],
            commence_utc=datetime.fromisoformat(r["commence_utc"]),
        )
        for r in rows
    ]


def context_exists(conn: sqlite3.Connection, game_id: str, payload_hash: str) -> bool:
    """Whether this exact context has already been stored for this game."""
    row = conn.execute(
        "SELECT 1 FROM contexts WHERE game_id = ? AND payload_hash = ? LIMIT 1",
        (game_id, payload_hash),
    ).fetchone()
    return row is not None


def insert_llm_call(
    conn: sqlite3.Connection,
    call,
    *,
    stage: int,
    created_utc: datetime | None = None,
) -> int:
    """Log one LLM call -- input, hash, params, raw response and outcome.

    Every call is logged whether it succeeded or not. A refusal or a malformed
    forecast is a result of the experiment, not an error to be swallowed.
    """
    cur = conn.execute(
        """
        INSERT INTO llm_calls (
            stage, seed_idx, model_id, prompt_version, params_json, input_hash,
            input_json, raw_response, stop_reason, created_utc, ok, error,
            input_tokens, output_tokens
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            stage,
            call.seed_idx,
            call.model_id,
            call.prompt_version,
            json.dumps(call.params, sort_keys=True),
            call.input_hash,
            call.input_text,
            call.raw_response,
            call.stop_reason,
            iso(created_utc or now_utc()),
            int(call.ok),
            call.error,
            call.input_tokens,
            call.output_tokens,
        ),
    )
    if cur.lastrowid is None:
        raise RuntimeError("failed to insert an llm_calls row")
    return cur.lastrowid


def insert_forecast(
    conn: sqlite3.Connection,
    game_id: str,
    llm_call_id: int,
    *,
    seed_idx: int,
    probabilities: Mapping[str, float],
) -> int:
    """Store one normalised blind forecast."""
    cur = conn.execute(
        """
        INSERT INTO forecasts (game_id, llm_call_id, seed_idx, p_home, p_away, p_draw)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            game_id,
            llm_call_id,
            seed_idx,
            probabilities["home"],
            probabilities["away"],
            probabilities.get("draw"),
        ),
    )
    if cur.lastrowid is None:
        raise RuntimeError("failed to insert a forecast row")
    return cur.lastrowid


def latest_contexts(conn: sqlite3.Connection, game_ids: Sequence[str]) -> dict[str, dict]:
    """The most recent stored context for each game, parsed back from JSON."""
    out: dict[str, dict] = {}
    for game_id in game_ids:
        row = conn.execute(
            """
            SELECT payload_json FROM contexts
             WHERE game_id = ? ORDER BY id DESC LIMIT 1
            """,
            (game_id,),
        ).fetchone()
        if row is not None:
            out[game_id] = json.loads(row["payload_json"])
    return out


def forecast_counts(conn: sqlite3.Connection, game_id: str) -> int:
    """How many successful draws this game already has."""
    row = conn.execute("SELECT COUNT(*) c FROM forecasts WHERE game_id = ?", (game_id,)).fetchone()
    return int(row["c"])


def llm_spend(conn: sqlite3.Connection, *, stage: int | None = None) -> dict[str, int]:
    """Total tokens logged, so spend can be audited against the budget."""
    clause = "WHERE stage = ?" if stage is not None else ""
    args = (stage,) if stage is not None else ()
    row = conn.execute(
        f"""
        SELECT COUNT(*) calls,
               COALESCE(SUM(input_tokens), 0) input_tokens,
               COALESCE(SUM(output_tokens), 0) output_tokens
          FROM llm_calls {clause}
        """,
        args,
    ).fetchone()
    return {
        "calls": int(row["calls"]),
        "input_tokens": int(row["input_tokens"]),
        "output_tokens": int(row["output_tokens"]),
    }
