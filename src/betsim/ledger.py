"""Bankroll accounting.

The ledger is **append-only**. Placing a bet appends a negative row, settling it
appends a positive one, and corrections are new rows rather than edits. Every
settled bet therefore leaves exactly two rows -- including a loss, which appends
a zero-delta row so that settlement is always visible in the audit trail rather
than inferred from its absence.

``balance`` is free cash: stakes leave at placement and payouts return at
settlement, so money riding on unsettled bets is not in the balance. Open
exposure is tracked separately.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from betsim.arms import ArmBet
from betsim.money import STARTING_BALANCE_MINOR, is_bust
from betsim.settlement import BetStatus
from betsim.store import iso, now_utc


@dataclass(frozen=True, slots=True)
class ArmState:
    arm: str
    balance_minor: int
    open_exposure_minor: int
    open_game_ids: frozenset[str]
    bust: bool

    @property
    def equity_minor(self) -> int:
        """Free cash plus money still riding on unsettled bets."""
        return self.balance_minor + self.open_exposure_minor


def balance(conn: sqlite3.Connection, arm: str) -> int:
    """Current free cash. An arm with no ledger history starts at 1,000 units."""
    row = conn.execute(
        "SELECT balance_minor FROM ledger WHERE arm = ? ORDER BY id DESC LIMIT 1", (arm,)
    ).fetchone()
    return STARTING_BALANCE_MINOR if row is None else int(row["balance_minor"])


def open_exposure(conn: sqlite3.Connection, arm: str) -> int:
    row = conn.execute(
        "SELECT COALESCE(SUM(stake_minor), 0) s FROM bets WHERE arm = ? AND status = 'open'",
        (arm,),
    ).fetchone()
    return int(row["s"])


def open_games(conn: sqlite3.Connection, arm: str) -> frozenset[str]:
    rows = conn.execute(
        "SELECT game_id FROM bets WHERE arm = ? AND status = 'open'", (arm,)
    ).fetchall()
    return frozenset(r["game_id"] for r in rows)


def arm_state(conn: sqlite3.Connection, arm: str) -> ArmState:
    current = balance(conn, arm)
    return ArmState(
        arm=arm,
        balance_minor=current,
        open_exposure_minor=open_exposure(conn, arm),
        open_game_ids=open_games(conn, arm),
        bust=is_bust(current),
    )


def _append(
    conn: sqlite3.Connection,
    arm: str,
    delta_minor: int,
    *,
    reason: str,
    bet_id: int | None,
    ts_utc: datetime | None = None,
) -> int:
    new_balance = balance(conn, arm) + delta_minor
    conn.execute(
        """
        INSERT INTO ledger (arm, ts_utc, delta_minor, balance_minor, reason, bet_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (arm, iso(ts_utc or now_utc()), delta_minor, new_balance, reason, bet_id),
    )
    return new_balance


def place_bet(
    conn: sqlite3.Connection,
    arm: str,
    bet: ArmBet,
    *,
    placed_utc: datetime | None = None,
    forecast_id: int | None = None,
    snapshot_id: int | None = None,
) -> int:
    """Record a bet and deduct its stake. Returns the bet id.

    Refuses to overdraw: an arm cannot stake money it does not have.
    """
    current = balance(conn, arm)
    if bet.stake_minor > current:
        raise ValueError(f"{arm} cannot stake {bet.stake_minor} with {current} free")
    stamp = placed_utc or now_utc()
    cur = conn.execute(
        """
        INSERT INTO bets (
            arm, game_id, forecast_id, snapshot_id, placed_utc, selection,
            price_decimal, tier, stake_minor, p_blind, p_revised, reason, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        """,
        (
            arm,
            bet.game_id,
            forecast_id,
            snapshot_id,
            iso(stamp),
            bet.selection,
            bet.price_decimal,
            str(bet.tier),
            bet.stake_minor,
            bet.p_blind,
            bet.p_revised,
            bet.reason,
        ),
    )
    bet_id = cur.lastrowid
    if bet_id is None:
        raise RuntimeError("failed to insert a bet row")
    _append(conn, arm, -bet.stake_minor, reason="stake", bet_id=bet_id, ts_utc=stamp)
    return bet_id


def settle_bet(
    conn: sqlite3.Connection,
    bet_id: int,
    *,
    status: BetStatus,
    payout_minor: int,
    settled_utc: datetime | None = None,
) -> int:
    """Settle a bet and credit its payout. Returns the arm's new balance.

    A losing bet still appends a zero-delta row, so every settled bet has exactly
    two ledger rows and settlement is never inferred from a missing one.
    """
    row = conn.execute(
        "SELECT arm, status, stake_minor FROM bets WHERE id = ?", (bet_id,)
    ).fetchone()
    if row is None:
        raise ValueError(f"no bet with id {bet_id}")
    if row["status"] != BetStatus.OPEN.value:
        raise ValueError(f"bet {bet_id} is already settled as {row['status']!r}")

    stamp = settled_utc or now_utc()
    conn.execute(
        "UPDATE bets SET status = ?, payout_minor = ?, settled_utc = ? WHERE id = ?",
        (BetStatus(status).value, payout_minor, iso(stamp), bet_id),
    )
    return _append(
        conn,
        row["arm"],
        payout_minor,
        reason=f"settle:{BetStatus(status).value}",
        bet_id=bet_id,
        ts_utc=stamp,
    )


def balance_path(conn: sqlite3.Connection, arm: str) -> list[int]:
    """Balance after every ledger entry, starting from the opening bankroll."""
    rows = conn.execute(
        "SELECT balance_minor FROM ledger WHERE arm = ? ORDER BY id ASC", (arm,)
    ).fetchall()
    return [STARTING_BALANCE_MINOR] + [int(r["balance_minor"]) for r in rows]


def arms_in_play(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT DISTINCT arm FROM ledger ORDER BY arm").fetchall()
    return [r["arm"] for r in rows]


def verify_ledger(conn: sqlite3.Connection, arm: str) -> None:
    """Check the ledger's running balance is internally consistent.

    Cheap to run and worth running: a drift here means the bankroll numbers the
    whole experiment reports are wrong.
    """
    rows = conn.execute(
        "SELECT id, delta_minor, balance_minor FROM ledger WHERE arm = ? ORDER BY id ASC",
        (arm,),
    ).fetchall()
    running = STARTING_BALANCE_MINOR
    for row in rows:
        running += int(row["delta_minor"])
        if running != int(row["balance_minor"]):
            raise AssertionError(
                f"{arm} ledger row {row['id']}: running total {running} "
                f"!= recorded balance {row['balance_minor']}"
            )


def open_bets(conn: sqlite3.Connection, arm: str | None = None) -> Sequence[sqlite3.Row]:
    clause = "WHERE status = 'open'" + (" AND arm = ?" if arm else "")
    args = (arm,) if arm else ()
    return conn.execute(f"SELECT * FROM bets {clause} ORDER BY placed_utc ASC", args).fetchall()
