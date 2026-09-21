"""Settling open bets and measuring closing line value.

Grading lives in :mod:`betsim.settlement` and bankroll movement in
:mod:`betsim.ledger`; this module joins them to the database and is where a bad
result is contained. A game the grader refuses to settle -- an NHL game reported
level, say, which cannot happen -- is recorded and skipped rather than allowed
to take down the rest of the night's settlement.

CLV is derived rather than stored: it is entirely determined by the price taken
and the closing price, both of which are already in ``odds_snapshots``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

from betsim.config import DESIGNATED_BOOKMAKER, MARKET
from betsim.ledger import settle_bet as ledger_settle
from betsim.metrics import clv as clv_of
from betsim.settlement import BetStatus, GameStatus, settle_bet
from betsim.sports import get_sport
from betsim.store import closing_prices, now_utc, open_bets_for_settlement


@dataclass(slots=True)
class SettleResult:
    settled: int = 0
    won: int = 0
    lost: int = 0
    void: int = 0
    by_arm: dict[str, int] = field(default_factory=dict)
    profit_minor: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def _count(self, status: BetStatus) -> None:
        self.settled += 1
        if status is BetStatus.WON:
            self.won += 1
        elif status is BetStatus.LOST:
            self.lost += 1
        else:
            self.void += 1


def settle_open_bets(
    conn: sqlite3.Connection,
    sport: str,
    *,
    now: datetime | None = None,
) -> SettleResult:
    """Grade and settle every open bet whose game has finished or been called off."""
    spec = get_sport(sport)
    stamp = now or now_utc()
    result = SettleResult()

    for row in open_bets_for_settlement(conn, sport):
        try:
            status, payout = settle_bet(
                spec,
                selection=row["selection"],
                price_decimal=row["price_decimal"],
                stake_minor=row["stake_minor"],
                game_status=GameStatus(row["game_status"]),
                home_score=row["home_score"],
                away_score=row["away_score"],
            )
        except ValueError as exc:
            # Bad data on one game must not strand the rest of the slate.
            result.errors.append(f"bet {row['id']} ({row['game_id']}): {exc}")
            continue

        ledger_settle(conn, row["id"], status=status, payout_minor=payout, settled_utc=stamp)
        result._count(status)
        arm = row["arm"]
        result.by_arm[arm] = result.by_arm.get(arm, 0) + 1
        result.profit_minor[arm] = result.profit_minor.get(arm, 0) + payout - row["stake_minor"]
    return result


@dataclass(frozen=True, slots=True)
class BetCLV:
    bet_id: int
    arm: str
    game_id: str
    selection: str
    price_taken: float
    price_close: float
    clv: float


def bet_clv(
    conn: sqlite3.Connection,
    *,
    bookmaker: str = DESIGNATED_BOOKMAKER,
    market: str = MARKET,
    arm: str | None = None,
) -> list[BetCLV]:
    """CLV for every bet that has a closing price to compare against.

    The null is **not** zero. A bet struck at the slate snapshot picks up some
    CLV from timing alone, so the comparison is against the `random` arm's
    distribution -- it bets at the same snapshots with no skill.
    """
    clause = " AND arm = ?" if arm else ""
    args: tuple = (arm,) if arm else ()
    rows = conn.execute(
        f"SELECT id, arm, game_id, selection, price_decimal FROM bets WHERE 1=1{clause}",
        args,
    ).fetchall()

    out: list[BetCLV] = []
    closes: dict[str, dict[str, float]] = {}
    for row in rows:
        game_id = row["game_id"]
        if game_id not in closes:
            closes[game_id] = closing_prices(conn, game_id, bookmaker=bookmaker, market=market)
        close = closes[game_id].get(row["selection"])
        if close is None:
            continue
        out.append(
            BetCLV(
                bet_id=row["id"],
                arm=row["arm"],
                game_id=game_id,
                selection=row["selection"],
                price_taken=row["price_decimal"],
                price_close=close,
                clv=clv_of(row["price_decimal"], close),
            )
        )
    return out


def clv_by_arm(values: Sequence[BetCLV]) -> dict[str, dict[str, float]]:
    """Mean CLV and bet count per arm."""
    grouped: dict[str, list[float]] = {}
    for entry in values:
        grouped.setdefault(entry.arm, []).append(entry.clv)
    return {arm: {"n": len(v), "mean_clv": sum(v) / len(v)} for arm, v in sorted(grouped.items())}
