"""Grading settled bets.

The Odds API returns scores but does not grade bets, so settlement is ours.
Payouts round **down**, consistent with stakes.
"""

from __future__ import annotations

from enum import StrEnum

from betsim.money import floor_minor
from betsim.odds import validate_decimal
from betsim.sports import SportSpec


class GameStatus(StrEnum):
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


class BetStatus(StrEnum):
    OPEN = "open"
    WON = "won"
    LOST = "lost"
    VOID = "void"


VOIDING_STATUSES = frozenset({GameStatus.POSTPONED, GameStatus.CANCELLED})


def winning_selection(spec: SportSpec, home_score: int, away_score: int) -> str | None:
    """The selection that wins, or ``None`` when the game pushes.

    Raises for a level score in a sport that cannot tie -- that is a data
    integrity problem, and silently treating it as a push would corrupt the ledger.
    """
    if home_score > away_score:
        return "home"
    if away_score > home_score:
        return "away"
    if spec.market_width == 3:
        return "draw"
    if spec.tie_is_push:
        return None
    raise ValueError(
        f"{spec.key} reported a level final score {home_score}-{away_score}, "
        "but this sport cannot tie (overtime/shootout). Refusing to settle."
    )


def settle_bet(
    spec: SportSpec,
    *,
    selection: str,
    price_decimal: float,
    stake_minor: int,
    game_status: GameStatus,
    home_score: int | None = None,
    away_score: int | None = None,
) -> tuple[BetStatus, int]:
    """Grade one bet.

    Returns ``(status, payout_minor)`` where ``payout_minor`` is the **total**
    returned to the bankroll: stake plus profit on a win, the stake on a void or
    push, zero on a loss. Profit is therefore ``payout_minor - stake_minor``.
    """
    if stake_minor < 0:
        raise ValueError(f"stake must be non-negative, got {stake_minor}")
    if not spec.validate_selection(selection):
        raise ValueError(f"{selection!r} is not a valid selection for {spec.key}")

    status = GameStatus(game_status)
    if status in VOIDING_STATUSES:
        return BetStatus.VOID, stake_minor
    if status is not GameStatus.COMPLETED:
        raise ValueError(f"cannot settle a game with status {status!r}")
    if home_score is None or away_score is None:
        raise ValueError("completed games must carry both scores")

    winner = winning_selection(spec, home_score, away_score)
    if winner is None:
        return BetStatus.VOID, stake_minor
    if winner == selection:
        return BetStatus.WON, floor_minor(stake_minor * validate_decimal(price_decimal))
    return BetStatus.LOST, 0


def profit_minor(status: BetStatus, stake_minor: int, payout_minor: int) -> int:
    """Signed profit for a settled bet. Zero for a void, and for a bet still open."""
    if BetStatus(status) is BetStatus.OPEN:
        return 0
    return payout_minor - stake_minor
