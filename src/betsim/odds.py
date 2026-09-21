"""Odds representation and conversion.

Decimal odds are the internal representation everywhere; American odds are
converted on ingest and only re-derived for display.
"""

from __future__ import annotations

import math

MIN_DECIMAL_ODDS = 1.0


def validate_decimal(d: float) -> float:
    """Decimal odds must exceed 1.0 (a price of 1.0 returns only the stake)."""
    if not isinstance(d, (int, float)) or math.isnan(d):
        raise ValueError(f"decimal odds must be a number, got {d!r}")
    if d <= MIN_DECIMAL_ODDS:
        raise ValueError(f"decimal odds must be > 1.0, got {d}")
    return float(d)


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal.

    ``+100`` and ``-100`` both mean evens and map to 2.0. There are no American
    prices strictly between -100 and +100.
    """
    a = float(american)
    if math.isnan(a):
        raise ValueError("American odds must be a number, got NaN")
    if abs(a) < 100:
        raise ValueError(f"American odds must satisfy |odds| >= 100, got {american}")
    if a < 0:
        return 1.0 + 100.0 / abs(a)
    return 1.0 + a / 100.0


def decimal_to_american(d: float) -> float:
    """Convert decimal odds to American. At exactly 2.0 the convention here is +100."""
    d = validate_decimal(d)
    if d >= 2.0:
        return (d - 1.0) * 100.0
    return -100.0 / (d - 1.0)


def implied_probability(d: float) -> float:
    """Raw implied probability, 1/d. Includes the bookmaker's margin; see :mod:`betsim.devig`."""
    return 1.0 / validate_decimal(d)


def booksum(prices: list[float]) -> float:
    """Sum of raw implied probabilities across a market's outcomes.

    Also called the overround or booksum. Above 1.0 for a normal market; the
    excess over 1.0 is the bookmaker's margin.
    """
    if not prices:
        raise ValueError("booksum requires at least one price")
    return sum(implied_probability(d) for d in prices)
