"""Fractional Kelly staking.

The Kelly shadow arms exist to separate forecasting skill from money management.
They take the *same* probabilities the LLM arm saw and the *same* caps, so a
difference between ``llm`` and ``kelly`` is attributable to sizing alone.
"""

from __future__ import annotations

from collections.abc import Mapping

from betsim.money import floor_minor
from betsim.odds import validate_decimal

DEFAULT_FRACTION = 0.25  # quarter Kelly


def kelly_fraction(p: float, price_decimal: float) -> float:
    """Full-Kelly fraction of bankroll: ``f = (b*p - q) / b``.

    ``b = d - 1`` (net odds), ``q = 1 - p``. Negative when the bet has no edge at
    this price; callers should not stake on a non-positive fraction.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {p}")
    b = validate_decimal(price_decimal) - 1.0
    return (b * p - (1.0 - p)) / b


def kelly_stake_minor(
    p: float,
    price_decimal: float,
    balance_minor: int,
    fraction: float = DEFAULT_FRACTION,
) -> int:
    """Fractional-Kelly stake in minor units, rounded **down**.

    Returns 0 when the edge is non-positive. Tier and exposure caps are applied
    by the caller, not here.
    """
    if balance_minor <= 0:
        return 0
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"Kelly fraction must be in (0, 1], got {fraction}")
    f = kelly_fraction(p, price_decimal)
    if f <= 0.0:
        return 0
    return max(0, floor_minor(fraction * f * balance_minor))


def best_outcome(
    probs: Mapping[str, float],
    prices: Mapping[str, float],
) -> tuple[str, float] | None:
    """Pick the outcome with the largest positive Kelly fraction.

    Returns ``(outcome, fraction)``, or ``None`` when no outcome has an edge --
    which is a normal and expected result, not a failure.
    """
    best: tuple[str, float] | None = None
    for outcome, p in probs.items():
        if outcome not in prices:
            continue
        f = kelly_fraction(p, prices[outcome])
        if f > 0.0 and (best is None or f > best[1]):
            best = (outcome, f)
    return best
