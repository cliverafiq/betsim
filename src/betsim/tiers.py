"""Tier assignment and per-tier stake caps.

Tiers are assigned by code from the designated bookmaker's price at placement,
never by the model. They describe *variance*, not value: at -110 the break-even
win rate is 52.38% and a coin-flipper loses 4.55% of stakes, so "safe" means
smaller swings, not positive expected value.

Boundaries are half-open so that no price can match two tiers:

    safe   : implied >= 0.60          (-150 or shorter)   cap 5% of balance
    medium : 0.40 <= implied < 0.60                       cap 3% of balance
    risky  : implied < 0.40           (longer than +150)  cap 1% of balance
"""

from __future__ import annotations

from enum import StrEnum

from betsim.money import pct_of_balance_minor
from betsim.odds import implied_probability

# Boundary comparisons are done with a tolerance because the exact boundary
# prices are not representable in binary floating point. 1/2.5000000000000004 is
# 0.3999999999999999, which without a guard classifies a +150 price as RISKY (1%
# cap) instead of MEDIUM (3%) on one unit in the last place of error. The
# tolerance is ~1e-9, far above float error (~1e-16) and far below any real
# price difference (-150 vs 1.6667 differ by ~1.2e-5, and still resolve apart).
TIER_EPS = 1e-9

SAFE_MIN_IMPLIED = 0.60
MEDIUM_MIN_IMPLIED = 0.40


class Tier(StrEnum):
    SAFE = "safe"
    MEDIUM = "medium"
    RISKY = "risky"


TIER_CAP_PCT: dict[Tier, float] = {
    Tier.SAFE: 0.05,
    Tier.MEDIUM: 0.03,
    Tier.RISKY: 0.01,
}


def assign_tier(price_decimal: float) -> Tier:
    """Assign a tier from a decimal price. Uses the raw implied probability, 1/d."""
    implied = implied_probability(price_decimal)
    if implied >= SAFE_MIN_IMPLIED - TIER_EPS:
        return Tier.SAFE
    if implied >= MEDIUM_MIN_IMPLIED - TIER_EPS:
        return Tier.MEDIUM
    return Tier.RISKY


def tier_cap_minor(tier: Tier, balance_minor: int) -> int:
    """Maximum stake for a tier at a given balance, in minor units (floored)."""
    return pct_of_balance_minor(balance_minor, TIER_CAP_PCT[Tier(tier)])
