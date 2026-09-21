"""Integer minor-unit money handling.

All balances and stakes are integers in *minor units* (1 unit = 100 minor units),
so no bankroll arithmetic ever touches a float. Every rounding in this project is
**floor**, for both stakes and payouts: it is conservative (never invents money)
and it makes the ledger reproducible.
"""

from __future__ import annotations

import math
from decimal import ROUND_FLOOR, Decimal

# Floor rounding on a float that *should* be a whole number can be short by one
# minor unit: kelly_fraction(0.6, 2.0) is 0.19999999999999996, so a quarter-Kelly
# stake of 5000 computes as 4999.999999999999 and floors to 4999. The epsilon is
# ~5 orders of magnitude above float error at realistic bankroll sizes (~1e-11)
# and ~6 below one minor unit, so it repairs the representation error without
# ever rounding a genuinely fractional stake up.
FLOOR_EPS = 1e-6

MINOR_PER_UNIT = 100
STARTING_BALANCE_MINOR = 1_000 * MINOR_PER_UNIT
MIN_STAKE_MINOR = 1 * MINOR_PER_UNIT
BUST_THRESHOLD_MINOR = 1 * MINOR_PER_UNIT


def floor_minor(value: float) -> int:
    """Floor a computed money amount to whole minor units, tolerant of float error."""
    if not math.isfinite(value):
        raise ValueError(f"amount must be finite, got {value!r}")
    return math.floor(value + FLOOR_EPS)


def units_to_minor(units: float | str | Decimal) -> int:
    """Convert units to minor units, rounding down.

    Goes through ``Decimal(str(x))`` rather than ``int(units * 100)`` because the
    naive form is wrong for ordinary decimal inputs: ``0.29 * 100`` is
    ``28.999999999999996`` in binary floating point, which floors to 28.
    """
    if isinstance(units, float) and not math.isfinite(units):
        raise ValueError(f"stake must be finite, got {units!r}")
    d = Decimal(str(units))
    if d < 0:
        raise ValueError(f"stake must be non-negative, got {units!r}")
    return int((d * MINOR_PER_UNIT).to_integral_value(rounding=ROUND_FLOOR))


def minor_to_units(minor: int) -> float:
    """Convert minor units back to units. For display and reporting only."""
    return minor / MINOR_PER_UNIT


def pct_of_balance_minor(balance_minor: int, pct: float) -> int:
    """Floor of ``pct`` of a balance, in minor units. ``pct`` is a fraction (0.05 = 5%)."""
    if balance_minor < 0:
        raise ValueError(f"balance must be non-negative, got {balance_minor}")
    if not 0.0 <= pct <= 1.0:
        raise ValueError(f"pct must be in [0, 1], got {pct}")
    return floor_minor(balance_minor * pct)


def is_bust(balance_minor: int) -> bool:
    """An arm is bust when its balance falls below one whole unit.

    Bust is absorbing: the arm stops betting and its ledger freezes, but the run
    continues and time-to-ruin is reported.
    """
    return balance_minor < BUST_THRESHOLD_MINOR
