"""Removing the bookmaker's margin from a market's prices.

Three methods, because the de-vigged consensus is the *yardstick* for the
headline forecast comparison and the choice of method biases that yardstick:

* ``multiplicative`` -- splits the margin in proportion to raw implied
  probability. Simplest, and the one most projects reach for, but it does not
  correct favourite-longshot bias, which is precisely the axis the tier system
  is built on.
* ``power`` -- solves ``sum(q_i ** k) == 1``. Corrects in the right direction.
* ``shin`` -- Shin's insider-trading model, solved for the insider proportion
  ``z``. The standard correction for favourite-longshot bias.

``shin`` is the project's primary method; the other two are reported as a
sensitivity check.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Literal

from betsim.odds import implied_probability

Method = Literal["multiplicative", "power", "shin"]
METHODS: tuple[Method, ...] = ("multiplicative", "power", "shin")

_SUM_TOLERANCE = 1e-9


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1e-14) -> float:
    """Bisect a continuous, monotonic ``f`` for its root on ``[lo, hi]``."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0:
        raise ValueError(f"no sign change on [{lo}, {hi}]: f(lo)={f_lo}, f(hi)={f_hi}")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if f_mid == 0.0 or 0.5 * (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _raw(prices: Sequence[float]) -> list[float]:
    if len(prices) < 2:
        raise ValueError(f"a market needs at least 2 outcomes, got {len(prices)}")
    return [implied_probability(d) for d in prices]


def multiplicative(prices: Sequence[float]) -> list[float]:
    """Scale raw implied probabilities so they sum to 1."""
    q = _raw(prices)
    total = sum(q)
    return [qi / total for qi in q]


def power(prices: Sequence[float]) -> list[float]:
    """Raise raw implied probabilities to the power ``k`` that makes them sum to 1."""
    q = _raw(prices)
    total = sum(q)
    if abs(total - 1.0) <= _SUM_TOLERANCE:
        return [qi / total for qi in q]

    def g(k: float) -> float:
        return sum(qi**k for qi in q) - 1.0

    k = _bisect(g, 1e-6, 100.0)
    p = [qi**k for qi in q]
    s = sum(p)
    return [pi / s for pi in p]


def shin(prices: Sequence[float]) -> list[float]:
    """Shin's method: solve for the insider proportion ``z`` that normalises the book.

    For each outcome with raw implied probability ``q_i`` and booksum ``B``::

        p_i(z) = (sqrt(z^2 + 4(1-z) q_i^2 / B) - z) / (2(1-z))

    ``sum_i p_i(z)`` decreases monotonically in ``z``, equals ``sqrt(B) >= 1`` at
    ``z = 0`` and falls below 1 as ``z -> 1``, so a unique root exists.
    """
    q = _raw(prices)
    b = sum(q)
    if b <= 1.0 + _SUM_TOLERANCE:
        return [qi / b for qi in q]

    def p_of(z: float) -> list[float]:
        den = 2.0 * (1.0 - z)
        return [(math.sqrt(z * z + 4.0 * (1.0 - z) * qi * qi / b) - z) / den for qi in q]

    def h(z: float) -> float:
        return sum(p_of(z)) - 1.0

    z = _bisect(h, 0.0, 1.0 - 1e-12)
    p = p_of(z)
    s = sum(p)
    return [pi / s for pi in p]


_DISPATCH: dict[str, Callable[[Sequence[float]], list[float]]] = {
    "multiplicative": multiplicative,
    "power": power,
    "shin": shin,
}


def devig(prices: Sequence[float], method: Method = "shin") -> list[float]:
    """De-vig one bookmaker's prices for one market. Output is aligned to input order."""
    try:
        fn = _DISPATCH[method]
    except KeyError:
        raise ValueError(f"unknown de-vig method {method!r}; expected one of {METHODS}") from None
    p = fn(prices)
    if abs(sum(p) - 1.0) > 1e-6:
        raise AssertionError(f"{method} de-vig did not normalise: sum={sum(p)}")
    return p


def consensus(
    books: Sequence[Mapping[str, float]],
    outcomes: Sequence[str],
    method: Method = "shin",
) -> dict[str, float]:
    """Mean de-vigged probability across bookmakers.

    Each entry in ``books`` maps outcome name to decimal price. Books that do not
    quote every outcome in ``outcomes`` are skipped rather than partially used --
    de-vigging an incomplete market would misstate the margin.
    """
    if not outcomes:
        raise ValueError("consensus requires at least one outcome")
    usable = [bk for bk in books if all(o in bk for o in outcomes)]
    if not usable:
        raise ValueError("no bookmaker quotes every outcome; cannot form a consensus")
    totals = dict.fromkeys(outcomes, 0.0)
    for bk in usable:
        probs = devig([bk[o] for o in outcomes], method)
        for outcome, p in zip(outcomes, probs, strict=True):
            totals[outcome] += p
    n = len(usable)
    return {o: totals[o] / n for o in outcomes}
