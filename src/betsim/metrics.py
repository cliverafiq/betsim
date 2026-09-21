"""Scoring rules, bankroll statistics, and uncertainty.

On the Brier convention: this project uses the **sum over outcomes** form,
``sum_k (p_k - o_k)^2``. Its scale depends on how many outcomes the market has --
a uniform 2-way forecast scores 0.500 and a uniform 3-way scores 0.667 -- so NHL
numbers are *not* comparable to the 3-way soccer literature (where market Brier
is around 0.469). Never put two market widths in one table.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

LOG_LOSS_EPS = 1e-15


def brier_score(probs: Mapping[str, float], actual: str) -> float:
    """Multi-class Brier score for one game, summed over outcomes. Lower is better."""
    if actual not in probs:
        raise ValueError(f"actual outcome {actual!r} not among forecast outcomes {sorted(probs)}")
    return sum((p - (1.0 if outcome == actual else 0.0)) ** 2 for outcome, p in probs.items())


def log_loss(probs: Mapping[str, float], actual: str, eps: float = LOG_LOSS_EPS) -> float:
    """Negative log likelihood of the realised outcome. Clipped to stay finite."""
    if actual not in probs:
        raise ValueError(f"actual outcome {actual!r} not among forecast outcomes {sorted(probs)}")
    return -math.log(min(max(probs[actual], eps), 1.0 - eps))


def roi(profits_minor: Sequence[int], stakes_minor: Sequence[int]) -> float | None:
    """Total profit over total staked. ``None`` when nothing was staked.

    Zero-bet days are a valid and expected outcome, so an undefined ROI is a
    normal state rather than an error.
    """
    if len(profits_minor) != len(stakes_minor):
        raise ValueError("profits and stakes must be the same length")
    total_stake = sum(stakes_minor)
    if total_stake == 0:
        return None
    return sum(profits_minor) / total_stake


def max_drawdown(balances_minor: Sequence[int]) -> float:
    """Largest peak-to-trough fall in the balance path, as a fraction of the peak."""
    if not balances_minor:
        return 0.0
    peak = balances_minor[0]
    worst = 0.0
    for balance in balances_minor:
        peak = max(peak, balance)
        if peak > 0:
            worst = max(worst, (peak - balance) / peak)
    return worst


def clv(price_taken: float, price_close: float) -> float:
    """Closing line value: ``d_taken / d_close - 1``.

    The primary metric for this experiment, because it is the best-evidenced
    skill signal and settles far faster than ROI. Note the null is **not** zero:
    bets placed at the slate snapshot pick up some CLV from timing alone, so the
    comparison is against the `random` arm's CLV distribution, which is generated
    at the same snapshots with no skill.
    """
    if price_taken <= 1.0 or price_close <= 1.0:
        raise ValueError(f"decimal prices must exceed 1.0, got {price_taken} and {price_close}")
    return price_taken / price_close - 1.0


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    mean_predicted: float
    observed_rate: float


def calibration_table(
    pairs: Sequence[tuple[float, bool]],
    n_bins: int = 10,
) -> list[CalibrationBin]:
    """Reliability table: predicted probability vs observed frequency.

    Expect overconfidence concentrated on high-probability outcomes -- that is
    the documented LLM failure mode, and the thing the `kelly_cal` arm tests.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, hit in pairs:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {p}")
        idx = min(int(p * n_bins), n_bins - 1)  # p == 1.0 belongs to the top bin
        buckets[idx].append((p, hit))
    table = []
    for i, bucket in enumerate(buckets):
        lower, upper = i / n_bins, (i + 1) / n_bins
        if bucket:
            mean_p = sum(p for p, _ in bucket) / len(bucket)
            rate = sum(1 for _, hit in bucket if hit) / len(bucket)
        else:
            mean_p = rate = float("nan")
        table.append(CalibrationBin(lower, upper, len(bucket), mean_p, rate))
    return table


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence. ``q`` in [0, 1]."""
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = q * (len(sorted_values) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    return float(sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo))


def bootstrap_ci[T](
    groups: Mapping[object, Sequence[T]],
    statistic: Callable[[Sequence[T]], float | None],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float] | None:
    """Bootstrap CI, resampling whole **groups** (days) with replacement.

    Same-day bets share a slate, a bankroll state and often correlated outcomes,
    so resampling individual bets would understate the interval. Resamples where
    the statistic is undefined (no bets drawn) are skipped.
    """
    keys = list(groups)
    if not keys:
        return None
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_resamples):
        sample: list[T] = []
        for _ in range(len(keys)):
            sample.extend(groups[keys[rng.randrange(len(keys))]])
        value = statistic(sample)
        if value is not None and math.isfinite(value):
            draws.append(value)
    if not draws:
        return None
    draws.sort()
    return percentile(draws, alpha / 2.0), percentile(draws, 1.0 - alpha / 2.0)
