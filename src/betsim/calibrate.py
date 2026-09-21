"""Post-hoc probability calibration (Platt scaling).

The documented LLM failure mode in forecasting is **overconfidence on
high-probability events** -- a model says 90% for things that happen 70% of the
time. That is correctable after the fact, and doing so moved a general
forecasting system to parity with the superforecaster median, so it is worth
testing directly rather than assuming the forecasts are beyond help.

``kelly_cal`` stakes recalibrated blind probabilities under the same rule as
``kelly``. If it wins, the forecasts carry signal the model's own confidence is
destroying.

Fitted with IRLS in plain Python -- two parameters on one feature does not
justify a numerical dependency. Platt's target smoothing is applied, which keeps
the fit from saturating on small samples.

**Out-of-sample discipline is the caller's job and it is not optional.** A map
fitted on a game and then applied to that same game is circular, and would make
``kelly_cal`` look good for no reason. Fit on games before a cutoff; apply only
after it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

EPS = 1e-12
MAX_ITERATIONS = 100
CONVERGENCE = 1e-10


class CalibrationError(ValueError):
    """Not enough signal to fit a calibration map."""


def logit(p: float) -> float:
    p = min(max(p, EPS), 1.0 - EPS)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass(frozen=True, slots=True)
class PlattParams:
    """``p_calibrated = sigmoid(a * logit(p) + b)``."""

    a: float
    b: float
    n: int

    def apply(self, p: float) -> float:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {p}")
        return sigmoid(self.a * logit(p) + self.b)

    @property
    def is_identity(self) -> bool:
        """A well-calibrated input needs no correction: a == 1, b == 0."""
        return abs(self.a - 1.0) < 1e-6 and abs(self.b) < 1e-6

    def to_dict(self) -> dict[str, float | int]:
        return {"a": self.a, "b": self.b, "n": self.n}

    @classmethod
    def from_dict(cls, data: dict) -> PlattParams:
        return cls(a=float(data["a"]), b=float(data["b"]), n=int(data.get("n", 0)))


IDENTITY = PlattParams(a=1.0, b=0.0, n=0)


def fit_platt(
    pairs: Sequence[tuple[float, bool]],
    *,
    smooth: bool = True,
) -> PlattParams:
    """Fit ``a`` and ``b`` by iteratively reweighted least squares.

    ``pairs`` is (forecast probability, whether it happened). Needs both
    outcomes present -- a sample where everything happened carries no
    information about calibration and would diverge.
    """
    if len(pairs) < 2:
        raise CalibrationError(f"need at least 2 observations, got {len(pairs)}")
    positives = sum(1 for _, hit in pairs if hit)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise CalibrationError(
            "every observation has the same outcome; nothing to calibrate against"
        )

    # Platt's target smoothing keeps the fit off 0 and 1 on small samples.
    hi = (positives + 1.0) / (positives + 2.0) if smooth else 1.0
    lo = 1.0 / (negatives + 2.0) if smooth else 0.0

    xs = [logit(p) for p, _ in pairs]
    ys = [hi if hit else lo for _, hit in pairs]

    a, b = 1.0, 0.0
    for _ in range(MAX_ITERATIONS):
        s_wxx = s_wx = s_w = s_wxz = s_wz = 0.0
        for x, y in zip(xs, ys, strict=True):
            eta = a * x + b
            mu = sigmoid(eta)
            w = max(mu * (1.0 - mu), 1e-10)
            z = eta + (y - mu) / w
            s_wxx += w * x * x
            s_wx += w * x
            s_w += w
            s_wxz += w * x * z
            s_wz += w * z

        det = s_wxx * s_w - s_wx * s_wx
        if abs(det) < 1e-15:
            break
        new_a = (s_wxz * s_w - s_wx * s_wz) / det
        new_b = (s_wxx * s_wz - s_wx * s_wxz) / det
        if abs(new_a - a) < CONVERGENCE and abs(new_b - b) < CONVERGENCE:
            a, b = new_a, new_b
            break
        a, b = new_a, new_b

    if not (math.isfinite(a) and math.isfinite(b)):
        raise CalibrationError("fit did not converge to finite parameters")
    return PlattParams(a=a, b=b, n=len(pairs))


def calibrate_distribution(probs: dict[str, float], params: PlattParams) -> dict[str, float]:
    """Recalibrate each outcome, then renormalise so the distribution still sums to 1."""
    mapped = {outcome: params.apply(p) for outcome, p in probs.items()}
    total = sum(mapped.values())
    if total <= 0:
        raise CalibrationError("calibration produced a degenerate distribution")
    return {outcome: p / total for outcome, p in mapped.items()}
