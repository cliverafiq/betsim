"""Pydantic schemas for every LLM input and output.

These are deliberately permissive about *experiment* rules and strict about
*structure*. Structure is enforced at parse time via structured outputs
(``output_config.format`` plus ``messages.parse()``), so schema-valid JSON is
guaranteed rather than parsed-and-retried. The experiment rule -- that a
forecast's probabilities must sum to within 0.02 of 1 -- lives in
:func:`normalize_probabilities` instead, as a tested pure function, so that a
violation can be *logged as data* rather than raised inside the API client.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Selection = Literal["home", "away", "draw"]

PROBABILITY_SUM_TOLERANCE = 0.02

# "Within 0.02 of 1" is inclusive, but the boundary is not representable: a
# forecast summing to exactly 1.02 differs from 1.0 by 0.020000000000000018 in
# binary, which would be rejected. Same guard, and same reasoning, as the tier
# boundary in betsim.tiers.
_TOLERANCE_EPS = 1e-9


class ForecastSumError(ValueError):
    """Raised when a forecast's probabilities are too far from summing to 1 to normalise."""


class Stage1Forecast(BaseModel):
    """Blind forecast for one game. Never sees odds or anything derived from them."""

    model_config = ConfigDict(extra="forbid")

    game_id: str
    p_home: float = Field(ge=0.0, le=1.0)
    p_away: float = Field(ge=0.0, le=1.0)
    p_draw: float | None = Field(default=None, ge=0.0, le=1.0)
    notes: str = Field(default="", max_length=300)

    def probabilities(self) -> dict[str, float]:
        probs = {"home": self.p_home, "away": self.p_away}
        if self.p_draw is not None:
            probs["draw"] = self.p_draw
        return probs


class Stage2Bet(BaseModel):
    """One proposed bet. ``p_revised`` measures anchoring and drives `kelly_revised`."""

    model_config = ConfigDict(extra="forbid")

    game_id: str
    selection: Selection
    stake_units: float = Field(ge=0.0)
    p_revised: float = Field(ge=0.0, le=1.0)
    reason: str = Field(default="", max_length=300)


class Stage2Decision(BaseModel):
    """A slate's decisions. An empty ``bets`` list is always valid and never an error."""

    model_config = ConfigDict(extra="forbid")

    bets: list[Stage2Bet] = Field(default_factory=list)
    day_notes: str = Field(default="", max_length=500)


def normalize_probabilities(
    probs: dict[str, float],
    tolerance: float = PROBABILITY_SUM_TOLERANCE,
) -> dict[str, float]:
    """Normalise a forecast to sum to exactly 1, if it is close enough to start with.

    Raises :class:`ForecastSumError` when the sum is further than ``tolerance``
    from 1 -- the caller logs that as a failed call rather than silently
    rescaling a forecast the model did not intend.
    """
    if not probs:
        raise ForecastSumError("forecast has no outcomes")
    total = sum(probs.values())
    if total <= 0.0:
        raise ForecastSumError(f"probabilities sum to {total}, cannot normalise")
    if abs(total - 1.0) > tolerance + _TOLERANCE_EPS:
        raise ForecastSumError(
            f"probabilities sum to {total:.4f}, outside tolerance {tolerance} of 1.0"
        )
    return {outcome: p / total for outcome, p in probs.items()}
