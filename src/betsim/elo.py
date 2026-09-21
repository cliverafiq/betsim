"""Elo baseline.

This is the non-LLM statistical arm. It exists because without one you cannot
tell whether the model is bad or the task is hard -- in KellyBench a plain
Dixon-Coles model beat 3 of the 5 frontier models tested.

Two-way sports only (NHL, NBA). A 3-way soccer version needs an explicit draw
model, which is v2 along with 3-way markets generally.

The constants below are **placeholders**. Tune ``k`` and ``home_advantage`` on
prior seasons only, then freeze them before opening night and record them in
``PREREGISTRATION.md``; tuning them on live results would contaminate the arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class EloConfig:
    k: float = 20.0
    home_advantage: float = 50.0
    initial: float = 1500.0
    # Between seasons, ratings regress toward the mean; teams change.
    season_carryover: float = 0.70


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for A against B on the standard 400-point logistic scale."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


@dataclass(slots=True)
class EloTable:
    cfg: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)

    def rating(self, team: str) -> float:
        return self.ratings.setdefault(team, self.cfg.initial)

    def probability(self, home: str, away: str) -> dict[str, float]:
        """Win probabilities for a two-way market, including home advantage."""
        p_home = expected_score(self.rating(home) + self.cfg.home_advantage, self.rating(away))
        return {"home": p_home, "away": 1.0 - p_home}

    def update(self, home: str, away: str, home_score_actual: float) -> None:
        """Apply one result. ``home_score_actual`` is 1.0 for a home win, 0.0 for away.

        The update is zero-sum: the two ratings move by equal and opposite amounts.
        """
        if not 0.0 <= home_score_actual <= 1.0:
            raise ValueError(f"actual score must be in [0, 1], got {home_score_actual}")
        r_home, r_away = self.rating(home), self.rating(away)
        expected_home = expected_score(r_home + self.cfg.home_advantage, r_away)
        delta = self.cfg.k * (home_score_actual - expected_home)
        self.ratings[home] = r_home + delta
        self.ratings[away] = r_away - delta

    def regress_to_mean(self) -> None:
        """Carry ratings into a new season, regressing toward the league mean."""
        if not self.ratings:
            return
        mean = sum(self.ratings.values()) / len(self.ratings)
        c = self.cfg.season_carryover
        self.ratings = {t: mean + c * (r - mean) for t, r in self.ratings.items()}
