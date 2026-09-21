"""Per-sport market and settlement conventions.

These differ in ways that change how a bet grades, so they are data rather than
scattered conditionals:

* NHL and NBA moneylines include overtime -- and the NHL's includes the shootout
  -- so a settled game **cannot** be level. A level final score is a data error.
* NFL moneylines can tie, which pushes and refunds the stake.
* Soccer h2h is 3-way and settles on regulation time, so the draw is a selectable
  outcome rather than a push.
"""

from __future__ import annotations

from dataclasses import dataclass

TWO_WAY: tuple[str, ...] = ("home", "away")
THREE_WAY: tuple[str, ...] = ("home", "away", "draw")


@dataclass(frozen=True, slots=True)
class SportSpec:
    key: str
    market_width: int
    ties_possible: bool
    tie_is_push: bool

    @property
    def selections(self) -> tuple[str, ...]:
        return THREE_WAY if self.market_width == 3 else TWO_WAY

    def validate_selection(self, selection: str) -> bool:
        return selection in self.selections


SPORTS: dict[str, SportSpec] = {
    # v1 primary: 2026-27 season opens 2026-09-29, 1,312 regular-season games.
    "icehockey_nhl": SportSpec("icehockey_nhl", 2, ties_possible=False, tie_is_push=False),
    # v1 secondary: 2026-27 season opens 2026-10-20, 1,230 games.
    "basketball_nba": SportSpec("basketball_nba", 2, ties_possible=False, tie_is_push=False),
    "americanfootball_nfl": SportSpec(
        "americanfootball_nfl", 2, ties_possible=True, tie_is_push=True
    ),
    # v2: 3-way markets, where the published work found the worst miscalibration.
    "soccer_epl": SportSpec("soccer_epl", 3, ties_possible=True, tie_is_push=False),
}


def get_sport(key: str) -> SportSpec:
    try:
        return SPORTS[key]
    except KeyError:
        raise ValueError(f"unknown sport key {key!r}; known: {sorted(SPORTS)}") from None
