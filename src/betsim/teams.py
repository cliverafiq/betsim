"""Matching team names between The Odds API and the NHL API.

The two feeds name the same franchise differently, and a mismatch is silent and
costly: the game gets no context, no Elo rating, and its forecast is worthless.
Real cases in the 2025-26 standings:

* ``Montréal Canadiens`` -- the NHL API uses an accented "é"; The Odds API is ASCII.
* ``St. Louis Blues`` -- punctuation differs between feeds.
* ``Utah Mammoth`` -- the franchise was renamed, so older names need aliasing.

Normalisation therefore strips accents and punctuation, collapses whitespace and
casefolds. An unresolvable name raises: skipping it quietly would drop a game
from the experiment without anyone noticing.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from betsim.nhl import TeamStanding


class UnknownTeam(KeyError):
    """A team name could not be matched to an NHL franchise."""


# Former names and common variants that no longer appear in the standings feed.
ALIASES: dict[str, str] = {
    "utah hockey club": "UTA",
    "utah club": "UTA",
    "arizona coyotes": "UTA",  # relocated to Utah
    "montreal canadians": "MTL",  # frequent misspelling
}


def normalise_team_name(name: str) -> str:
    """Fold a team name to a comparable key.

    ``Montréal Canadiens`` and ``Montreal Canadiens`` both become
    ``montreal canadiens``; ``St. Louis Blues`` and ``St Louis Blues`` both
    become ``st louis blues``.
    """
    decomposed = unicodedata.normalize("NFKD", name)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    alphanumeric = "".join(c if (c.isalnum() or c.isspace()) else " " for c in without_accents)
    return " ".join(alphanumeric.split()).casefold()


@dataclass(frozen=True, slots=True)
class TeamIndex:
    """Two-way index between full team names and NHL tricodes."""

    _by_name: dict[str, str] = field(default_factory=dict)
    _by_tricode: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_standings(cls, standings: Sequence[TeamStanding]) -> TeamIndex:
        """Build the index from the live standings, rather than a hardcoded table.

        The standings feed carries the full name and the tricode together, so the
        mapping stays correct through renames and relocations without a code change.
        """
        if not standings:
            raise ValueError("cannot build a team index from empty standings")
        by_name: dict[str, str] = {}
        by_tricode: dict[str, str] = {}
        for team in standings:
            by_name[normalise_team_name(team.name)] = team.tricode
            by_tricode[team.tricode.upper()] = team.name
        for alias, tricode in ALIASES.items():
            if tricode in by_tricode:
                by_name.setdefault(alias, tricode)
        return cls(by_name, by_tricode)

    def tricode(self, name: str) -> str:
        """Resolve a full team name to its tricode, raising if it cannot be matched."""
        key = normalise_team_name(name)
        try:
            return self._by_name[key]
        except KeyError:
            raise UnknownTeam(
                f"{name!r} (normalised {key!r}) does not match any NHL team. "
                "Add an alias in betsim.teams.ALIASES if the feed renamed it."
            ) from None

    def name(self, tricode: str) -> str:
        try:
            return self._by_tricode[tricode.upper()]
        except KeyError:
            raise UnknownTeam(f"unknown tricode {tricode!r}") from None

    def known(self, name: str) -> bool:
        return normalise_team_name(name) in self._by_name

    def unresolved(self, names: Iterable[str]) -> list[str]:
        """Names that would fail to resolve -- for checking a slate before a run."""
        return [n for n in names if not self.known(n)]

    def __len__(self) -> int:
        return len(self._by_tricode)
