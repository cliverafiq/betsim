"""NHL official API client and parsing.

``api-web.nhle.com`` is free, needs no key and has no quota, which is why it is
the context source for v1. Two things about it matter:

* ``/v1/standings/now`` answers **307** and redirects to a dated endpoint, so the
  client must follow redirects. Before a season opens it redirects to the *end of
  the previous season*, which is exactly what seeds Elo.
* Many strings are localised as ``{"default": "..."}`` rather than plain strings.

Only ``/standings`` and ``/club-schedule-season`` are used. ``/v1/score/{date}``
is deliberately avoided: it carries an ``oddsPartners`` field, and nothing
odds-derived may come near the blind Stage 1 context.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Self

import httpx

NHL_BASE = "https://api-web.nhle.com"
DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 2

# gameType 1 is preseason, 2 regular season, 3 playoffs. Elo is seeded on
# regular-season games only -- preseason lineups are not representative.
REGULAR_SEASON = 2
FINISHED_STATES = frozenset({"OFF", "FINAL"})


class NhlApiError(RuntimeError):
    """The NHL API could not be reached or returned an unusable response."""


class NhlParseError(ValueError):
    """A payload did not have the shape the API returns."""


def localised(value: Any, field: str = "value") -> str:
    """Unwrap the API's ``{"default": "..."}`` localisation envelope."""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping) and "default" in value:
        return str(value["default"])
    raise NhlParseError(f"{field} is not a localised string: {value!r}")


@dataclass(frozen=True, slots=True)
class TeamStanding:
    tricode: str
    name: str
    games_played: int
    wins: int
    losses: int
    ot_losses: int
    points: int
    goals_for: int
    goals_against: int
    goal_differential: int
    l10_wins: int
    l10_losses: int
    l10_ot_losses: int
    streak_code: str
    streak_count: int
    division: str
    conference: str

    @property
    def record(self) -> str:
        return f"{self.wins}-{self.losses}-{self.ot_losses}"

    @property
    def last_10(self) -> str:
        return f"{self.l10_wins}-{self.l10_losses}-{self.l10_ot_losses}"

    @property
    def streak(self) -> str:
        return f"{self.streak_code}{self.streak_count}" if self.streak_code else "none"


@dataclass(frozen=True, slots=True)
class GoalieLine:
    name: str
    games_played: int
    goals_against_average: float | None
    save_pct: float | None
    record: str
    shutouts: int


@dataclass(frozen=True, slots=True)
class TeamForm:
    """Season-level team stats with league ranks, from a stated season."""

    season: int
    goals_for_per_game: float | None
    goals_for_rank: int | None
    goals_against_per_game: float | None
    goals_against_rank: int | None
    power_play_pct: float | None
    power_play_rank: int | None
    penalty_kill_pct: float | None
    penalty_kill_rank: int | None
    faceoff_win_pct: float | None
    faceoff_rank: int | None


@dataclass(frozen=True, slots=True)
class NhlGameResult:
    id: int
    season: int
    game_type: int
    start_utc: datetime
    home_tricode: str
    away_tricode: str
    home_score: int
    away_score: int
    last_period_type: str

    @property
    def home_won(self) -> bool:
        return self.home_score > self.away_score

    @property
    def went_beyond_regulation(self) -> bool:
        """NHL moneylines include overtime and the shootout, so these still settle."""
        return self.last_period_type in {"OT", "SO"}


def _int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise NhlParseError(f"{key}={value!r} is not an integer") from exc


def parse_standings(payload: Mapping[str, Any]) -> list[TeamStanding]:
    """Parse /v1/standings. ``teamName.default`` is the full name, which is what
    The Odds API also uses, so this doubles as the team-name index source."""
    rows = payload.get("standings")
    if not isinstance(rows, list):
        raise NhlParseError("standings payload has no 'standings' list")
    out = []
    for row in rows:
        out.append(
            TeamStanding(
                tricode=localised(row.get("teamAbbrev"), "teamAbbrev"),
                name=localised(row.get("teamName"), "teamName"),
                games_played=_int(row, "gamesPlayed"),
                wins=_int(row, "wins"),
                losses=_int(row, "losses"),
                ot_losses=_int(row, "otLosses"),
                points=_int(row, "points"),
                goals_for=_int(row, "goalFor"),
                goals_against=_int(row, "goalAgainst"),
                goal_differential=_int(row, "goalDifferential"),
                l10_wins=_int(row, "l10Wins"),
                l10_losses=_int(row, "l10Losses"),
                l10_ot_losses=_int(row, "l10OtLosses"),
                streak_code=str(row.get("streakCode") or ""),
                streak_count=_int(row, "streakCount"),
                division=localised(row.get("divisionName", ""), "divisionName"),
                conference=localised(row.get("conferenceName", ""), "conferenceName"),
            )
        )
    return out


def parse_club_schedule(
    payload: Mapping[str, Any],
    *,
    game_types: Sequence[int] = (REGULAR_SEASON,),
) -> list[NhlGameResult]:
    """Parse finished games from /v1/club-schedule-season.

    Unfinished games are skipped rather than returned with partial scores, and
    a finished game missing a score raises -- a silently dropped result would
    leave Elo mis-seeded.
    """
    games = payload.get("games")
    if not isinstance(games, list):
        raise NhlParseError("schedule payload has no 'games' list")
    wanted = set(game_types)
    out = []
    for game in games:
        if game.get("gameType") not in wanted:
            continue
        if game.get("gameState") not in FINISHED_STATES:
            continue
        home, away = game.get("homeTeam") or {}, game.get("awayTeam") or {}
        if home.get("score") is None or away.get("score") is None:
            raise NhlParseError(f"finished game {game.get('id')!r} is missing a score")
        start = game.get("startTimeUTC")
        try:
            start_utc = datetime.fromisoformat(str(start))
        except ValueError as exc:
            raise NhlParseError(f"bad startTimeUTC {start!r}") from exc
        out.append(
            NhlGameResult(
                id=_int(game, "id"),
                season=_int(game, "season"),
                game_type=_int(game, "gameType"),
                start_utc=start_utc,
                home_tricode=localised(home.get("abbrev"), "homeTeam.abbrev"),
                away_tricode=localised(away.get("abbrev"), "awayTeam.abbrev"),
                home_score=int(home["score"]),
                away_score=int(away["score"]),
                last_period_type=str((game.get("gameOutcome") or {}).get("lastPeriodType", "")),
            )
        )
    return out


class NhlClient:
    """A thin client for the free NHL web API. No key, no quota."""

    def __init__(
        self,
        *,
        base_url: str = NHL_BASE,
        timeout: float = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.retries = retries
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            # /v1/standings/now answers 307; without this every call returns empty.
            follow_redirects=True,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def standings(self, date: str = "now") -> dict[str, Any]:
        return self._get(f"/v1/standings/{date}")

    def score_for_date(self, date: str) -> dict[str, Any]:
        """/v1/score/{YYYY-MM-DD} -- used only to map games to NHL ids.

        Note this endpoint carries an ``oddsPartners`` field, so only the game
        identifiers are taken from it and nothing else reaches a context.
        """
        return self._get(f"/v1/score/{date}")

    def gamecenter_landing(self, game_id: int) -> dict[str, Any]:
        return self._get(f"/v1/gamecenter/{game_id}/landing")

    def gamecenter_right_rail(self, game_id: int) -> dict[str, Any]:
        return self._get(f"/v1/gamecenter/{game_id}/right-rail")

    def club_schedule_season(self, tricode: str, season: str | int) -> dict[str, Any]:
        """``season`` is YYYYYYYY, e.g. 20252026."""
        return self._get(f"/v1/club-schedule-season/{tricode.upper()}/{season}")

    def _get(self, path: str) -> dict[str, Any]:
        for attempt in range(self.retries + 1):
            try:
                response = self._client.get(path)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise NhlApiError(f"{path}: transport failure after retries: {exc}") from exc
            if response.status_code >= 500 and attempt < self.retries:
                time.sleep(2**attempt)
                continue
            if response.status_code >= 400:
                raise NhlApiError(f"{path}: HTTP {response.status_code}")
            payload = response.json()
            if not isinstance(payload, dict):
                raise NhlApiError(f"{path}: expected a JSON object")
            return payload
        raise NhlApiError(f"{path}: exhausted retries")


def _num(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _rank(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def parse_goalies(payload: Mapping[str, Any]) -> dict[str, tuple[int, list[GoalieLine]]]:
    """Parse ``matchup.goalieComparison`` from /v1/gamecenter/{id}/landing.

    Returns ``{"home"|"away": (season, [GoalieLine, ...])}``. This is the team's
    goalie **depth chart with season statistics** -- it is NOT the confirmed
    starter, which the NHL feed does not publish this far ahead. Callers must
    label it as such: a model shown "39 games played" beside a name will
    otherwise read it as tonight's starter.
    """
    matchup = payload.get("matchup") or {}
    comparison = matchup.get("goalieComparison") or {}
    season = _rank(comparison.get("contextSeason")) or 0
    out: dict[str, tuple[int, list[GoalieLine]]] = {}
    for api_side, side in (("homeTeam", "home"), ("awayTeam", "away")):
        lines = []
        for g in (comparison.get(api_side) or {}).get("leaders") or []:
            lines.append(
                GoalieLine(
                    name=localised(g.get("name"), "goalie.name"),
                    games_played=_rank(g.get("gamesPlayed")) or 0,
                    goals_against_average=_num(g.get("gaa")),
                    save_pct=_num(g.get("savePctg")),
                    record=str(g.get("record") or ""),
                    shutouts=_rank(g.get("shutouts")) or 0,
                )
            )
        out[side] = (season, lines)
    return out


def parse_team_form(payload: Mapping[str, Any]) -> dict[str, TeamForm]:
    """Parse ``teamSeasonStats`` from /v1/gamecenter/{id}/right-rail."""
    stats = payload.get("teamSeasonStats") or {}
    season = _rank(stats.get("contextSeason")) or 0
    out: dict[str, TeamForm] = {}
    for api_side, side in (("homeTeam", "home"), ("awayTeam", "away")):
        s = stats.get(api_side) or {}
        out[side] = TeamForm(
            season=season,
            goals_for_per_game=_num(s.get("goalsForPerGamePlayed")),
            goals_for_rank=_rank(s.get("goalsForPerGamePlayedRank")),
            goals_against_per_game=_num(s.get("goalsAgainstPerGamePlayed")),
            goals_against_rank=_rank(s.get("goalsAgainstPerGamePlayedRank")),
            power_play_pct=_num(s.get("ppPctg")),
            power_play_rank=_rank(s.get("ppPctgRank")),
            penalty_kill_pct=_num(s.get("pkPctg")),
            penalty_kill_rank=_rank(s.get("pkPctgRank")),
            faceoff_win_pct=_num(s.get("faceoffWinningPctg")),
            faceoff_rank=_rank(s.get("faceoffWinningPctgRank")),
        )
    return out


def parse_scratches(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    """Parse ``gameInfo.scratches`` from /v1/gamecenter/{id}/right-rail.

    Empty until close to puck drop, which is exactly why a late context refresh
    is worth scheduling.
    """
    info = payload.get("gameInfo") or {}
    out: dict[str, list[str]] = {}
    for api_side, side in (("homeTeam", "home"), ("awayTeam", "away")):
        names = []
        for entry in (info.get(api_side) or {}).get("scratches") or []:
            first = localised(entry.get("firstName", ""), "scratch.firstName")
            last = localised(entry.get("lastName", ""), "scratch.lastName")
            names.append(f"{first} {last}".strip())
        out[side] = names
    return out


def parse_schedule_ids(payload: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    """Map ``(home_tricode, away_tricode)`` to the NHL game id for one date.

    The Odds API and the NHL use different identifiers for the same game, so
    they have to be matched on teams and date.
    """
    out: dict[tuple[str, str], int] = {}
    for game in payload.get("games") or []:
        home = (game.get("homeTeam") or {}).get("abbrev")
        away = (game.get("awayTeam") or {}).get("abbrev")
        gid = _rank(game.get("id"))
        if home and away and gid:
            out[(str(home), str(away))] = gid
    return out
