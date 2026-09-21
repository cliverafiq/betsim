"""Parsing The Odds API payloads into domain objects.

Pure functions, no network. The HTTP client lives in :mod:`betsim.oddsapi`; this
module is what turns its JSON into something the rest of the experiment can use,
and it is where the API's two sharp edges are handled:

* ``outcomes[].name`` is the **team name**, not ``"home"``/``"away"``. Selections
  are resolved against the event's ``home_team`` and ``away_team``.
* ``scores[].score`` is a **string**, and ``scores`` is a list keyed by team name
  which is ``null`` until a game starts.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

DRAW_NAMES = frozenset({"draw", "tie"})


class IngestError(ValueError):
    """A payload did not have the shape the API documents."""


@dataclass(frozen=True, slots=True)
class IngestedGame:
    id: str
    sport_key: str
    home_team: str
    away_team: str
    commence_utc: datetime


@dataclass(frozen=True, slots=True)
class IngestedPrice:
    game_id: str
    bookmaker: str
    market: str
    selection: str
    price_decimal: float
    last_update: datetime | None


@dataclass(frozen=True, slots=True)
class IngestedScore:
    game_id: str
    completed: bool
    home_score: int | None
    away_score: int | None


def parse_utc(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime."""
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise IngestError(f"could not parse timestamp {value!r}: {exc}") from exc
    if parsed.tzinfo is None:
        raise IngestError(f"timestamp {value!r} is missing a timezone")
    return parsed


def _require(event: Mapping[str, Any], key: str) -> Any:
    if key not in event:
        raise IngestError(f"event is missing required field {key!r}")
    return event[key]


def parse_event(event: Mapping[str, Any]) -> IngestedGame:
    """Parse one entry from /events or /odds into a game."""
    commence = parse_utc(_require(event, "commence_time"))
    if commence is None:
        raise IngestError("commence_time must not be null")
    return IngestedGame(
        id=str(_require(event, "id")),
        sport_key=str(_require(event, "sport_key")),
        home_team=str(_require(event, "home_team")),
        away_team=str(_require(event, "away_team")),
        commence_utc=commence,
    )


def parse_events(payload: Sequence[Mapping[str, Any]]) -> list[IngestedGame]:
    """Parse a /events response. This endpoint is free, so it carries the schedule."""
    return [parse_event(event) for event in payload]


def outcome_to_selection(name: str, home_team: str, away_team: str) -> str:
    """Map an outcome's team name onto ``home``/``away``/``draw``.

    Falls back to a case- and whitespace-insensitive comparison before giving up,
    because a bookmaker feed occasionally differs from the event header in
    punctuation or spacing. An unresolvable name raises rather than guessing --
    silently mislabelling a selection would invert a bet.
    """
    if name == home_team:
        return "home"
    if name == away_team:
        return "away"
    if name.strip().lower() in DRAW_NAMES:
        return "draw"

    def norm(value: str) -> str:
        return " ".join(value.split()).casefold()

    if norm(name) == norm(home_team):
        return "home"
    if norm(name) == norm(away_team):
        return "away"
    raise IngestError(
        f"outcome {name!r} matches neither {home_team!r} nor {away_team!r} and is not a draw"
    )


def parse_odds(
    payload: Sequence[Mapping[str, Any]],
    *,
    market: str = "h2h",
    bookmakers: Sequence[str] | None = None,
) -> tuple[list[IngestedGame], list[IngestedPrice]]:
    """Parse an /odds response into games and one price row per outcome per book.

    Every bookmaker returned is kept, because the de-vigged consensus needs the
    full cross-section; ``bookmakers`` narrows that only when explicitly asked.
    """
    wanted = {b.casefold() for b in bookmakers} if bookmakers else None
    games: list[IngestedGame] = []
    prices: list[IngestedPrice] = []

    for event in payload:
        game = parse_event(event)
        games.append(game)
        for book in event.get("bookmakers") or []:
            key = str(book.get("key", ""))
            if wanted is not None and key.casefold() not in wanted:
                continue
            last_update = parse_utc(book.get("last_update"))
            for mkt in book.get("markets") or []:
                if mkt.get("key") != market:
                    continue
                for outcome in mkt.get("outcomes") or []:
                    if "name" not in outcome or "price" not in outcome:
                        raise IngestError(f"malformed outcome in {key!r}: {outcome!r}")
                    prices.append(
                        IngestedPrice(
                            game_id=game.id,
                            bookmaker=key,
                            market=market,
                            selection=outcome_to_selection(
                                str(outcome["name"]), game.home_team, game.away_team
                            ),
                            price_decimal=float(outcome["price"]),
                            last_update=last_update,
                        )
                    )
    return games, prices


def _score_for(scores: Sequence[Mapping[str, Any]], team: str) -> int | None:
    """Pull one team's score out of the list, tolerating the string encoding."""
    for entry in scores:
        if str(entry.get("name", "")) == team:
            raw = entry.get("score")
            if raw is None:
                return None
            try:
                return int(str(raw))
            except ValueError as exc:
                raise IngestError(f"score {raw!r} for {team!r} is not an integer") from exc
    return None


def parse_scores(payload: Sequence[Mapping[str, Any]]) -> list[IngestedScore]:
    """Parse a /scores response.

    Games that have not started carry ``scores: null``; those come back with
    ``None`` on both sides and ``completed`` false, and must not be settled.
    """
    out: list[IngestedScore] = []
    for event in payload:
        home_team = str(_require(event, "home_team"))
        away_team = str(_require(event, "away_team"))
        scores = event.get("scores") or []
        completed = bool(event.get("completed", False))
        home = _score_for(scores, home_team)
        away = _score_for(scores, away_team)
        if completed and (home is None or away is None):
            raise IngestError(
                f"game {event.get('id')!r} is marked completed but is missing a score"
            )
        out.append(
            IngestedScore(
                game_id=str(_require(event, "id")),
                completed=completed,
                home_score=home,
                away_score=away,
            )
        )
    return out
