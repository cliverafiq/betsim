"""Assembling the blind Stage 1 context.

Stage 1 must never see odds, spreads, totals, or anything derived from them --
if the price leaks in, the model's "probability" echoes the market and every
edge estimate downstream is meaningless. That rule is the experiment's single
most load-bearing invariant, so it is enforced in code rather than trusted.

The guard is an **allowlist** of permitted keys, not a denylist of banned words.
A denylist fails open: the first time someone adds a field nobody thought to ban,
it sails through. An allowlist fails closed -- anything new has to be named here
deliberately, which is exactly the review step that should happen.

The payload is stored verbatim alongside its hash, so what the model saw can be
reconstructed exactly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime

from betsim.ingest import IngestedGame
from betsim.nhl import GoalieLine, NhlGameResult, TeamForm, TeamStanding
from betsim.teams import TeamIndex

SCHEMA_VERSION = 2
DEFAULT_RECENT_GAMES = 5

ALLOWED_KEYS = frozenset(
    {
        "schema_version",
        "sport",
        "game_id",
        "start_utc",
        "home",
        "away",
        # team block
        "team",
        "tricode",
        "record",
        "points",
        "games_played",
        "goals_for",
        "goals_against",
        "goal_differential",
        "last_10",
        "streak",
        "division",
        "conference",
        "rest_days",
        "recent_games",
        # recent-game block
        "date",
        "season",
        "opponent",
        "at_home",
        "result",
        "finish",
        # team form: season-level rates with league ranks
        "team_stats",
        "goals_for_per_game",
        "goals_for_rank",
        "goals_against_per_game",
        "goals_against_rank",
        "power_play_pct",
        "power_play_rank",
        "penalty_kill_pct",
        "penalty_kill_rank",
        "faceoff_win_pct",
        "faceoff_rank",
        # goaltending: a depth chart with season stats, not tonight's starter
        "goalies",
        "confirmed_starter",
        "depth",
        "name",
        "goals_against_average",
        "save_pct",
        "shutouts",
        # availability, empty until close to puck drop
        "scratches",
    }
)

# Betting vocabulary that must never appear in a stored string value, as a second
# line of defence behind the key allowlist.
BANNED_VALUE_TERMS = (
    "odds",
    "moneyline",
    "money line",
    "spread",
    "handicap",
    "implied",
    "vig",
    "juice",
    "overround",
    "bookmaker",
    "sportsbook",
    "payout",
    "underdog",
    "favourite",
    "favorite",
    "wager",
    "stake",
)


class OddsLeakError(AssertionError):
    """Odds-derived data reached a blind-stage context."""


def assert_no_odds_leak(payload: object, path: str = "context") -> None:
    """Recursively verify a context payload carries nothing odds-derived."""
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if key not in ALLOWED_KEYS:
                raise OddsLeakError(
                    f"{path}.{key} is not on the Stage 1 allowlist. If it is genuinely "
                    "odds-free, add it to betsim.context.ALLOWED_KEYS deliberately."
                )
            assert_no_odds_leak(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for i, item in enumerate(payload):
            assert_no_odds_leak(item, f"{path}[{i}]")
    elif isinstance(payload, str):
        lowered = payload.casefold()
        for term in BANNED_VALUE_TERMS:
            if term in lowered:
                raise OddsLeakError(f"{path} contains betting vocabulary: {payload!r}")


def context_hash(payload: Mapping[str, object]) -> str:
    """Stable SHA-256 of a context, so what the model saw is reconstructible."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def index_results_by_team(
    results: Sequence[NhlGameResult],
) -> dict[str, list[NhlGameResult]]:
    """Group finished games by team tricode, oldest first, de-duplicated by game id."""
    seen: dict[int, NhlGameResult] = {r.id: r for r in results}
    by_team: dict[str, list[NhlGameResult]] = {}
    for game in sorted(seen.values(), key=lambda g: g.start_utc):
        by_team.setdefault(game.home_tricode, []).append(game)
        by_team.setdefault(game.away_tricode, []).append(game)
    return by_team


def _recent_form(
    tricode: str,
    history: Sequence[NhlGameResult],
    before: datetime,
    limit: int,
) -> list[dict[str, object]]:
    """The team's last ``limit`` finished games before ``before``, newest first.

    Each entry carries its ``season``. On opening night the most recent hockey a
    team played was the *previous* season, against a roster that has since
    changed -- without the season stamp the model would read those as current
    form, and the long ``rest_days`` beside them would look like an anomaly
    rather than an offseason.
    """
    played = [g for g in history if g.start_utc < before]
    out = []
    for game in reversed(played[-limit:]):
        at_home = game.home_tricode == tricode
        scored = game.home_score if at_home else game.away_score
        conceded = game.away_score if at_home else game.home_score
        out.append(
            {
                "date": game.start_utc.date().isoformat(),
                "season": game.season,
                "opponent": game.away_tricode if at_home else game.home_tricode,
                "at_home": at_home,
                "goals_for": scored,
                "goals_against": conceded,
                "result": "W" if scored > conceded else "L",
                "finish": game.last_period_type or "REG",
            }
        )
    return out


def _rest_days(history: Sequence[NhlGameResult], before: datetime) -> int | None:
    played = [g for g in history if g.start_utc < before]
    if not played:
        return None
    return (before - played[-1].start_utc).days


def _goalie_block(
    season: int, lines: Sequence[GoalieLine], starter: str | None
) -> dict[str, object]:
    """Goaltending, stamped with the season the numbers come from.

    ``confirmed_starter`` is explicitly present and usually ``null``: the NHL
    feed does not publish the starter until close to puck drop. Without that
    field a model shown "39 games played" beside a name will read it as
    tonight's starter rather than a depth chart.
    """
    return {
        "season": season,
        "confirmed_starter": starter,
        "depth": [
            {
                "name": g.name,
                "games_played": g.games_played,
                "goals_against_average": g.goals_against_average,
                "save_pct": g.save_pct,
                "record": g.record,
                "shutouts": g.shutouts,
            }
            for g in lines
        ],
    }


def _form_block(form: TeamForm) -> dict[str, object]:
    return {
        "season": form.season,
        "goals_for_per_game": form.goals_for_per_game,
        "goals_for_rank": form.goals_for_rank,
        "goals_against_per_game": form.goals_against_per_game,
        "goals_against_rank": form.goals_against_rank,
        "power_play_pct": form.power_play_pct,
        "power_play_rank": form.power_play_rank,
        "penalty_kill_pct": form.penalty_kill_pct,
        "penalty_kill_rank": form.penalty_kill_rank,
        "faceoff_win_pct": form.faceoff_win_pct,
        "faceoff_rank": form.faceoff_rank,
    }


def _team_block(
    tricode: str,
    standing: TeamStanding,
    history: Sequence[NhlGameResult],
    start_utc: datetime,
    recent: int,
    *,
    goalies: tuple[int, Sequence[GoalieLine]] | None = None,
    form: TeamForm | None = None,
    scratches: Sequence[str] | None = None,
) -> dict[str, object]:
    extra: dict[str, object] = {}
    if goalies is not None:
        extra["goalies"] = _goalie_block(goalies[0], goalies[1], None)
    if form is not None:
        extra["team_stats"] = _form_block(form)
    if scratches is not None:
        extra["scratches"] = list(scratches)
    return {
        **extra,
        "team": standing.name,
        "tricode": tricode,
        "record": standing.record,
        "points": standing.points,
        "games_played": standing.games_played,
        "goals_for": standing.goals_for,
        "goals_against": standing.goals_against,
        "goal_differential": standing.goal_differential,
        "last_10": standing.last_10,
        "streak": standing.streak,
        "division": standing.division,
        "conference": standing.conference,
        "rest_days": _rest_days(history, start_utc),
        "recent_games": _recent_form(tricode, history, start_utc, recent),
    }


def build_context(
    game: IngestedGame,
    *,
    index: TeamIndex,
    standings: Mapping[str, TeamStanding],
    results_by_team: Mapping[str, Sequence[NhlGameResult]],
    recent: int = DEFAULT_RECENT_GAMES,
    goalies: Mapping[str, tuple[int, Sequence[GoalieLine]]] | None = None,
    form: Mapping[str, TeamForm] | None = None,
    scratches: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, object]:
    """Build the blind context for one game.

    ``standings`` is keyed by tricode. Raises :class:`betsim.teams.UnknownTeam`
    if either side cannot be resolved -- a game with no context must fail loudly
    rather than be forecast blind in the wrong sense.
    """
    home_code = index.tricode(game.home_team)
    away_code = index.tricode(game.away_team)
    for code in (home_code, away_code):
        if code not in standings:
            raise KeyError(f"no standings row for {code!r}; cannot build a context")

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "sport": "NHL",
        "game_id": game.id,
        "start_utc": game.commence_utc.isoformat(),
        "home": _team_block(
            home_code,
            standings[home_code],
            results_by_team.get(home_code, []),
            game.commence_utc,
            recent,
            goalies=(goalies or {}).get("home"),
            form=(form or {}).get("home"),
            scratches=(scratches or {}).get("home"),
        ),
        "away": _team_block(
            away_code,
            standings[away_code],
            results_by_team.get(away_code, []),
            game.commence_utc,
            recent,
            goalies=(goalies or {}).get("away"),
            form=(form or {}).get("away"),
            scratches=(scratches or {}).get("away"),
        ),
    }
    assert_no_odds_leak(payload)
    return payload


def seed_elo_from_results(results: Sequence[NhlGameResult], table) -> int:
    """Walk finished games in chronological order, updating Elo. Returns the count.

    NHL moneylines pay in full on an overtime or shootout win, so the target being
    forecast is "wins including OT/SO" and every result counts in full. A partial
    credit for OT/SO wins is a defensible refinement; it belongs in EloConfig and
    must be tuned on prior seasons, never on live results.
    """
    seen: dict[int, NhlGameResult] = {r.id: r for r in results}
    ordered = sorted(seen.values(), key=lambda g: g.start_utc)
    for game in ordered:
        table.update(game.home_tricode, game.away_tricode, 1.0 if game.home_won else 0.0)
    return len(ordered)
