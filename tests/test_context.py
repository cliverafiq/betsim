import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from betsim.context import (
    OddsLeakError,
    assert_no_odds_leak,
    build_context,
    context_hash,
    index_results_by_team,
    seed_elo_from_results,
)
from betsim.elo import EloTable
from betsim.ingest import IngestedGame, parse_odds
from betsim.teams import UnknownTeam


@pytest.fixture
def results(nhl_results):
    return nhl_results


@pytest.fixture
def history(results):
    return index_results_by_team(results)


@pytest.fixture
def standings_by_code(standings):
    return {t.tricode: t for t in standings}


@pytest.fixture
def game(odds_payload):
    games, _ = parse_odds(odds_payload)
    return games[0]  # Carolina v Florida


@pytest.fixture
def col_game(history):
    """Colorado v Boston -- both have 82 recorded games, unlike the Odds API
    fixture's Carolina v Florida, which appear only as COL/BOS opponents."""
    last = max(g.start_utc for g in history["COL"])
    return IngestedGame(
        id="nhl_col_bos",
        sport_key="icehockey_nhl",
        home_team="Colorado Avalanche",
        away_team="Boston Bruins",
        commence_utc=last + timedelta(days=2),
    )


@pytest.fixture
def context(game, team_index, standings_by_code, history):
    return build_context(
        game, index=team_index, standings=standings_by_code, results_by_team=history
    )


# --- the leak guard ---------------------------------------------------------


def test_a_clean_context_passes(context):
    assert_no_odds_leak(context)


def test_an_unlisted_key_is_rejected(context):
    # An allowlist fails closed: anything new must be named deliberately.
    context["closing_price"] = 1.91
    with pytest.raises(OddsLeakError, match="not on the Stage 1 allowlist"):
        assert_no_odds_leak(context)


def test_a_nested_unlisted_key_is_rejected(context):
    context["home"]["implied_probability"] = 0.52
    with pytest.raises(OddsLeakError, match="context.home.implied_probability"):
        assert_no_odds_leak(context)


def test_an_unlisted_key_inside_a_list_is_rejected(context):
    context["home"]["recent_games"][0]["market_price"] = 2.0
    with pytest.raises(OddsLeakError, match=r"recent_games\[0\].market_price"):
        assert_no_odds_leak(context)


@pytest.mark.parametrize(
    "leaked", ["favourite by two goals", "the sportsbook has them short", "implied 62%"]
)
def test_betting_vocabulary_in_a_value_is_rejected(context, leaked):
    context["home"]["streak"] = leaked
    with pytest.raises(OddsLeakError, match="betting vocabulary"):
        assert_no_odds_leak(context)


def test_build_context_runs_the_guard_itself(game, team_index, standings_by_code, history):
    # The guard is not optional at the call site.
    assert build_context(
        game, index=team_index, standings=standings_by_code, results_by_team=history
    )


# --- content ----------------------------------------------------------------


def test_context_identifies_the_game(context, game):
    assert context["game_id"] == game.id
    assert context["sport"] == "NHL"
    assert context["home"]["team"] == "Carolina Hurricanes"
    assert context["away"]["team"] == "Florida Panthers"
    assert context["home"]["tricode"] == "CAR"


def test_context_carries_standings_but_nothing_price_derived(context):
    home = context["home"]
    assert home["record"] == "53-22-7"
    assert home["points"] == 113
    assert home["goal_differential"] == home["goals_for"] - home["goals_against"]


def test_recent_games_are_newest_first_and_limited(
    col_game, team_index, standings_by_code, history
):
    ctx = build_context(
        col_game, index=team_index, standings=standings_by_code, results_by_team=history, recent=3
    )
    recent = ctx["home"]["recent_games"]
    assert len(recent) == 3
    dates = [r["date"] for r in recent]
    assert dates == sorted(dates, reverse=True)


def test_recent_games_carry_their_season(col_game, team_index, standings_by_code, history):
    context = build_context(
        col_game, index=team_index, standings=standings_by_code, results_by_team=history
    )
    # On opening night the most recent hockey was LAST season, against a roster
    # that has changed. Without this stamp the model reads it as current form.
    assert context["home"]["recent_games"][0]["season"] == 20252026


def test_only_games_before_kickoff_are_included(game, team_index, standings_by_code, history):
    early = replace(game, commence_utc=datetime(2025, 10, 10, tzinfo=UTC))
    ctx = build_context(
        early, index=team_index, standings=standings_by_code, results_by_team=history
    )
    for entry in ctx["home"]["recent_games"] + ctx["away"]["recent_games"]:
        assert entry["date"] < "2025-10-10"


def test_a_team_with_no_prior_games_has_no_rest_days(team_index, standings_by_code):
    game = IngestedGame(
        id="g",
        sport_key="icehockey_nhl",
        home_team="Carolina Hurricanes",
        away_team="Florida Panthers",
        commence_utc=datetime(2020, 1, 1, tzinfo=UTC),
    )
    ctx = build_context(game, index=team_index, standings=standings_by_code, results_by_team={})
    assert ctx["home"]["rest_days"] is None
    assert ctx["home"]["recent_games"] == []


def test_rest_days_counts_from_the_previous_game(col_game, team_index, standings_by_code, history):
    ctx = build_context(
        col_game, index=team_index, standings=standings_by_code, results_by_team=history
    )
    assert ctx["home"]["rest_days"] == 2


def test_an_unresolvable_team_fails_loudly(team_index, standings_by_code, history):
    game = IngestedGame(
        id="g",
        sport_key="icehockey_nhl",
        home_team="Quebec Nordiques",
        away_team="Florida Panthers",
        commence_utc=datetime(2026, 9, 29, tzinfo=UTC),
    )
    with pytest.raises(UnknownTeam):
        build_context(game, index=team_index, standings=standings_by_code, results_by_team=history)


def test_a_missing_standings_row_fails_loudly(game, team_index, history):
    with pytest.raises(KeyError, match="no standings row"):
        build_context(game, index=team_index, standings={}, results_by_team=history)


# --- hashing ----------------------------------------------------------------


def test_hash_is_stable_across_serialisation(context):
    assert context_hash(context) == context_hash(json.loads(json.dumps(context)))


def test_hash_is_order_independent(context):
    reordered = dict(reversed(list(context.items())))
    assert context_hash(reordered) == context_hash(context)


def test_hash_changes_when_content_changes(context):
    before = context_hash(context)
    context["home"]["points"] += 1
    assert context_hash(context) != before


# --- Elo seeding ------------------------------------------------------------


def test_seeding_walks_every_game_once(results):
    # COL and BOS played each other, so the two recorded schedules overlap and
    # the unique count is genuinely lower than the row count.
    unique = len({r.id for r in results})
    assert unique < len(results)
    table = EloTable()
    assert seed_elo_from_results(results, table) == unique


def test_seeding_de_duplicates_repeated_games(results):
    # Each game appears in both clubs' schedules, so the seed must dedupe by id.
    table = EloTable()
    unique = len({r.id for r in results})
    assert seed_elo_from_results(list(results) + list(results), table) == unique


def test_seeding_is_order_independent_of_input(results):
    forward, shuffled = EloTable(), EloTable()
    seed_elo_from_results(results, forward)
    seed_elo_from_results(list(reversed(results)), shuffled)
    assert forward.ratings == pytest.approx(shuffled.ratings)


def test_seeding_conserves_the_mean_rating(results):
    table = EloTable()
    seed_elo_from_results(results, table)
    mean = sum(table.ratings.values()) / len(table.ratings)
    assert mean == pytest.approx(1500.0)


def test_seeding_separates_good_teams_from_bad(results, standings_by_code):
    table = EloTable()
    seed_elo_from_results(results, table)
    ranked = sorted(table.ratings.items(), key=lambda kv: -kv[1])
    best, worst = ranked[0][0], ranked[-1][0]
    assert standings_by_code[best].points > standings_by_code[worst].points
