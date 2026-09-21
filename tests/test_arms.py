from datetime import UTC, datetime, timedelta

import pytest

from betsim.arms import elo_probabilities, favourite_slate, kelly_slate, random_slate
from betsim.config import FLAT_STAKE_MINOR
from betsim.elo import EloTable
from betsim.money import STARTING_BALANCE_MINOR
from betsim.tiers import Tier
from betsim.validator import GameRef

NOW = datetime(2026, 9, 29, 22, 0, tzinfo=UTC)
EVENS = {"home": 1.91, "away": 1.91}
SAFE = {"home": 1.50, "away": 2.80}


def game(gid, *, starts_in_h=1, prices=None):
    return GameRef(gid, NOW + timedelta(hours=starts_in_h), NOW, prices or EVENS)


# --- Kelly ------------------------------------------------------------------


def test_kelly_stakes_only_where_there_is_an_edge():
    games = [game("g1", prices=EVENS), game("g2", starts_in_h=2, prices=EVENS)]
    probs = {"g1": {"home": 0.65, "away": 0.35}, "g2": {"home": 0.50, "away": 0.50}}
    bets = kelly_slate(games, probs, balance_minor=STARTING_BALANCE_MINOR)
    assert [b.game_id for b in bets] == ["g1"]
    assert bets[0].selection == "home"
    assert bets[0].p_blind == 0.65


def test_kelly_respects_the_tier_cap():
    # A huge edge on a SAFE price would stake far more than 5% without the cap.
    games = [game("g1", prices=SAFE)]
    probs = {"g1": {"home": 0.95, "away": 0.05}}
    bets = kelly_slate(games, probs, balance_minor=STARTING_BALANCE_MINOR)
    assert bets[0].tier is Tier.SAFE
    assert bets[0].stake_minor == 5_000  # exactly the 5% cap


def test_kelly_respects_the_exposure_cap():
    games = [game(f"g{i}", starts_in_h=i + 1, prices=SAFE) for i in range(8)]
    probs = {f"g{i}": {"home": 0.95, "away": 0.05} for i in range(8)}
    bets = kelly_slate(games, probs, balance_minor=STARTING_BALANCE_MINOR)
    assert sum(b.stake_minor for b in bets) == 20_000  # 20% of bankroll
    assert len(bets) == 4


def test_kelly_processes_games_in_start_time_order():
    # Deterministic, rather than dependent on dictionary ordering.
    games = [game("late", starts_in_h=9, prices=SAFE), game("early", starts_in_h=1, prices=SAFE)]
    probs = {g: {"home": 0.95, "away": 0.05} for g in ("late", "early")}
    bets = kelly_slate(games, probs, balance_minor=STARTING_BALANCE_MINOR)
    assert [b.game_id for b in bets] == ["early", "late"]


def test_kelly_counts_existing_open_exposure():
    games = [game("g1", prices=SAFE)]
    probs = {"g1": {"home": 0.95, "away": 0.05}}
    bets = kelly_slate(
        games, probs, balance_minor=STARTING_BALANCE_MINOR, open_exposure_minor=19_000
    )
    assert bets[0].stake_minor == 1_000  # only 1,000 of headroom left


def test_kelly_skips_a_game_it_already_has_a_bet_on():
    games = [game("g1", prices=SAFE)]
    probs = {"g1": {"home": 0.95, "away": 0.05}}
    assert (
        kelly_slate(
            games,
            probs,
            balance_minor=STARTING_BALANCE_MINOR,
            open_game_ids=frozenset({"g1"}),
        )
        == []
    )


def test_kelly_skips_a_game_with_no_forecast():
    assert kelly_slate([game("g1")], {}, balance_minor=STARTING_BALANCE_MINOR) == []


def test_a_bust_arm_stakes_nothing():
    games = [game("g1", prices=SAFE)]
    probs = {"g1": {"home": 0.95, "away": 0.05}}
    assert kelly_slate(games, probs, balance_minor=50) == []


def test_kelly_skips_stakes_below_one_unit():
    # A sliver of an edge on a tiny bankroll rounds below the minimum stake.
    games = [game("g1", prices=EVENS)]
    probs = {"g1": {"home": 0.51, "away": 0.49}}
    assert kelly_slate(games, probs, balance_minor=1_000) == []


# --- flat arms --------------------------------------------------------------


def test_favourite_backs_the_shortest_price():
    bets = favourite_slate([game("g1", prices=SAFE)])
    assert bets[0].selection == "home"
    assert bets[0].stake_minor == FLAT_STAKE_MINOR


def test_flat_arms_ignore_the_tier_and_exposure_caps():
    # They are reference lines, not managed bankrolls. Ten risky bets at 10 units
    # each would breach both caps for a managed arm.
    longshot = {"home": 5.0, "away": 1.15}
    games = [game(f"g{i}", starts_in_h=i + 1, prices=longshot) for i in range(10)]
    bets = favourite_slate(games)
    assert len(bets) == 10
    assert sum(b.stake_minor for b in bets) == 10 * FLAT_STAKE_MINOR
    assert all(b.stake_minor == FLAT_STAKE_MINOR for b in bets)


def test_random_is_reproducible_for_a_seed():
    games = [game(f"g{i}", starts_in_h=i + 1) for i in range(6)]
    first = [b.selection for b in random_slate(games, seed=7)]
    again = [b.selection for b in random_slate(games, seed=7)]
    assert first == again


def test_random_differs_between_seeds():
    games = [game(f"g{i}", starts_in_h=i + 1) for i in range(12)]
    a = [b.selection for b in random_slate(games, seed=1)]
    b = [b.selection for b in random_slate(games, seed=2)]
    assert a != b


def test_random_pick_does_not_depend_on_slate_order():
    # Seeded per game id, so re-running a slate in a different order is stable.
    games = [game(f"g{i}", starts_in_h=i + 1) for i in range(6)]
    forward = {b.game_id: b.selection for b in random_slate(games, seed=3)}
    backward = {b.game_id: b.selection for b in random_slate(list(reversed(games)), seed=3)}
    assert forward == backward


def test_random_actually_picks_both_sides():
    games = [game(f"g{i}", starts_in_h=i + 1) for i in range(30)]
    picks = {b.selection for b in random_slate(games, seed=11)}
    assert picks == {"home", "away"}


# --- Elo --------------------------------------------------------------------


def test_elo_probabilities_are_keyed_by_game():
    table = EloTable()
    table.ratings.update({"CAR": 1600.0, "FLA": 1500.0})
    probs = elo_probabilities([game("g1")], table, {"g1": ("CAR", "FLA")})
    assert probs["g1"]["home"] > 0.5
    assert sum(probs["g1"].values()) == pytest.approx(1.0)


def test_elo_skips_games_it_has_no_mapping_for():
    assert elo_probabilities([game("g1")], EloTable(), {}) == {}


def test_elo_feeds_kelly_like_any_other_probability_source():
    table = EloTable()
    table.ratings.update({"CAR": 1750.0, "FLA": 1400.0})
    probs = elo_probabilities([game("g1", prices=EVENS)], table, {"g1": ("CAR", "FLA")})
    bets = kelly_slate([game("g1", prices=EVENS)], probs, balance_minor=STARTING_BALANCE_MINOR)
    assert bets and bets[0].selection == "home"


# --- line shopping ----------------------------------------------------------


def test_best_price_takes_the_shortest_selection_at_the_longest_price():
    from betsim.arms import best_price_slate

    g = game("g1", prices={"home": 1.50, "away": 2.80})  # designated book
    best = {"g1": {"home": (1.62, "bovada"), "away": (2.90, "betmgm")}}
    (bet,) = best_price_slate([g], best)
    # Same selection as `fav` -- the pair isolates line shopping alone.
    assert bet.selection == "home"
    assert bet.price_decimal == 1.62
    assert "bovada" in bet.reason


def test_best_price_matches_fav_when_no_book_beats_the_designated_one():
    from betsim.arms import best_price_slate

    g = game("g1", prices={"home": 1.50, "away": 2.80})
    best = {"g1": {"home": (1.50, "draftkings")}}
    (shopped,) = best_price_slate([g], best)
    (flat,) = favourite_slate([g])
    assert shopped.price_decimal == flat.price_decimal
    assert shopped.selection == flat.selection


def test_best_price_skips_a_game_with_no_shopped_price():
    from betsim.arms import best_price_slate

    assert best_price_slate([game("g1")], {}) == []
