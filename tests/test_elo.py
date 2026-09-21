import pytest

from betsim.elo import EloConfig, EloTable, expected_score


def test_expected_score_is_even_between_equal_ratings():
    assert expected_score(1500, 1500) == pytest.approx(0.5)


def test_expected_score_is_symmetric():
    assert expected_score(1600, 1500) + expected_score(1500, 1600) == pytest.approx(1.0)


def test_400_points_is_ten_to_one():
    assert expected_score(1900, 1500) == pytest.approx(10 / 11, abs=1e-9)


def test_home_advantage_favours_the_home_team_between_equal_sides():
    table = EloTable()
    probs = table.probability("BOS", "MTL")
    assert probs["home"] > 0.5
    assert probs["home"] + probs["away"] == pytest.approx(1.0)


def test_no_home_advantage_means_a_coin_flip():
    table = EloTable(EloConfig(home_advantage=0.0))
    assert table.probability("BOS", "MTL")["home"] == pytest.approx(0.5)


def test_update_is_zero_sum():
    table = EloTable()
    before = table.rating("BOS") + table.rating("MTL")
    table.update("BOS", "MTL", 1.0)
    after = table.ratings["BOS"] + table.ratings["MTL"]
    assert after == pytest.approx(before)


def test_winning_raises_the_winner_and_lowers_the_loser():
    table = EloTable()
    table.update("BOS", "MTL", 1.0)
    assert table.ratings["BOS"] > 1500 > table.ratings["MTL"]


def test_beating_a_stronger_opponent_moves_ratings_more():
    weak, strong = EloTable(), EloTable()
    strong.ratings["MTL"] = 1900
    weak.update("BOS", "MTL", 1.0)
    strong.update("BOS", "MTL", 1.0)
    assert strong.ratings["BOS"] - 1500 > weak.ratings["BOS"] - 1500


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_update_rejects_impossible_scores(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EloTable().update("BOS", "MTL", bad)


def test_regress_to_mean_preserves_the_mean_and_shrinks_the_spread():
    table = EloTable(EloConfig(season_carryover=0.70))
    table.ratings = {"A": 1600.0, "B": 1400.0}
    table.regress_to_mean()
    assert sum(table.ratings.values()) / 2 == pytest.approx(1500.0)
    assert table.ratings["A"] == pytest.approx(1570.0)
    assert table.ratings["B"] == pytest.approx(1430.0)


def test_regress_to_mean_on_an_empty_table_is_a_no_op():
    table = EloTable()
    table.regress_to_mean()
    assert table.ratings == {}
