import pytest

from betsim.kelly import DEFAULT_FRACTION, best_outcome, kelly_fraction, kelly_stake_minor


def test_kelly_fraction_known_value():
    # p=0.6 at evens: b=1, q=0.4 -> f = (1*0.6 - 0.4)/1 = 0.2
    assert kelly_fraction(0.6, 2.0) == pytest.approx(0.2)


def test_kelly_fraction_is_zero_at_fair_odds():
    assert kelly_fraction(0.5, 2.0) == pytest.approx(0.0)


def test_kelly_fraction_negative_without_edge():
    assert kelly_fraction(0.4, 2.0) < 0


def test_quarter_kelly_stake():
    # 0.25 * 0.2 * 100_000 = 5_000
    assert kelly_stake_minor(0.6, 2.0, 100_000) == 5_000
    assert DEFAULT_FRACTION == 0.25


def test_no_stake_without_edge():
    assert kelly_stake_minor(0.4, 2.0, 100_000) == 0
    assert kelly_stake_minor(0.5, 2.0, 100_000) == 0


def test_stake_rounds_down():
    # f = (0.9*0.55 - 0.45)/0.9 = 0.05; 0.25 * 0.05 * 1001 = 12.5125 -> 12
    assert kelly_stake_minor(0.55, 1.9, 1001) == 12


def test_zero_balance_stakes_nothing():
    assert kelly_stake_minor(0.9, 2.0, 0) == 0


@pytest.mark.parametrize("bad_p", [-0.1, 1.1])
def test_rejects_out_of_range_probability(bad_p):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        kelly_fraction(bad_p, 2.0)


@pytest.mark.parametrize("bad_fraction", [0.0, -0.5, 1.5])
def test_rejects_bad_kelly_fraction(bad_fraction):
    with pytest.raises(ValueError, match="Kelly fraction"):
        kelly_stake_minor(0.6, 2.0, 100_000, fraction=bad_fraction)


def test_best_outcome_picks_largest_positive_edge():
    probs = {"home": 0.60, "away": 0.40}
    prices = {"home": 2.00, "away": 2.60}
    outcome, f = best_outcome(probs, prices)
    assert outcome == "home"
    assert f == pytest.approx(0.2)


def test_best_outcome_returns_none_without_an_edge():
    # No bet is a normal, expected result -- not a failure.
    assert best_outcome({"home": 0.5, "away": 0.5}, {"home": 1.91, "away": 1.91}) is None


def test_best_outcome_ignores_outcomes_without_a_price():
    assert best_outcome({"home": 0.9, "away": 0.1}, {"away": 5.0}) is None
