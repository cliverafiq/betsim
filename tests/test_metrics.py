import math

import pytest

from betsim.metrics import (
    bootstrap_ci,
    brier_score,
    calibration_table,
    clv,
    log_loss,
    max_drawdown,
    percentile,
    roi,
)


def test_brier_uniform_two_way_is_one_half():
    assert brier_score({"home": 0.5, "away": 0.5}, "home") == pytest.approx(0.5)


def test_brier_uniform_three_way_is_two_thirds():
    # The sum-over-outcomes convention makes the scale depend on market width, so
    # NHL (2-way) numbers are NOT comparable to the 3-way soccer literature.
    third = 1 / 3
    probs = {"home": third, "away": third, "draw": third}
    assert brier_score(probs, "home") == pytest.approx(2 / 3)


def test_brier_perfect_and_worst():
    assert brier_score({"home": 1.0, "away": 0.0}, "home") == pytest.approx(0.0)
    assert brier_score({"home": 0.0, "away": 1.0}, "home") == pytest.approx(2.0)


def test_brier_rewards_the_better_forecast():
    good = brier_score({"home": 0.8, "away": 0.2}, "home")
    bad = brier_score({"home": 0.4, "away": 0.6}, "home")
    assert good < bad


def test_brier_rejects_an_outcome_it_did_not_forecast():
    with pytest.raises(ValueError, match="not among forecast outcomes"):
        brier_score({"home": 0.5, "away": 0.5}, "draw")


def test_log_loss_known_values():
    assert log_loss({"home": 0.5, "away": 0.5}, "home") == pytest.approx(math.log(2))
    assert log_loss({"home": 1.0, "away": 0.0}, "home") == pytest.approx(0.0, abs=1e-12)


def test_log_loss_stays_finite_on_a_confident_miss():
    value = log_loss({"home": 0.0, "away": 1.0}, "home")
    assert math.isfinite(value) and value > 30


def test_roi_basic():
    assert roi([100, -50], [1_000, 1_000]) == pytest.approx(0.025)


def test_roi_is_undefined_with_no_stakes():
    # Zero-bet days are valid and expected, so this is a normal state.
    assert roi([], []) is None
    assert roi([0], [0]) is None


def test_roi_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        roi([1], [1, 2])


def test_max_drawdown():
    assert max_drawdown([100, 120, 60, 90]) == pytest.approx(0.5)
    assert max_drawdown([100, 110, 120]) == pytest.approx(0.0)
    assert max_drawdown([]) == 0.0


def test_clv_sign():
    assert clv(2.10, 2.00) == pytest.approx(0.05)  # beat the close
    assert clv(1.90, 2.00) == pytest.approx(-0.05)  # beaten by the close
    assert clv(2.00, 2.00) == pytest.approx(0.0)


@pytest.mark.parametrize(("taken", "close"), [(1.0, 2.0), (2.0, 1.0), (0.5, 2.0)])
def test_clv_rejects_non_prices(taken, close):
    with pytest.raises(ValueError, match="exceed 1.0"):
        clv(taken, close)


def test_calibration_table_bins_and_counts():
    pairs = [(0.05, False), (0.15, True), (0.95, True), (0.95, False)]
    table = calibration_table(pairs, n_bins=10)
    assert len(table) == 10
    assert table[0].count == 1 and table[0].observed_rate == pytest.approx(0.0)
    assert table[1].count == 1 and table[1].observed_rate == pytest.approx(1.0)
    assert table[9].count == 2 and table[9].observed_rate == pytest.approx(0.5)
    assert math.isnan(table[5].mean_predicted)  # empty bin


def test_calibration_puts_probability_one_in_the_top_bin():
    table = calibration_table([(1.0, True)], n_bins=10)
    assert table[9].count == 1


def test_calibration_detects_overconfidence():
    # The documented LLM failure mode: high stated probabilities that do not
    # verify at the stated rate.
    pairs = [(0.95, i < 6) for i in range(10)]
    table = calibration_table(pairs, n_bins=10)
    top = table[9]
    assert top.mean_predicted > top.observed_rate


def test_calibration_rejects_out_of_range_probabilities():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        calibration_table([(1.5, True)])


def test_percentile():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert percentile(values, 0.0) == 1.0
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 1.0) == 5.0
    assert percentile([7.0], 0.5) == 7.0


def test_percentile_validates():
    with pytest.raises(ValueError):
        percentile([], 0.5)
    with pytest.raises(ValueError):
        percentile([1.0], 1.5)


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    # Day means must actually differ, or there is no between-day variance to
    # resample and the interval is legitimately zero-width (see the test below).
    groups = {day: [0.1 + 0.01 * ((day % 7) - 3)] for day in range(40)}
    observed_mean = _mean([v for values in groups.values() for v in values])
    first = bootstrap_ci(groups, _mean, n_resamples=500, seed=7)
    again = bootstrap_ci(groups, _mean, n_resamples=500, seed=7)
    assert first == again, "same seed must give the same interval"
    lo, hi = first
    assert lo < observed_mean < hi
    assert hi > lo


def test_bootstrap_ci_is_zero_width_without_between_day_variance():
    # Every day has the same mean, so every resample does too. A zero-width
    # interval is the correct answer here, not a bug.
    groups = {day: [0.1 + 0.001 * day, 0.1 - 0.001 * day] for day in range(40)}
    lo, hi = bootstrap_ci(groups, _mean, n_resamples=200, seed=7)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.1)


def test_bootstrap_ci_widens_with_fewer_days():
    many = {d: [float(d % 5)] for d in range(60)}
    few = {d: [float(d % 5)] for d in range(6)}
    lo_m, hi_m = bootstrap_ci(many, _mean, n_resamples=400, seed=1)
    lo_f, hi_f = bootstrap_ci(few, _mean, n_resamples=400, seed=1)
    assert (hi_f - lo_f) > (hi_m - lo_m)


def test_bootstrap_ci_on_no_groups_is_none():
    assert bootstrap_ci({}, _mean, n_resamples=10) is None


def test_bootstrap_ci_skips_undefined_resamples():
    # A statistic that is undefined (no bets drawn) must not poison the interval.
    groups = {0: [1.0], 1: [2.0]}
    result = bootstrap_ci(groups, lambda xs: None if len(xs) < 2 else _mean(xs), n_resamples=50)
    assert result is not None
