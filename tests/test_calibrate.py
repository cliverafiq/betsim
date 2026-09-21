import math
import random

import pytest

from betsim.calibrate import (
    IDENTITY,
    CalibrationError,
    PlattParams,
    calibrate_distribution,
    fit_platt,
    logit,
    sigmoid,
)


def test_logit_and_sigmoid_are_inverses():
    for p in (0.01, 0.25, 0.5, 0.75, 0.99):
        assert sigmoid(logit(p)) == pytest.approx(p)


def test_logit_clamps_the_endpoints_instead_of_diverging():
    assert math.isfinite(logit(0.0))
    assert math.isfinite(logit(1.0))


def test_sigmoid_is_stable_at_large_magnitudes():
    assert sigmoid(-800) == pytest.approx(0.0)
    assert sigmoid(800) == pytest.approx(1.0)


def test_identity_leaves_a_probability_alone():
    assert IDENTITY.apply(0.62) == pytest.approx(0.62)
    assert IDENTITY.is_identity


def _sample(mapping, n=3000, seed=0):
    rng = random.Random(seed)
    stated = list(mapping)
    return [(s, rng.random() < mapping[s]) for s in (rng.choice(stated) for _ in range(n))]


def test_fit_recovers_systematic_overconfidence():
    # The documented LLM failure: high stated probabilities that under-deliver.
    truth = {0.55: 0.52, 0.70: 0.60, 0.90: 0.70}
    params = fit_platt(_sample(truth))
    assert params.a < 1.0, "an overconfident model needs its confidence shrunk"
    for stated, observed in truth.items():
        assert params.apply(stated) == pytest.approx(observed, abs=0.04)


def test_fit_leaves_a_well_calibrated_model_nearly_alone():
    truth = {0.30: 0.30, 0.50: 0.50, 0.70: 0.70}
    params = fit_platt(_sample(truth, n=4000, seed=3))
    for stated in truth:
        assert params.apply(stated) == pytest.approx(stated, abs=0.05)


def test_fit_sharpens_an_underconfident_model():
    truth = {0.45: 0.35, 0.55: 0.65}
    params = fit_platt(_sample(truth, n=4000, seed=5))
    assert params.a > 1.0


def test_fit_needs_both_outcomes():
    with pytest.raises(CalibrationError, match="same outcome"):
        fit_platt([(0.6, True), (0.7, True), (0.8, True)])


def test_fit_needs_enough_observations():
    with pytest.raises(CalibrationError, match="at least 2"):
        fit_platt([(0.6, True)])


def test_apply_validates_its_input():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        IDENTITY.apply(1.5)


def test_calibrate_distribution_renormalises():
    params = PlattParams(a=0.5, b=0.0, n=100)
    out = calibrate_distribution({"home": 0.9, "away": 0.1}, params)
    assert sum(out.values()) == pytest.approx(1.0)
    # Shrinking confidence moves the favourite toward the centre.
    assert out["home"] < 0.9


def test_calibrate_distribution_preserves_ordering():
    params = PlattParams(a=0.4, b=0.1, n=100)
    out = calibrate_distribution({"home": 0.7, "away": 0.3}, params)
    assert out["home"] > out["away"]


def test_params_round_trip_through_a_dict():
    params = PlattParams(a=0.42, b=-0.13, n=250)
    assert PlattParams.from_dict(params.to_dict()) == params
