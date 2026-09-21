import pytest
from pydantic import ValidationError

from betsim.models import (
    ForecastSumError,
    Stage1Forecast,
    Stage2Bet,
    Stage2Decision,
    normalize_probabilities,
)


def test_zero_bets_is_a_valid_decision():
    # The prompts must never require a minimum number of bets, and the schema
    # must not either. Selectivity is a measurement, not a failure.
    decision = Stage2Decision()
    assert decision.bets == []
    assert Stage2Decision.model_validate({"bets": [], "day_notes": "no edge today"}).bets == []


def test_stage1_forecast_round_trip():
    f = Stage1Forecast(game_id="g1", p_home=0.55, p_away=0.45, notes="rested goalie")
    assert f.probabilities() == {"home": 0.55, "away": 0.45}


def test_draw_is_only_present_when_forecast():
    two_way = Stage1Forecast(game_id="g1", p_home=0.55, p_away=0.45)
    three_way = Stage1Forecast(game_id="g2", p_home=0.4, p_away=0.3, p_draw=0.3)
    assert "draw" not in two_way.probabilities()
    assert three_way.probabilities()["draw"] == 0.3


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_probabilities_must_be_in_range(bad):
    with pytest.raises(ValidationError):
        Stage1Forecast(game_id="g1", p_home=bad, p_away=0.5)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Stage1Forecast(game_id="g1", p_home=0.5, p_away=0.5, p_market=0.52)


def test_notes_are_length_capped():
    with pytest.raises(ValidationError):
        Stage1Forecast(game_id="g1", p_home=0.5, p_away=0.5, notes="x" * 301)


def test_stage2_bet_rejects_an_unknown_selection():
    with pytest.raises(ValidationError):
        Stage2Bet(game_id="g1", selection="over", stake_units=10, p_revised=0.5)


def test_stage2_bet_rejects_a_negative_stake():
    with pytest.raises(ValidationError):
        Stage2Bet(game_id="g1", selection="home", stake_units=-1, p_revised=0.5)


def test_normalize_within_tolerance():
    out = normalize_probabilities({"home": 0.50, "away": 0.51})
    assert sum(out.values()) == pytest.approx(1.0)
    assert out["home"] == pytest.approx(0.50 / 1.01)


def test_already_normalised_is_unchanged():
    out = normalize_probabilities({"home": 0.6, "away": 0.4})
    assert out == pytest.approx({"home": 0.6, "away": 0.4})


def test_normalize_rejects_sums_outside_tolerance():
    # Beyond 0.02 the caller logs a failed call rather than silently rescaling a
    # forecast the model did not intend.
    with pytest.raises(ForecastSumError, match="outside tolerance"):
        normalize_probabilities({"home": 0.5, "away": 0.6})


def test_normalize_rejects_degenerate_input():
    with pytest.raises(ForecastSumError):
        normalize_probabilities({})
    with pytest.raises(ForecastSumError):
        normalize_probabilities({"home": 0.0, "away": 0.0})


def test_tolerance_boundary():
    normalize_probabilities({"home": 0.5, "away": 0.52})  # sum 1.02, exactly at tolerance
    with pytest.raises(ForecastSumError):
        normalize_probabilities({"home": 0.5, "away": 0.531})
