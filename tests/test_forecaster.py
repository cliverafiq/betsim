import json
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from betsim.context import OddsLeakError
from betsim.forecaster import Stage1Forecaster, summarise
from betsim.models import Stage1Forecast

CONTEXT = {
    "schema_version": 1,
    "sport": "NHL",
    "game_id": "nhl_car_fla",
    "start_utc": "2026-09-29T21:00:00+00:00",
    "home": {"team": "Carolina Hurricanes", "tricode": "CAR", "record": "53-22-7"},
    "away": {"team": "Florida Panthers", "tricode": "FLA", "record": "47-28-7"},
}


def response(parsed=None, *, stop_reason="end_turn", details=None, tokens=(1200, 300)):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        stop_details=details,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
        model_dump_json=lambda: json.dumps({"stop_reason": stop_reason}),
    )


def forecast(p_home=0.58, p_away=0.42):
    return Stage1Forecast(
        game_id="nhl_car_fla", p_home=p_home, p_away=p_away, notes="rested, home ice"
    )


class StubClient:
    """Minimal stand-in for the Anthropic client."""

    def __init__(self, *responses, raises=None):
        self.calls = []
        self._responses = list(responses)
        self._raises = raises
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def make(*responses, raises=None):
    client = StubClient(*responses, raises=raises)
    return Stage1Forecaster(client=client), client


# --- request shape ----------------------------------------------------------


def test_request_uses_structured_outputs_and_adaptive_thinking():
    fc, client = make(response(forecast()))
    fc.forecast(CONTEXT)
    sent = client.calls[0]
    assert sent["model"] == "claude-opus-5"
    assert sent["output_format"] is Stage1Forecast
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}


def test_no_sampling_parameters_are_sent():
    # temperature, top_p, top_k and budget_tokens all return HTTP 400 on Opus 5.
    fc, client = make(response(forecast()))
    fc.forecast(CONTEXT)
    sent = client.calls[0]
    for banned in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert banned not in sent


def test_logged_params_record_no_temperature():
    fc, _ = make(response(forecast()))
    params = fc.forecast(CONTEXT).params
    assert "temperature" not in params
    assert params["effort"] == "high"
    assert params["thinking"] == {"type": "adaptive"}
    assert len(params["prompt_sha256"]) == 64


# --- the blind guarantee ----------------------------------------------------


def test_rendering_runs_the_leak_guard():
    fc, _ = make(response(forecast()))
    leaky = {**CONTEXT, "closing_price": 1.91}
    with pytest.raises(OddsLeakError):
        fc.render(leaky)


def test_rendered_prompt_carries_the_context():
    fc, _ = make(response(forecast()))
    text = fc.render(CONTEXT)
    assert "Carolina Hurricanes" in text
    assert "nhl_car_fla" in text


# --- k independent draws ----------------------------------------------------


def test_k_draws_share_one_input_hash():
    # Identical input across draws is what makes them independent draws rather
    # than k different questions.
    fc, _ = make(response(forecast()))
    calls = fc.forecast_k(CONTEXT, k=5)
    assert len({c.input_hash for c in calls}) == 1
    assert [c.seed_idx for c in calls] == [0, 1, 2, 3, 4]


def test_k_draws_make_k_calls():
    fc, client = make(response(forecast()))
    fc.forecast_k(CONTEXT, k=3)
    assert len(client.calls) == 3


def test_k_must_be_positive():
    fc, _ = make(response(forecast()))
    with pytest.raises(ValueError, match="at least 1"):
        fc.forecast_k(CONTEXT, k=0)


# --- outcomes ---------------------------------------------------------------


def test_successful_forecast_is_normalised():
    fc, _ = make(response(forecast(0.50, 0.51)))  # sums to 1.01, within tolerance
    call = fc.forecast(CONTEXT)
    assert call.ok
    assert sum(call.probabilities.values()) == pytest.approx(1.0)
    assert call.input_tokens == 1200
    assert call.output_tokens == 300


def test_cost_is_estimated_from_usage():
    fc, _ = make(response(forecast(), tokens=(1_000_000, 1_000_000)))
    # Opus 5 is $5/MTok in and $25/MTok out.
    assert fc.forecast(CONTEXT).cost_usd == pytest.approx(30.0)


def test_a_refusal_is_logged_not_raised():
    # Gambling-adjacent prompts make this non-hypothetical; a nightly run must
    # survive one rather than crashing.
    details = SimpleNamespace(category="gambling", explanation="declined")
    fc, _ = make(response(None, stop_reason="refusal", details=details))
    call = fc.forecast(CONTEXT)
    assert call.ok is False
    assert call.stop_reason == "refusal"
    assert "gambling" in call.error
    assert call.raw_response is not None


def test_an_api_error_is_logged_not_raised():
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    fc, _ = make(raises=err)
    call = fc.forecast(CONTEXT)
    assert call.ok is False
    assert "APIConnectionError" in call.error
    assert call.input_hash  # still logged, so the failure is auditable


def test_an_unnormalisable_forecast_is_rejected_but_kept():
    fc, _ = make(response(forecast(0.50, 0.90)))  # sums to 1.40
    call = fc.forecast(CONTEXT)
    assert call.ok is False
    assert "unnormalisable" in call.error
    assert call.forecast is not None  # the raw output is still recorded as data


def test_missing_parsed_output_is_reported():
    fc, _ = make(response(None, stop_reason="max_tokens"))
    call = fc.forecast(CONTEXT)
    assert call.ok is False
    assert "no parsed output" in call.error


# --- summary ----------------------------------------------------------------


def test_summarise_counts_and_measures_dispersion():
    fc, _ = make(response(forecast(0.55, 0.45)), response(forecast(0.65, 0.35)))
    s = summarise(fc.forecast_k(CONTEXT, k=2))
    assert s["calls"] == 2 and s["ok"] == 2 and s["failed"] == 0 and s["refusals"] == 0
    assert s["cost_usd"] == pytest.approx(0.027)
    assert s["mean_p_home"] == pytest.approx(0.6)
    assert s["max_spread_within_game"] == pytest.approx(0.1)


def test_dispersion_is_measured_within_a_game_not_across_the_slate():
    # Pooling every game's p_home measures how varied the slate is, not how much
    # the model disagrees with itself on identical input -- which is the signal.
    import dataclasses

    fc, _ = make(response(forecast(0.55, 0.45)), response(forecast(0.57, 0.43)))
    calls = fc.forecast_k(CONTEXT, k=2)
    far_away = [
        dataclasses.replace(c, game_id="other", probabilities={"home": 0.20, "away": 0.80})
        for c in calls
    ]
    s = summarise([*calls, *far_away])
    assert s["max_spread_within_game"] == pytest.approx(0.02)


def test_summarise_handles_an_all_failed_batch():
    fc, _ = make(response(None, stop_reason="refusal", details=None))
    s = summarise(fc.forecast_k(CONTEXT, k=2))
    assert s["ok"] == 0 and s["failed"] == 2 and s["refusals"] == 2
    assert "mean_p_home" not in s
