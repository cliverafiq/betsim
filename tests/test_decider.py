from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import anthropic
import httpx
import pytest

from betsim.decider import Stage2Decider, build_slate_payload
from betsim.ledger import ArmState
from betsim.models import Stage2Bet, Stage2Decision
from betsim.money import STARTING_BALANCE_MINOR
from betsim.prompts import load_prompt
from betsim.validator import GameRef

NOW = datetime(2026, 9, 29, 22, 0, tzinfo=UTC)
PRICES = {"home": 1.91, "away": 2.00}


def game(gid="g1", *, starts_in_h=1, prices=None):
    return GameRef(gid, NOW + timedelta(hours=starts_in_h), NOW, prices or PRICES)


def state(balance=STARTING_BALANCE_MINOR, exposure=0, open_games=frozenset()):
    return ArmState("llm_s1", balance, exposure, open_games, bust=False)


def payload(**kw):
    return build_slate_payload(
        kw.pop("games", [game()]),
        p_blind=kw.pop("p_blind", {"g1": {"home": 0.58, "away": 0.42}}),
        teams=kw.pop("teams", {"g1": ("Carolina Hurricanes", "Florida Panthers")}),
        state=kw.pop("state", state()),
        record=kw.pop("record", None),
    )


def response(parsed=None, *, stop_reason="end_turn", details=None, tokens=(3000, 500)):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        stop_details=details,
        usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]),
        model_dump_json=lambda: '{"ok": true}',
    )


class StubClient:
    def __init__(self, *responses, raises=None):
        self.calls = []
        self._responses = list(responses)
        self._raises = raises
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises:
            raise self._raises
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


# --- the prompt -------------------------------------------------------------


def test_the_prompt_says_zero_bets_is_fine():
    # Prompts must never require a minimum number of bets. KellyBench forced one
    # bet per matchday, which manufactures losses.
    template = load_prompt("stage2_v1").template
    assert "Placing no bets at all is a perfectly good outcome" in template
    assert "no minimum" in template


def test_the_prompt_explains_that_bets_are_rejected_not_adjusted():
    template = load_prompt("stage2_v1").template
    assert "rejected, not" in template
    assert "order you return them" in template  # exposure cap binds in order


# --- what Stage 2 is shown --------------------------------------------------


def test_payload_precomputes_caps_in_units():
    # The model is not asked to do bankroll arithmetic it cannot be checked on.
    p = payload()
    assert p["balance_units"] == 1000.0
    assert p["tier_caps_units"] == {"safe": 50.0, "medium": 30.0, "risky": 10.0}
    assert p["remaining_exposure_units"] == 200.0


def test_payload_carries_both_price_formats_and_the_assigned_tier():
    selections = {s["selection"]: s for s in payload()["games"][0]["selections"]}
    assert selections["home"]["price_decimal"] == 1.91
    assert selections["home"]["price_american"] == -110
    assert selections["away"]["price_decimal"] == 2.00
    assert selections["away"]["price_american"] == 100
    assert selections["home"]["tier"] == "medium"  # assigned by code, never the model
    assert selections["home"]["tier_cap_units"] == 30.0


def test_payload_carries_the_blind_forecast():
    selections = {s["selection"]: s for s in payload()["games"][0]["selections"]}
    assert selections["home"]["p_blind"] == 0.58
    assert selections["away"]["p_blind"] == 0.42


def test_payload_excludes_games_the_arm_already_has_a_bet_on():
    p = payload(state=state(open_games=frozenset({"g1"})))
    assert p["games"] == []
    assert p["open_bets"] == ["g1"]


def test_payload_shrinks_remaining_exposure_by_open_bets():
    assert payload(state=state(exposure=15_000))["remaining_exposure_units"] == 50.0


def test_remaining_exposure_never_goes_negative():
    assert payload(state=state(exposure=99_000))["remaining_exposure_units"] == 0.0


def test_caps_scale_with_a_drawn_down_bankroll():
    p = payload(state=state(balance=50_000))
    assert p["tier_caps_units"] == {"safe": 25.0, "medium": 15.0, "risky": 5.0}


def test_games_are_listed_in_start_time_order():
    games = [game("late", starts_in_h=9), game("early", starts_in_h=1)]
    p = payload(games=games, p_blind={}, teams={})
    assert [g["game_id"] for g in p["games"]] == ["early", "late"]


# --- decisions --------------------------------------------------------------


def test_a_decision_becomes_validator_proposals():
    decision = Stage2Decision(
        bets=[
            Stage2Bet(
                game_id="g1", selection="home", stake_units=20.0, p_revised=0.56, reason="home ice"
            )
        ],
        day_notes="one bet",
    )
    decider = Stage2Decider(client=StubClient(response(decision)))
    call = decider.decide(payload())
    assert call.ok
    assert len(call.proposals) == 1
    assert call.proposals[0].game_id == "g1"
    assert call.proposals[0].stake_units == 20.0
    assert call.day_notes == "one bet"


def test_zero_bets_is_a_success_not_a_failure():
    decider = Stage2Decider(client=StubClient(response(Stage2Decision(day_notes="no value"))))
    call = decider.decide(payload())
    assert call.ok is True
    assert call.proposals == ()
    assert call.day_notes == "no value"


def test_a_refusal_is_recorded_not_raised():
    details = SimpleNamespace(category="gambling", explanation="declined")
    decider = Stage2Decider(
        client=StubClient(response(None, stop_reason="refusal", details=details))
    )
    call = decider.decide(payload())
    assert call.ok is False
    assert "gambling" in call.error
    assert call.proposals == ()


def test_an_api_error_is_recorded_not_raised():
    err = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    decider = Stage2Decider(client=StubClient(raises=err))
    call = decider.decide(payload())
    assert call.ok is False
    assert "APIConnectionError" in call.error
    assert call.input_hash


def test_no_sampling_parameters_are_sent_or_logged():
    client = StubClient(response(Stage2Decision()))
    call = Stage2Decider(client=client).decide(payload())
    for banned in ("temperature", "top_p", "top_k", "budget_tokens"):
        assert banned not in client.calls[0]
        assert banned not in call.params
    assert client.calls[0]["output_config"] == {"effort": "high"}
    assert client.calls[0]["thinking"] == {"type": "adaptive"}


def test_the_rendered_prompt_contains_the_slate():
    decider = Stage2Decider(client=StubClient(response(Stage2Decision())))
    text = decider.render(payload())
    assert "Carolina Hurricanes" in text
    assert "tier_caps_units" in text
    assert "p_blind" in text


def test_cost_is_estimated_from_usage():
    decider = Stage2Decider(
        client=StubClient(response(Stage2Decision(), tokens=(1_000_000, 1_000_000)))
    )
    assert decider.decide(payload()).cost_usd == pytest.approx(30.0)
