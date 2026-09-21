"""Stage 2: the priced decision.

Stage 2 sees everything Stage 1 was denied -- the prices, the tier caps, the
bankroll and the recent record -- and decides what to bet. Splitting the two is
the point of the experiment: with the price visible throughout, a loss cannot be
attributed to the forecast or to the sizing, which is exactly the ambiguity that
left prior work unable to say why its agents lost money.

As in Stage 1, a refusal or a malformed decision is recorded rather than raised,
and zero bets is a valid, expected outcome that is never treated as a failure.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import anthropic

from betsim.config import EFFORT, MAX_TOKENS, MODEL_ID, estimate_cost_usd
from betsim.ledger import ArmState
from betsim.llmcall import invoke_parse
from betsim.models import Stage2Decision
from betsim.money import minor_to_units
from betsim.odds import decimal_to_american
from betsim.prompts import load_prompt
from betsim.tiers import Tier, assign_tier, tier_cap_minor
from betsim.validator import BetProposal, GameRef, exposure_cap_minor

STAGE = 2
DEFAULT_PROMPT_VERSION = "stage2_v1"


@dataclass(frozen=True, slots=True)
class DecisionCall:
    """Everything needed to log one Stage 2 call, successful or not."""

    seed_idx: int
    ok: bool
    model_id: str
    prompt_version: str
    params: dict[str, Any]
    input_hash: str
    input_text: str
    raw_response: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    decision: Stage2Decision | None = None
    proposals: tuple[BetProposal, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def day_notes(self) -> str:
        return self.decision.day_notes if self.decision else ""

    @property
    def cost_usd(self) -> float:
        if self.input_tokens is None or self.output_tokens is None:
            return 0.0
        return estimate_cost_usd(self.input_tokens, self.output_tokens, self.model_id)


def build_slate_payload(
    games: Sequence[GameRef],
    *,
    p_blind: Mapping[str, Mapping[str, float]],
    teams: Mapping[str, tuple[str, str]],
    state: ArmState,
    record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble what Stage 2 is shown.

    Caps are precomputed in units so the model is not asked to do bankroll
    arithmetic it cannot be checked on, and every selection carries the tier the
    code assigned -- tiers are never the model's to choose.
    """
    balance = state.balance_minor
    remaining = max(0, exposure_cap_minor(balance) - state.open_exposure_minor)

    entries = []
    for game in sorted(games, key=lambda g: (g.commence_utc, g.game_id)):
        if game.game_id in state.open_game_ids:
            continue
        home, away = teams.get(game.game_id, ("", ""))
        selections = []
        for selection, price in sorted(game.prices.items()):
            tier = assign_tier(price)
            selections.append(
                {
                    "selection": selection,
                    "price_decimal": round(price, 4),
                    "price_american": round(decimal_to_american(price)),
                    "tier": str(tier),
                    "tier_cap_units": minor_to_units(tier_cap_minor(tier, balance)),
                    "p_blind": round(p_blind.get(game.game_id, {}).get(selection, 0.0), 4),
                }
            )
        entries.append(
            {
                "game_id": game.game_id,
                "home": home,
                "away": away,
                "start_utc": game.commence_utc.isoformat(),
                "selections": selections,
            }
        )

    return {
        "balance_units": minor_to_units(balance),
        "remaining_exposure_units": minor_to_units(remaining),
        "tier_caps_units": {
            str(tier): minor_to_units(tier_cap_minor(tier, balance)) for tier in Tier
        },
        "open_bets": sorted(state.open_game_ids),
        "recent_record": dict(record or {}),
        "games": entries,
    }


class Stage2Decider:
    """Runs priced decisions. Pass ``client`` to inject a stub in tests."""

    def __init__(
        self,
        client: Any | None = None,
        *,
        model: str = MODEL_ID,
        effort: str = EFFORT,
        max_tokens: int = MAX_TOKENS,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
    ) -> None:
        self._client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.prompt = load_prompt(prompt_version)

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def params(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "effort": self.effort,
            "thinking": {"type": "adaptive"},
            "max_tokens": self.max_tokens,
            "prompt_version": self.prompt.version,
            "prompt_sha256": self.prompt.sha256,
        }

    def render(self, payload: Mapping[str, Any]) -> str:
        return self.prompt.render(
            slate=json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        )

    def decide(self, payload: Mapping[str, Any], *, seed_idx: int = 0) -> DecisionCall:
        """One priced decision. Never raises for a model or API failure."""
        text = self.render(payload)
        base = {
            "seed_idx": seed_idx,
            "model_id": self.model,
            "prompt_version": self.prompt.version,
            "params": self.params,
            "input_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "input_text": text,
        }

        result = invoke_parse(
            self.client,
            model=self.model,
            max_tokens=self.max_tokens,
            effort=self.effort,
            text=text,
            output_format=Stage2Decision,
        )
        tokens = {"input_tokens": result.input_tokens, "output_tokens": result.output_tokens}
        if not result.ok:
            return DecisionCall(
                **base,
                **tokens,
                ok=False,
                stop_reason=result.stop_reason,
                raw_response=result.raw_response,
                error=result.error,
            )

        decision: Stage2Decision = result.parsed
        proposals = tuple(
            BetProposal(
                game_id=bet.game_id,
                selection=bet.selection,
                stake_units=bet.stake_units,
                p_revised=bet.p_revised,
                reason=bet.reason,
            )
            for bet in decision.bets
        )
        return DecisionCall(
            **base,
            **tokens,
            ok=True,
            stop_reason=result.stop_reason,
            raw_response=result.raw_response,
            decision=decision,
            proposals=proposals,
        )
