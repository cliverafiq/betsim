"""Stage 1: the blind forecast.

One call per game per draw, with no odds anywhere in the input. Three things
this module is careful about:

**k independent draws.** Current Claude models expose no sampling parameters --
``temperature``, ``top_p``, ``top_k`` and ``budget_tokens`` all return HTTP 400
on Opus 5 -- and there is no seed. Repeated variance therefore has to come from
repeated calls. All k draws for a game share one ``input_hash``, which is what
proves they were independent draws on identical input rather than k different
questions.

**Nothing is raised away.** A refusal, a malformed forecast or an API failure is
recorded as a failed call and returned, not thrown. Rule violations and refusals
are data; one bad game must not take down a slate.

**No sampling parameters are logged.** ``params`` records the model, effort,
thinking config and token ceiling. There is deliberately no ``temperature``
field, because no such parameter exists on the models under test.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import anthropic

from betsim.config import EFFORT, K_SEEDS, MAX_TOKENS, MODEL_ID, estimate_cost_usd
from betsim.context import assert_no_odds_leak
from betsim.models import ForecastSumError, Stage1Forecast, normalize_probabilities
from betsim.prompts import load_prompt

STAGE = 1
DEFAULT_PROMPT_VERSION = "stage1_v1"


@dataclass(frozen=True, slots=True)
class ForecastCall:
    """Everything needed to log one Stage 1 call, successful or not."""

    game_id: str
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
    forecast: Stage1Forecast | None = None
    probabilities: dict[str, float] | None = field(default=None)
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def cost_usd(self) -> float:
        if self.input_tokens is None or self.output_tokens is None:
            return 0.0
        return estimate_cost_usd(self.input_tokens, self.output_tokens, self.model_id)


def _hash_input(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Stage1Forecaster:
    """Runs blind forecasts. Pass ``client`` to inject a stub in tests."""

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
        """Constructed lazily so that importing this module needs no API key."""
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    @property
    def params(self) -> dict[str, Any]:
        """Request parameters, for the audit log.

        Note the absence of a sampling parameter: there is none to record.
        """
        return {
            "model": self.model,
            "effort": self.effort,
            "thinking": {"type": "adaptive"},
            "max_tokens": self.max_tokens,
            "prompt_version": self.prompt.version,
            "prompt_sha256": self.prompt.sha256,
        }

    def render(self, context: dict[str, Any]) -> str:
        """Render the blind prompt for one game, re-checking it carries no odds."""
        assert_no_odds_leak(context)
        return self.prompt.render(
            context=json.dumps(context, indent=2, sort_keys=True, ensure_ascii=False)
        )

    def forecast(self, context: dict[str, Any], *, seed_idx: int = 0) -> ForecastCall:
        """One blind forecast. Never raises for a model or API failure."""
        game_id = str(context.get("game_id", ""))
        text = self.render(context)
        base = {
            "game_id": game_id,
            "seed_idx": seed_idx,
            "model_id": self.model,
            "prompt_version": self.prompt.version,
            "params": self.params,
            "input_hash": _hash_input(text),
            "input_text": text,
        }

        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=self.max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                messages=[{"role": "user", "content": text}],
                output_format=Stage1Forecast,
            )
        except anthropic.APIError as exc:
            return ForecastCall(**base, ok=False, error=f"{type(exc).__name__}: {exc}")

        usage = getattr(response, "usage", None)
        tokens = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
        }
        stop_reason = getattr(response, "stop_reason", None)
        raw = _serialise(response)

        # A refusal is a logged outcome, not a crash. Gambling-adjacent prompts
        # make this non-hypothetical, and a nightly run must survive one.
        if stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            explanation = getattr(details, "explanation", "")
            return ForecastCall(
                **base,
                **tokens,
                ok=False,
                stop_reason=stop_reason,
                raw_response=raw,
                error=f"refusal (category={category}): {explanation}",
            )

        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            return ForecastCall(
                **base,
                **tokens,
                ok=False,
                stop_reason=stop_reason,
                raw_response=raw,
                error=f"no parsed output (stop_reason={stop_reason})",
            )

        try:
            probabilities = normalize_probabilities(parsed.probabilities())
        except ForecastSumError as exc:
            return ForecastCall(
                **base,
                **tokens,
                ok=False,
                stop_reason=stop_reason,
                raw_response=raw,
                forecast=parsed,
                error=f"unnormalisable forecast: {exc}",
            )

        return ForecastCall(
            **base,
            **tokens,
            ok=True,
            stop_reason=stop_reason,
            raw_response=raw,
            forecast=parsed,
            probabilities=probabilities,
        )

    def forecast_k(self, context: dict[str, Any], *, k: int = K_SEEDS) -> list[ForecastCall]:
        """``k`` independent draws on identical input.

        A single live forward path is one seed, and KellyBench's per-model seed
        spread was +34.1% to -32.9% on the same data, so one draw cannot separate
        skill from luck.
        """
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        return [self.forecast(context, seed_idx=i) for i in range(k)]


def _serialise(response: Any) -> str:
    """Best-effort JSON of the raw response, for the audit log."""
    for attr in ("model_dump_json", "to_json"):
        method = getattr(response, attr, None)
        if callable(method):
            try:
                return str(method())
            except (TypeError, ValueError):
                continue
    return repr(response)


def summarise(calls: list[ForecastCall]) -> dict[str, Any]:
    """Aggregate a batch of draws: success rate, spend, and forecast dispersion."""
    ok = [c for c in calls if c.ok and c.probabilities]
    summary: dict[str, Any] = {
        "calls": len(calls),
        "ok": len(ok),
        "failed": len(calls) - len(ok),
        "cost_usd": round(sum(c.cost_usd for c in calls), 4),
        "refusals": sum(1 for c in calls if c.stop_reason == "refusal"),
    }
    if ok:
        home = [c.probabilities["home"] for c in ok]
        summary["mean_p_home"] = round(sum(home) / len(home), 4)
        # Dispersion across draws is a free uncertainty signal -- wide
        # disagreement means the model is guessing.
        summary["spread_p_home"] = round(max(home) - min(home), 4)
    return summary
