"""Shared plumbing for a single structured-output model call.

Stage 1 and Stage 2 differ in what they ask for and what they do with the
answer, but their failure handling is identical and duplicating it is how the
two drift apart. Both go through :func:`invoke_parse`.

Nothing here raises for a model or transport failure. A refusal, a missing
parse or an API error comes back as a record with ``ok=False`` -- rule
violations and refusals are results of the experiment, and one bad call must not
take down a slate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic


@dataclass(frozen=True, slots=True)
class Invocation:
    ok: bool
    parsed: Any | None = None
    raw_response: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"


def serialise(response: Any) -> str:
    """Best-effort JSON of the raw response, for the audit log."""
    for attr in ("model_dump_json", "to_json"):
        method = getattr(response, attr, None)
        if callable(method):
            try:
                return str(method())
            except (TypeError, ValueError):
                continue
    return repr(response)


def invoke_parse(
    client: Any,
    *,
    model: str,
    max_tokens: int,
    effort: str,
    text: str,
    output_format: type,
) -> Invocation:
    """One structured-output call.

    Note what is *not* passed: no ``temperature``, ``top_p``, ``top_k`` or
    ``budget_tokens``. All four return HTTP 400 on the models under test.
    """
    try:
        response = client.messages.parse(
            model=model,
            max_tokens=max_tokens,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            messages=[{"role": "user", "content": text}],
            output_format=output_format,
        )
    except anthropic.APIError as exc:
        return Invocation(ok=False, error=f"{type(exc).__name__}: {exc}")

    usage = getattr(response, "usage", None)
    tokens = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    stop_reason = getattr(response, "stop_reason", None)
    raw = serialise(response)

    if stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", "")
        return Invocation(
            ok=False,
            raw_response=raw,
            stop_reason=stop_reason,
            **tokens,
            error=f"refusal (category={category}): {explanation}",
        )

    parsed = getattr(response, "parsed_output", None)
    if parsed is None:
        return Invocation(
            ok=False,
            raw_response=raw,
            stop_reason=stop_reason,
            **tokens,
            error=f"no parsed output (stop_reason={stop_reason})",
        )

    return Invocation(ok=True, parsed=parsed, raw_response=raw, stop_reason=stop_reason, **tokens)
