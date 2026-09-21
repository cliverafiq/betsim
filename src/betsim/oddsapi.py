"""The Odds API v4 client.

Deliberately thin: it makes requests, tracks the credit quota, and optionally
records raw responses as test fixtures. Turning payloads into domain objects is
:mod:`betsim.ingest`, which is pure and carries the tests.

Credit costs (free tier is 500/month):

===================  =====================================
endpoint             cost
===================  =====================================
``/sports``          free
``/sports/../events``  free -- so scheduling never spends
``/sports/../odds``  ``len(markets) x len(regions)``
``/sports/../scores``  1, or 2 with ``daysFrom``
===================  =====================================

The client refuses a costed call that would take the balance below
``reserve_credits``, so a run can never strand the experiment mid-season with no
credits left for settlement.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import httpx

BASE_URL = "https://api.the-odds-api.com"
API_VERSION = "v4"

# A day on one sport costs about 6 credits (1 slate + ~3 closing + 2 settlement),
# so this keeps roughly four days of headroom in reserve.
DEFAULT_RESERVE_CREDITS = 25
DEFAULT_TIMEOUT = 20.0
DEFAULT_RETRIES = 2

REMAINING_HEADER = "x-requests-remaining"
USED_HEADER = "x-requests-used"
LAST_HEADER = "x-requests-last"


class OddsApiError(RuntimeError):
    """Base class for Odds API failures."""


class AuthError(OddsApiError):
    """The API key was missing, malformed or rejected (401)."""


class BadRequestError(OddsApiError):
    """The request parameters were rejected (422)."""


class RateLimited(OddsApiError):
    """Throttled by the API (429)."""


class ServerError(OddsApiError):
    """The API returned 5xx after exhausting retries."""


class QuotaExhausted(OddsApiError):
    """A costed call was refused because it would breach the reserve."""


@dataclass(frozen=True, slots=True)
class Quota:
    remaining: int | None = None
    used: int | None = None
    last_cost: int | None = None

    @classmethod
    def from_headers(cls, headers: Mapping[str, str]) -> Quota:
        def get(name: str) -> int | None:
            raw = headers.get(name)
            if raw is None:
                return None
            try:
                return int(float(raw))
            except ValueError:
                return None

        return cls(get(REMAINING_HEADER), get(USED_HEADER), get(LAST_HEADER))


def _count(value: str | Sequence[str] | None) -> int:
    if value is None:
        return 0
    items = value.split(",") if isinstance(value, str) else list(value)
    return len([i for i in items if i.strip()])


def estimate_odds_cost(
    markets: str | Sequence[str],
    regions: str | Sequence[str] | None = None,
    bookmakers: str | Sequence[str] | None = None,
) -> int:
    """``markets x regions``. Ten bookmakers count as one region when given instead."""
    n_markets = _count(markets)
    if n_markets == 0:
        raise ValueError("at least one market is required")
    if bookmakers:
        n_regions = math.ceil(_count(bookmakers) / 10)
    else:
        n_regions = _count(regions)
    if n_regions == 0:
        raise ValueError("at least one region (or bookmaker) is required")
    return n_markets * n_regions


def estimate_scores_cost(days_from: int | None = None) -> int:
    """1 for live and upcoming games, 2 when reaching back for completed ones."""
    return 2 if days_from else 1


class OddsApiClient:
    """A synchronous client for The Odds API v4."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        reserve_credits: int = DEFAULT_RESERVE_CREDITS,
        retries: int = DEFAULT_RETRIES,
        record_dir: Path | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise AuthError("an Odds API key is required")
        self._api_key = api_key
        self.reserve_credits = reserve_credits
        self.retries = retries
        self.record_dir = Path(record_dir) if record_dir else None
        self.quota = Quota()
        self._client = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # -- quota ---------------------------------------------------------------

    def check_affordable(self, cost: int) -> None:
        """Raise if a costed call would take the balance below the reserve.

        The quota is unknown until the first response, so the very first call
        proceeds; :meth:`refresh_quota` learns it up front for free.
        """
        if cost <= 0 or self.quota.remaining is None:
            return
        if self.quota.remaining - cost < self.reserve_credits:
            raise QuotaExhausted(
                f"{self.quota.remaining} credits remaining; a {cost}-credit call would "
                f"breach the reserve of {self.reserve_credits}"
            )

    def refresh_quota(self) -> Quota:
        """Learn the current quota using a free endpoint."""
        self.list_sports()
        return self.quota

    # -- endpoints -----------------------------------------------------------

    def list_sports(self, *, all_sports: bool = False) -> list[dict[str, Any]]:
        """GET /v4/sports -- free."""
        params: dict[str, Any] = {}
        if all_sports:
            params["all"] = "true"
        return self._get("/v4/sports/", params, cost=0, label="sports")

    def list_events(
        self,
        sport: str,
        *,
        commence_time_from: str | None = None,
        commence_time_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """GET /v4/sports/{sport}/events -- **free**, so scheduling costs nothing."""
        params: dict[str, Any] = {"dateFormat": "iso"}
        if commence_time_from:
            params["commenceTimeFrom"] = commence_time_from
        if commence_time_to:
            params["commenceTimeTo"] = commence_time_to
        return self._get(f"/v4/sports/{sport}/events", params, cost=0, label=f"events_{sport}")

    def fetch_odds(
        self,
        sport: str,
        *,
        regions: str = "eu",
        markets: str = "h2h",
        bookmakers: str | None = None,
        odds_format: str = "decimal",
    ) -> list[dict[str, Any]]:
        """GET /v4/sports/{sport}/odds -- costs ``markets x regions``."""
        cost = estimate_odds_cost(markets, regions, bookmakers)
        params: dict[str, Any] = {
            "markets": markets,
            "oddsFormat": odds_format,
            "dateFormat": "iso",
        }
        if bookmakers:
            params["bookmakers"] = bookmakers
        else:
            params["regions"] = regions
        return self._get(f"/v4/sports/{sport}/odds/", params, cost=cost, label=f"odds_{sport}")

    def fetch_scores(self, sport: str, *, days_from: int | None = None) -> list[dict[str, Any]]:
        """GET /v4/sports/{sport}/scores -- 1 credit, or 2 with ``days_from``."""
        if days_from is not None and not 1 <= days_from <= 3:
            raise ValueError(f"days_from must be between 1 and 3, got {days_from}")
        params: dict[str, Any] = {"dateFormat": "iso"}
        if days_from is not None:
            params["daysFrom"] = days_from
        cost = estimate_scores_cost(days_from)
        return self._get(f"/v4/sports/{sport}/scores/", params, cost=cost, label=f"scores_{sport}")

    # -- plumbing ------------------------------------------------------------

    def _get(
        self, path: str, params: Mapping[str, Any], *, cost: int, label: str
    ) -> list[dict[str, Any]]:
        self.check_affordable(cost)
        request_params = {**params, "apiKey": self._api_key}

        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._client.get(path, params=request_params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise ServerError(f"{label}: transport failure after retries: {exc}") from exc

            # Quota headers come back on every response, including errors.
            quota = Quota.from_headers(response.headers)
            if quota.remaining is not None or quota.used is not None:
                self.quota = quota

            if response.status_code >= 500:
                last_exc = OddsApiError(f"{label}: HTTP {response.status_code}")
                if attempt < self.retries:
                    time.sleep(2**attempt)
                    continue
                raise ServerError(f"{label}: HTTP {response.status_code} after retries")

            self._raise_for_client_error(response, label)
            payload = response.json()
            if not isinstance(payload, list):
                raise OddsApiError(f"{label}: expected a JSON list, got {type(payload).__name__}")
            if self.record_dir is not None:
                self._record(label, payload, response.headers)
            return payload

        raise ServerError(f"{label}: exhausted retries") from last_exc

    @staticmethod
    def _raise_for_client_error(response: httpx.Response, label: str) -> None:
        if response.status_code == 401:
            raise AuthError(f"{label}: API key rejected (401)")
        if response.status_code == 422:
            raise BadRequestError(f"{label}: parameters rejected (422) -- {response.text[:200]}")
        if response.status_code == 429:
            raise RateLimited(f"{label}: throttled (429)")
        if response.status_code >= 400:
            raise OddsApiError(f"{label}: HTTP {response.status_code} -- {response.text[:200]}")

    def _record(
        self, label: str, payload: list[dict[str, Any]], headers: Mapping[str, str]
    ) -> None:
        """Write a response to disk as a test fixture.

        Only the body and the three quota headers are stored. The request URL is
        never written, because the API key travels in the query string.
        """
        self.record_dir.mkdir(parents=True, exist_ok=True)
        destination = self.record_dir / f"{label}.json"
        destination.write_text(
            json.dumps(
                {
                    "quota": {
                        REMAINING_HEADER: headers.get(REMAINING_HEADER),
                        USED_HEADER: headers.get(USED_HEADER),
                        LAST_HEADER: headers.get(LAST_HEADER),
                    },
                    "payload": payload,
                },
                indent=2,
            )
        )
