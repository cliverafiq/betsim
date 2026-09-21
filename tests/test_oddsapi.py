import json

import httpx
import pytest

from betsim.oddsapi import (
    AuthError,
    BadRequestError,
    OddsApiClient,
    OddsApiError,
    Quota,
    QuotaExhausted,
    RateLimited,
    ServerError,
    estimate_odds_cost,
    estimate_scores_cost,
)

API_KEY = "secret-key-do-not-leak"
QUOTA_HEADERS = {
    "x-requests-remaining": "480",
    "x-requests-used": "20",
    "x-requests-last": "1",
}


def client_with(handler, **kwargs) -> OddsApiClient:
    return OddsApiClient(API_KEY, transport=httpx.MockTransport(handler), **kwargs)


def ok(payload=None, headers=QUOTA_HEADERS):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload if payload is not None else [], headers=headers)

    return handler


# --- cost model -------------------------------------------------------------


def test_odds_cost_is_markets_times_regions():
    assert estimate_odds_cost("h2h", "eu") == 1
    assert estimate_odds_cost("h2h,spreads", "eu") == 2
    assert estimate_odds_cost("h2h", "us,uk,eu") == 3
    assert estimate_odds_cost("h2h,spreads,totals", "us,eu") == 6


def test_ten_bookmakers_count_as_one_region():
    books = ",".join(f"book{i}" for i in range(12))
    assert estimate_odds_cost("h2h", bookmakers=books) == 2


def test_odds_cost_requires_markets_and_regions():
    with pytest.raises(ValueError, match="at least one market"):
        estimate_odds_cost("", "eu")
    with pytest.raises(ValueError, match="at least one region"):
        estimate_odds_cost("h2h", "")


def test_scores_cost_doubles_when_reaching_back():
    assert estimate_scores_cost() == 1
    assert estimate_scores_cost(3) == 2


def test_v1_slate_call_costs_one_credit():
    assert estimate_odds_cost("h2h", "eu") == 1


# --- quota tracking ---------------------------------------------------------


def test_quota_headers_are_recorded():
    with client_with(ok()) as c:
        c.fetch_odds("icehockey_nhl")
        assert c.quota == Quota(remaining=480, used=20, last_cost=1)


def test_missing_quota_headers_leave_the_quota_unknown():
    with client_with(ok(headers={})) as c:
        c.fetch_odds("icehockey_nhl")
        assert c.quota.remaining is None


def test_malformed_quota_headers_do_not_crash_the_run():
    with client_with(ok(headers={"x-requests-remaining": "lots"})) as c:
        c.fetch_odds("icehockey_nhl")
        assert c.quota.remaining is None


def test_costed_call_is_refused_when_it_would_breach_the_reserve():
    with client_with(ok(headers={**QUOTA_HEADERS, "x-requests-remaining": "26"})) as c:
        c.fetch_odds("icehockey_nhl")  # learns remaining == 26
        with pytest.raises(QuotaExhausted, match="breach the reserve"):
            c.fetch_scores("icehockey_nhl", days_from=3)  # costs 2, reserve 25


def test_free_endpoints_are_never_blocked_by_the_reserve():
    # /events is free, so scheduling must keep working even at zero credits.
    with client_with(ok(headers={**QUOTA_HEADERS, "x-requests-remaining": "0"})) as c:
        c.list_sports()
        assert c.list_events("icehockey_nhl") == []


def test_first_call_proceeds_while_the_quota_is_unknown():
    with client_with(ok()) as c:
        assert c.quota.remaining is None
        c.fetch_odds("icehockey_nhl")


def test_refresh_quota_learns_the_balance_for_free():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json=[], headers=QUOTA_HEADERS)

    with client_with(handler) as c:
        assert c.refresh_quota().remaining == 480
    assert seen == ["/v4/sports/"]


# --- requests ---------------------------------------------------------------


def test_api_key_and_params_are_sent():
    seen = {}

    def handler(request):
        seen["params"] = dict(request.url.params)
        seen["path"] = request.url.path
        return httpx.Response(200, json=[], headers=QUOTA_HEADERS)

    with client_with(handler) as c:
        c.fetch_odds("icehockey_nhl", regions="eu", markets="h2h")

    assert seen["path"] == "/v4/sports/icehockey_nhl/odds/"
    assert seen["params"]["apiKey"] == API_KEY
    assert seen["params"]["regions"] == "eu"
    assert seen["params"]["oddsFormat"] == "decimal"


def test_bookmakers_replace_regions_when_given():
    seen = {}

    def handler(request):
        seen.update(dict(request.url.params))
        return httpx.Response(200, json=[], headers=QUOTA_HEADERS)

    with client_with(handler) as c:
        c.fetch_odds("icehockey_nhl", bookmakers="pinnacle")

    assert seen["bookmakers"] == "pinnacle"
    assert "regions" not in seen


def test_days_from_is_validated():
    with client_with(ok()) as c, pytest.raises(ValueError, match="between 1 and 3"):
        c.fetch_scores("icehockey_nhl", days_from=4)


def test_an_empty_api_key_is_refused():
    with pytest.raises(AuthError, match="API key is required"):
        OddsApiClient("")


def test_a_non_list_payload_is_rejected():
    def handler(request):
        return httpx.Response(200, json={"message": "nope"}, headers=QUOTA_HEADERS)

    with client_with(handler) as c, pytest.raises(OddsApiError, match="expected a JSON list"):
        c.fetch_odds("icehockey_nhl")


# --- errors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, AuthError), (422, BadRequestError), (429, RateLimited), (403, OddsApiError)],
)
def test_client_errors_map_to_typed_exceptions(status, expected):
    def handler(request):
        return httpx.Response(status, text="denied", headers=QUOTA_HEADERS)

    with client_with(handler) as c, pytest.raises(expected):
        c.fetch_odds("icehockey_nhl")


def test_quota_is_still_read_from_an_error_response():
    def handler(request):
        return httpx.Response(429, text="slow down", headers=QUOTA_HEADERS)

    with client_with(handler) as c:
        with pytest.raises(RateLimited):
            c.fetch_odds("icehockey_nhl")
        assert c.quota.remaining == 480


def test_server_errors_are_retried_then_raised(monkeypatch):
    monkeypatch.setattr("betsim.oddsapi.time.sleep", lambda _: None)
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503, text="unavailable")

    with client_with(handler, retries=2) as c, pytest.raises(ServerError, match="after retries"):
        c.fetch_odds("icehockey_nhl")
    assert len(calls) == 3  # initial attempt plus two retries


def test_a_transient_failure_is_retried_and_succeeds(monkeypatch):
    # A dropped closing-snapshot call would cost the primary metric for that game.
    monkeypatch.setattr("betsim.oddsapi.time.sleep", lambda _: None)
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(200, json=[{"id": "g1"}], headers=QUOTA_HEADERS)

    with client_with(handler, retries=2) as c:
        assert c.fetch_odds("icehockey_nhl") == [{"id": "g1"}]
    assert len(attempts) == 2


def test_client_errors_are_not_retried():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(422, text="bad params")

    with client_with(handler, retries=2) as c, pytest.raises(BadRequestError):
        c.fetch_odds("icehockey_nhl")
    assert len(calls) == 1


# --- fixture recording ------------------------------------------------------


def test_recorder_writes_payload_and_quota(tmp_path):
    payload = [{"id": "g1", "home_team": "A", "away_team": "B"}]
    with client_with(ok(payload), record_dir=tmp_path) as c:
        c.fetch_odds("icehockey_nhl")

    written = json.loads((tmp_path / "odds_icehockey_nhl.json").read_text())
    assert written["payload"] == payload
    assert written["quota"]["x-requests-remaining"] == "480"


def test_recorder_never_writes_the_api_key(tmp_path):
    # The key travels in the query string, so recording a URL would leak it into
    # a file destined for version control.
    with client_with(ok([{"id": "g1"}]), record_dir=tmp_path) as c:
        c.fetch_odds("icehockey_nhl")

    text = (tmp_path / "odds_icehockey_nhl.json").read_text()
    assert API_KEY not in text
    assert "apiKey" not in text


def test_nothing_is_recorded_without_a_record_dir(tmp_path):
    with client_with(ok([{"id": "g1"}])) as c:
        c.fetch_odds("icehockey_nhl")
    assert list(tmp_path.iterdir()) == []
