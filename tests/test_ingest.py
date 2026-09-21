from datetime import UTC, datetime

import pytest

from betsim.devig import consensus
from betsim.ingest import (
    IngestError,
    outcome_to_selection,
    parse_events,
    parse_odds,
    parse_scores,
    parse_utc,
)

HOME, AWAY = "Carolina Hurricanes", "Florida Panthers"


def test_parse_utc_returns_aware_datetimes():
    assert parse_utc("2026-09-29T21:00:00Z") == datetime(2026, 9, 29, 21, tzinfo=UTC)
    assert parse_utc(None) is None


def test_parse_utc_refuses_naive_timestamps():
    with pytest.raises(IngestError, match="missing a timezone"):
        parse_utc("2026-09-29T21:00:00")


def test_parse_utc_reports_garbage():
    with pytest.raises(IngestError, match="could not parse"):
        parse_utc("not a date")


def test_parse_events(events_payload):
    games = parse_events(events_payload)
    assert [g.id for g in games] == ["nhl_car_fla", "nhl_tor_mtl", "nhl_bos_nyr"]
    assert games[0].home_team == HOME
    assert games[0].commence_utc.tzinfo is not None


def test_missing_required_field_is_reported():
    with pytest.raises(IngestError, match="missing required field 'home_team'"):
        parse_events(
            [
                {
                    "id": "x",
                    "sport_key": "y",
                    "commence_time": "2026-09-29T21:00:00Z",
                    "away_team": AWAY,
                }
            ]
        )


# --- the sharp edge: outcomes are named by team, not by home/away ------------


def test_outcome_names_map_to_selections():
    assert outcome_to_selection(HOME, HOME, AWAY) == "home"
    assert outcome_to_selection(AWAY, HOME, AWAY) == "away"
    assert outcome_to_selection("Draw", HOME, AWAY) == "draw"


def test_outcome_matching_tolerates_spacing_and_case():
    assert outcome_to_selection("carolina  hurricanes", HOME, AWAY) == "home"


def test_unrecognised_outcome_raises_rather_than_guessing():
    # Silently mislabelling a selection would invert a bet, so this must never
    # fall through to a default.
    with pytest.raises(IngestError, match="matches neither"):
        outcome_to_selection("Ottawa Senators", HOME, AWAY)


def test_parse_odds_maps_every_book(odds_payload):
    games, prices = parse_odds(odds_payload)
    assert len(games) == 3
    assert len(prices) == 12  # 3 games x 2 books x 2 outcomes
    assert {p.bookmaker for p in prices} == {"pinnacle", "betfair_ex_eu"}
    assert {p.selection for p in prices} == {"home", "away"}
    pin_home = next(
        p
        for p in prices
        if p.game_id == "nhl_car_fla" and p.bookmaker == "pinnacle" and p.selection == "home"
    )
    assert pin_home.price_decimal == pytest.approx(2.10)


def test_parse_odds_can_narrow_to_one_book(odds_payload):
    _, prices = parse_odds(odds_payload, bookmakers=["pinnacle"])
    assert {p.bookmaker for p in prices} == {"pinnacle"}
    assert len(prices) == 6


def test_parse_odds_ignores_other_markets(odds_payload):
    _, prices = parse_odds(odds_payload, market="spreads")
    assert prices == []


def test_parse_odds_rejects_a_malformed_outcome(odds_payload):
    odds_payload[0]["bookmakers"][0]["markets"][0]["outcomes"][0].pop("price")
    with pytest.raises(IngestError, match="malformed outcome"):
        parse_odds(odds_payload)


def test_parse_odds_tolerates_a_book_with_no_markets(odds_payload):
    odds_payload[0]["bookmakers"][0]["markets"] = []
    _, prices = parse_odds(odds_payload)
    assert len(prices) == 10


def test_ingested_prices_feed_the_devig_consensus(odds_payload):
    _, prices = parse_odds(odds_payload)
    books = {}
    for p in prices:
        if p.game_id == "nhl_car_fla":
            books.setdefault(p.bookmaker, {})[p.selection] = p.price_decimal
    c = consensus(list(books.values()), ("home", "away"), "shin")
    assert sum(c.values()) == pytest.approx(1.0)
    assert 0.0 < c["home"] < 1.0


# --- the other sharp edge: scores are strings, and null before puck drop -----


def test_parse_scores_converts_string_scores_to_integers(scores_payload):
    scores = {s.game_id: s for s in parse_scores(scores_payload)}
    car = scores["nhl_car_fla"]
    assert car.completed is True
    assert (car.home_score, car.away_score) == (4, 1)
    assert isinstance(car.home_score, int)


def test_parse_scores_handles_a_game_that_has_not_started(scores_payload):
    scores = {s.game_id: s for s in parse_scores(scores_payload)}
    upcoming = scores["nhl_bos_nyr"]
    assert upcoming.completed is False
    assert upcoming.home_score is None and upcoming.away_score is None


def test_completed_game_missing_a_score_is_rejected(scores_payload):
    scores_payload[0]["scores"] = [{"name": HOME, "score": "4"}]  # away missing
    with pytest.raises(IngestError, match="marked completed but is missing a score"):
        parse_scores(scores_payload)


def test_non_numeric_score_is_rejected(scores_payload):
    scores_payload[0]["scores"][0]["score"] = "four"
    with pytest.raises(IngestError, match="is not an integer"):
        parse_scores(scores_payload)


def test_scores_null_is_treated_as_no_scores(scores_payload):
    scores_payload[0]["completed"] = False
    scores_payload[0]["scores"] = None
    parsed = parse_scores(scores_payload)[0]
    assert parsed.home_score is None
