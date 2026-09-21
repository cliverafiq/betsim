import httpx
import pytest

from betsim.nhl import (
    REGULAR_SEASON,
    NhlApiError,
    NhlClient,
    NhlParseError,
    localised,
    parse_club_schedule,
    parse_standings,
)


def test_localised_unwraps_the_default_envelope():
    assert localised({"default": "COL", "fr": "COL"}) == "COL"
    assert localised("COL") == "COL"


def test_localised_rejects_anything_else():
    with pytest.raises(NhlParseError, match="not a localised string"):
        localised(42, "teamAbbrev")


def test_parse_standings_reads_all_teams(standings):
    assert len(standings) == 32
    col = next(t for t in standings if t.tricode == "COL")
    assert col.name == "Colorado Avalanche"
    assert col.games_played == 82
    assert col.points == 121
    assert col.record == "55-16-11"
    assert col.last_10 == "7-2-1"
    assert col.streak == "W3"
    assert col.division == "Central"


def test_standings_carry_the_full_team_name(standings):
    # The full name is what The Odds API also uses, which is what makes the
    # team index derivable rather than hardcoded.
    names = {t.name for t in standings}
    assert "Carolina Hurricanes" in names
    assert "Montréal Canadiens" in names


def test_parse_standings_rejects_a_payload_without_standings():
    with pytest.raises(NhlParseError, match="no 'standings' list"):
        parse_standings({"wildCardIndicator": False})


def test_parse_club_schedule_returns_only_finished_regular_season_games(nhl_schedule_payload):
    games = parse_club_schedule(nhl_schedule_payload)
    assert len(games) == 82  # a full regular season
    assert {g.game_type for g in games} == {REGULAR_SEASON}
    assert all(g.home_score is not None and g.away_score is not None for g in games)


def test_preseason_and_playoff_games_are_excluded_by_default(nhl_schedule_payload):
    all_types = {g.get("gameType") for g in nhl_schedule_payload["games"]}
    assert all_types == {1, 2, 3}, "fixture should contain preseason and playoff games"
    assert {g.game_type for g in parse_club_schedule(nhl_schedule_payload)} == {2}


def test_playoff_games_can_be_included_explicitly(nhl_schedule_payload):
    games = parse_club_schedule(nhl_schedule_payload, game_types=(2, 3))
    assert len(games) > 82


def test_overtime_and_shootout_finishes_are_recorded(nhl_schedule_payload):
    games = parse_club_schedule(nhl_schedule_payload)
    finishes = {g.last_period_type for g in games}
    assert finishes == {"REG", "OT", "SO"}
    beyond = [g for g in games if g.went_beyond_regulation]
    assert len(beyond) == 18  # NHL moneylines still settle on these


def test_home_won_matches_the_score(nhl_schedule_payload):
    for game in parse_club_schedule(nhl_schedule_payload):
        assert game.home_won == (game.home_score > game.away_score)
        # NHL games cannot end level -- overtime and the shootout guarantee it.
        assert game.home_score != game.away_score


def test_a_finished_game_missing_a_score_raises(nhl_schedule_payload):
    for game in nhl_schedule_payload["games"]:
        if game.get("gameType") == 2 and game.get("gameState") in ("OFF", "FINAL"):
            game["homeTeam"]["score"] = None
            break
    with pytest.raises(NhlParseError, match="missing a score"):
        parse_club_schedule(nhl_schedule_payload)


def test_parse_club_schedule_rejects_a_payload_without_games():
    with pytest.raises(NhlParseError, match="no 'games' list"):
        parse_club_schedule({"currentSeason": 20252026})


# --- client -----------------------------------------------------------------


def test_client_follows_the_standings_redirect():
    # /v1/standings/now answers 307. Without follow_redirects every call is empty.
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if request.url.path == "/v1/standings/now":
            return httpx.Response(307, headers={"location": "/v1/standings/2026-04-17"})
        return httpx.Response(200, json={"standings": []})

    with NhlClient(transport=httpx.MockTransport(handler)) as nhl:
        assert nhl.standings() == {"standings": []}
    assert seen == ["/v1/standings/now", "/v1/standings/2026-04-17"]


def test_club_schedule_builds_the_expected_path():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"games": []})

    with NhlClient(transport=httpx.MockTransport(handler)) as nhl:
        nhl.club_schedule_season("col", 20252026)
    assert seen == ["/v1/club-schedule-season/COL/20252026"]


def test_http_errors_raise():
    def handler(request):
        return httpx.Response(404, text="not found")

    with (
        NhlClient(transport=httpx.MockTransport(handler)) as nhl,
        pytest.raises(NhlApiError, match="HTTP 404"),
    ):
        nhl.standings()


def test_transport_failures_are_retried(monkeypatch):
    monkeypatch.setattr("betsim.nhl.time.sleep", lambda _: None)
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json={"standings": []})

    with NhlClient(transport=httpx.MockTransport(handler), retries=2) as nhl:
        assert nhl.standings() == {"standings": []}
    assert len(attempts) == 2


def test_a_non_object_response_is_rejected():
    def handler(request):
        return httpx.Response(200, json=[1, 2, 3])

    with (
        NhlClient(transport=httpx.MockTransport(handler)) as nhl,
        pytest.raises(NhlApiError, match="expected a JSON object"),
    ):
        nhl.standings()


# --- pre-game detail --------------------------------------------------------


def test_parse_goalies_reads_the_depth_chart(nhl_landing_payload):
    from betsim.nhl import parse_goalies

    goalies = parse_goalies(nhl_landing_payload)
    assert set(goalies) == {"home", "away"}
    season, lines = goalies["home"]
    assert season == 20252026  # LAST season -- the stats are not current
    assert len(lines) >= 1
    g = lines[0]
    assert g.games_played > 0
    assert 0.0 < g.save_pct < 1.0
    assert "-" in g.record


def test_parse_team_form_reads_rates_and_league_ranks(nhl_right_rail_payload):
    from betsim.nhl import parse_team_form

    form = parse_team_form(nhl_right_rail_payload)
    home = form["home"]
    assert home.season == 20252026
    assert home.goals_for_per_game > 0
    assert 1 <= home.goals_for_rank <= 32
    assert 0.0 < home.power_play_pct < 1.0


def test_parse_scratches_is_empty_before_puck_drop(nhl_right_rail_payload):
    from betsim.nhl import parse_scratches

    # Scratches populate close to game time, which is why a late context
    # refresh is worth scheduling.
    assert parse_scratches(nhl_right_rail_payload) == {"home": [], "away": []}


def test_schedule_ids_map_teams_to_the_nhl_game_id(nhl_score_payload):
    from betsim.nhl import parse_schedule_ids

    ids = parse_schedule_ids(nhl_score_payload)
    assert ids, "expected at least one game"
    (home, away), gid = next(iter(ids.items()))
    assert len(home) == 3 and len(away) == 3
    assert isinstance(gid, int)


def test_parsers_tolerate_a_payload_without_the_block():
    from betsim.nhl import parse_goalies, parse_schedule_ids, parse_scratches, parse_team_form

    assert parse_goalies({}) == {"home": (0, []), "away": (0, [])}
    assert parse_team_form({})["home"].goals_for_per_game is None
    assert parse_scratches({}) == {"home": [], "away": []}
    assert parse_schedule_ids({}) == {}
