import pytest

from betsim.teams import TeamIndex, UnknownTeam, normalise_team_name


def test_normalisation_strips_accents():
    assert normalise_team_name("Montréal Canadiens") == "montreal canadiens"
    assert normalise_team_name("Montreal Canadiens") == "montreal canadiens"


def test_normalisation_strips_punctuation():
    assert normalise_team_name("St. Louis Blues") == "st louis blues"
    assert normalise_team_name("St Louis Blues") == "st louis blues"


def test_normalisation_collapses_case_and_whitespace():
    assert normalise_team_name("  BOSTON   bruins ") == "boston bruins"


def test_index_is_built_from_live_standings(team_index):
    # Derived, not hardcoded, so renames and relocations do not need a code change.
    assert len(team_index) == 32


@pytest.mark.parametrize(
    ("odds_api_name", "tricode"),
    [
        ("Carolina Hurricanes", "CAR"),
        ("Montreal Canadiens", "MTL"),  # accent mismatch between the two feeds
        ("Montréal Canadiens", "MTL"),
        ("St Louis Blues", "STL"),  # punctuation mismatch
        ("St. Louis Blues", "STL"),
        ("Toronto Maple Leafs", "TOR"),
        ("Vegas Golden Knights", "VGK"),
    ],
)
def test_real_feed_mismatches_resolve(team_index, odds_api_name, tricode):
    assert team_index.tricode(odds_api_name) == tricode


def test_renamed_franchises_resolve_through_aliases(team_index):
    assert team_index.tricode("Utah Hockey Club") == "UTA"
    assert team_index.tricode("Arizona Coyotes") == "UTA"
    assert team_index.tricode("Utah Mammoth") == "UTA"


def test_unknown_team_raises_rather_than_dropping_the_game(team_index):
    # A silent miss would leave a game with no context and no Elo rating, and
    # nothing downstream would notice.
    with pytest.raises(UnknownTeam, match="does not match any NHL team"):
        team_index.tricode("Quebec Nordiques")


def test_error_points_at_the_fix(team_index):
    with pytest.raises(UnknownTeam, match="ALIASES"):
        team_index.tricode("Hartford Whalers")


def test_reverse_lookup(team_index):
    assert team_index.name("col") == "Colorado Avalanche"
    with pytest.raises(UnknownTeam, match="unknown tricode"):
        team_index.name("XXX")


def test_known_and_unresolved(team_index):
    assert team_index.known("Boston Bruins")
    assert not team_index.known("Quebec Nordiques")
    names = ["Boston Bruins", "Quebec Nordiques", "Hartford Whalers"]
    assert team_index.unresolved(names) == ["Quebec Nordiques", "Hartford Whalers"]


def test_every_standings_name_resolves_to_its_own_tricode(standings, team_index):
    for team in standings:
        assert team_index.tricode(team.name) == team.tricode


def test_empty_standings_is_refused():
    with pytest.raises(ValueError, match="empty standings"):
        TeamIndex.from_standings([])
