import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> list[dict]:
    """Return the payload from a recorded (or synthetic) API fixture."""
    return json.loads((FIXTURES / f"{name}.json").read_text())["payload"]


@pytest.fixture
def odds_payload() -> list[dict]:
    return load_fixture("odds_icehockey_nhl")


@pytest.fixture
def events_payload() -> list[dict]:
    return load_fixture("events_icehockey_nhl")


@pytest.fixture
def scores_payload() -> list[dict]:
    return load_fixture("scores_icehockey_nhl")


def load_nhl_fixture(name: str) -> dict:
    """Real recorded NHL API responses (the API is free, so these are not synthetic)."""
    return json.loads((FIXTURES / "nhl" / f"{name}.json").read_text())


@pytest.fixture
def nhl_standings_payload() -> dict:
    return load_nhl_fixture("standings")


@pytest.fixture
def nhl_schedule_payload() -> dict:
    return load_nhl_fixture("club_schedule_COL")


@pytest.fixture
def standings(nhl_standings_payload):
    from betsim.nhl import parse_standings

    return parse_standings(nhl_standings_payload)


@pytest.fixture
def team_index(standings):
    from betsim.teams import TeamIndex

    return TeamIndex.from_standings(standings)


@pytest.fixture
def nhl_results():
    """Finished games from both recorded club schedules, so COL and BOS each
    have a full 82-game history to test recent form against."""
    from betsim.nhl import parse_club_schedule

    out = []
    for tricode in ("COL", "BOS"):
        out.extend(parse_club_schedule(load_nhl_fixture(f"club_schedule_{tricode}")))
    return out
