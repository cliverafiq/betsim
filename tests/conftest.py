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
