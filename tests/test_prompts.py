import pytest

from betsim.context import BANNED_VALUE_TERMS
from betsim.prompts import load_prompt


def test_stage1_prompt_loads():
    prompt = load_prompt("stage1_v1")
    assert prompt.version == "stage1_v1"
    assert "{context}" in prompt.template
    assert len(prompt.sha256) == 64


def test_the_blind_prompt_contains_no_betting_vocabulary():
    # Stage 1 must never see a price, and that starts with the prompt itself.
    template = load_prompt("stage1_v1").template.casefold()
    assert [term for term in BANNED_VALUE_TERMS if term in template] == []


def test_the_blind_prompt_states_the_constraints_the_market_would_otherwise_imply():
    template = load_prompt("stage1_v1").template
    assert "sum to 1" in template  # NHL games cannot end level
    assert "season" in template  # recent games may be from a prior season
    assert "rest_days" in template


def test_render_fills_the_placeholder():
    rendered = load_prompt("stage1_v1").render(context='{"game_id": "g1"}')
    assert '"game_id": "g1"' in rendered
    assert "{context}" not in rendered


def test_render_reports_a_missing_placeholder():
    with pytest.raises(KeyError, match="needs a value"):
        load_prompt("stage1_v1").render()


def test_hash_is_stable_across_loads():
    assert load_prompt("stage1_v1").sha256 == load_prompt("stage1_v1").sha256


def test_unknown_version_lists_what_exists():
    with pytest.raises(FileNotFoundError, match="available:"):
        load_prompt("stage1_v999")
