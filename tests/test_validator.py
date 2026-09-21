from datetime import UTC, datetime, timedelta

import pytest

from betsim.sports import get_sport
from betsim.tiers import Tier
from betsim.validator import BetProposal, GameRef, RejectReason, validate_bets

NHL = get_sport("icehockey_nhl")
NOW = datetime(2026, 9, 29, 22, 0, tzinfo=UTC)
BALANCE = 100_000  # 1,000 units
EVENS = {"home": 1.91, "away": 1.91}
SAFE_PRICE = {"home": 1.50, "away": 2.80}  # implied 0.667 -> SAFE, cap 5% = 5,000


def game(gid, *, starts_in_min=120, snapshot_age_min=5, prices=None):
    return GameRef(
        game_id=gid,
        commence_utc=NOW + timedelta(minutes=starts_in_min),
        snapshot_captured_utc=NOW - timedelta(minutes=snapshot_age_min),
        prices=prices if prices is not None else EVENS,
    )


def run(proposals, games, **kw):
    kw.setdefault("balance_minor", BALANCE)
    return validate_bets(proposals, spec=NHL, games=games, now=NOW, **kw)


def test_accepts_a_clean_bet():
    res = run([BetProposal("g1", "home", 20.0, p_revised=0.55)], {"g1": game("g1")})
    assert not res.rejected
    (bet,) = res.accepted
    assert bet.stake_minor == 2_000
    assert bet.price_decimal == 1.91
    assert bet.tier is Tier.MEDIUM
    assert bet.p_revised == 0.55


def test_zero_proposals_is_valid_and_not_an_error():
    res = run([], {})
    assert res.accepted == [] and res.rejected == []


def test_rejects_unknown_game():
    res = run([BetProposal("nope", "home", 10.0)], {"g1": game("g1")})
    assert res.rejected[0].reason is RejectReason.GAME_NOT_FOUND


def test_rejects_game_that_has_started():
    res = run([BetProposal("g1", "home", 10.0)], {"g1": game("g1", starts_in_min=-1)})
    assert res.rejected[0].reason is RejectReason.GAME_ALREADY_STARTED


def test_rejects_stale_snapshot():
    res = run([BetProposal("g1", "home", 10.0)], {"g1": game("g1", snapshot_age_min=61)})
    assert res.rejected[0].reason is RejectReason.SNAPSHOT_STALE


def test_accepts_snapshot_exactly_at_the_age_limit():
    res = run([BetProposal("g1", "home", 10.0)], {"g1": game("g1", snapshot_age_min=60)})
    assert res.accepted and not res.rejected


def test_rejects_draw_in_a_two_way_market():
    res = run([BetProposal("g1", "draw", 10.0)], {"g1": game("g1")})
    assert res.rejected[0].reason is RejectReason.INVALID_SELECTION


def test_rejects_selection_with_no_price():
    res = run([BetProposal("g1", "away", 10.0)], {"g1": game("g1", prices={"home": 1.91})})
    assert res.rejected[0].reason is RejectReason.NO_PRICE


def test_rejects_second_bet_on_the_same_game():
    proposals = [BetProposal("g1", "home", 10.0), BetProposal("g1", "away", 10.0)]
    res = run(proposals, {"g1": game("g1")})
    assert len(res.accepted) == 1
    assert res.rejected[0].reason is RejectReason.DUPLICATE_GAME


def test_rejects_game_that_already_has_an_open_bet():
    res = run([BetProposal("g1", "home", 10.0)], {"g1": game("g1")}, open_game_ids={"g1"})
    assert res.rejected[0].reason is RejectReason.DUPLICATE_GAME


def test_rejects_stake_below_one_unit():
    res = run([BetProposal("g1", "home", 0.5)], {"g1": game("g1")})
    assert res.rejected[0].reason is RejectReason.STAKE_BELOW_MIN


def test_rejects_stake_above_tier_cap_without_clipping():
    # SAFE cap is 5% of 100,000 = 5,000 minor. 60 units = 6,000.
    res = run([BetProposal("g1", "home", 60.0)], {"g1": game("g1", prices=SAFE_PRICE)})
    assert res.accepted == [], "the validator must reject, never clip"
    assert res.rejected[0].reason is RejectReason.STAKE_EXCEEDS_TIER_CAP


def test_exposure_cap_binds_and_rejects_later_bets_in_order():
    # Exposure cap is 20% of 100,000 = 20,000. Four 5,000 bets fill it exactly.
    games = {f"g{i}": game(f"g{i}", prices=SAFE_PRICE) for i in range(1, 6)}
    proposals = [BetProposal(f"g{i}", "home", 50.0) for i in range(1, 6)]
    res = run(proposals, games)
    assert [b.game_id for b in res.accepted] == ["g1", "g2", "g3", "g4"]
    assert res.staked_minor == 20_000
    assert len(res.rejected) == 1
    assert res.rejected[0].proposal.game_id == "g5"
    assert res.rejected[0].reason is RejectReason.STAKE_EXCEEDS_EXPOSURE


def test_open_exposure_from_earlier_slates_counts_against_the_cap():
    games = {"g1": game("g1", prices=SAFE_PRICE)}
    res = run([BetProposal("g1", "home", 50.0)], games, open_exposure_minor=18_000)
    assert res.rejected[0].reason is RejectReason.STAKE_EXCEEDS_EXPOSURE


def test_bust_arm_rejects_everything():
    res = run([BetProposal("g1", "home", 10.0)], {"g1": game("g1")}, balance_minor=99)
    assert res.accepted == []
    assert res.rejected[0].reason is RejectReason.ARM_BUST


def test_rejection_order_and_count_are_preserved():
    games = {"g1": game("g1"), "g2": game("g2")}
    proposals = [
        BetProposal("g1", "home", 10.0),
        BetProposal("bad", "home", 10.0),
        BetProposal("g2", "draw", 10.0),
    ]
    res = run(proposals, games)
    assert [b.game_id for b in res.accepted] == ["g1"]
    assert [r.reason for r in res.rejected] == [
        RejectReason.GAME_NOT_FOUND,
        RejectReason.INVALID_SELECTION,
    ]


def test_naive_datetime_is_refused():
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_bets([], spec=NHL, games={}, balance_minor=BALANCE, now=datetime(2026, 9, 29))  # noqa: DTZ001 -- naive input is what this test asserts against
