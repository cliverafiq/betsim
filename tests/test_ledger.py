from datetime import UTC, datetime

import pytest

from betsim.arms import ArmBet
from betsim.db import connect, init_db
from betsim.ledger import (
    arm_state,
    balance,
    balance_path,
    open_exposure,
    place_bet,
    settle_bet,
    verify_ledger,
)
from betsim.money import STARTING_BALANCE_MINOR
from betsim.settlement import BetStatus
from betsim.tiers import Tier

PLACED = datetime(2026, 9, 29, 20, tzinfo=UTC)
GAMES = [
    (
        "nhl_car_fla",
        "icehockey_nhl",
        "Carolina Hurricanes",
        "Florida Panthers",
        "2026-09-29T21:00:00Z",
    ),
    (
        "nhl_tor_mtl",
        "icehockey_nhl",
        "Toronto Maple Leafs",
        "Montreal Canadiens",
        "2026-09-29T23:00:00Z",
    ),
]


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_db(c)
    for g in GAMES:
        c.execute(
            "INSERT INTO games (id, sport_key, home, away, commence_utc) VALUES (?,?,?,?,?)", g
        )
    c.commit()
    yield c
    c.close()


def bet(game_id="nhl_car_fla", stake=2_000, price=1.91):
    return ArmBet(
        game_id=game_id,
        selection="home",
        stake_minor=stake,
        price_decimal=price,
        tier=Tier.MEDIUM,
        p_blind=0.58,
        p_revised=0.56,
        reason="home ice",
    )


def test_a_new_arm_starts_at_one_thousand_units(conn):
    assert balance(conn, "llm_s1") == STARTING_BALANCE_MINOR
    assert open_exposure(conn, "llm_s1") == 0


def test_placing_a_bet_deducts_the_stake(conn):
    place_bet(conn, "llm_s1", bet(stake=2_000), placed_utc=PLACED)
    assert balance(conn, "llm_s1") == STARTING_BALANCE_MINOR - 2_000
    assert open_exposure(conn, "llm_s1") == 2_000


def test_a_winning_bet_credits_the_payout(conn):
    bet_id = place_bet(conn, "llm_s1", bet(stake=1_000, price=2.0), placed_utc=PLACED)
    new_balance = settle_bet(conn, bet_id, status=BetStatus.WON, payout_minor=2_000)
    assert new_balance == STARTING_BALANCE_MINOR + 1_000
    assert open_exposure(conn, "llm_s1") == 0


def test_a_losing_bet_returns_nothing(conn):
    bet_id = place_bet(conn, "llm_s1", bet(stake=1_000), placed_utc=PLACED)
    assert settle_bet(conn, bet_id, status=BetStatus.LOST, payout_minor=0) == (
        STARTING_BALANCE_MINOR - 1_000
    )


def test_a_void_refunds_the_stake(conn):
    bet_id = place_bet(conn, "llm_s1", bet(stake=1_000), placed_utc=PLACED)
    settle_bet(conn, bet_id, status=BetStatus.VOID, payout_minor=1_000)
    assert balance(conn, "llm_s1") == STARTING_BALANCE_MINOR


def test_every_settled_bet_leaves_exactly_two_ledger_rows(conn):
    # Including a loss: the zero-delta row makes settlement visible in the audit
    # trail rather than inferred from a missing row.
    bet_id = place_bet(conn, "llm_s1", bet(stake=1_000), placed_utc=PLACED)
    settle_bet(conn, bet_id, status=BetStatus.LOST, payout_minor=0)
    rows = conn.execute(
        "SELECT delta_minor, reason FROM ledger WHERE bet_id = ? ORDER BY id", (bet_id,)
    ).fetchall()
    assert [(r["delta_minor"], r["reason"]) for r in rows] == [
        (-1_000, "stake"),
        (0, "settle:lost"),
    ]


def test_the_ledger_is_append_only_across_a_full_cycle(conn):
    a = place_bet(conn, "llm_s1", bet("nhl_car_fla", stake=2_000), placed_utc=PLACED)
    b = place_bet(conn, "llm_s1", bet("nhl_tor_mtl", stake=1_000), placed_utc=PLACED)
    settle_bet(conn, a, status=BetStatus.WON, payout_minor=3_820)
    settle_bet(conn, b, status=BetStatus.LOST, payout_minor=0)
    assert conn.execute("SELECT COUNT(*) c FROM ledger").fetchone()["c"] == 4
    verify_ledger(conn, "llm_s1")


def test_running_balance_is_internally_consistent(conn):
    for i, stake in enumerate((1_000, 2_000)):
        bet_id = place_bet(conn, "llm_s1", bet(GAMES[i][0], stake=stake), placed_utc=PLACED)
        settle_bet(conn, bet_id, status=BetStatus.WON, payout_minor=stake * 2)
    verify_ledger(conn, "llm_s1")


def test_verify_ledger_catches_a_tampered_balance(conn):
    bet_id = place_bet(conn, "llm_s1", bet(stake=1_000), placed_utc=PLACED)
    settle_bet(conn, bet_id, status=BetStatus.WON, payout_minor=1_910)
    conn.execute("UPDATE ledger SET balance_minor = 999999 WHERE id = 1")
    with pytest.raises(AssertionError, match="!= recorded balance"):
        verify_ledger(conn, "llm_s1")


def test_an_arm_cannot_stake_money_it_does_not_have(conn):
    with pytest.raises(ValueError, match="cannot stake"):
        place_bet(conn, "llm_s1", bet(stake=STARTING_BALANCE_MINOR + 1), placed_utc=PLACED)


def test_a_bet_cannot_be_settled_twice(conn):
    bet_id = place_bet(conn, "llm_s1", bet(stake=1_000), placed_utc=PLACED)
    settle_bet(conn, bet_id, status=BetStatus.WON, payout_minor=1_910)
    with pytest.raises(ValueError, match="already settled"):
        settle_bet(conn, bet_id, status=BetStatus.LOST, payout_minor=0)


def test_settling_an_unknown_bet_is_refused(conn):
    with pytest.raises(ValueError, match="no bet with id"):
        settle_bet(conn, 999, status=BetStatus.WON, payout_minor=1)


def test_arms_have_independent_bankrolls(conn):
    place_bet(conn, "llm_s1", bet(stake=5_000), placed_utc=PLACED)
    assert balance(conn, "llm_s1") == STARTING_BALANCE_MINOR - 5_000
    assert balance(conn, "kelly_s1") == STARTING_BALANCE_MINOR


def test_balance_path_starts_from_the_opening_bankroll(conn):
    bet_id = place_bet(conn, "llm_s1", bet(stake=1_000), placed_utc=PLACED)
    settle_bet(conn, bet_id, status=BetStatus.WON, payout_minor=1_910)
    assert balance_path(conn, "llm_s1") == [100_000, 99_000, 100_910]


def test_arm_state_summarises_the_position(conn):
    place_bet(conn, "llm_s1", bet("nhl_car_fla", stake=2_000), placed_utc=PLACED)
    state = arm_state(conn, "llm_s1")
    assert state.balance_minor == 98_000
    assert state.open_exposure_minor == 2_000
    assert state.open_game_ids == {"nhl_car_fla"}
    assert state.equity_minor == STARTING_BALANCE_MINOR
    assert state.bust is False


def test_an_arm_that_loses_everything_is_bust(conn):
    bet_id = place_bet(conn, "llm_s1", bet(stake=STARTING_BALANCE_MINOR), placed_utc=PLACED)
    settle_bet(conn, bet_id, status=BetStatus.LOST, payout_minor=0)
    state = arm_state(conn, "llm_s1")
    assert state.balance_minor == 0
    assert state.bust is True
