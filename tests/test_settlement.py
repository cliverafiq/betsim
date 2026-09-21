import pytest

from betsim.settlement import (
    BetStatus,
    GameStatus,
    profit_minor,
    settle_bet,
    winning_selection,
)
from betsim.sports import get_sport

NHL = get_sport("icehockey_nhl")
NFL = get_sport("americanfootball_nfl")
EPL = get_sport("soccer_epl")


def test_winning_selection_two_way():
    assert winning_selection(NHL, 3, 2) == "home"
    assert winning_selection(NHL, 2, 3) == "away"


def test_nhl_cannot_tie_and_a_level_score_is_a_data_error():
    # NHL moneylines include overtime and the shootout. A level settled score is
    # bad data, and silently treating it as a push would corrupt the ledger.
    with pytest.raises(ValueError, match="cannot tie"):
        winning_selection(NHL, 2, 2)


def test_nfl_tie_pushes():
    assert winning_selection(NFL, 17, 17) is None


def test_soccer_level_score_is_a_draw_not_a_push():
    assert winning_selection(EPL, 1, 1) == "draw"


def test_winning_bet_pays_stake_times_price():
    status, payout = settle_bet(
        NHL,
        selection="home",
        price_decimal=2.0,
        stake_minor=1_000,
        game_status=GameStatus.COMPLETED,
        home_score=3,
        away_score=2,
    )
    assert status is BetStatus.WON
    assert payout == 2_000
    assert profit_minor(status, 1_000, payout) == 1_000


def test_payout_rounds_down():
    # -110 is 1.909090..., so 1,000 minor units returns 1,909, not 1,910.
    status, payout = settle_bet(
        NHL,
        selection="home",
        price_decimal=1 + 100 / 110,
        stake_minor=1_000,
        game_status=GameStatus.COMPLETED,
        home_score=1,
        away_score=0,
    )
    assert status is BetStatus.WON
    assert payout == 1_909


def test_losing_bet_returns_nothing():
    status, payout = settle_bet(
        NHL,
        selection="away",
        price_decimal=2.0,
        stake_minor=1_000,
        game_status=GameStatus.COMPLETED,
        home_score=3,
        away_score=2,
    )
    assert status is BetStatus.LOST
    assert payout == 0
    assert profit_minor(status, 1_000, payout) == -1_000


@pytest.mark.parametrize("status", [GameStatus.POSTPONED, GameStatus.CANCELLED])
def test_postponed_and_cancelled_games_void_and_refund(status):
    result, payout = settle_bet(
        NHL,
        selection="home",
        price_decimal=2.0,
        stake_minor=1_000,
        game_status=status,
    )
    assert result is BetStatus.VOID
    assert payout == 1_000
    assert profit_minor(result, 1_000, payout) == 0


def test_nfl_push_voids_and_refunds():
    status, payout = settle_bet(
        NFL,
        selection="home",
        price_decimal=1.91,
        stake_minor=1_000,
        game_status=GameStatus.COMPLETED,
        home_score=17,
        away_score=17,
    )
    assert status is BetStatus.VOID
    assert payout == 1_000


def test_soccer_draw_can_be_backed_and_won():
    status, payout = settle_bet(
        EPL,
        selection="draw",
        price_decimal=3.40,
        stake_minor=1_000,
        game_status=GameStatus.COMPLETED,
        home_score=1,
        away_score=1,
    )
    assert status is BetStatus.WON
    assert payout == 3_400


def test_draw_is_not_a_valid_selection_in_a_two_way_market():
    with pytest.raises(ValueError, match="not a valid selection"):
        settle_bet(
            NHL,
            selection="draw",
            price_decimal=2.0,
            stake_minor=1_000,
            game_status=GameStatus.COMPLETED,
            home_score=1,
            away_score=0,
        )


@pytest.mark.parametrize("status", [GameStatus.SCHEDULED, GameStatus.IN_PROGRESS])
def test_cannot_settle_an_unfinished_game(status):
    with pytest.raises(ValueError, match="cannot settle"):
        settle_bet(NHL, selection="home", price_decimal=2.0, stake_minor=1_000, game_status=status)


def test_completed_game_must_carry_scores():
    with pytest.raises(ValueError, match="must carry both scores"):
        settle_bet(
            NHL,
            selection="home",
            price_decimal=2.0,
            stake_minor=1_000,
            game_status=GameStatus.COMPLETED,
        )
