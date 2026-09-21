"""End-to-end walk through the pure logic for one synthetic NHL slate.

The unit tests check each module alone; this one checks they compose -- prices
in, tiers and stakes out, settled, and the ledger balancing. No network, no LLM.
"""

from datetime import UTC, datetime, timedelta

import pytest

from betsim.config import EXPOSURE_CAP_PCT
from betsim.devig import consensus
from betsim.kelly import best_outcome, kelly_stake_minor
from betsim.metrics import brier_score, clv, max_drawdown, roi
from betsim.money import STARTING_BALANCE_MINOR, pct_of_balance_minor
from betsim.settlement import GameStatus, profit_minor, settle_bet
from betsim.sports import get_sport
from betsim.tiers import Tier, assign_tier, tier_cap_minor
from betsim.validator import BetProposal, GameRef, RejectReason, validate_bets

NHL = get_sport("icehockey_nhl")
NOW = datetime(2026, 9, 29, 22, 0, tzinfo=UTC)

# Three games at a designated book, plus a second book for the consensus.
PINNACLE = {
    "g1": {"home": 1.50, "away": 2.80},
    "g2": {"home": 1.91, "away": 1.91},
    "g3": {"home": 3.20, "away": 1.38},
}
OTHER_BOOK = {
    "g1": {"home": 1.52, "away": 2.70},
    "g2": {"home": 1.95, "away": 1.87},
    "g3": {"home": 3.10, "away": 1.40},
}

# Stage 1 blind forecasts (synthetic, but the shapes a real run would produce).
BLIND = {
    "g1": {"home": 0.70, "away": 0.30},
    "g2": {"home": 0.58, "away": 0.42},
    "g3": {"home": 0.25, "away": 0.75},
}

RESULTS = {  # home_score, away_score
    "g1": (4, 1),  # home wins
    "g2": (2, 3),  # away wins
    "g3": (0, 2),  # away wins
}


def _games():
    return {
        gid: GameRef(
            game_id=gid,
            commence_utc=NOW + timedelta(hours=1),
            snapshot_captured_utc=NOW - timedelta(minutes=5),
            prices=prices,
        )
        for gid, prices in PINNACLE.items()
    }


def test_consensus_is_a_probability_distribution_per_game():
    for gid, pinnacle in PINNACLE.items():
        c = consensus([pinnacle, OTHER_BOOK[gid]], ("home", "away"), "shin")
        assert sum(c.values()) == pytest.approx(1.0)
        assert all(0 < p < 1 for p in c.values())


def test_tiers_are_assigned_from_the_designated_book_price():
    assert assign_tier(PINNACLE["g1"]["home"]) is Tier.SAFE  # 1.50 -> implied 0.667
    assert assign_tier(PINNACLE["g2"]["home"]) is Tier.MEDIUM  # 1.91 -> implied 0.524
    assert assign_tier(PINNACLE["g3"]["home"]) is Tier.RISKY  # 3.20 -> implied 0.313
    assert assign_tier(PINNACLE["g3"]["away"]) is Tier.SAFE  # 1.38 -> implied 0.725


def test_kelly_arm_builds_a_legal_slate():
    balance = STARTING_BALANCE_MINOR
    staked = 0
    picks = {}
    for gid, prices in PINNACLE.items():
        pick = best_outcome(BLIND[gid], prices)
        assert pick is not None, f"{gid} should show an edge on these synthetic numbers"
        outcome, _ = pick
        stake = kelly_stake_minor(BLIND[gid][outcome], prices[outcome], balance)
        tier = assign_tier(prices[outcome])
        assert 0 < stake <= tier_cap_minor(tier, balance), "stake must respect its tier cap"
        picks[gid] = (outcome, stake)
        staked += stake
    assert staked <= pct_of_balance_minor(balance, EXPOSURE_CAP_PCT)
    # The forecasts favour the side the market underprices in each game.
    assert picks["g1"][0] == "home"
    assert picks["g2"][0] == "home"
    assert picks["g3"][0] == "away"


def test_full_slate_places_settles_and_the_ledger_balances():
    balance = STARTING_BALANCE_MINOR
    opening_balance = balance

    proposals = [
        BetProposal("g1", "home", 25.0, p_revised=0.68, reason="rest edge"),
        BetProposal("g2", "home", 20.0, p_revised=0.56, reason="goalie"),
        BetProposal("g3", "away", 10.0, p_revised=0.72, reason="road form"),
    ]
    result = validate_bets(proposals, spec=NHL, games=_games(), balance_minor=balance, now=NOW)
    assert not result.rejected
    assert len(result.accepted) == 3

    # Stakes leave the balance at placement.
    balance -= result.staked_minor
    path = [opening_balance, balance]

    total_payout = 0
    profits, stakes = [], []
    for bet in result.accepted:
        home_score, away_score = RESULTS[bet.game_id]
        status, payout = settle_bet(
            NHL,
            selection=bet.selection,
            price_decimal=bet.price_decimal,
            stake_minor=bet.stake_minor,
            game_status=GameStatus.COMPLETED,
            home_score=home_score,
            away_score=away_score,
        )
        total_payout += payout
        profits.append(profit_minor(status, bet.stake_minor, payout))
        stakes.append(bet.stake_minor)
        balance += payout
        path.append(balance)

    # The ledger invariant: no money is created or destroyed.
    assert balance == opening_balance - result.staked_minor + total_payout
    assert sum(profits) == balance - opening_balance

    # g1 (home, 4-1) and g3 (away, 0-2) won; g2 (home, 2-3) lost. The single
    # loss is the largest stake, so the slate still finishes down.
    assert profits[0] > 0 and profits[1] < 0 and profits[2] > 0
    assert sum(profits) < 0
    assert roi(profits, stakes) == pytest.approx(sum(profits) / sum(stakes))
    assert balance < opening_balance
    assert max_drawdown(path) > 0


def test_exposure_cap_binds_before_the_balance_does():
    # Eight safe-priced bets at the tier cap would be 40% of bankroll; the
    # exposure cap must stop it at 20% long before the balance runs out.
    games = {
        f"g{i}": GameRef(
            f"g{i}",
            NOW + timedelta(hours=1),
            NOW - timedelta(minutes=5),
            {"home": 1.50, "away": 2.80},
        )
        for i in range(8)
    }
    proposals = [BetProposal(f"g{i}", "home", 50.0) for i in range(8)]
    res = validate_bets(
        proposals, spec=NHL, games=games, balance_minor=STARTING_BALANCE_MINOR, now=NOW
    )
    assert res.staked_minor == 20_000
    assert len(res.accepted) == 4
    assert all(r.reason is RejectReason.STAKE_EXCEEDS_EXPOSURE for r in res.rejected)


def test_forecast_scoring_against_the_market_consensus():
    # The headline comparison: Stage 1 versus the de-vigged consensus, scored on
    # every game rather than only the ones bet on.
    blind_scores, market_scores = [], []
    for gid, pinnacle in PINNACLE.items():
        home_score, away_score = RESULTS[gid]
        actual = "home" if home_score > away_score else "away"
        market = consensus([pinnacle, OTHER_BOOK[gid]], ("home", "away"), "shin")
        blind_scores.append(brier_score(BLIND[gid], actual))
        market_scores.append(brier_score(market, actual))
    assert all(0.0 <= s <= 2.0 for s in blind_scores + market_scores)
    assert len(blind_scores) == len(market_scores) == 3


def test_clv_is_measured_against_the_closing_price():
    # Bet taken at 1.91, closed at 1.85 -> the price shortened, so positive CLV.
    assert clv(1.91, 1.85) > 0
    # Taken at 1.91, closed at 2.00 -> drifted out, negative CLV.
    assert clv(1.91, 2.00) < 0


def test_a_bust_arm_stops_betting():
    res = validate_bets(
        [BetProposal("g1", "home", 10.0)],
        spec=NHL,
        games=_games(),
        balance_minor=50,  # below one unit
        now=NOW,
    )
    assert res.accepted == []
    assert res.rejected[0].reason is RejectReason.ARM_BUST
