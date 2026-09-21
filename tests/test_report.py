from datetime import UTC, datetime, timedelta

import pytest

from betsim.arms import ArmBet
from betsim.config import DESIGNATED_BOOKMAKER
from betsim.db import connect, init_db
from betsim.ingest import IngestedPrice
from betsim.ledger import place_bet, settle_bet
from betsim.report import (
    arm_reports,
    behaviour_report,
    fit_calibration,
    forecast_quality,
    render,
    tier_breakdown,
)
from betsim.settlement import BetStatus
from betsim.store import insert_snapshots
from betsim.tiers import Tier

DAY1 = datetime(2026, 9, 29, 20, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_db(c)
    yield c
    c.close()


def add_game(conn, gid, *, home_score=None, away_score=None, status="scheduled"):
    conn.execute(
        "INSERT INTO games (id, sport_key, home, away, commence_utc, status, "
        "home_score, away_score) VALUES (?,?,?,?,?,?,?,?)",
        (
            gid,
            "icehockey_nhl",
            f"{gid} home",
            f"{gid} away",
            "2026-09-29T21:00:00Z",
            status,
            home_score,
            away_score,
        ),
    )


def add_prices(conn, gid, home, away, *, closing=False):
    insert_snapshots(
        conn,
        [
            IngestedPrice(gid, DESIGNATED_BOOKMAKER, "h2h", "home", home, None),
            IngestedPrice(gid, DESIGNATED_BOOKMAKER, "h2h", "away", away, None),
            IngestedPrice(gid, "bovada", "h2h", "home", home * 1.01, None),
            IngestedPrice(gid, "bovada", "h2h", "away", away * 1.01, None),
        ],
        captured_utc=DAY1,
        closing_game_ids={gid} if closing else set(),
    )


def add_forecast(conn, gid, p_home, *, seed=0):
    cur = conn.execute(
        "INSERT INTO llm_calls (stage, seed_idx, model_id, prompt_version, params_json, "
        "input_hash, input_json, created_utc, ok) VALUES (1,?,'m','v','{}','h','{}',?,1)",
        (seed, DAY1.isoformat()),
    )
    conn.execute(
        "INSERT INTO forecasts (game_id, llm_call_id, seed_idx, p_home, p_away) VALUES (?,?,?,?,?)",
        (gid, cur.lastrowid, seed, p_home, 1 - p_home),
    )


def bet(gid, *, stake=2_000, price=2.0, tier=Tier.MEDIUM):
    return ArmBet(gid, "home", stake, price, tier, p_blind=0.6, reason="x")


def test_arm_reports_on_an_empty_database(conn):
    assert arm_reports(conn) == []


def test_arm_reports_summarise_a_settled_arm(conn):
    add_game(conn, "g1", home_score=4, away_score=1, status="completed")
    add_game(conn, "g2", home_score=1, away_score=4, status="completed")
    b1 = place_bet(conn, "llm_s1", bet("g1", stake=2_000), placed_utc=DAY1)
    b2 = place_bet(conn, "llm_s1", bet("g2", stake=1_000), placed_utc=DAY2)
    settle_bet(conn, b1, status=BetStatus.WON, payout_minor=4_000)
    settle_bet(conn, b2, status=BetStatus.LOST, payout_minor=0)

    (report,) = arm_reports(conn, bootstrap=False)
    assert report.arm == "llm_s1"
    assert report.bets == 2
    assert report.staked_minor == 3_000
    assert report.profit_minor == 1_000
    assert report.roi == pytest.approx(1_000 / 3_000)
    assert report.balance_minor == 101_000
    assert not report.bust


def test_roi_interval_is_bootstrapped_by_day(conn):
    for i in range(8):
        gid = f"g{i}"
        add_game(
            conn,
            gid,
            home_score=4 if i % 2 else 1,
            away_score=1 if i % 2 else 4,
            status="completed",
        )
        bid = place_bet(conn, "llm_s1", bet(gid, stake=1_000), placed_utc=DAY1 + timedelta(days=i))
        settle_bet(
            conn,
            bid,
            status=BetStatus.WON if i % 2 else BetStatus.LOST,
            payout_minor=2_000 if i % 2 else 0,
        )
    (report,) = arm_reports(conn)
    assert report.roi_ci is not None
    lo, hi = report.roi_ci
    assert lo < report.roi < hi


def test_tier_breakdown_splits_by_tier(conn):
    add_game(conn, "g1", home_score=4, away_score=1, status="completed")
    add_game(conn, "g2", home_score=1, away_score=4, status="completed")
    b1 = place_bet(conn, "a", bet("g1", tier=Tier.SAFE, stake=2_000), placed_utc=DAY1)
    b2 = place_bet(conn, "a", bet("g2", tier=Tier.RISKY, stake=500), placed_utc=DAY1)
    settle_bet(conn, b1, status=BetStatus.WON, payout_minor=4_000)
    settle_bet(conn, b2, status=BetStatus.LOST, payout_minor=0)

    tiers = tier_breakdown(conn, "a")
    assert tiers["safe"]["bets"] == 1 and tiers["safe"]["roi"] == pytest.approx(1.0)
    assert tiers["risky"]["roi"] == pytest.approx(-1.0)


def test_forecast_quality_compares_blind_against_the_market(conn):
    for i in range(6):
        gid = f"g{i}"
        home_won = i % 2 == 0
        add_game(
            conn,
            gid,
            home_score=4 if home_won else 1,
            away_score=1 if home_won else 4,
            status="completed",
        )
        add_prices(conn, gid, 1.90, 2.00)
        add_forecast(conn, gid, 0.55)

    q = forecast_quality(conn)
    assert q.games == 6
    assert 0.0 <= q.brier_blind <= 2.0
    assert 0.0 <= q.brier_market <= 2.0
    assert q.brier_gap is not None
    assert len(q.pairs) == 6


def test_forecast_quality_averages_across_the_k_draws(conn):
    add_game(conn, "g1", home_score=4, away_score=1, status="completed")
    add_prices(conn, "g1", 1.90, 2.00)
    add_forecast(conn, "g1", 0.50, seed=0)
    add_forecast(conn, "g1", 0.70, seed=1)
    q = forecast_quality(conn)
    assert q.pairs[0][0] == pytest.approx(0.60)  # the ensemble mean


def test_forecast_quality_skips_games_without_a_market(conn):
    add_game(conn, "g1", home_score=4, away_score=1, status="completed")
    add_forecast(conn, "g1", 0.6)
    assert forecast_quality(conn).games == 0


def test_forecast_quality_scores_all_games_not_only_those_bet_on(conn):
    # Selection must not flatter the forecast.
    for i in range(4):
        add_game(conn, f"g{i}", home_score=4, away_score=1, status="completed")
        add_prices(conn, f"g{i}", 1.90, 2.00)
        add_forecast(conn, f"g{i}", 0.6)
    place_bet(conn, "llm_s1", bet("g0"), placed_utc=DAY1)
    assert forecast_quality(conn).games == 4


def test_behaviour_report_separates_stakes_after_wins_and_losses(conn):
    # Loss chasing: a bigger stake the day after a losing day.
    add_game(conn, "g1", home_score=1, away_score=4, status="completed")
    add_game(conn, "g2", home_score=4, away_score=1, status="completed")
    b1 = place_bet(conn, "llm_s1", bet("g1", stake=1_000), placed_utc=DAY1)
    settle_bet(conn, b1, status=BetStatus.LOST, payout_minor=0)
    b2 = place_bet(conn, "llm_s1", bet("g2", stake=4_000), placed_utc=DAY2)
    settle_bet(conn, b2, status=BetStatus.WON, payout_minor=8_000)

    behaviour = behaviour_report(conn, "llm_s1")
    assert behaviour["days_with_bets"] == 2
    assert behaviour["mean_stake_after_losing_day"] == 4_000
    assert behaviour["mean_stake_after_winning_day"] is None


def test_behaviour_report_counts_rejections(conn):
    conn.execute(
        "INSERT INTO rejections (arm, payload_json, reason, created_utc) "
        "VALUES ('llm_s1','{}','stake_exceeds_tier_cap',?)",
        (DAY1.isoformat(),),
    )
    assert behaviour_report(conn, "llm_s1")["rejections"] == {"stake_exceeds_tier_cap": 1}


def test_calibration_respects_the_burn_in(conn):
    add_game(conn, "g1", home_score=4, away_score=1, status="completed")
    add_prices(conn, "g1", 1.90, 2.00)
    add_forecast(conn, "g1", 0.6)
    params, note = fit_calibration(conn, min_games=200)
    assert params is None
    assert "burn-in is 200" in note


def test_calibration_fits_once_there_is_enough_data(conn):
    import random

    rng = random.Random(1)
    for i in range(300):
        gid = f"g{i}"
        stated = rng.choice([0.55, 0.70, 0.90])
        happened = rng.random() < {0.55: 0.52, 0.70: 0.60, 0.90: 0.70}[stated]
        add_game(
            conn,
            gid,
            home_score=4 if happened else 1,
            away_score=1 if happened else 4,
            status="completed",
        )
        add_prices(conn, gid, 1.90, 2.00)
        add_forecast(conn, gid, stated)

    params, note = fit_calibration(conn, min_games=200)
    assert params is not None
    assert "300 games" in note
    assert params.a < 1.0  # the simulated model is overconfident


def test_render_is_safe_on_an_empty_database(conn):
    text = render(conn)
    assert "CLOSING LINE VALUE" in text
    assert "no bets placed yet" in text


def test_render_includes_every_section_once_populated(conn):
    add_game(conn, "g1", home_score=4, away_score=1, status="completed")
    add_prices(conn, "g1", 1.90, 2.00, closing=True)
    add_forecast(conn, "g1", 0.6)
    bid = place_bet(conn, "random", bet("g1", price=1.95), placed_utc=DAY1)
    settle_bet(conn, bid, status=BetStatus.WON, payout_minor=3_900)

    text = render(conn)
    assert "mean" in text
    assert "Brier" in text
    assert "no-bet baseline" in text
    assert "<- null" in text  # random is labelled as the CLV null
