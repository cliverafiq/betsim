from datetime import UTC, datetime, timedelta

import pytest

from betsim.arms import ArmBet
from betsim.db import connect, init_db
from betsim.ingest import IngestedPrice
from betsim.ledger import balance, place_bet, verify_ledger
from betsim.money import STARTING_BALANCE_MINOR
from betsim.settle import bet_clv, clv_by_arm, settle_open_bets
from betsim.store import insert_snapshots
from betsim.tiers import Tier

PLACED = datetime(2026, 9, 29, 20, tzinfo=UTC)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    init_db(c)
    for gid, home, away in (
        ("g_win", "Carolina Hurricanes", "Florida Panthers"),
        ("g_loss", "Toronto Maple Leafs", "Montreal Canadiens"),
        ("g_void", "Boston Bruins", "New York Rangers"),
    ):
        c.execute(
            "INSERT INTO games (id, sport_key, home, away, commence_utc) VALUES (?,?,?,?,?)",
            (gid, "icehockey_nhl", home, away, "2026-09-29T21:00:00Z"),
        )
    c.commit()
    yield c
    c.close()


def bet(game_id, *, stake=2_000, price=2.0, selection="home"):
    return ArmBet(
        game_id=game_id,
        selection=selection,
        stake_minor=stake,
        price_decimal=price,
        tier=Tier.MEDIUM,
        p_blind=0.6,
        reason="x",
    )


def complete(conn, game_id, home, away):
    conn.execute(
        "UPDATE games SET status='completed', home_score=?, away_score=?, completed_utc=? "
        "WHERE id=?",
        (home, away, "2026-09-30T02:00:00Z", game_id),
    )


def test_settles_wins_losses_and_voids(conn):
    place_bet(conn, "llm_s1", bet("g_win"), placed_utc=PLACED)
    place_bet(conn, "llm_s1", bet("g_loss"), placed_utc=PLACED)
    place_bet(conn, "llm_s1", bet("g_void"), placed_utc=PLACED)
    complete(conn, "g_win", 4, 1)
    complete(conn, "g_loss", 1, 4)
    conn.execute("UPDATE games SET status='postponed' WHERE id='g_void'")

    result = settle_open_bets(conn, "icehockey_nhl")
    assert (result.settled, result.won, result.lost, result.void) == (3, 1, 1, 1)
    # +2,000 on the win, -2,000 on the loss, 0 on the void.
    assert result.profit_minor["llm_s1"] == 0
    assert balance(conn, "llm_s1") == STARTING_BALANCE_MINOR
    verify_ledger(conn, "llm_s1")


def test_leaves_unfinished_games_alone(conn):
    place_bet(conn, "llm_s1", bet("g_win"), placed_utc=PLACED)
    assert settle_open_bets(conn, "icehockey_nhl").settled == 0
    row = conn.execute("SELECT status FROM bets").fetchone()
    assert row["status"] == "open"


def test_a_bet_is_settled_only_once(conn):
    place_bet(conn, "llm_s1", bet("g_win"), placed_utc=PLACED)
    complete(conn, "g_win", 4, 1)
    assert settle_open_bets(conn, "icehockey_nhl").settled == 1
    assert settle_open_bets(conn, "icehockey_nhl").settled == 0


def test_bad_data_on_one_game_does_not_strand_the_rest(conn):
    # An NHL game reported level cannot happen; it must be recorded and skipped
    # rather than taking down the night's settlement.
    place_bet(conn, "llm_s1", bet("g_win"), placed_utc=PLACED)
    place_bet(conn, "llm_s1", bet("g_loss"), placed_utc=PLACED)
    complete(conn, "g_win", 2, 2)  # impossible
    complete(conn, "g_loss", 1, 4)

    result = settle_open_bets(conn, "icehockey_nhl")
    assert result.settled == 1
    assert len(result.errors) == 1
    assert "cannot tie" in result.errors[0]
    verify_ledger(conn, "llm_s1")


def test_each_arm_settles_against_its_own_bankroll(conn):
    place_bet(conn, "llm_s1", bet("g_win", stake=1_000), placed_utc=PLACED)
    place_bet(conn, "fav", bet("g_win", stake=1_000), placed_utc=PLACED)
    complete(conn, "g_win", 4, 1)
    result = settle_open_bets(conn, "icehockey_nhl")
    assert result.by_arm == {"llm_s1": 1, "fav": 1}
    for arm in ("llm_s1", "fav"):
        assert balance(conn, arm) == STARTING_BALANCE_MINOR + 1_000
        verify_ledger(conn, arm)


# --- CLV --------------------------------------------------------------------


def _closing(conn, game_id, selection, price):
    insert_snapshots(
        conn,
        [IngestedPrice(game_id, "pinnacle", "h2h", selection, price, None)],
        captured_utc=PLACED + timedelta(hours=1),
        closing_game_ids={game_id},
    )


def test_clv_is_positive_when_the_price_shortens(conn):
    place_bet(conn, "llm_s1", bet("g_win", price=2.10), placed_utc=PLACED)
    _closing(conn, "g_win", "home", 2.00)
    (entry,) = bet_clv(conn)
    assert entry.clv == pytest.approx(0.05)
    assert entry.price_taken == 2.10 and entry.price_close == 2.00


def test_clv_is_negative_when_the_price_drifts(conn):
    place_bet(conn, "llm_s1", bet("g_win", price=1.90), placed_utc=PLACED)
    _closing(conn, "g_win", "home", 2.00)
    assert bet_clv(conn)[0].clv < 0


def test_bets_without_a_closing_price_are_skipped(conn):
    place_bet(conn, "llm_s1", bet("g_win"), placed_utc=PLACED)
    assert bet_clv(conn) == []


def test_only_closing_snapshots_count(conn):
    # A mid-afternoon snapshot is not the close.
    place_bet(conn, "llm_s1", bet("g_win", price=2.10), placed_utc=PLACED)
    insert_snapshots(
        conn,
        [IngestedPrice("g_win", "pinnacle", "h2h", "home", 2.00, None)],
        captured_utc=PLACED,
        closing_game_ids=set(),
    )
    assert bet_clv(conn) == []


def test_clv_groups_by_arm_with_random_as_the_null(conn):
    place_bet(conn, "llm_s1", bet("g_win", price=2.20), placed_utc=PLACED)
    place_bet(conn, "random", bet("g_win", price=2.05), placed_utc=PLACED)
    _closing(conn, "g_win", "home", 2.00)

    summary = clv_by_arm(bet_clv(conn))
    assert summary["llm_s1"]["mean_clv"] == pytest.approx(0.10)
    assert summary["random"]["mean_clv"] == pytest.approx(0.025)
    # The skill signal is the gap to the random arm, not the raw number: betting
    # early earns some CLV with no skill at all.
    assert summary["llm_s1"]["mean_clv"] > summary["random"]["mean_clv"]


def test_clv_can_be_filtered_to_one_arm(conn):
    place_bet(conn, "llm_s1", bet("g_win", price=2.20), placed_utc=PLACED)
    place_bet(conn, "fav", bet("g_win", price=2.05), placed_utc=PLACED)
    _closing(conn, "g_win", "home", 2.00)
    assert {e.arm for e in bet_clv(conn, arm="fav")} == {"fav"}
