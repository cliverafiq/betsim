import pytest

from betsim.odds import (
    american_to_decimal,
    booksum,
    decimal_to_american,
    implied_probability,
    validate_decimal,
)


@pytest.mark.parametrize(
    ("american", "expected"),
    [(100, 2.0), (-100, 2.0), (-110, 1 + 100 / 110), (150, 2.5), (-150, 1 + 100 / 150), (200, 3.0)],
)
def test_american_to_decimal(american, expected):
    assert american_to_decimal(american) == pytest.approx(expected)


@pytest.mark.parametrize("bad", [0, 99, -99, 50, -1])
def test_american_rejects_impossible_prices(bad):
    with pytest.raises(ValueError, match=r"\|odds\| >= 100"):
        american_to_decimal(bad)


@pytest.mark.parametrize("american", [-500, -150, -110, 100, 150, 250, 1000])
def test_american_decimal_round_trip(american):
    assert decimal_to_american(american_to_decimal(american)) == pytest.approx(american)


def test_decimal_to_american_uses_plus_100_at_evens():
    assert decimal_to_american(2.0) == pytest.approx(100.0)


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -2.0])
def test_validate_decimal_rejects_non_prices(bad):
    with pytest.raises(ValueError, match="> 1.0"):
        validate_decimal(bad)


def test_implied_probability():
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(american_to_decimal(-110)) == pytest.approx(0.5238095, abs=1e-6)


def test_breakeven_win_rate_at_minus_110():
    # The spec's stated 52.4% break-even and 4.55% coin-flipper edge.
    d = american_to_decimal(-110)
    assert implied_probability(d) == pytest.approx(0.523810, abs=1e-6)
    coin_flipper_ev = 0.5 * (d - 1) - 0.5
    assert coin_flipper_ev == pytest.approx(-0.045455, abs=1e-6)


def test_booksum_exceeds_one_when_there_is_margin():
    assert booksum([2.0, 2.0]) == pytest.approx(1.0)
    assert booksum([american_to_decimal(-110)] * 2) == pytest.approx(1.047619, abs=1e-6)


def test_booksum_requires_prices():
    with pytest.raises(ValueError):
        booksum([])
