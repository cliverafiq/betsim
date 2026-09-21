import pytest

from betsim.devig import METHODS, consensus, devig, multiplicative, power, shin


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("prices", [[1.20, 4.50], [1.91, 1.91], [2.10, 3.40, 3.80], [1.05, 15.0]])
def test_all_methods_normalise(method, prices):
    p = devig(prices, method)
    assert sum(p) == pytest.approx(1.0, abs=1e-9)
    assert all(0.0 < x < 1.0 for x in p)
    assert len(p) == len(prices)


@pytest.mark.parametrize("method", METHODS)
def test_symmetric_market_splits_evenly(method):
    p = devig([1.91, 1.91], method)
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(0.5)


@pytest.mark.parametrize("method", METHODS)
def test_no_vig_book_is_identity(method):
    assert devig([2.0, 2.0], method) == pytest.approx([0.5, 0.5])


def test_shin_and_power_correct_favourite_longshot_bias():
    # Longshots are overbet, so their raw implied probability overstates the truth.
    # Correcting for that must move probability *toward* the favourite relative to
    # the proportional (multiplicative) split. This is the whole reason the project
    # does not use multiplicative de-vig for its primary numbers.
    prices = [1.20, 4.50]
    m, p, s = multiplicative(prices), power(prices), shin(prices)
    assert s[0] > m[0], "Shin should lift the favourite above the proportional split"
    assert p[0] > m[0], "power should lift the favourite above the proportional split"
    assert s[1] < m[1] and p[1] < m[1], "and shrink the longshot"
    assert m[0] == pytest.approx(0.789474, abs=1e-5)
    assert s[0] == pytest.approx(0.805556, abs=1e-5)
    assert p[0] == pytest.approx(0.815013, abs=1e-5)


def test_three_way_market():
    p = devig([2.10, 3.40, 3.80], "shin")
    assert sum(p) == pytest.approx(1.0)
    assert p[0] > p[1] > p[2]


def test_devig_rejects_unknown_method():
    with pytest.raises(ValueError, match="unknown de-vig method"):
        devig([2.0, 2.0], "bogus")


def test_devig_requires_at_least_two_outcomes():
    with pytest.raises(ValueError, match="at least 2 outcomes"):
        devig([2.0], "shin")


def test_consensus_averages_across_bookmakers():
    books = [{"home": 1.90, "away": 2.00}, {"home": 2.00, "away": 1.90}]
    c = consensus(books, ("home", "away"), "multiplicative")
    assert c["home"] == pytest.approx(0.5, abs=1e-9)
    assert c["home"] + c["away"] == pytest.approx(1.0)


def test_consensus_skips_books_missing_an_outcome():
    # A book quoting only one side would misstate the margin if de-vigged alone.
    books = [{"home": 1.90, "away": 2.00}, {"home": 1.50}]
    c = consensus(books, ("home", "away"), "shin")
    only_complete = devig([1.90, 2.00], "shin")
    assert c["home"] == pytest.approx(only_complete[0])


def test_consensus_raises_when_no_book_is_complete():
    with pytest.raises(ValueError, match="no bookmaker quotes every outcome"):
        consensus([{"home": 1.5}], ("home", "away"), "shin")
