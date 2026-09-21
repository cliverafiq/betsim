import math

import pytest

from betsim.money import (
    BUST_THRESHOLD_MINOR,
    MINOR_PER_UNIT,
    STARTING_BALANCE_MINOR,
    is_bust,
    minor_to_units,
    pct_of_balance_minor,
    units_to_minor,
)


def test_starting_balance_is_1000_units():
    assert STARTING_BALANCE_MINOR == 100_000
    assert minor_to_units(STARTING_BALANCE_MINOR) == 1000.0


@pytest.mark.parametrize(
    ("units", "expected"),
    [(0, 0), (1, 100), (10, 1000), (12.5, 1250), (3.7, 370), (0.999, 99)],
)
def test_units_to_minor_floors(units, expected):
    assert units_to_minor(units) == expected


def test_units_to_minor_avoids_binary_float_trap():
    # int(0.29 * 100) == 28 in binary floating point. The Decimal path must give 29.
    assert int(0.29 * MINOR_PER_UNIT) == 28  # documents the trap
    assert units_to_minor(0.29) == 29
    assert units_to_minor(12.29) == 1229


@pytest.mark.parametrize("bad", [-1, -0.01])
def test_units_to_minor_rejects_negative(bad):
    with pytest.raises(ValueError, match="non-negative"):
        units_to_minor(bad)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_units_to_minor_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        units_to_minor(bad)


def test_pct_of_balance_floors():
    assert pct_of_balance_minor(100_000, 0.05) == 5_000
    assert pct_of_balance_minor(100_000, 0.20) == 20_000
    assert pct_of_balance_minor(999, 0.01) == 9  # 9.99 floors to 9


@pytest.mark.parametrize(("balance", "pct"), [(-1, 0.05), (100, 1.5), (100, -0.1)])
def test_pct_of_balance_validates(balance, pct):
    with pytest.raises(ValueError):
        pct_of_balance_minor(balance, pct)


def test_bust_boundary_is_below_one_unit():
    assert is_bust(BUST_THRESHOLD_MINOR - 1)
    assert not is_bust(BUST_THRESHOLD_MINOR)
    assert not is_bust(STARTING_BALANCE_MINOR)


def test_floor_minor_repairs_float_representation_error():
    # Regression: quarter Kelly on p=0.6 at evens is exactly 5% of bankroll, but
    # 0.6 - 0.4 is 0.19999999999999996 in binary, so the naive floor is short by
    # one minor unit. Floor must not systematically shave stakes.
    import math

    from betsim.money import floor_minor

    raw = 0.25 * (0.6 - 0.4) * 100_000
    assert math.floor(raw) == 4_999  # documents the trap
    assert floor_minor(raw) == 5_000


def test_floor_minor_still_floors_genuine_fractions():
    from betsim.money import floor_minor

    assert floor_minor(1909.0909) == 1909
    assert floor_minor(12.5125) == 12
    assert floor_minor(0.9) == 0
