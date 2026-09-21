import pytest

from betsim.odds import american_to_decimal, implied_probability
from betsim.tiers import TIER_CAP_PCT, Tier, assign_tier, tier_cap_minor


def test_tier_boundaries_at_the_stated_american_prices():
    assert assign_tier(american_to_decimal(-150)) is Tier.SAFE
    assert assign_tier(american_to_decimal(150)) is Tier.MEDIUM
    assert assign_tier(american_to_decimal(-200)) is Tier.SAFE
    assert assign_tier(american_to_decimal(200)) is Tier.RISKY


def test_boundary_survives_one_ulp_of_float_error():
    # Without the epsilon guard this price classifies as RISKY (1% cap) rather
    # than MEDIUM (3%), purely from one unit in the last place of float error.
    hair_above_2_5 = 2.5000000000000004
    assert implied_probability(hair_above_2_5) < 0.40  # the raw comparison fails
    assert assign_tier(hair_above_2_5) is Tier.MEDIUM  # the guarded one does not


def test_epsilon_does_not_absorb_a_real_price_difference():
    # 1.6667 is genuinely a worse price than -150 (implied 0.59999 vs 0.60), and
    # must still fall to MEDIUM. The guard is sized for float error, not for
    # rounding real prices together.
    assert assign_tier(1.6667) is Tier.MEDIUM
    assert assign_tier(american_to_decimal(-150)) is Tier.SAFE


def test_no_price_maps_to_two_tiers():
    # Exhaustive sweep: every price resolves to exactly one tier.
    d = 1.001
    while d < 50.0:
        assert assign_tier(d) in (Tier.SAFE, Tier.MEDIUM, Tier.RISKY)
        d = round(d + 0.001, 6)


def test_tiers_are_monotonic_in_price():
    order = {Tier.SAFE: 0, Tier.MEDIUM: 1, Tier.RISKY: 2}
    prices = [1.05, 1.4, 1.66, 1.8, 2.0, 2.4, 2.5, 3.0, 10.0]
    ranks = [order[assign_tier(d)] for d in prices]
    assert ranks == sorted(ranks), "longer prices must never map to a safer tier"


def test_caps_match_the_spec():
    assert TIER_CAP_PCT[Tier.SAFE] == 0.05
    assert TIER_CAP_PCT[Tier.MEDIUM] == 0.03
    assert TIER_CAP_PCT[Tier.RISKY] == 0.01


def test_tier_cap_minor_floors():
    assert tier_cap_minor(Tier.SAFE, 100_000) == 5_000
    assert tier_cap_minor(Tier.MEDIUM, 100_000) == 3_000
    assert tier_cap_minor(Tier.RISKY, 100_000) == 1_000
    assert tier_cap_minor(Tier.RISKY, 999) == 9


@pytest.mark.parametrize("bad", [1.0, 0.0, -1.0])
def test_assign_tier_rejects_non_prices(bad):
    with pytest.raises(ValueError):
        assign_tier(bad)
