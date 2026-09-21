"""The shadow arms.

Every arm is built from the *same* price snapshot the LLM saw, so differences
between them are attributable rather than incidental:

* ``kelly_si`` stakes seed *i*'s own blind probabilities under a fixed rule.
  Compared against ``llm_si``, which saw those same probabilities and the price,
  it isolates **bet sizing** from forecasting -- the decomposition no prior work
  in this area could make, because they showed the model the price throughout.
* ``kelly_revised`` stakes ``p_revised`` instead, so it answers whether seeing
  the price *improved* the forecast or merely anchored it.
* ``elo`` is the non-LLM statistical baseline. Without one you cannot tell
  whether the model is bad or the task is hard.
* ``fav`` flat-stakes the market favourite. In WC2026-Agents this out-earned all
  four frontier models, so it is the baseline to beat.
* ``random`` flat-stakes a uniformly random outcome. It is the floor, and --
  because it bets at the same snapshots with no skill -- it is also the null
  distribution for CLV.

``fav`` and ``random`` are exempt from **both** the tier caps and the exposure
cap: they are flat-stake references, not managed bankrolls.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from betsim.config import EXPOSURE_CAP_PCT, FLAT_STAKE_MINOR, KELLY_FRACTION
from betsim.kelly import best_outcome, kelly_stake_minor
from betsim.money import MIN_STAKE_MINOR, is_bust, pct_of_balance_minor
from betsim.tiers import Tier, assign_tier, tier_cap_minor
from betsim.validator import GameRef


@dataclass(frozen=True, slots=True)
class ArmBet:
    """A bet an arm wants to place, already sized and tier-assigned."""

    game_id: str
    selection: str
    stake_minor: int
    price_decimal: float
    tier: Tier
    p_blind: float | None = None
    p_revised: float | None = None
    reason: str = ""


def _by_start_time(games: Sequence[GameRef]) -> list[GameRef]:
    """Kelly processes games in start-time order so the exposure cap binds
    deterministically rather than depending on dictionary ordering."""
    return sorted(games, key=lambda g: (g.commence_utc, g.game_id))


def kelly_slate(
    games: Sequence[GameRef],
    probabilities: Mapping[str, Mapping[str, float]],
    *,
    balance_minor: int,
    fraction: float = KELLY_FRACTION,
    open_exposure_minor: int = 0,
    open_game_ids: frozenset[str] = frozenset(),
    reason: str = "",
) -> list[ArmBet]:
    """Fractional-Kelly bets for one slate, under the same caps as the LLM arm.

    Returns an empty list when nothing has an edge, which is a normal outcome.
    """
    if is_bust(balance_minor):
        return []

    exposure_cap = pct_of_balance_minor(balance_minor, EXPOSURE_CAP_PCT)
    committed = 0
    bets: list[ArmBet] = []

    for game in _by_start_time(games):
        if game.game_id in open_game_ids:
            continue
        probs = probabilities.get(game.game_id)
        if not probs:
            continue
        pick = best_outcome(probs, game.prices)
        if pick is None:
            continue
        selection, _ = pick
        price = game.prices[selection]

        stake = kelly_stake_minor(probs[selection], price, balance_minor, fraction)
        tier = assign_tier(price)
        stake = min(stake, tier_cap_minor(tier, balance_minor))
        stake = min(stake, exposure_cap - open_exposure_minor - committed)
        stake = min(stake, balance_minor - committed)
        if stake < MIN_STAKE_MINOR:
            continue

        committed += stake
        bets.append(
            ArmBet(
                game_id=game.game_id,
                selection=selection,
                stake_minor=stake,
                price_decimal=price,
                tier=tier,
                p_blind=probs[selection],
                reason=reason,
            )
        )
    return bets


def favourite_slate(
    games: Sequence[GameRef],
    *,
    stake_minor: int = FLAT_STAKE_MINOR,
    open_game_ids: frozenset[str] = frozenset(),
) -> list[ArmBet]:
    """Flat stake on the shortest price in every game.

    Exempt from the tier and exposure caps by design -- this is a reference line,
    not a managed bankroll.
    """
    bets = []
    for game in _by_start_time(games):
        if game.game_id in open_game_ids or not game.prices:
            continue
        selection = min(game.prices, key=lambda o: game.prices[o])
        price = game.prices[selection]
        bets.append(
            ArmBet(
                game_id=game.game_id,
                selection=selection,
                stake_minor=stake_minor,
                price_decimal=price,
                tier=assign_tier(price),
                reason="flat stake on the market favourite",
            )
        )
    return bets


def random_slate(
    games: Sequence[GameRef],
    *,
    seed: int,
    stake_minor: int = FLAT_STAKE_MINOR,
    open_game_ids: frozenset[str] = frozenset(),
) -> list[ArmBet]:
    """Flat stake on a uniformly random outcome, seeded per game.

    Seeding on ``(seed, game_id)`` rather than on iteration order keeps each
    game's pick reproducible however the slate is ordered or re-run.
    """
    bets = []
    for game in _by_start_time(games):
        if game.game_id in open_game_ids or not game.prices:
            continue
        rng = random.Random(f"{seed}:{game.game_id}")
        selection = rng.choice(sorted(game.prices))
        price = game.prices[selection]
        bets.append(
            ArmBet(
                game_id=game.game_id,
                selection=selection,
                stake_minor=stake_minor,
                price_decimal=price,
                tier=assign_tier(price),
                reason="flat stake on a seeded random outcome",
            )
        )
    return bets


def elo_probabilities(
    games: Sequence[GameRef],
    table,
    home_away: Mapping[str, tuple[str, str]],
) -> dict[str, dict[str, float]]:
    """Win probabilities from the Elo table, keyed by game id.

    ``home_away`` maps game id to its (home, away) tricodes.
    """
    out: dict[str, dict[str, float]] = {}
    for game in games:
        pair = home_away.get(game.game_id)
        if pair is None:
            continue
        out[game.game_id] = table.probability(*pair)
    return out


def best_price_slate(
    games: Sequence[GameRef],
    best: Mapping[str, Mapping[str, tuple[float, str]]],
    *,
    stake_minor: int = FLAT_STAKE_MINOR,
    open_game_ids: frozenset[str] = frozenset(),
) -> list[ArmBet]:
    """Flat stake on the market favourite, taken at the **best price anywhere**.

    Deliberately the same selection as ``fav``, so the pair isolates line
    shopping alone. Any difference between them is a structural edge available
    without forecasting anything -- and if there is none, that is worth knowing
    before anyone builds a strategy on it.
    """
    bets = []
    for game in _by_start_time(games):
        if game.game_id in open_game_ids or not game.prices:
            continue
        selection = min(game.prices, key=lambda o: game.prices[o])
        shopped = (best.get(game.game_id) or {}).get(selection)
        if shopped is None:
            continue
        price, bookmaker = shopped
        bets.append(
            ArmBet(
                game_id=game.game_id,
                selection=selection,
                stake_minor=stake_minor,
                price_decimal=price,
                tier=assign_tier(price),
                reason=f"favourite at the best of all books ({bookmaker})",
            )
        )
    return bets
