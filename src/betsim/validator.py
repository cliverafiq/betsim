"""Bet validation.

The validator **rejects** invalid bets and logs the reason. It never clips a
stake down to fit, because a clipped bet silently converts a rule violation into
a valid-looking row -- and rule violations are data.

Two conventions the draft spec left open, pinned down here:

**Ordering.** Bets are processed in the order the model returned them, and when
the exposure cap binds the later ones are rejected. Reordering (say, cheapest
first) would make the result depend on an arbitrary implementation choice; and
whether the model manages its own exposure is itself a measurement.

**Accounting.** ``balance_minor`` is *free cash* at slate open -- stakes are
deducted from the ledger at placement and payouts credited at settlement, so
money riding on unsettled bets is not in the balance. Both the tier caps and the
exposure cap are percentages of that slate-open balance, and ``open_exposure_minor``
carries the stakes of bets from earlier slates that have not yet settled.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from betsim.config import EXPOSURE_CAP_PCT, MAX_SNAPSHOT_AGE
from betsim.money import MIN_STAKE_MINOR, is_bust, pct_of_balance_minor, units_to_minor
from betsim.sports import SportSpec
from betsim.tiers import Tier, assign_tier, tier_cap_minor


class RejectReason(StrEnum):
    ARM_BUST = "arm_bust"
    GAME_NOT_FOUND = "game_not_found"
    DUPLICATE_GAME = "duplicate_game"
    INVALID_SELECTION = "invalid_selection"
    NO_PRICE = "no_price"
    GAME_ALREADY_STARTED = "game_already_started"
    SNAPSHOT_STALE = "snapshot_stale"
    STAKE_BELOW_MIN = "stake_below_min"
    STAKE_EXCEEDS_BALANCE = "stake_exceeds_balance"
    STAKE_EXCEEDS_TIER_CAP = "stake_exceeds_tier_cap"
    STAKE_EXCEEDS_EXPOSURE = "stake_exceeds_exposure"
    MALFORMED_STAKE = "malformed_stake"


@dataclass(frozen=True, slots=True)
class BetProposal:
    """One bet as the model proposed it, before any checking."""

    game_id: str
    selection: str
    stake_units: float
    p_revised: float | None = None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class GameRef:
    """What the validator needs to know about a game to check a bet against it."""

    game_id: str
    commence_utc: datetime
    snapshot_captured_utc: datetime
    prices: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class AcceptedBet:
    game_id: str
    selection: str
    stake_minor: int
    price_decimal: float
    tier: Tier
    p_revised: float | None
    reason: str


@dataclass(frozen=True, slots=True)
class RejectedBet:
    proposal: BetProposal
    reason: RejectReason
    detail: str = ""


@dataclass(slots=True)
class ValidationResult:
    accepted: list[AcceptedBet] = field(default_factory=list)
    rejected: list[RejectedBet] = field(default_factory=list)

    @property
    def staked_minor(self) -> int:
        return sum(b.stake_minor for b in self.accepted)


def _require_aware(dt: datetime, label: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware UTC, got naive {dt!r}")
    return dt


def validate_bets(
    proposals: Sequence[BetProposal],
    *,
    spec: SportSpec,
    games: Mapping[str, GameRef],
    balance_minor: int,
    now: datetime,
    open_exposure_minor: int = 0,
    open_game_ids: Iterable[str] = (),
    max_snapshot_age: timedelta = MAX_SNAPSHOT_AGE,
) -> ValidationResult:
    """Validate a slate of proposed bets, in the order given.

    Zero bets in is a valid input and returns an empty result; zero bets out is a
    valid outcome and is never an error.
    """
    _require_aware(now, "now")
    result = ValidationResult()

    if is_bust(balance_minor):
        result.rejected.extend(
            RejectedBet(p, RejectReason.ARM_BUST, f"balance {balance_minor} is below one unit")
            for p in proposals
        )
        return result

    exposure_cap = pct_of_balance_minor(balance_minor, EXPOSURE_CAP_PCT)
    seen: set[str] = set(open_game_ids)
    committed = 0

    for proposal in proposals:
        game = games.get(proposal.game_id)
        if game is None:
            result.rejected.append(RejectedBet(proposal, RejectReason.GAME_NOT_FOUND))
            continue
        if proposal.game_id in seen:
            result.rejected.append(
                RejectedBet(proposal, RejectReason.DUPLICATE_GAME, "one bet per game per arm")
            )
            continue
        if not spec.validate_selection(proposal.selection):
            result.rejected.append(
                RejectedBet(
                    proposal,
                    RejectReason.INVALID_SELECTION,
                    f"{proposal.selection!r} invalid for a {spec.market_width}-way market",
                )
            )
            continue
        price = game.prices.get(proposal.selection)
        if price is None:
            result.rejected.append(RejectedBet(proposal, RejectReason.NO_PRICE))
            continue
        if _require_aware(game.commence_utc, "commence_utc") <= now:
            result.rejected.append(
                RejectedBet(proposal, RejectReason.GAME_ALREADY_STARTED, "forward-only")
            )
            continue
        age = now - _require_aware(game.snapshot_captured_utc, "snapshot_captured_utc")
        if age > max_snapshot_age:
            result.rejected.append(
                RejectedBet(proposal, RejectReason.SNAPSHOT_STALE, f"snapshot age {age}")
            )
            continue

        try:
            stake_minor = units_to_minor(proposal.stake_units)
        except (ValueError, ArithmeticError, TypeError) as exc:
            result.rejected.append(RejectedBet(proposal, RejectReason.MALFORMED_STAKE, str(exc)))
            continue

        if stake_minor < MIN_STAKE_MINOR:
            result.rejected.append(
                RejectedBet(proposal, RejectReason.STAKE_BELOW_MIN, f"{stake_minor} minor units")
            )
            continue
        if stake_minor > balance_minor - committed:
            result.rejected.append(
                RejectedBet(
                    proposal,
                    RejectReason.STAKE_EXCEEDS_BALANCE,
                    f"{stake_minor} > {balance_minor - committed} free",
                )
            )
            continue

        tier = assign_tier(price)
        cap = tier_cap_minor(tier, balance_minor)
        if stake_minor > cap:
            result.rejected.append(
                RejectedBet(
                    proposal,
                    RejectReason.STAKE_EXCEEDS_TIER_CAP,
                    f"{stake_minor} > {tier} cap {cap}",
                )
            )
            continue

        available_exposure = exposure_cap - open_exposure_minor - committed
        if stake_minor > available_exposure:
            result.rejected.append(
                RejectedBet(
                    proposal,
                    RejectReason.STAKE_EXCEEDS_EXPOSURE,
                    f"{stake_minor} > {max(0, available_exposure)} remaining",
                )
            )
            continue

        seen.add(proposal.game_id)
        committed += stake_minor
        result.accepted.append(
            AcceptedBet(
                game_id=proposal.game_id,
                selection=proposal.selection,
                stake_minor=stake_minor,
                price_decimal=float(price),
                tier=tier,
                p_revised=proposal.p_revised,
                reason=proposal.reason,
            )
        )

    return result


def cap_for(tier: Tier, balance_minor: int) -> int:
    """Per-tier cap in minor units, precomputed for the Stage 2 prompt."""
    return tier_cap_minor(tier, balance_minor)


def exposure_cap_minor(balance_minor: int) -> int:
    return math.floor(balance_minor * EXPOSURE_CAP_PCT)
