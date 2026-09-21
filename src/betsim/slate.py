"""The daily loop: one price snapshot in, every arm's bets out.

Every arm is built from the **same** snapshot, which is what makes the
comparisons between them mean anything. The pairing matters too: seed *i*'s
Stage 2 sees seed *i*'s blind forecast and nothing else, and ``kelly_si`` stakes
that same forecast under a fixed rule. ``llm_si`` versus ``kelly_si`` is
therefore a clean read on bet sizing alone, holding the forecast constant --
the decomposition prior work could not make.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from betsim.arms import ArmBet, best_price_slate, favourite_slate, kelly_slate, random_slate
from betsim.config import DESIGNATED_BOOKMAKER, MARKET, PRIMARY_DEVIG_METHOD
from betsim.decider import Stage2Decider, build_slate_payload
from betsim.devig import consensus
from betsim.ledger import arm_state, place_bet
from betsim.sports import get_sport
from betsim.store import (
    best_prices,
    forecast_ids_by_seed,
    forecasts_by_seed,
    game_teams,
    insert_llm_call,
    insert_rejection,
    now_utc,
    slate_prices,
    upcoming_games,
)
from betsim.validator import GameRef, validate_bets


@dataclass(slots=True)
class SlateResult:
    games: int = 0
    seeds: int = 0
    placed: dict[str, int] = field(default_factory=dict)
    staked_minor: dict[str, int] = field(default_factory=dict)
    rejected: dict[str, int] = field(default_factory=dict)
    cost_usd: float = 0.0
    refusals: int = 0
    notes: list[str] = field(default_factory=list)

    def record(self, arm: str, bets: Sequence[ArmBet]) -> None:
        if not bets:
            self.placed.setdefault(arm, 0)
            return
        self.placed[arm] = self.placed.get(arm, 0) + len(bets)
        self.staked_minor[arm] = self.staked_minor.get(arm, 0) + sum(b.stake_minor for b in bets)


def build_game_refs(
    conn: sqlite3.Connection,
    games: Sequence,
    *,
    bookmaker: str = DESIGNATED_BOOKMAKER,
    market: str = MARKET,
) -> list[GameRef]:
    """Attach the latest designated-book prices to each upcoming game.

    Games with no price from the designated bookmaker are dropped: without its
    price there is no tier, no stake cap, and no closing line to measure CLV
    against later.
    """
    prices = slate_prices(conn, [g.id for g in games], bookmaker=bookmaker, market=market)
    refs = []
    for game in games:
        entry = prices.get(game.id)
        if not entry or not entry[1]:
            continue
        captured, outcomes = entry
        refs.append(
            GameRef(
                game_id=game.id,
                commence_utc=game.commence_utc,
                snapshot_captured_utc=captured,
                prices=outcomes,
            )
        )
    return refs


def market_consensus(
    conn: sqlite3.Connection,
    game_ids: Sequence[str],
    outcomes: Sequence[str],
    *,
    method: str = PRIMARY_DEVIG_METHOD,
) -> dict[str, dict[str, float]]:
    """De-vigged consensus per game, across every bookmaker stored."""
    out: dict[str, dict[str, float]] = {}
    for gid in game_ids:
        books: dict[str, dict[str, float]] = {}
        for row in conn.execute(
            "SELECT bookmaker, outcome, price_decimal FROM odds_snapshots "
            "WHERE game_id = ? AND market = 'h2h'",
            (gid,),
        ):
            books.setdefault(row["bookmaker"], {})[row["outcome"]] = row["price_decimal"]
        usable = [b for b in books.values() if all(o in b for o in outcomes)]
        if usable:
            out[gid] = consensus(usable, outcomes, method)
    return out


def _place_all(
    conn: sqlite3.Connection,
    arm: str,
    bets: Sequence[ArmBet],
    *,
    placed_utc: datetime,
    forecast_ids: Mapping[str, int] | None = None,
) -> list[ArmBet]:
    placed = []
    for bet in bets:
        place_bet(
            conn,
            arm,
            bet,
            placed_utc=placed_utc,
            forecast_id=(forecast_ids or {}).get(bet.game_id),
        )
        placed.append(bet)
    return placed


def run_slate(
    conn: sqlite3.Connection,
    *,
    sport: str,
    decider: Stage2Decider | None,
    anchored: Stage2Decider | None = None,
    k: int = 1,
    refs: Sequence[GameRef] | None = None,
    now: datetime | None = None,
    elo_probs: Mapping[str, Mapping[str, float]] | None = None,
    random_seed: int = 0,
) -> SlateResult:
    """Run one slate across every arm.

    ``decider`` may be ``None`` to compute the shadow arms alone, which is how a
    dry run exercises the whole pipeline without spending anything.
    """
    stamp = now or now_utc()
    spec = get_sport(sport)
    result = SlateResult()

    # The caller usually resolves the slate itself (applying the horizon); fall
    # back to every upcoming game only when it did not.
    if refs is None:
        refs = build_game_refs(conn, upcoming_games(conn, sport, after=stamp))
    if not refs:
        result.notes.append("no games with designated-bookmaker prices")
        return result

    result.games = len(refs)
    by_id = {g.game_id: g for g in refs}
    teams = game_teams(conn, list(by_id))
    blind = forecasts_by_seed(conn, list(by_id))
    fids = forecast_ids_by_seed(conn, list(by_id))
    result.seeds = k

    for seed in range(k):
        probs = blind.get(seed, {})
        if not probs:
            result.notes.append(f"seed {seed}: no blind forecasts, skipped")
            continue

        # --- the LLM arm ---
        llm_arm = f"llm_s{seed + 1}"
        state = arm_state(conn, llm_arm)
        revised: dict[str, dict[str, float]] = {}
        if decider is not None and not state.bust:
            payload = build_slate_payload(refs, p_blind=probs, teams=teams, state=state)
            call = decider.decide(payload, seed_idx=seed)
            call_id = insert_llm_call(conn, call, stage=2, created_utc=stamp)
            result.cost_usd += call.cost_usd
            if call.stop_reason == "refusal":
                result.refusals += 1
            if call.ok:
                validated = validate_bets(
                    call.proposals,
                    spec=spec,
                    games=by_id,
                    balance_minor=state.balance_minor,
                    now=stamp,
                    open_exposure_minor=state.open_exposure_minor,
                    open_game_ids=state.open_game_ids,
                )
                for rejection in validated.rejected:
                    insert_rejection(conn, llm_arm, rejection, llm_call_id=call_id)
                    key = str(rejection.reason)
                    result.rejected[key] = result.rejected.get(key, 0) + 1
                accepted = [
                    ArmBet(
                        game_id=b.game_id,
                        selection=b.selection,
                        stake_minor=b.stake_minor,
                        price_decimal=b.price_decimal,
                        tier=b.tier,
                        p_blind=probs.get(b.game_id, {}).get(b.selection),
                        p_revised=b.p_revised,
                        reason=b.reason,
                    )
                    for b in validated.accepted
                ]
                result.record(
                    llm_arm,
                    _place_all(
                        conn, llm_arm, accepted, placed_utc=stamp, forecast_ids=fids.get(seed)
                    ),
                )
                for bet in accepted:
                    if bet.p_revised is not None:
                        revised.setdefault(bet.game_id, {})[bet.selection] = bet.p_revised
            else:
                result.record(llm_arm, [])

        # --- the same decision, but anchored to the market ---
        if anchored is not None:
            _run_anchored(
                conn,
                f"llm_anchored_s{seed + 1}",
                refs,
                probs,
                spec,
                by_id,
                teams,
                anchored,
                seed,
                stamp,
                result,
                fids.get(seed),
            )

        # --- the sizing counterfactual, on the same probabilities ---
        _run_shadow(
            conn,
            f"kelly_s{seed + 1}",
            refs,
            probs,
            result,
            stamp,
            fids.get(seed),
            "quarter Kelly on the blind forecast",
        )

        # --- did seeing the price improve the forecast, or just anchor it? ---
        if revised:
            _run_shadow(
                conn,
                f"kelly_revised_s{seed + 1}",
                refs,
                revised,
                result,
                stamp,
                fids.get(seed),
                "quarter Kelly on the revised probability",
            )

    # --- arms that need no forecast at all ---
    if elo_probs:
        _run_shadow(conn, "elo", refs, elo_probs, result, stamp, None, "quarter Kelly on Elo")

    shopped = best_prices(conn, [r.game_id for r in refs])
    for arm, bets in (
        ("fav", favourite_slate(refs, open_game_ids=arm_state(conn, "fav").open_game_ids)),
        (
            "best_price",
            best_price_slate(
                refs, shopped, open_game_ids=arm_state(conn, "best_price").open_game_ids
            ),
        ),
        (
            "random",
            random_slate(
                refs, seed=random_seed, open_game_ids=arm_state(conn, "random").open_game_ids
            ),
        ),
    ):
        state = arm_state(conn, arm)
        affordable = [b for b in bets if b.stake_minor <= state.balance_minor]
        result.record(arm, _place_all(conn, arm, affordable, placed_utc=stamp))

    return result


def _run_shadow(
    conn: sqlite3.Connection,
    arm: str,
    refs: Sequence[GameRef],
    probs: Mapping[str, Mapping[str, float]],
    result: SlateResult,
    stamp: datetime,
    forecast_ids: Mapping[str, int] | None,
    reason: str,
) -> None:
    state = arm_state(conn, arm)
    if state.bust:
        result.record(arm, [])
        return
    bets = kelly_slate(
        refs,
        probs,
        balance_minor=state.balance_minor,
        open_exposure_minor=state.open_exposure_minor,
        open_game_ids=state.open_game_ids,
        reason=reason,
    )
    result.record(arm, _place_all(conn, arm, bets, placed_utc=stamp, forecast_ids=forecast_ids))


def _run_anchored(
    conn: sqlite3.Connection,
    arm: str,
    refs: Sequence[GameRef],
    probs: Mapping[str, Mapping[str, float]],
    spec,
    by_id: Mapping[str, GameRef],
    teams: Mapping[str, tuple[str, str]],
    decider: Stage2Decider,
    seed: int,
    stamp: datetime,
    result: SlateResult,
    forecast_ids: Mapping[str, int] | None,
) -> None:
    """The market-anchored decision: same forecast, shown the consensus too.

    Tests whether thinking like a bettor -- treating the price as the prior and
    moving only for a nameable reason -- beats forecasting like an analyst.
    """
    state = arm_state(conn, arm)
    if state.bust:
        result.record(arm, [])
        return
    payload = build_slate_payload(
        refs,
        p_blind=probs,
        teams=teams,
        state=state,
        p_market=market_consensus(conn, [r.game_id for r in refs], ("home", "away")),
    )
    call = decider.decide(payload, seed_idx=seed)
    call_id = insert_llm_call(conn, call, stage=2, created_utc=stamp)
    result.cost_usd += call.cost_usd
    if call.stop_reason == "refusal":
        result.refusals += 1
    if not call.ok:
        result.record(arm, [])
        return

    validated = validate_bets(
        call.proposals,
        spec=spec,
        games=by_id,
        balance_minor=state.balance_minor,
        now=stamp,
        open_exposure_minor=state.open_exposure_minor,
        open_game_ids=state.open_game_ids,
    )
    for rejection in validated.rejected:
        insert_rejection(conn, arm, rejection, llm_call_id=call_id)
        key = str(rejection.reason)
        result.rejected[key] = result.rejected.get(key, 0) + 1
    accepted = [
        ArmBet(
            game_id=b.game_id,
            selection=b.selection,
            stake_minor=b.stake_minor,
            price_decimal=b.price_decimal,
            tier=b.tier,
            p_blind=probs.get(b.game_id, {}).get(b.selection),
            p_revised=b.p_revised,
            reason=b.reason,
        )
        for b in validated.accepted
    ]
    result.record(arm, _place_all(conn, arm, accepted, placed_utc=stamp, forecast_ids=forecast_ids))
