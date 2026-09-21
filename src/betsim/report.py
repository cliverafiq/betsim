"""Reporting: what the experiment actually found.

Ordered by how fast each number becomes trustworthy, not by how interesting it
sounds. CLV and Brier settle far faster than ROI -- 1,000 flat bets at -110
still leave a roughly +/-6 point ROI interval -- so ROI is reported last and
labelled underpowered until the bet count justifies otherwise.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from betsim.calibrate import CalibrationError, fit_platt
from betsim.config import BOOTSTRAP_RESAMPLES, CALIBRATION_BINS, PRIMARY_DEVIG_METHOD
from betsim.devig import consensus
from betsim.ledger import balance_path
from betsim.metrics import (
    CalibrationBin,
    bootstrap_ci,
    brier_score,
    calibration_table,
    log_loss,
    max_drawdown,
    roi,
)
from betsim.money import STARTING_BALANCE_MINOR, is_bust, minor_to_units
from betsim.settle import bet_clv, clv_by_arm

SETTLED = ("won", "lost", "void")


@dataclass(frozen=True, slots=True)
class ArmReport:
    arm: str
    bets: int
    open_bets: int
    staked_minor: int
    profit_minor: int
    balance_minor: int
    max_drawdown: float
    bust: bool
    roi: float | None
    roi_ci: tuple[float, float] | None = None

    @property
    def balance_units(self) -> float:
        return minor_to_units(self.balance_minor)


def _settled_rows(conn: sqlite3.Connection, arm: str | None = None) -> Sequence[sqlite3.Row]:
    clause = " AND b.arm = ?" if arm else ""
    args = (arm,) if arm else ()
    return conn.execute(
        f"""
        SELECT b.*, substr(b.placed_utc, 1, 10) AS day
          FROM bets b
         WHERE b.status IN ('won', 'lost', 'void'){clause}
         ORDER BY b.id
        """,
        args,
    ).fetchall()


def _profit(row: sqlite3.Row) -> int:
    return (row["payout_minor"] or 0) - row["stake_minor"]


def arm_reports(conn: sqlite3.Connection, *, bootstrap: bool = True) -> list[ArmReport]:
    """One row per arm, with a bootstrap ROI interval resampled by day."""
    arms = [r["arm"] for r in conn.execute("SELECT DISTINCT arm FROM bets ORDER BY arm").fetchall()]
    out = []
    for arm in arms:
        rows = _settled_rows(conn, arm)
        staked = sum(r["stake_minor"] for r in rows)
        profit = sum(_profit(r) for r in rows)
        path = balance_path(conn, arm)
        open_count = conn.execute(
            "SELECT COUNT(*) c FROM bets WHERE arm = ? AND status = 'open'", (arm,)
        ).fetchone()["c"]

        ci = None
        if bootstrap and rows:
            by_day: dict[str, list[sqlite3.Row]] = {}
            for row in rows:
                by_day.setdefault(row["day"], []).append(row)
            ci = bootstrap_ci(
                by_day,
                lambda sample: roi(
                    [_profit(r) for r in sample], [r["stake_minor"] for r in sample]
                ),
                n_resamples=min(BOOTSTRAP_RESAMPLES, 2000),
            )

        out.append(
            ArmReport(
                arm=arm,
                bets=len(rows),
                open_bets=open_count,
                staked_minor=staked,
                profit_minor=profit,
                balance_minor=path[-1],
                max_drawdown=max_drawdown(path),
                bust=is_bust(path[-1]),
                roi=roi([_profit(r) for r in rows], [r["stake_minor"] for r in rows]),
                roi_ci=ci,
            )
        )
    return out


def tier_breakdown(conn: sqlite3.Connection, arm: str) -> dict[str, dict[str, float | int]]:
    """Bets, stake and ROI per tier -- tiers describe variance, not value."""
    out: dict[str, dict[str, float | int]] = {}
    for row in _settled_rows(conn, arm):
        entry = out.setdefault(row["tier"], {"bets": 0, "staked": 0, "profit": 0})
        entry["bets"] += 1
        entry["staked"] += row["stake_minor"]
        entry["profit"] += _profit(row)
    for entry in out.values():
        entry["roi"] = entry["profit"] / entry["staked"] if entry["staked"] else None
    return out


def _actual_outcome(row: sqlite3.Row) -> str | None:
    if row["home_score"] is None or row["away_score"] is None:
        return None
    if row["home_score"] > row["away_score"]:
        return "home"
    if row["away_score"] > row["home_score"]:
        return "away"
    return "draw"


def _market_consensus(
    conn: sqlite3.Connection, game_id: str, outcomes: Sequence[str], method: str
) -> dict[str, float] | None:
    rows = conn.execute(
        """
        SELECT bookmaker, outcome, price_decimal FROM odds_snapshots
         WHERE game_id = ? AND market = 'h2h'
        """,
        (game_id,),
    ).fetchall()
    books: dict[str, dict[str, float]] = {}
    for row in rows:
        books.setdefault(row["bookmaker"], {})[row["outcome"]] = row["price_decimal"]
    usable = [b for b in books.values() if all(o in b for o in outcomes)]
    if not usable:
        return None
    try:
        return consensus(usable, outcomes, method)
    except ValueError:
        return None


@dataclass(slots=True)
class ForecastQuality:
    games: int = 0
    brier_blind: float | None = None
    brier_market: float | None = None
    logloss_blind: float | None = None
    logloss_market: float | None = None
    pairs: list[tuple[float, bool]] = field(default_factory=list)

    @property
    def brier_gap(self) -> float | None:
        """Positive means the market is better, which is the expected result."""
        if self.brier_blind is None or self.brier_market is None:
            return None
        return self.brier_blind - self.brier_market


def forecast_quality(
    conn: sqlite3.Connection, *, method: str = PRIMARY_DEVIG_METHOD
) -> ForecastQuality:
    """Stage 1 against the de-vigged consensus, on **all** settled games.

    Scored on every game rather than only the ones bet on, so selection does not
    flatter the forecast. Uses the mean across the k draws as the ensemble.
    """
    games = conn.execute(
        """
        SELECT id, home_score, away_score FROM games
         WHERE status = 'completed' AND home_score IS NOT NULL
        """
    ).fetchall()
    result = ForecastQuality()
    blind_briers, market_briers, blind_lls, market_lls = [], [], [], []

    for game in games:
        actual = _actual_outcome(game)
        if actual is None:
            continue
        draws = conn.execute(
            "SELECT p_home, p_away FROM forecasts WHERE game_id = ?", (game["id"],)
        ).fetchall()
        if not draws:
            continue
        p_home = sum(d["p_home"] for d in draws) / len(draws)
        blind = {"home": p_home, "away": 1.0 - p_home}

        market = _market_consensus(conn, game["id"], ("home", "away"), method)
        if market is None:
            continue

        result.games += 1
        blind_briers.append(brier_score(blind, actual))
        market_briers.append(brier_score(market, actual))
        blind_lls.append(log_loss(blind, actual))
        market_lls.append(log_loss(market, actual))
        result.pairs.append((p_home, actual == "home"))

    if result.games:
        result.brier_blind = sum(blind_briers) / len(blind_briers)
        result.brier_market = sum(market_briers) / len(market_briers)
        result.logloss_blind = sum(blind_lls) / len(blind_lls)
        result.logloss_market = sum(market_lls) / len(market_lls)
    return result


def calibration_report(
    pairs: Sequence[tuple[float, bool]], *, bins: int = CALIBRATION_BINS
) -> list[CalibrationBin]:
    return calibration_table(pairs, n_bins=bins)


def behaviour_report(conn: sqlite3.Connection, arm: str) -> dict[str, object]:
    """Loss chasing, tier drift and selectivity -- the behavioural outcomes.

    KellyBench's most damaging findings were behavioural rather than analytical,
    so these are instrumented deliberately rather than left to be noticed.
    """
    rows = _settled_rows(conn, arm)
    by_day: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_day.setdefault(row["day"], []).append(row)

    days = sorted(by_day)
    after_loss, after_win = [], []
    for previous, current in pairwise(days):
        day_profit = sum(_profit(r) for r in by_day[previous])
        stakes = [r["stake_minor"] for r in by_day[current]]
        (after_loss if day_profit < 0 else after_win).extend(stakes)

    rejections = conn.execute(
        "SELECT reason, COUNT(*) c FROM rejections WHERE arm = ? GROUP BY reason", (arm,)
    ).fetchall()

    return {
        "days_with_bets": len(days),
        "mean_stake_after_losing_day": (sum(after_loss) / len(after_loss) if after_loss else None),
        "mean_stake_after_winning_day": (sum(after_win) / len(after_win) if after_win else None),
        "tier_mix": {tier: entry["bets"] for tier, entry in tier_breakdown(conn, arm).items()},
        "rejections": {r["reason"]: r["c"] for r in rejections},
    }


def fit_calibration(conn: sqlite3.Connection, *, min_games: int) -> tuple[object | None, str]:
    """Fit a Platt map on settled games. Returns ``(params, explanation)``."""
    quality = forecast_quality(conn)
    if len(quality.pairs) < min_games:
        return None, (
            f"{len(quality.pairs)} settled games with a forecast; the burn-in is {min_games}"
        )
    try:
        return fit_platt(quality.pairs), f"fitted on {len(quality.pairs)} games"
    except CalibrationError as exc:
        return None, str(exc)


def render(conn: sqlite3.Connection, *, method: str = PRIMARY_DEVIG_METHOD) -> str:
    """The full text report."""
    lines: list[str] = []
    add = lines.append

    add("CLOSING LINE VALUE   (primary -- settles fastest)")
    clv_values = bet_clv(conn)
    if not clv_values:
        add("  no bets have a closing price yet")
    else:
        summary = clv_by_arm(clv_values)
        null = summary.get("random", {}).get("mean_clv")
        for arm, stats in summary.items():
            gap = f"  gap to null {stats['mean_clv'] - null:+.4f}" if null is not None else ""
            tag = "  <- null" if arm == "random" else gap
            add(f"  {arm:22} n={stats['n']:4}  mean {stats['mean_clv']:+.4f}{tag}")
        if null is not None:
            add("  The null is `random`, not zero: betting early earns CLV with no skill.")

    add("")
    add("FORECAST QUALITY     (Stage 1 vs the de-vigged consensus, all settled games)")
    q = forecast_quality(conn, method=method)
    if not q.games:
        add("  no settled games with both a forecast and a market price yet")
    else:
        add(f"  games {q.games}   de-vig method: {method}")
        add(
            f"  Brier    blind {q.brier_blind:.4f}   market {q.brier_market:.4f}"
            f"   gap {q.brier_gap:+.4f}"
        )
        add(f"  log loss blind {q.logloss_blind:.4f}   market {q.logloss_market:.4f}")
        add("  Sum-over-outcomes convention: a uniform 2-way forecast scores 0.500.")
        add("")
        add("  calibration (expect overconfidence at the top)")
        for b in calibration_report(q.pairs):
            if b.count:
                add(
                    f"    {b.lower:.1f}-{b.upper:.1f}  n={b.count:4}  "
                    f"said {b.mean_predicted:.3f}  happened {b.observed_rate:.3f}"
                )

    add("")
    add("BANKROLL             (underpowered until the bet count is large)")
    reports = arm_reports(conn)
    if not reports:
        add("  no bets placed yet")
    for r in reports:
        roi_text = "n/a" if r.roi is None else f"{r.roi:+.2%}"
        ci_text = f"  95% CI [{r.roi_ci[0]:+.2%}, {r.roi_ci[1]:+.2%}]" if r.roi_ci else ""
        bust = "  BUST" if r.bust else ""
        add(
            f"  {r.arm:22} {r.bets:4} bets  {r.balance_units:8.2f} u  "
            f"ROI {roi_text:>8}{ci_text}  maxDD {r.max_drawdown:.1%}{bust}"
        )
    add(f"  (no-bet baseline: {minor_to_units(STARTING_BALANCE_MINOR):.2f} u)")
    return "\n".join(lines)
