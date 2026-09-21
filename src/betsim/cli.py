"""Command-line entry point.

Subcommands land with their milestones. M1 ships the data layer: ``quota`` and
``events`` are free, ``odds`` and ``scores`` cost credits and record what they
spent. ``slate`` needs Stage 1 and Stage 2, which are M3 and M4.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from betsim.config import DESIGNATED_BOOKMAKER, MARKET, REGION
from betsim.context import build_context, context_hash, index_results_by_team, seed_elo_from_results
from betsim.db import connect, init_db
from betsim.elo import EloTable
from betsim.ingest import parse_events, parse_odds, parse_scores
from betsim.nhl import NhlApiError, NhlClient, parse_club_schedule, parse_standings
from betsim.oddsapi import OddsApiClient, OddsApiError
from betsim.store import (
    apply_scores,
    context_exists,
    finish_run,
    insert_context,
    insert_snapshots,
    save_elo_ratings,
    start_run,
    upcoming_games,
    upsert_games,
)
from betsim.teams import TeamIndex, UnknownTeam

DEFAULT_DB = Path("betsim.db")
DEFAULT_SPORT = "icehockey_nhl"
NOT_YET = {"slate": "M3/M4", "report": "M6", "calibrate": "M6"}
PRIOR_SEASON = 20252026
NHL_CALL_DELAY = 0.15  # be a considerate client of a free public API


def _api_key() -> str:
    load_dotenv()
    key = os.environ.get("ODDS_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "ODDS_API_KEY is not set. Copy .env.example to .env and add your key "
            "(free tier: https://the-odds-api.com)."
        )
    return key


def _open_db(path: Path) -> sqlite3.Connection:
    conn = connect(path)
    init_db(conn)
    return conn


def cmd_init(args: argparse.Namespace) -> int:
    _open_db(args.db).close()
    print(f"initialised {args.db}")
    return 0


def cmd_quota(args: argparse.Namespace) -> int:
    """Check the key and report remaining credits. Free -- spends nothing."""
    with OddsApiClient(_api_key()) as api:
        quota = api.refresh_quota()
    if quota.remaining is None:
        print("connected, but the API did not report a quota")
        return 0
    print(f"credits remaining: {quota.remaining}   used: {quota.used}")
    daily = 6
    print(f"at ~{daily}/day for one sport that is ~{quota.remaining // daily} days of headroom")
    return 0


def cmd_events(args: argparse.Namespace) -> int:
    """List and store the upcoming schedule. Free -- /events costs no credits."""
    conn = _open_db(args.db)
    try:
        with OddsApiClient(_api_key()) as api:
            games = parse_events(api.list_events(args.sport))
        with conn:
            upsert_games(conn, games)
        for game in sorted(games, key=lambda g: g.commence_utc):
            print(f"  {game.commence_utc:%Y-%m-%d %H:%M}Z  {game.away_team} @ {game.home_team}")
        print(f"{len(games)} games stored (0 credits)")
    finally:
        conn.close()
    return 0


def cmd_odds(args: argparse.Namespace) -> int:
    """Fetch and store a price snapshot. Costs markets x regions -- 1 credit for v1."""
    conn = _open_db(args.db)
    kind = "close" if args.closing else "slate"
    try:
        with conn:
            run_id = start_run(conn, kind)
            with OddsApiClient(_api_key(), record_dir=args.record) as api:
                payload = api.fetch_odds(args.sport, regions=args.regions, markets=MARKET)
                games, prices = parse_odds(payload, market=MARKET)
                upsert_games(conn, games)
                n = insert_snapshots(
                    conn, prices, captured_utc=datetime.now(UTC), is_closing=args.closing
                )
                finish_run(
                    conn,
                    run_id,
                    credits_remaining=api.quota.remaining,
                    notes=f"{kind}: {len(games)} games, {n} prices, cost {api.quota.last_cost}",
                )
        designated = sum(1 for p in prices if p.bookmaker == DESIGNATED_BOOKMAKER)
        print(f"{len(games)} games, {n} prices from {len({p.bookmaker for p in prices})} books")
        if not designated:
            print(f"  WARNING: no prices from {DESIGNATED_BOOKMAKER}, the designated bookmaker")
        print(f"credits remaining: {api.quota.remaining}")
    finally:
        conn.close()
    return 0


def cmd_scores(args: argparse.Namespace) -> int:
    """Fetch and store final scores. 1 credit, or 2 with --days-from."""
    conn = _open_db(args.db)
    try:
        with conn:
            run_id = start_run(conn, "settle")
            with OddsApiClient(_api_key(), record_dir=args.record) as api:
                payload = api.fetch_scores(args.sport, days_from=args.days_from)
                scores = parse_scores(payload)
                completed = apply_scores(conn, scores)
                finish_run(
                    conn,
                    run_id,
                    credits_remaining=api.quota.remaining,
                    notes=f"settle: {completed}/{len(scores)} completed, "
                    f"cost {api.quota.last_cost}",
                )
        print(f"{completed} of {len(scores)} games completed and stored")
        print(f"credits remaining: {api.quota.remaining}")
    finally:
        conn.close()
    return 0


def _nhl_context_sources(tricodes: list[str], season: int) -> tuple[dict, list]:
    """Fetch standings and the club schedules for just the teams involved."""
    with NhlClient() as nhl:
        standings = parse_standings(nhl.standings())
        results = []
        for code in tricodes:
            results.extend(parse_club_schedule(nhl.club_schedule_season(code, season)))
            time.sleep(NHL_CALL_DELAY)
    return {t.tricode: t for t in standings}, results


def cmd_seed_elo(args: argparse.Namespace) -> int:
    """Seed the Elo baseline from a completed season. Free -- the NHL API has no quota."""
    conn = _open_db(args.db)
    try:
        with NhlClient() as nhl:
            standings = parse_standings(nhl.standings())
            results = []
            for team in standings:
                results.extend(
                    parse_club_schedule(nhl.club_schedule_season(team.tricode, args.season))
                )
                time.sleep(NHL_CALL_DELAY)
        table = EloTable()
        n = seed_elo_from_results(results, table)
        table.regress_to_mean()
        with conn:
            save_elo_ratings(conn, table.ratings, games_seeded=n, season=args.season)
        ranked = sorted(table.ratings.items(), key=lambda kv: -kv[1])
        print(f"seeded Elo from {n} regular-season games of {args.season}")
        print(f"  strongest: {ranked[0][0]} {ranked[0][1]:.0f}")
        print(f"  weakest:   {ranked[-1][0]} {ranked[-1][1]:.0f}")
        print(f"stored {len(table.ratings)} ratings (0 credits)")
    finally:
        conn.close()
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    """Build and store the blind Stage 1 context for upcoming games. Free."""
    conn = _open_db(args.db)
    try:
        games = upcoming_games(conn, args.sport)
        if not games:
            print("no upcoming games; run `betsim events` first")
            return 0

        with NhlClient() as nhl:
            standings = parse_standings(nhl.standings())
        index = TeamIndex.from_standings(standings)

        names = {n for g in games for n in (g.home_team, g.away_team)}
        unresolved = index.unresolved(names)
        if unresolved:
            print(f"cannot resolve: {unresolved}", file=sys.stderr)
            return 1

        tricodes = sorted({index.tricode(n) for n in names})
        by_code, results = _nhl_context_sources(tricodes, args.season)
        history = index_results_by_team(results)

        stored = skipped = 0
        with conn:
            for game in games:
                payload = build_context(
                    game,
                    index=index,
                    standings=by_code,
                    results_by_team=history,
                )
                digest = context_hash(payload)
                if context_exists(conn, game.id, digest):
                    skipped += 1
                    continue
                insert_context(
                    conn,
                    game.id,
                    payload,
                    payload_hash=digest,
                    sources="api-web.nhle.com/v1: standings, club-schedule-season",
                )
                stored += 1
        print(f"{stored} contexts stored, {skipped} unchanged ({len(tricodes)} teams, 0 credits)")
    finally:
        conn.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="betsim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, *, sport: bool = True) -> argparse.ArgumentParser:
        p.add_argument("--db", type=Path, default=DEFAULT_DB)
        if sport:
            p.add_argument("--sport", default=DEFAULT_SPORT)
        return p

    common(sub.add_parser("init", help="create the database"), sport=False).set_defaults(
        func=cmd_init
    )
    sub.add_parser("quota", help="check the key and credits (free)").set_defaults(func=cmd_quota)
    common(sub.add_parser("events", help="fetch the schedule (free)")).set_defaults(func=cmd_events)

    p_odds = common(sub.add_parser("odds", help="fetch a price snapshot (costs credits)"))
    p_odds.add_argument("--regions", default=REGION)
    p_odds.add_argument("--closing", action="store_true", help="mark this snapshot as closing")
    p_odds.add_argument(
        "--record", type=Path, default=None, help="record the response as a fixture"
    )
    p_odds.set_defaults(func=cmd_odds)

    p_scores = common(sub.add_parser("scores", help="fetch final scores (costs credits)"))
    p_scores.add_argument("--days-from", type=int, default=3, choices=(1, 2, 3))
    p_scores.add_argument("--record", type=Path, default=None)
    p_scores.set_defaults(func=cmd_scores)

    p_seed = common(sub.add_parser("seed-elo", help="seed the Elo baseline (free)"), sport=False)
    p_seed.add_argument("--season", type=int, default=PRIOR_SEASON)
    p_seed.set_defaults(func=cmd_seed_elo)

    p_ctx = common(sub.add_parser("context", help="build blind Stage 1 contexts (free)"))
    p_ctx.add_argument("--season", type=int, default=PRIOR_SEASON)
    p_ctx.set_defaults(func=cmd_context)

    for name, milestone in NOT_YET.items():
        common(sub.add_parser(name, help=f"not implemented yet ({milestone})")).set_defaults(
            func=None, milestone=milestone
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "func", None) is None:
        print(
            f"{args.command!r} is not implemented yet -- it lands in {args.milestone}. "
            "See docs/PLAN.md.",
            file=sys.stderr,
        )
        return 2
    try:
        return args.func(args)
    except OddsApiError as exc:
        print(f"Odds API error: {exc}", file=sys.stderr)
        return 1
    except NhlApiError as exc:
        print(f"NHL API error: {exc}", file=sys.stderr)
        return 1
    except UnknownTeam as exc:
        print(f"Unmapped team: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
