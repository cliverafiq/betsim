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
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from betsim.config import DESIGNATED_BOOKMAKER, MARKET, REGION
from betsim.db import connect, init_db
from betsim.ingest import parse_events, parse_odds, parse_scores
from betsim.oddsapi import OddsApiClient, OddsApiError
from betsim.store import apply_scores, finish_run, insert_snapshots, start_run, upsert_games

DEFAULT_DB = Path("betsim.db")
DEFAULT_SPORT = "icehockey_nhl"
NOT_YET = {"slate": "M3/M4", "report": "M6", "calibrate": "M6"}


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


if __name__ == "__main__":
    raise SystemExit(main())
