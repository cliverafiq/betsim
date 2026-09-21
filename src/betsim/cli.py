"""Command-line entry point.

Subcommands land with their milestones. M1 ships the data layer: ``quota`` and
``events`` are free, ``odds`` and ``scores`` cost credits and record what they
spent. ``slate`` needs Stage 1 and Stage 2, which are M3 and M4.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from betsim.arms import elo_probabilities
from betsim.config import (
    CLOSING_WINDOW,
    DESIGNATED_BOOKMAKER,
    EFFORT,
    K_SEEDS,
    MARKET,
    MODEL_ID,
    REGION,
)
from betsim.context import build_context, context_hash, index_results_by_team, seed_elo_from_results
from betsim.db import connect, init_db
from betsim.decider import Stage2Decider
from betsim.elo import EloTable
from betsim.forecaster import Stage1Forecaster, summarise
from betsim.ingest import parse_events, parse_odds, parse_scores
from betsim.ledger import arm_state, arms_in_play, verify_ledger
from betsim.money import minor_to_units
from betsim.nhl import NhlApiError, NhlClient, parse_club_schedule, parse_standings
from betsim.oddsapi import OddsApiClient, OddsApiError
from betsim.prompts import load_prompt
from betsim.settle import bet_clv, clv_by_arm, settle_open_bets
from betsim.slate import build_game_refs, run_slate
from betsim.store import (
    apply_scores,
    context_exists,
    finish_run,
    forecast_counts,
    games_starting_within,
    insert_context,
    insert_forecast,
    insert_llm_call,
    insert_snapshots,
    latest_contexts,
    llm_spend,
    load_elo_ratings,
    save_elo_ratings,
    start_run,
    upcoming_games,
    upsert_games,
)
from betsim.teams import TeamIndex, UnknownTeam

DEFAULT_DB = Path("betsim.db")
DEFAULT_SPORT = "icehockey_nhl"
NOT_YET = {"report": "M6", "calibrate": "M6"}
PRIOR_SEASON = 20252026
NHL_CALL_DELAY = 0.15  # be a considerate client of a free public API
CHARS_PER_TOKEN = 4  # rough, for the dry run only


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
                closing_ids = (
                    games_starting_within(conn, args.sport, CLOSING_WINDOW)
                    if args.closing
                    else set()
                )
                n = insert_snapshots(
                    conn,
                    prices,
                    captured_utc=datetime.now(UTC),
                    closing_game_ids=closing_ids,
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


def cmd_forecast(args: argparse.Namespace) -> int:
    """Run k blind Stage 1 draws per game. **This spends money on LLM calls.**"""
    conn = _open_db(args.db)
    try:
        games = upcoming_games(conn, args.sport)
        contexts = latest_contexts(conn, [g.id for g in games])
        pending = [g for g in games if g.id in contexts]
        missing = [g.id for g in games if g.id not in contexts]
        if missing:
            print(f"  {len(missing)} games have no context yet; run `betsim context`")
        if not pending:
            print("nothing to forecast")
            return 0

        forecaster = Stage1Forecaster(model=args.model, effort=args.effort)

        if args.dry_run:
            chars = sum(len(forecaster.render(contexts[g.id])) for g in pending)
            calls = len(pending) * args.k
            approx_in = chars * args.k // CHARS_PER_TOKEN
            print(f"dry run: {calls} calls ({len(pending)} games x k={args.k})")
            print(f"  ~{approx_in:,} input tokens, model {args.model} at effort {args.effort}")
            print("  no API calls made, nothing spent")
            return 0

        run_id = start_run(conn, "forecast")
        all_calls = []
        with conn:
            for game in pending:
                have = forecast_counts(conn, game.id)
                if have >= args.k and not args.force:
                    continue
                calls = forecaster.forecast_k(contexts[game.id], k=args.k)
                all_calls.extend(calls)
                for call in calls:
                    call_id = insert_llm_call(conn, call, stage=1)
                    if call.ok and call.probabilities:
                        insert_forecast(
                            conn,
                            game.id,
                            call_id,
                            seed_idx=call.seed_idx,
                            probabilities=call.probabilities,
                        )
            summary = summarise(all_calls)
            finish_run(conn, run_id, notes=json.dumps(summary, sort_keys=True))

        print(
            f"{summary['ok']}/{summary['calls']} draws succeeded"
            f" ({summary['failed']} failed, {summary['refusals']} refused)"
        )
        if "mean_p_home" in summary:
            print(
                f"  mean p_home {summary['mean_p_home']}, "
                f"widest spread across draws {summary['spread_p_home']}"
            )
        print(f"  estimated spend ${summary['cost_usd']}")
        total = llm_spend(conn, stage=1)
        print(
            f"  stage 1 to date: {total['calls']} calls, "
            f"{total['input_tokens']:,} in / {total['output_tokens']:,} out"
        )
    finally:
        conn.close()
    return 0


def _elo_probabilities(conn, refs) -> dict:
    """Elo win probabilities, using the tricodes already stored in each context."""
    ratings = load_elo_ratings(conn)
    if not ratings:
        return {}
    table = EloTable()
    table.ratings.update(ratings)
    contexts = latest_contexts(conn, [r.game_id for r in refs])
    pairs = {
        gid: (ctx["home"]["tricode"], ctx["away"]["tricode"])
        for gid, ctx in contexts.items()
        if "home" in ctx and "away" in ctx
    }
    return elo_probabilities(refs, table, pairs)


def cmd_slate(args: argparse.Namespace) -> int:
    """Run the daily loop across every arm. Spends money unless --dry-run."""
    conn = _open_db(args.db)
    try:
        games = upcoming_games(conn, args.sport)
        refs = build_game_refs(conn, games)
        if not refs:
            print("no games with designated-bookmaker prices; run `betsim odds` first")
            return 0

        decider = None if args.dry_run else Stage2Decider(model=args.model, effort=args.effort)
        if args.dry_run:
            print(f"dry run: {len(refs)} games, shadow arms only, no LLM calls")

        run_id = start_run(conn, "slate")
        with conn:
            result = run_slate(
                conn,
                sport=args.sport,
                decider=decider,
                k=args.k,
                elo_probs=_elo_probabilities(conn, refs),
                random_seed=args.seed,
            )
            finish_run(
                conn,
                run_id,
                notes=json.dumps(
                    {
                        "games": result.games,
                        "placed": result.placed,
                        "cost_usd": round(result.cost_usd, 4),
                    },
                    sort_keys=True,
                ),
            )

        print(f"{result.games} games, k={result.seeds}")
        for arm in sorted(result.placed):
            staked = minor_to_units(result.staked_minor.get(arm, 0))
            state = arm_state(conn, arm)
            print(
                f"  {arm:20} {result.placed[arm]:2} bets  {staked:7.2f} u staked  "
                f"balance {minor_to_units(state.balance_minor):8.2f} u"
                f"{'  BUST' if state.bust else ''}"
            )
        if result.rejected:
            print(
                "  rejections: " + ", ".join(f"{k}={v}" for k, v in sorted(result.rejected.items()))
            )
        for note in result.notes:
            print(f"  note: {note}")
        if not args.dry_run:
            print(
                f"  estimated spend ${result.cost_usd:.4f}"
                f"{f', {result.refusals} refusals' if result.refusals else ''}"
            )

        for arm in arms_in_play(conn):
            verify_ledger(conn, arm)
        print("  ledger verified")
    finally:
        conn.close()
    return 0


def cmd_close(args: argparse.Namespace) -> int:
    """Take a closing snapshot. Only games inside the window are marked closing."""
    args.closing = True
    args.regions = getattr(args, "regions", REGION)
    args.record = getattr(args, "record", None)
    return cmd_odds(args)


def cmd_settle(args: argparse.Namespace) -> int:
    """Fetch scores, grade open bets and append the ledger rows. Costs 2 credits."""
    conn = _open_db(args.db)
    try:
        with conn:
            run_id = start_run(conn, "settle")
            with OddsApiClient(_api_key(), record_dir=args.record) as api:
                payload = api.fetch_scores(args.sport, days_from=args.days_from)
                scores = parse_scores(payload)
                completed = apply_scores(conn, scores)
            result = settle_open_bets(conn, args.sport)
            finish_run(
                conn,
                run_id,
                credits_remaining=api.quota.remaining,
                notes=f"settle: {completed} games, {result.settled} bets, "
                f"cost {api.quota.last_cost}",
            )

        print(f"{completed} of {len(scores)} games completed")
        print(
            f"{result.settled} bets settled: {result.won} won, {result.lost} lost, "
            f"{result.void} void"
        )
        for arm in sorted(result.by_arm):
            profit = minor_to_units(result.profit_minor.get(arm, 0))
            state = arm_state(conn, arm)
            print(
                f"  {arm:20} {result.by_arm[arm]:2} settled  {profit:+8.2f} u  "
                f"balance {minor_to_units(state.balance_minor):8.2f} u"
                f"{'  BUST' if state.bust else ''}"
            )
        for err in result.errors:
            print(f"  ERROR {err}", file=sys.stderr)

        for arm in arms_in_play(conn):
            verify_ledger(conn, arm)
        print("  ledger verified")
        print(f"credits remaining: {api.quota.remaining}")
    finally:
        conn.close()
    return 0


def cmd_clv(args: argparse.Namespace) -> int:
    """Closing line value per arm. Free -- derived from data already stored."""
    conn = _open_db(args.db)
    try:
        values = bet_clv(conn)
        if not values:
            print("no bets have a closing price yet; run `betsim close` near puck drop")
            return 0
        summary = clv_by_arm(values)
        baseline = summary.get("random", {}).get("mean_clv")
        print(f"CLV over {len(values)} bets with a closing price:")
        for arm, stats in summary.items():
            marker = "  <- null" if arm == "random" else ""
            print(f"  {arm:20} n={stats['n']:3}  mean CLV {stats['mean_clv']:+.4f}{marker}")
        if baseline is not None:
            print(f"\n  The null is the `random` arm ({baseline:+.4f}), not zero: a bet struck")
            print("  at the slate snapshot picks up some CLV from timing alone.")
    finally:
        conn.close()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Check credentials and setup without printing or transmitting any secret."""
    import re
    import subprocess

    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    problems = 0

    print("credentials")
    if not env_path.exists():
        print(f"  .env               MISSING -- copy {root / '.env.example'} to .env")
        problems += 1
    else:
        print("  .env               present")
        ignored = (
            subprocess.run(
                ["git", "check-ignore", "-q", ".env"], cwd=root, capture_output=True, check=False
            ).returncode
            == 0
        )
        tracked = (
            subprocess.run(
                ["git", "ls-files", "--error-unmatch", ".env"],
                cwd=root,
                capture_output=True,
                check=False,
            ).returncode
            == 0
        )
        print(f"  gitignored         {'yes' if ignored else 'NO -- your key would be published'}")
        print(f"  tracked by git     {'YES -- REMOVE IT NOW' if tracked else 'no'}")
        problems += (not ignored) + tracked

        from dotenv import dotenv_values

        values = dotenv_values(env_path)
        checks = {
            "ODDS_API_KEY": (r"^[0-9a-f]{32}$", "32 lowercase hex characters"),
            "ANTHROPIC_API_KEY": (r"^sk-ant-\S{20,}$", "starts with sk-ant-"),
        }
        for name, (pattern, shape) in checks.items():
            raw = values.get(name)
            if not raw or not raw.strip():
                print(f"  {name:18} missing or empty")
                problems += 1
                continue
            value = raw.strip()
            masked = f"{value[:6]}...{value[-4:]}" if len(value) > 12 else "(very short)"
            notes = []
            if value != raw:
                notes.append("surrounding whitespace")
            if value[0] in "\"'":
                notes.append("remove the quotes")
            if not re.match(pattern, value):
                notes.append(f"expected {shape}")
            status = "; ".join(notes) if notes else "looks right"
            problems += bool(notes)
            print(f"  {name:18} {masked}  len={len(value)}  {status}")

    print("\nsetup")
    db = args.db
    print(
        f"  database           {'present' if Path(db).exists() else 'not created -- run `betsim init`'}"
    )
    try:
        load_prompt("stage1_v1"), load_prompt("stage2_v1")
        print("  prompts            stage1_v1, stage2_v1 found")
    except FileNotFoundError as exc:
        print(f"  prompts            {exc}")
        problems += 1

    print(
        "\nready"
        if not problems
        else f"\n{problems} problem(s) above. Nothing was printed in full and nothing was sent."
    )
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

    p_fc = common(sub.add_parser("forecast", help="run blind Stage 1 draws (SPENDS MONEY)"))
    p_fc.add_argument("--k", type=int, default=K_SEEDS, help="independent draws per game")
    p_fc.add_argument("--model", default=MODEL_ID)
    p_fc.add_argument("--effort", default=EFFORT, choices=("low", "medium", "high", "xhigh", "max"))
    p_fc.add_argument("--dry-run", action="store_true", help="estimate without calling the API")
    p_fc.add_argument(
        "--force", action="store_true", help="re-forecast games that already have draws"
    )
    p_fc.set_defaults(func=cmd_forecast)

    p_sl = common(sub.add_parser("slate", help="run the daily loop across every arm"))
    p_sl.add_argument("--k", type=int, default=K_SEEDS)
    p_sl.add_argument("--model", default=MODEL_ID)
    p_sl.add_argument("--effort", default=EFFORT, choices=("low", "medium", "high", "xhigh", "max"))
    p_sl.add_argument("--seed", type=int, default=0, help="seed for the random arm")
    p_sl.add_argument("--dry-run", action="store_true", help="shadow arms only, no LLM calls")
    p_sl.set_defaults(func=cmd_slate)

    p_cl = common(sub.add_parser("close", help="closing snapshot near puck drop (costs credits)"))
    p_cl.add_argument("--regions", default=REGION)
    p_cl.add_argument("--record", type=Path, default=None)
    p_cl.set_defaults(func=cmd_close)

    p_st = common(sub.add_parser("settle", help="score, grade and settle (costs credits)"))
    p_st.add_argument("--days-from", type=int, default=3, choices=(1, 2, 3))
    p_st.add_argument("--record", type=Path, default=None)
    p_st.set_defaults(func=cmd_settle)

    common(sub.add_parser("clv", help="closing line value per arm (free)")).set_defaults(
        func=cmd_clv
    )

    common(
        sub.add_parser("doctor", help="check credentials and setup (free, prints no secrets)"),
        sport=False,
    ).set_defaults(func=cmd_doctor)

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
