"""Command-line entry point.

Subcommands land with their milestones; M0 ships the skeleton and ``init`` only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from betsim.db import connect, init_db

DEFAULT_DB = Path("betsim.db")

COMMANDS = ("init", "slate", "close", "settle", "report", "calibrate")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="betsim", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--sport", default="icehockey_nhl")
    args = parser.parse_args(argv)

    if args.command == "init":
        with connect(args.db) as conn:
            init_db(conn)
        print(f"initialised {args.db}")
        return 0

    parser.error(f"{args.command!r} is not implemented yet (see docs/PLAN.md milestones)")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
