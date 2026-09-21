# betsim

A paper-trading experiment testing whether an LLM can manage a hypothetical
sports-betting bankroll. **No real money, ever** -- the project never integrates
with a sportsbook account and never places a real bet.

## What it measures

Two stages, deliberately separated:

1. **Stage 1 (blind)** forecasts each game without ever seeing odds, spreads,
   totals, or anything derived from them.
2. **Stage 2 (priced)** sees the prices, the bankroll, the caps and the recent
   record, and decides what to bet -- including nothing at all.

That split is the point. Published work in this area showed the model the price
throughout, so when those agents lost money there was no way to tell whether the
forecast or the bet sizing was at fault. Here a Kelly shadow arm re-stakes the
*same* probabilities under a fixed rule, so the difference is attributable to
sizing alone.

Expect the model to lose slowly to the bookmaker's margin. The interesting
results are calibration, sizing discipline, selectivity, and behaviour under
drawdown -- not whether it turns a profit.

Read [`docs/PLAN.md`](docs/PLAN.md) for the evidence behind every decision and
[`docs/design.md`](docs/design.md) for the design. [`CLAUDE.md`](CLAUDE.md) holds
the rules that apply to every session.

## Status

**M0 complete** -- schema and pure logic with tests. Nothing touches the network
yet. Milestones are tracked in `docs/PLAN.md`.

| Module | What it does |
|---|---|
| `money` | Integer minor units; floor rounding for every stake and payout |
| `odds` | American/decimal conversion, implied probability, booksum |
| `devig` | Multiplicative, power and Shin de-vig, plus a cross-book consensus |
| `tiers` | Half-open tier boundaries and per-tier stake caps |
| `kelly` | Fractional-Kelly staking for the shadow arms |
| `validator` | Rejects invalid bets -- never clips them -- and records the reason |
| `settlement` | Grades bets, with per-sport push and void rules |
| `elo` | The non-LLM statistical baseline arm |
| `metrics` | Brier, log loss, ROI, drawdown, CLV, calibration, bootstrap CIs |
| `models` | Pydantic schemas for every LLM input and output |
| `db` | SQLite schema; the ledger is append-only |

## Developing

```bash
uv sync                  # install
uv run pytest            # 198 tests, no network, no API keys needed
uv run ruff check src tests
uv run betsim init       # create an empty database
```

Tests never call a paid API or an LLM. Run `pytest` before every commit.

## Setup

Copy `.env.example` to `.env` and fill in the two keys. `.env` is gitignored.
