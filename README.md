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

**M3 complete** -- pure logic, the Odds API client, the NHL context builder
with a seeded Elo baseline, and blind Stage 1 forecasting. Stage 2 and the
betting arms (M4) are next. Milestones are tracked in `docs/PLAN.md`.

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
| `oddsapi` | The Odds API v4 client, credit accounting, fixture recording |
| `ingest` | Parsing API payloads into domain objects |
| `store` | Persisting games, snapshots, scores, contexts and credit spend |
| `nhl` | NHL official API client and parsing (free, no key) |
| `teams` | Matching team names between the two feeds |
| `context` | Blind Stage 1 context, with an allowlist leak guard |
| `prompts` | Versioned, hashed prompt templates |
| `forecaster` | Stage 1 blind forecasting, k independent draws |

## Developing

```bash
uv sync                  # install
uv run pytest            # 360 tests, no network, no API keys needed
uv run ruff check src tests
uv run betsim init       # create an empty database
```

Tests never call a paid API or an LLM. Run `pytest` before every commit.

## Setup

Copy `.env.example` to `.env` and add your keys. `.env` is gitignored.

```bash
uv run betsim init             # create the database
uv run betsim quota            # check the key and credits  (free)
uv run betsim events           # fetch the NHL schedule     (free)
uv run betsim odds             # price snapshot             (1 credit)
uv run betsim odds --closing   # closing snapshot, for CLV  (1 credit)
uv run betsim scores           # final scores               (2 credits)
uv run betsim seed-elo         # seed the Elo baseline      (free)
uv run betsim context          # blind Stage 1 contexts     (free)
uv run betsim forecast --dry-run   # estimate Stage 1, spend nothing
uv run betsim forecast             # blind Stage 1 draws    (SPENDS MONEY)
```

`seed-elo` and `context` use the NHL's own API, which is free and needs no key,
so neither spends Odds API credits.

### Credit budget

The free tier is 500 credits a month. `/sports` and `/events` are free, `/odds`
costs `markets x regions`, and `/scores` costs 2 when reaching back for
completed games. One NHL day is about **6 credits** -- one slate call, roughly
three closing snapshots, and settlement -- so **~180 a month**, which the free
tier covers comfortably. Adding NBA roughly doubles that and leaves no margin,
so move to the $30 tier at that point.

The client refuses any costed call that would drop the balance below a 25-credit
reserve, and every run records the credits remaining so the spend is auditable.

### Stage 1 and the k draws

Current Claude models expose **no sampling parameters** -- `temperature`,
`top_p`, `top_k` and `budget_tokens` all return HTTP 400 on Opus 5 -- and there
is no seed. Repeated variance therefore comes from repeated calls: `k`
independent draws per game, all sharing one `input_hash`, which is what proves
they were independent draws on identical input rather than `k` different
questions. One draw cannot separate skill from luck; KellyBench's per-model seed
spread was +34.1% to -32.9% on identical data.

Every call is logged whether or not it succeeded. A refusal, a malformed
forecast or an API failure is recorded and returned, never thrown -- rule
violations and refusals are results, and one bad game must not take down a
slate. `params_json` records the model, effort, thinking config and token
ceiling, and deliberately has no `temperature` field.

Measured cost, at the real rendered prompt size of ~1,000 input tokens:
roughly **$0.03-0.06 per call**, so Stage 1 for a full NHL season at k=5 is
about **$160-350** depending on how much the model thinks. Run
`betsim forecast --dry-run` first -- it renders every prompt and reports the
call count without touching the API.

### Keeping odds out of the blind stage

Stage 1 must never see a price. If it does, the model's "probability" echoes the
market and every edge estimate downstream is meaningless, so the rule is enforced
in code: `betsim.context.assert_no_odds_leak` runs on every context before it is
stored, and `build_context` calls it itself.

The guard is an **allowlist** of permitted keys rather than a denylist of banned
words. A denylist fails open -- the first field nobody thought to ban sails
through. An allowlist fails closed: anything new has to be added to
`ALLOWED_KEYS` deliberately, which is the review step that should happen. There
is a second check on string values for betting vocabulary.

The NHL's `/v1/score/{date}` endpoint is deliberately unused because it carries
an `oddsPartners` field; only `/standings` and `/club-schedule-season` feed the
context, and both are verified odds-free.

### Matching team names

The two feeds name the same franchise differently, and a mismatch is silent: the
game gets no context, no Elo rating, and a worthless forecast. Real cases --
`Montréal Canadiens` (accented in the NHL feed, ASCII in The Odds API),
`St. Louis Blues` (punctuation), and `Utah Mammoth` (renamed franchise).
`betsim.teams` normalises accents, punctuation, whitespace and case, and **raises**
on anything it cannot resolve. The index is derived from the live standings
rather than hardcoded, so renames do not need a code change.

### Fixtures

The NHL fixtures in `tests/fixtures/nhl/` are **real recorded responses** --
that API is free, so there was no reason to fake them.

The Odds API fixtures in `tests/fixtures/` are **synthetic** -- hand-built to the
documented v4 schema, because no key was available. Re-record them against the
real thing on the first authenticated call:

```bash
uv run betsim odds --record tests/fixtures
```

The recorder stores only the response body and the three quota headers. The API
key travels in the query string, so the request URL is never written to disk.
