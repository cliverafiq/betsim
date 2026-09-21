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

**M6 complete -- the build is done.** Every milestone in `docs/PLAN.md` has
landed: both API clients, the blind context, Stage 1 and Stage 2, the validator,
an append-only ledger, every shadow arm, settlement, CLV, reporting and
calibration. The predictions, thresholds and frozen parameters are committed in
[`PREREGISTRATION.md`](PREREGISTRATION.md) and tagged `prereg-v1`.

First live slate: **NHL opening night, 2026-09-29**.

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
| `llmcall` | Shared structured-output call plumbing |
| `decider` | Stage 2 priced decisions |
| `arms` | Kelly, Elo, favourite and random shadow arms |
| `ledger` | Append-only bankroll accounting |
| `slate` | The daily loop across every arm |
| `settle` | Grading into the ledger, and CLV |
| `calibrate` | Platt scaling for the `kelly_cal` arm |
| `report` | Metrics, calibration curves and bootstrap intervals |

## Developing

```bash
uv sync                  # install
uv run pytest            # 482 tests, no network, no API keys needed
uv run ruff check src tests
uv run betsim init       # create an empty database
```

Tests never call a paid API or an LLM. Run `pytest` before every commit.

## Setup

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env && open -e .env
```

Paste each key after the `=`, with no quotes and no spaces around the `=`. Then
check it with `uv run betsim doctor`, which validates both keys' shape without
printing them.

`.env` is gitignored. Note that `.gitignore` itself **is** committed -- it lists
filenames for git to skip, so a secret written into it would be published rather
than protected.

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
uv run betsim slate --dry-run      # shadow arms only, no LLM calls
uv run betsim slate                # the full daily loop    (SPENDS MONEY)
uv run betsim close                # closing snapshot       (1 credit)
uv run betsim settle               # score, grade, settle   (2 credits)
uv run betsim clv                  # closing line value     (free)
uv run betsim doctor               # check setup and keys   (free)
uv run betsim report               # metrics for every arm  (free)
uv run betsim calibrate            # refit the kelly_cal map (free)
```

A full day is `odds` -> `context` -> `forecast` -> `slate` -> `close` (near each
puck drop) -> `settle`. `betsim doctor` checks credentials and setup without
printing or transmitting any secret.

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

**Measured cost**, calibrated against a real call (2,149 input / 294 output
tokens for Stage 1, $0.0181):

| | per night (7 games) | per season (180 nights) |
|---|---|---|
| k=1 | $0.16 | $29 |
| k=3 | $0.49 | $88 |
| **k=5** | **$0.81** | **$146** |

A slate reaches only `SLATE_HORIZON` (36 hours) ahead. The odds feed returns
everything it has, often ten days out; forecasting that far ahead both wastes
money and uses standings and form that will have moved by game time. Override
with `--within-hours`, or `--within-hours 0` for no limit.

`betsim forecast --dry-run` renders every prompt and reports the call count
without touching the API.

### The arms

Every arm is built from the **same** price snapshot, which is what makes
comparing them mean anything. The pairing matters too: seed *i*'s Stage 2 sees
seed *i*'s blind forecast and nothing else, and `kelly_si` stakes that same
forecast under a fixed rule -- so `llm_si` versus `kelly_si` reads on bet sizing
alone, holding the forecast constant.

| Arm | Probabilities | Sizing |
|---|---|---|
| `llm_s1..k` | — (Stage 2 decides) | Stage 2, after validation |
| `kelly_s1..k` | `p_blind` from draw *i* | quarter Kelly, same caps |
| `kelly_revised_s1..k` | `p_revised` | quarter Kelly, same caps |
| `elo` | Elo + home ice | quarter Kelly, same caps |
| `fav` | — | flat 10 units on the favourite |
| `random` | — | flat 10 units, seeded |
| `llm_anchored_s1..k` * | — (shown the consensus) | Stage 2, after validation |
| `best_price` * | — | flat 10 units at the best price anywhere |

`*` exploratory, added after the pre-registration and recorded in
[`DEVIATIONS.md`](DEVIATIONS.md). They are not part of the confirmatory
hypotheses.

`fav` and `random` are exempt from **both** the tier and exposure caps: they are
reference lines, not managed bankrolls. `random` is also the null distribution
for CLV, since it bets at the same snapshots with no skill.

The ledger is append-only. Placing appends a negative row, settling appends a
positive one, and a **loss appends a zero-delta row** so every settled bet leaves
exactly two rows and settlement is never inferred from a missing one.
`verify_ledger` re-derives the running balance and runs after every slate.

### Calibration

The documented LLM failure in forecasting is **overconfidence on
high-probability events**. That is correctable after the fact, so rather than
assume the forecasts are beyond help, `kelly_cal` stakes Platt-scaled
probabilities under the same rule as `kelly`. If it wins, the forecasts carry
signal the model's own confidence is destroying.

The map is fitted by IRLS in plain Python, after a 200-game burn-in, and is
**strictly out-of-sample**: `valid_from_utc` records the point past which it may
be applied, because fitting on a game and then applying to that same game is
circular and would flatter the arm for nothing.

### Market width: NHL is two different markets

European books quote NHL h2h as **3-way on regulation time**, with a Draw.
North American books quote the **2-way moneyline** including overtime and the
shootout. They share an endpoint and are not interchangeable: a 3-way bet on a
team *loses* when that team wins in OT. `parse_odds(market_width=...)` drops
books quoting the wrong width, and `betsim odds` reports what it dropped.

The designated bookmaker is **DraftKings, region `us`** -- chosen by measuring a
live 33-game slate, not assumed. Pinnacle (the original plan) is not served on
this account, and the EU region is mostly 3-way. See the correction in
`docs/PLAN.md`.

### Closing snapshots and CLV

A closing line only means anything near the off, so `close` marks a snapshot as
closing **only for games starting within 15 minutes**. Marking a whole slate
closing hours early would record the wrong price as the close and make CLV --
the primary metric -- meaningless.

CLV is derived, not stored: it follows entirely from the price taken and the
closing price, both already in `odds_snapshots`. `betsim clv` reports it per
arm, with the `random` arm labelled as the null -- a bet struck at the slate
snapshot picks up some CLV from timing alone, so the skill signal is the gap to
`random`, not the raw number.

### What Stage 1 is shown

Per team, all odds-free and all from the NHL's own free feed: standings and
record, recent results **stamped with the season they belong to**, rest days,
season-level rates with league ranks (goals for/against per game, power play,
penalty kill, faceoffs), and the goaltending depth chart.

Two labels matter more than they look. Goaltending carries
`confirmed_starter`, normally `null` — the NHL does not publish the starter
until close to puck drop, and a model shown "39 games played" beside a name
reads it as tonight's starter unless the absence is stated. `scratches` is
present and empty for the same reason: it populates near game time, which is
what makes a late context refresh worth scheduling.

Every block carries the season its numbers come from. On opening night all of
it describes *last* season, against rosters that have changed.

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
