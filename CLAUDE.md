# betsim: LLM bankroll-management experiment

Paper-trading experiment testing whether an LLM can manage a hypothetical bankroll across safe, medium, and risky sports bets. No real money, ever.

Full rationale, prior work, prompt schemas, data model, and build milestones live in `docs/design.md`; the research-backed build plan is in `docs/PLAN.md`. Read both before changing experiment logic, prompts, or the schema.

## Hard rules (breaking these invalidates the experiment)

- Paper trading only. Never integrate with sportsbook accounts or place real bets.
- Forward-only. Bet only on games that have not started; reject any bet whose game's `commence_time` has passed. Never evaluate a model on games played before its training cutoff.
- Stage 1 (blind forecast) never sees odds, spreads, totals, or anything derived from them. Its context is assembled by our pipeline and stored verbatim. No web search in Stage 1 for v1.
- Tiers are assigned by code from the price, never by the LLM.
- The validator rejects invalid bets (never clips them) and logs the reason. Rule violations are data.
- Zero bets is always a valid Stage 2 output. Prompts must say so and must never require a minimum number of bets.
- Every shadow arm uses the same caps as the LLM arm and probabilities from the same Stage 1 draw.
- Run `k` independent Stage 1 draws and `k` independent Stage 2 decisions per slate, each Stage 2 with its own bankroll. A single path cannot separate skill from luck. `k = 5` (3 if budget-constrained).
- Prompts are versioned files in `prompts/` (`stage1_v1.md`, `stage2_v1.md`, ...). Never edit a prompt that has been used in a run; add a new version. Store `prompt_version` and the exact `model_id` on every LLM call.
- Log the exact input payload, its hash, the request params, and the raw response for every LLM call.
- The ledger is append-only. Corrections are new rows, never updates or deletes.
- `PREREGISTRATION.md` is committed and tagged before the first live bet, and is not edited afterwards.

## Bankroll rules

- Each arm starts at 1,000 units. Store money as integer minor units (1 unit = 100).
- Tier comes from the designated bookmaker's decimal price `d` at placement (implied probability = 1/d). Intervals are half-open — no price may match two tiers:
  - safe: implied >= 60% (-150 or shorter); max stake 5% of current balance
  - medium: 40% <= implied < 60%; max stake 3%
  - risky: implied < 40% (longer than +150); max stake 1%
- Total open exposure <= 20% of current balance. Minimum stake 1 unit.
- Stakes round **down** to whole minor units, in every arm.
- When the exposure cap binds, process the LLM's bets in the order the model returned them and reject those that breach. Never reorder — order-dependence would make the result irreproducible, and self-managed exposure is itself a measurement.
- An arm is bust when its balance drops below 1 unit. Bust is **absorbing**: the arm stops betting, its ledger freezes, the run continues, and time-to-ruin is reported.
- At most one bet per game per arm.
- v1 markets: moneyline (h2h) singles only. Spreads, totals, and parlays are v2 (see design doc).

## Arms

- `llm_s1`…`llm_sk`: Stage 2 decisions, after validation. Independent bankrolls.
- `kelly_s1`…`kelly_sk`: quarter-Kelly on `p_blind` from the matching Stage 1 draw. `f = (b*p - q) / b` with `b = d - 1`, `q = 1 - p`; stake = `0.25 * f * balance`, capped by tier and exposure; bet only when `f > 0`.
- `kelly_revised`: quarter-Kelly on `p_revised`. Tests whether seeing the price improves the forecast or merely anchors it.
- `kelly_cal`: quarter-Kelly on Platt-scaled `p_blind`. Fitting starts after a 200-game burn-in, refits weekly, and is always strictly out-of-sample.
- `elo`: quarter-Kelly on an Elo model with home-ice advantage, seeded from the prior season and updated after each game. The non-LLM statistical baseline.
- `fav`: flat 10 units on the market favorite in every game.
- `random`: flat 10 units on a uniformly random outcome, seeded.
- `fav` and `random` are exempt from **both** the tier caps and the exposure cap — they are flat-stake references, not managed bankrolls.
- No-bet baseline: constant 1,000 units.

## Metrics (per arm and per tier)

Priority order — the first two settle fast enough to produce an answer; ROI does not.

1. CLV = `d_taken / d_close - 1` at the designated bookmaker. The null is the **`random` arm's CLV distribution**, not zero: bets placed at the slate snapshot accrue some CLV from timing alone, and `random` bets at the same snapshots with no skill.
2. Brier score and log loss of Stage 1 on all games vs the market's de-vigged consensus, plus a 10-bin reliability curve. Use the **sum-over-outcomes** Brier convention and never mix 2-way and 3-way markets in one table — a uniform 2-way forecast scores 0.500, a uniform 3-way scores 0.667.
3. Sizing decomposition: `llm_si` vs `kelly_si` on identical probabilities.
4. Behavior: stake % of balance after losing vs winning days, tier mix vs drawdown depth, share of slates with zero bets, rejection rate by reason, anchoring (`p_revised` vs `p_blind`, distance to market).
5. ROI = total profit / total staked; balance path; max drawdown; bet count; time-to-ruin. Report as underpowered until the bet count justifies otherwise — 1,000 flat bets at -110 still leave a ±6-point ROI CI.

All with bootstrap 95% CIs resampling by day.

## Data

- Odds: The Odds API v4. Free tier is 500 credits/month. `/odds` costs markets × regions, so v1 (h2h, one region) is 1 credit per call; `/scores` with `daysFrom` costs 2; **`/events` and `/sports` are free** — use `/events` for scheduling. Read `x-requests-remaining` from response headers, record it per run, and stop with headroom before it reaches 0.
- Designated bookmaker: **DraftKings**, region `us`. Pinnacle is not served on this account (it returns games with no bookmaker entry), and the EU region quotes NHL h2h as **3-way on regulation time**, which is a different market from the 2-way moneyline. DraftKings is the only US book covering a full slate, at ~4.3% margin. Store every bookmaker returned, for the consensus.
- **Never mix market widths.** European NHL h2h is 3-way with a Draw and settles on regulation time; North American h2h is 2-way including overtime and the shootout. A 3-way bet on a team loses when that team wins in OT. `parse_odds(market_width=...)` filters, and the slate must only ever use the sport's own width.
- De-vig with three methods — multiplicative, power, and Shin. Report **Shin** as primary. The de-vigged consensus is the yardstick for the headline Brier comparison, and multiplicative de-vig is biased on exactly the favorite-longshot axis the tiers are built on.
- The Odds API returns scores but doesn't grade bets; settlement is ours. Postponed or cancelled games are void (stake refunded). NHL and NBA moneylines cannot push (OT/shootout); NFL ties push and refund.
- UTC timestamps everywhere. Decimal odds internally; convert American odds on ingest.

## LLM call conventions

- Model under test: `claude-opus-5`. Adaptive thinking (`thinking: {type: "adaptive"}`).
- **There are no sampling parameters.** `temperature`, `top_p`, `top_k`, and `budget_tokens` return HTTP 400 on Opus 5, Sonnet 5, Opus 4.8/4.7 and Fable 5/5.1. The controllable knob is `output_config.effort` (`low`…`max`); run at `high`. There is no seed parameter — reproducibility by seed is unavailable, which is why `k` independent draws are mandatory.
- Log `model_id`, `effort`, the `thinking` config, and `max_tokens` in `params_json`. Never log a `temperature` field.
- Enforce output schemas with **structured outputs** (`output_config.format` plus `messages.parse()`) against the Pydantic model, rather than parsing free text and retrying.
- Handle `stop_reason: "refusal"` as a distinct logged outcome. Check `stop_reason` before reading content; a refusal must never crash a nightly run, and consecutive refusals should alert.

## Stack and conventions

- Python 3.13+, `uv` for env and dependencies (`uv init --package`, `uv run`, committed lockfile), SQLite, Pydantic v2 for every LLM input/output schema, httpx, pytest.
- Code in `src/betsim/`, tests in `tests/`, recorded API fixtures in `tests/fixtures/`.
- Odds math, de-vig, tier assignment, Kelly, Elo, validation, settlement, and metrics are pure functions with unit tests.
- Tests never call paid APIs or LLMs; use recorded fixtures.
- Secrets live in `.env` (gitignored). Keep `.env.example` current.
- Run `pytest` before every commit.

## Commands (planned; update this section when they change)

- `python -m betsim slate --sport <key>`: fetch odds and context, run k Stage 1 draws, k Stage 2 decisions, and all shadow arms.
- `python -m betsim close --sport <key>`: take closing-odds snapshots for games starting soon.
- `python -m betsim settle`: fetch scores, settle open bets, append ledger rows.
- `python -m betsim report`: metrics tables and plots.
- `python -m betsim calibrate`: refit the Platt scaling used by `kelly_cal` (out-of-sample only).

## Decisions closed (see docs/PLAN.md for the evidence)

Sport: NHL (`icehockey_nhl`) from 2026-09-29, NBA (`basketball_nba`) from 2026-10-20.
Bookmaker: DraftKings, region `us` (verified live; Pinnacle unavailable, EU is 3-way). Model: `claude-opus-5` at effort `high`, k=5.
Context: `api-web.nhle.com/v1/` (free, no key).
