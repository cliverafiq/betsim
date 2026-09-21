# Design: LLM bankroll-management experiment

Hypothetical money only. `CLAUDE.md` holds the rules that apply to every session; `PLAN.md` holds
the build plan and the evidence behind the decisions; this file holds the reasoning and detail.

## Research questions

1. How large is the gap between the model's blind forecasts and the de-vigged market consensus,
   with a confidence interval — and is it concentrated by tier, or on underdogs?
2. Given its own forecasts, does it size bets better or worse than quarter-Kelly?
3. Does seeing the price improve the forecast (`kelly_revised` vs `kelly`) or merely anchor it?
4. Does post-hoc recalibration rescue the forecast (`kelly_cal` vs `kelly`)?
5. How does it behave in drawdowns — loss chasing, drift toward risky bets, and does it adapt at all
   across a season?
6. Does any arm beat the `fav` and `elo` baselines?

Note that (1) is deliberately *not* framed as a yes/no. The published work has effectively answered
the yes/no: closed-book LLM forecasts are worse than the market, and only open-book forecasting has
reached parity. The quantity and its location are what remain open.

## Prior work and expectations

- **KellyBench** (General Reasoning, April 2026; arXiv 2604.27865). Five frontier models × five seeds
  ran a simulated 2023-24 Premier League season. Every model lost money on average: ROI −7.9%
  (GPT-5.4) to −89.6% (Kimi K2.5), with ruin in 6 of 25 seeds. A human quant analyst made **+5.1% on
  just 39 bets**; a plain Dixon-Coles model (−15.4%) beat 3 of the 5 models. Seeds that adapted
  intra-season averaged −11.1% against −70.0% for static seeds. Draw/longshot miscalibration appeared
  in 22 of 25 seeds. The failures were mostly behavioral rather than analytical — models wrote correct
  Kelly code and never invoked it, diagnosed their own miscalibration in writing and never fixed it,
  and in one case declared the season complete six separate times while it was still running.
  https://arxiv.org/abs/2604.27865
- **WC2026-Agents** (July 2026; arXiv 2607.17765). Four frontier models forecast and placed virtual
  bets on all 104 World Cup matches, contamination-free because every match postdated their training
  cutoffs. None beat the market's Brier score of 0.469; ROI ranged −18% to +10%; **flat-staking the
  market favorite out-earned all four**; market-citation rates ranged 12% to 100%.
  https://arxiv.org/abs/2607.17765 (data and code: https://github.com/graphuofm/FIFA2026LLM)
- **LLM-SoccerArena** (2026; arXiv 2607.24573). Seven models × open/closed book × prompt style ×
  horizon, 8,736 forecasts. Brier 0.506–0.546 with no significant between-model differences.
  **Open-book improved Brier by 0.0228**, and the best open-book run (0.497) essentially tied the
  de-vigged market consensus (0.498). Forecast horizon barely mattered (0.0021 from T-24h to T-2h).
- **AI World Cup 2026** (August 2026; arXiv 2608.03416). Self-reported confidence did not track
  accuracy (r = −0.060). The authors give eight recommendations for future work; this design adopts
  their #1 (pre-register), #2 (repeated independent forecasts), #3 (proper scoring rules),
  #6 (document information access), #7 (strong non-LLM baselines incl. market), and #8 (automated
  pipeline validation).
- **ForecastBench / AIA Forecaster.** Outside sports, LLMs are systematically **overconfident on
  high-probability events**, and post-hoc calibration (Platt scaling, extremization) has moved a
  system to statistical parity with the superforecaster median. This is the direct motivation for
  the `kelly_cal` arm.

**Expect the model to lose slowly to the bookmaker's margin, its blind Brier to trail the de-vigged
consensus, and `fav` to beat it on ROI.** The interesting results are calibration, sizing discipline,
selectivity, and behavior under drawdown — not whether it turns a profit.

## Why the design looks like this

- **Two stages.** If the model sees the price first, its "probability" echoes the market and any edge
  estimate is meaningless. Stage 1 is blind; Stage 2 sees prices and decides. No prior work does
  this — KellyBench and WC2026 both showed the price throughout, so neither could say whether a loss
  came from the forecast or the sizing. This separation is the experiment's main contribution.
- **Kelly shadow on identical probabilities.** Separates forecasting skill from money management.
  If both arms lose, the forecasts are the problem; if `kelly` beats `llm`, the sizing is.
- **k independent seeds.** KellyBench's per-model seed spread was +34.1% to −32.9% on the same season
  and the same data. A single live forward path is one seed and cannot distinguish skill from luck.
  k independent Stage 1 draws and k independent Stage 2 bankrolls cost k× LLM tokens and **zero extra
  odds credits**, since one snapshot feeds all of them. Draw dispersion also gives a free uncertainty
  signal.
- **A non-LLM statistical baseline (`elo`).** Without one you cannot tell whether the LLM is bad or
  the task is hard. Dixon-Coles beat most of the frontier models in KellyBench.
- **Zero bets always allowed.** KellyBench *required* one bet per matchday, which manufactures losses.
  The human quant's +5.1% came from 39 bets in a season. Selectivity is a measurement, not a failure.
- **Forward-only.** A model can remember results of past games from its training data. Live paper
  trading avoids contamination by construction.
- **No web search in Stage 1 (v1).** Previews and articles quote betting lines, which would leak the
  market into the blind stage. This is a real cost — open-book is the configuration that reached
  market parity in the literature — so a web-enabled Stage 1 is the highest-value v2 arm, logging
  every query and result.
- **Moneyline only (v1).** A spread or total line is itself the market's estimate, so showing it to
  Stage 1 breaks blindness. It also keeps each odds call at 1 credit.
- **Tiers describe variance, not value.** At -110 the break-even win rate is 52.38% and a coin-flipper
  loses 4.55% of stakes; a two-leg parlay of -110 legs carries a house edge of 8.88%. "Safe" means
  smaller swings, not positive expected value.

## Daily loop

1. `slate` (once per day per sport): one h2h odds call; build Stage 1 context per game; run k Stage 1
   draws; run k Stage 2 decisions with designated-book prices; validate; write `llm_si` bets; compute
   `kelly_si`, `kelly_revised`, `kelly_cal`, `elo`, `fav` and `random` bets from the same snapshot.
2. `close` (around start windows): one odds call per window; mark snapshots for games starting within
   15 minutes as closing.
3. `settle` (after games finish): fetch scores (`/scores` with `daysFrom`, 2 credits); settle open
   bets; append ledger rows.
4. `report` (on demand): metrics per arm and tier with CIs, calibration table, balance curves.

Bets are priced at the snapshot shown to the model. Stage 2 must run within 60 minutes of that
snapshot — this is about price staleness and CLV integrity, not forecast quality, since horizon has
been shown to matter very little to accuracy.

## Stage 1: blind forecast

Inputs, assembled by the pipeline and stored verbatim: league, home and away teams, start time,
recent results, standings, and injuries or availability where a source exists. Nothing derived from
odds. For NHL, all of this comes free and keyless from `api-web.nhle.com/v1/`.

Output per game:

```json
{"game_id": "str", "p_home": 0.0, "p_away": 0.0, "p_draw": null, "notes": "max 300 chars"}
```

Enforced with structured outputs (`output_config.format`) against the Pydantic model, so schema-valid
JSON is guaranteed rather than parsed-and-retried. Probabilities in [0, 1]; if the sum is within 0.02
of 1, normalize, otherwise log a failure. Run k independent draws and store each one; the mean is
available as an ensemble and the spread as an uncertainty measure.

## Stage 2: bet decision

Inputs: current balance, open bets, remaining exposure, per-tier caps in units (precomputed), a record
summary for the last 14 days (bets, ROI by tier, current streak, balance path), and today's games with
`p_blind`, designated-book prices in decimal and American, and the code-assigned tier for each
selection. The prompt states that placing zero bets is fine.

Output:

```json
{"bets": [{"game_id": "str", "selection": "home|away|draw", "stake_units": 0.0, "p_revised": 0.0, "reason": "max 300 chars"}], "day_notes": "max 500 chars"}
```

`p_revised` measures anchoring — how far the model moves from `p_blind` toward the market once it sees
prices — and drives the `kelly_revised` arm.

## Validator

Reject, never clip, and log each rejection with its reason:

- game exists, hasn't started, and the snapshot is at most 60 minutes old
- selection is valid for the market (draw only in 3-way markets)
- stake is at least 1 unit, within the tier cap, within remaining exposure, and within balance
- at most one bet per game

Bets are processed in the order the model returned them. When the exposure cap binds, later bets are
rejected rather than reordered or scaled — reordering would make results irreproducible, and whether
the model manages its own exposure is itself a measurement.

## Shadow arms

**Kelly.** For each game and outcome, `f = (b*p - q) / b` with `b = d - 1`, `q = 1 - p`. Take the
outcome with the largest positive `f`, stake `0.25 * f * balance` rounded **down** to whole minor
units, then apply tier and exposure caps, processing games in start-time order. Skip if the stake is
under 1 unit. `kelly_si` uses `p_blind` from draw *i*; `kelly_revised` uses `p_revised`; `kelly_cal`
uses Platt-scaled `p_blind`.

**Calibration (`kelly_cal`).** After a 200-game burn-in, fit Platt scaling on accumulated
(`p_blind`, outcome) pairs, refit weekly, and apply only to games after the fitting window. Strictly
out-of-sample. This tests the documented overconfidence failure directly: if `kelly_cal` beats
`kelly`, the forecasts carry signal that the model's own confidence is destroying.

**Elo.** Standard Elo with home-ice advantage, seeded from prior-season results, K tuned on prior
seasons only, updated after each settled game. A pure function with unit tests.

## Metrics and statistics

- **CLV** = `d_taken / d_close - 1` at the designated bookmaker (DraftKings,
  region `us`; see the correction in `PLAN.md` -- Pinnacle is not served on
  this account and the EU region quotes NHL as a 3-way market). Primary metric: best-evidenced skill
  signal in the literature and far faster to settle than ROI. Compare against the **`random` arm's
  CLV distribution**, not zero — betting at the slate snapshot accrues CLV from timing alone, and
  `random` bets at the same snapshots with no skill, which makes it the correct null.
- **De-vig** per bookmaker by three methods: multiplicative `p_i = (1/d_i) / sum_j (1/d_j)`, power,
  and Shin. Consensus = mean of de-vigged probabilities across bookmakers. **Shin is primary**;
  multiplicative does not correct favorite-longshot bias, which is the exact axis the tier system is
  built on, and the consensus is the yardstick for the headline comparison. Report all three.
- **Brier (sum over outcomes)**: `sum_k (p_k - o_k)^2` per game, averaged over games. Log loss:
  `-ln p(actual)`. Compare Stage 1 with the consensus on all games rather than only those bet on.
  Under this convention a uniform 2-way forecast scores 0.500 and a uniform 3-way 0.667 — **NHL
  numbers are not comparable to the 3-way soccer literature**, so never mix widths in one table.
- **Calibration**: bucket `p_blind` into 10-point bins and compare with observed frequencies. Expect
  overconfidence concentrated on high-probability outcomes.
- **Sizing decomposition**: `llm_si` vs `kelly_si` on identical probabilities, per seed.
- **ROI** = sum of profit / sum of stakes. **Max drawdown** = max over time of
  (running peak − balance) / running peak. **Time-to-ruin** where an arm busts.
- **Uncertainty**: bootstrap 95% CIs resampling by day, since same-day bets are correlated. For scale,
  1,000 flat bets at -110 leave an ROI CI of about ±5.9 points, and longshot-heavy tiers need 2-3×
  as many bets for the same precision. Brier differences and CLV settle much faster than ROI — which
  is why they lead.
- **Behavior**: mean stake as % of balance after losing days vs winning days, tier mix vs drawdown
  depth, share of days with zero bets, rejection rate by reason, and whether any strategy drift is
  detectable across the season (KellyBench's static-vs-adaptive split was the single largest
  performance difference it found).

## Data model (SQLite)

- `games(id, sport_key, home, away, commence_utc, status, home_score, away_score, completed_utc)`
- `odds_snapshots(id, game_id, captured_utc, bookmaker, market, outcome, price_decimal, is_closing)`
- `contexts(id, game_id, built_utc, payload_json, payload_hash, sources)`
- `llm_calls(id, stage, seed_idx, model_id, prompt_version, params_json, input_hash, input_json, raw_response, stop_reason, created_utc, ok, error)`
- `forecasts(id, game_id, llm_call_id, seed_idx, p_home, p_away, p_draw)`
- `bets(id, arm, game_id, forecast_id, snapshot_id, placed_utc, selection, price_decimal, tier, stake_minor, p_blind, p_revised, reason, status, payout_minor, settled_utc)`
- `rejections(id, arm, llm_call_id, payload_json, reason, created_utc)`
- `ledger(id, arm, ts_utc, delta_minor, balance_minor, reason, bet_id)`
- `calibrations(id, fitted_utc, method, params_json, n_games, valid_from_utc)`
- `runs(id, kind, started_utc, finished_utc, credits_remaining, notes)`

`params_json` records `model_id`, `effort`, the `thinking` config and `max_tokens`. It must not record
a `temperature` — no such parameter exists on the models under test.

## API budget (The Odds API)

Free tier is 500 credits/month. `/odds` costs markets × regions, so v1 (h2h, one region) is 1 credit.
`/scores` with `daysFrom` costs 2. **`/events` and `/sports` are free** — use `/events` for scheduling
so only odds calls consume quota. A day on one sport is 1 slate + ~3 closing windows + 2 settlement
= **6 credits**, about 180/month. NHL alone fits the free tier comfortably; NHL + NBA is ~360/month
and fits with no margin for error, so move to the $30/month 20K tier when NBA comes on.

## Sport choice and settlement

Volume drives statistical power. NHL is v1: the 2026-27 season opens **2026-09-29** with 1,312
regular-season games (~7 a night through April 2027), the market is 2-way, and
`api-web.nhle.com/v1/` serves standings, schedule, results and rosters free with no key. The clean
season boundary also makes pre-registration meaningful. NBA follows on **2026-10-20** (1,230 games).

Settlement conventions to encode in tests: NHL and NBA moneylines include overtime — and the NHL's
the shootout — so they cannot push. NFL moneylines can tie, which pushes and refunds the stake.
Soccer h2h is 3-way and settles on regulation time; it stays in v2, because the draw is exactly where
the published work found the worst miscalibration.

## Scheduling

A small always-on machine with cron is simplest because the SQLite file stays put. GitHub Actions
schedules also work, but the runner's disk is ephemeral (the database has to live elsewhere) and
scheduled runs can start late, which matters for closing snapshots and therefore for the primary
metric.

## Milestones

- M0: skeleton (uv, src layout, Python 3.13+), config, schema, and pure logic with tests — odds
  conversion, three de-vig methods, tiers, Kelly, Elo, validator, settlement, metrics.
- M1: Odds API client with fixture recording, ingestion, credit tracking, free `/events` scheduling.
- M2: NHL context builder, Elo seeded from prior season.
- M3: Stage 1 with structured outputs, k draws, refusal handling, storage.
- M4: Stage 2, validator, ledger, and the `kelly_*`, `elo`, `fav`, `random` arms.
- M5: closing snapshots and settlement.
- M6: report (tables, calibration, balance curves), `kelly_cal` fitting, `PREREGISTRATION.md`
  committed and tagged before the first live bet, scheduling.
- Later: **web-search Stage 1 as its own arm** (the configuration that reached market parity), spreads
  and totals via expected margin, parlays, multi-model tournament.

## v2 markets

- **Spreads and totals**: Stage 1 forecasts expected margin and total (ideally a distribution); code
  converts those into cover probabilities for whatever line is posted.
- **Parlays (risky tier)**: legs from different games only; price = product of leg decimal prices; a
  void or pushed leg drops out and the parlay reprices on the remaining legs; any losing leg loses the
  parlay. The Kelly shadow needs the joint probability (product of leg probabilities, assuming
  independence).

## Remaining open decisions

- `k` = 5 or 3, depending on the LLM budget actually approved.
- Whether to add `claude-sonnet-5` as a second model under test, or spend the same budget on a larger
  `k` for one model. (Larger `k` is the better science; a second model is the better headline.)
- Whether to run an `effort: "low"` vs `"high"` contrast as a second factor.
- Elo K-factor and home-ice constant — tune on prior seasons only, freeze before opening night.
