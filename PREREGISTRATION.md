# Pre-registration

**Committed before the first live bet. Not edited afterwards.**

Registered: 2026-09-21. First live slate: NHL opening night, 2026-09-29.
Code state: see the git tag `prereg-v1`.

This document exists because the alternative is choosing an analysis after
seeing the data. Everything below — the arms, the metrics, the thresholds, and
the predictions — is fixed now. Deviations get recorded in a separate
`DEVIATIONS.md`, with reasons and dates, rather than silently absorbed.

---

## 1. What is being tested

Whether a large language model can manage a hypothetical sports-betting bankroll
across a full NHL regular season, and — the part no prior work could answer —
whether any failure lies in the **forecasts** or in the **bet sizing**.

The design separates the two. Stage 1 forecasts each game having never seen a
price. Stage 2 sees the price and decides what to stake. A shadow arm then
stakes Stage 1's *identical* probabilities under a fixed quarter-Kelly rule, so
`llm_si` versus `kelly_si` isolates sizing with the forecast held constant.

Paper trading only. No real money at any point.

## 2. Predictions, stated in advance

These are the priors. Recording them now is what makes the result informative
either way.

1. **The model loses money.** All five frontier models in KellyBench lost over a
   season; flat-staking the favourite out-earned all four agents in
   WC2026-Agents. Expected `llm` ROI is negative, in the region of the
   bookmaker's margin (~4%) or worse.
2. **Blind forecasts are worse than the market.** Closed-book LLM forecasts have
   consistently trailed de-vigged consensus; only open-book forecasting has
   reached parity. Expected Brier gap: positive (market better), plausibly
   0.005–0.030 on the sum-over-outcomes scale.
3. **The model is overconfident at the top of the range.** High stated
   probabilities will verify at a lower rate than stated.
4. **`fav` beats `llm` on ROI.** It did in WC2026-Agents.
5. **`elo` is competitive with or better than `llm`.** A plain Dixon–Coles model
   beat 3 of 5 frontier models in KellyBench.

A result contradicting any of these is the interesting outcome.

## 3. Arms

Frozen. All are built from the same price snapshot; seed *i*'s Stage 2 sees seed
*i*'s blind forecast and nothing else.

| Arm | Probabilities | Sizing |
|---|---|---|
| `llm_s1..s5` | — (Stage 2 decides) | Stage 2, post-validation |
| `kelly_s1..s5` | `p_blind` draw *i* | quarter Kelly, tier + exposure caps |
| `kelly_revised_s1..s5` | `p_revised` | quarter Kelly, same caps |
| `kelly_cal` | Platt-scaled `p_blind` | quarter Kelly, same caps |
| `elo` | Elo + home ice | quarter Kelly, same caps |
| `fav` | — | flat 10 units on the favourite |
| `random` | — | flat 10 units, seeded |
| no-bet | — | constant 1,000 units |

`k = 5` independent draws. There is no seed parameter and no sampling
parameter on the models under test, so variance comes from repeated calls.

## 4. Frozen parameters

| | |
|---|---|
| Sport | NHL (`icehockey_nhl`), 2026-27 regular season |
| Model under test | `claude-opus-5`, `effort: high`, adaptive thinking |
| Prompts | `stage1_v1`, `stage2_v1` (hashed; a used prompt is never edited) |
| Designated bookmaker | DraftKings, region `us`, h2h 2-way |
| Starting bankroll | 1,000 units per arm, integer minor units |
| Tier caps | safe ≥60% → 5%; medium 40–60% → 3%; risky <40% → 1% |
| Exposure cap | 20% of balance; `fav`/`random` exempt |
| Kelly fraction | 0.25 |
| Primary de-vig | Shin (multiplicative and power reported as sensitivity) |
| Elo | K=20, home advantage 50, carryover 0.70, seeded on 2025-26 |
| Calibration burn-in | 200 settled games, refit weekly, strictly out-of-sample |

## 5. Metrics, in priority order

1. **CLV** (`d_taken / d_close − 1`) at the designated bookmaker. Primary.
   The null is the **`random` arm's distribution**, not zero — a bet struck at
   the slate snapshot earns CLV from timing alone.
2. **Brier and log loss** of Stage 1 against the Shin-de-vigged consensus, on
   **all** settled games rather than only those bet on. Sum-over-outcomes
   convention; 2-way and 3-way markets are never mixed in one table.
3. **Calibration**: 10-bin reliability curve.
4. **Sizing decomposition**: `llm_si` vs `kelly_si`.
5. **Behaviour**: stake % after losing vs winning days, tier mix vs drawdown,
   zero-bet rate, rejection counts by reason.
6. **ROI, max drawdown, time-to-ruin.** Reported last and labelled underpowered
   until the bet count justifies otherwise.

All intervals are bootstrap 95% CIs resampling **by day**, since same-day bets
share a slate and a bankroll state.

## 6. What would count as a positive result

Set now, so it cannot be moved later.

- **Forecasting**: Stage 1's Brier beats the de-vigged consensus with a 95% CI
  excluding zero, over ≥ 400 settled games.
- **Skill**: mean CLV of any `llm_si` exceeds the `random` arm's mean CLV with a
  95% CI on the difference excluding zero.
- **Sizing**: `llm_si` beats `kelly_si` on terminal bankroll in ≥ 4 of 5 seeds.
- **Recoverability**: `kelly_cal` beats `kelly` on ROI with a CI excluding zero
  — which would mean the forecasts carry signal the model's confidence destroys.

Anything short of these is a negative result and will be reported as one.

## 7. Analysis plan

- Run to the end of the NHL regular season (2027-04-11) or until every `llm`
  seed is bust, whichever comes first. Bust is absorbing; the arm stops and
  time-to-ruin is recorded.
- No arm is added, removed or re-tuned mid-season. Elo constants and the Kelly
  fraction are frozen above.
- The calibration map is fitted only on games before its `valid_from_utc`.
- Zero-bet days are a valid outcome and are counted, not excluded.
- Rejected bets are recorded with a reason and analysed; they are not discarded.

## 8. Known limitations, acknowledged now

- **One live path per seed.** k=5 gives a variance estimate but the season
  itself is a single realisation; a shared schedule correlates the seeds.
- **Closed-book Stage 1 is the weakest configuration** for forecast quality.
  That is deliberate — it is the only way to keep the price out — but it means
  prediction 2 is close to a foregone conclusion.
- **DraftKings is a softer closing line than Pinnacle**, which is unavailable on
  this account, so CLV is a weaker benchmark than originally planned.
- **ROI will very likely be underpowered.** At 1,000 flat bets at -110 the 95%
  interval is about ±6 points; the season will not deliver that many bets from a
  selective model.
- **The market may be slower early in the season**, when public information is
  thinnest. Any edge concentrated in October should be treated with suspicion.
