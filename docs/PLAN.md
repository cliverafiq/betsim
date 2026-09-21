# betsim — research-backed build plan

Written 2026-09-21 after reviewing the published work on LLM sports forecasting and betting,
the current Odds API terms, the current Claude API surface, and the 2026-27 season calendar.

Supersedes the open decisions and several rules in the draft `CLAUDE.md` / `design.md`.
The draft's *arithmetic* is correct in every particular I checked (break-even at -110,
coin-flipper edge, parlay edge, tier boundaries at ±150, the ±6-point ROI CI at 1,000 bets).
The problems are structural, and they are fixable before the first bet is placed.

---

## 0. Headline

Four changes decide whether this experiment produces a publishable result or an anecdote:

1. **Run k independent seeds per slate.** A single live forward path is one seed. KellyBench's
   per-model seed spread was +34.1% to −32.9% — the same model, same season, same data. One path
   cannot distinguish skill from luck, and the newest paper in this area lists "collect repeated
   independent forecasts, not single runs" as its second recommendation for future work.
2. **Add a non-LLM statistical baseline arm (Elo).** In KellyBench a plain Dixon-Coles model beat
   3 of 5 frontier models. Without a statistical arm you cannot tell whether the LLM is bad or
   the task is hard. Also the seventh explicit recommendation of the AI World Cup 2026 authors.
3. **Make CLV the primary metric, not ROI.** The draft's own math shows ROI needs ~1,000 bets for
   a ±6-point CI. At NHL volume with a selective model, that is most of a season for one usable
   number. CLV is the best-evidenced skill metric in the literature and settles far faster.
4. **Drop "fixed sampling params" — they no longer exist.** `temperature`, `top_p`, `top_k` and
   `budget_tokens` all return HTTP 400 on Claude Opus 5, Sonnet 5, Opus 4.8/4.7 and Fable 5/5.1.
   The controllable knob is `output_config.effort`. There is no seed parameter, so reproducibility
   by seed is unavailable and k independent draws are the only route to a variance estimate.

Everything else in the draft survives review, and two of its choices are better than the
published work: allowing zero bets, and separating a blind forecast from a priced decision.

---

## 1. Evidence base

Every work below was checked against its primary source.

| Work | Setup | Result that matters here |
|---|---|---|
| **KellyBench** (arXiv 2604.27865, Apr 2026) | 5 frontier models × 5 seeds, simulated 2023-24 EPL (~380 matches), £100k bankroll, agent writes its own code, **minimum one bet per matchday**, 5.3% vig | Every model lost money on average. ROI −7.9% (GPT-5.4) to −89.6% (Kimi K2.5); ruin in 6/25 seeds. A human quant made **+5.1% on 39 bets**. Dixon-Coles baseline (−15.4%) beat 3/5 models. Adaptive seeds −11.1% vs static seeds −70.0%. Draw/longshot miscalibration in **22/25** seeds; no principled staking at execution in 9/25. |
| **WC2026-Agents** (arXiv 2607.17765, Jul 2026) | 4 models, all 104 World Cup matches, flat $100 virtual bets, search-act-reflect | None beat the market's Brier (0.469). ROI −18% to +10%. **Flat-staking the market favorite out-earned all four.** Market-citation rate ranged 12%–100%. |
| **LLM-SoccerArena** (arXiv 2607.24573) | 7 models × open/closed book × prompt style × horizon; 8,736 forecasts | Brier 0.506–0.546, no significant difference between models. **Open-book improved Brier by 0.0228**; best open-book run (0.497) essentially tied the de-vigged market consensus (0.498). Forecast horizon barely mattered (0.0021 from T-24h to T-2h). |
| **AI World Cup 2026** (arXiv 2608.03416, Aug 2026) | 10 configurations, single pre-tournament forecast | Self-reported confidence did not track accuracy (r = −0.060). Authors give 8 recommendations for future work; #1 pre-register, **#2 repeated independent forecasts**, #3 proper scoring rules, **#7 strong non-LLM baselines including market comparison**. |
| **ForecastBench / AIA Forecaster** | General (non-sports) forecasting | LLMs are **overconfident on high-probability events**. Post-hoc calibration (Platt scaling / extremization) moved a system to statistical parity with the superforecaster median. |
| **meo0138/llm_sports_betting_benchmark** | €1,000 bankrolls, European football, leaderboard | Models see odds directly — no blind stage; unrestricted staking; no Kelly shadow. |
| **michaeltimbs / Betswaps** | $1,000 + top-ups, MLB/boxing/AFL/NRL | No calibration, no CLV, no market anchoring measure. Bankroll top-ups destroy ruin analysis. |

**Expected result for betsim, stated up front:** the model loses slowly to the margin, its blind
Brier is worse than the de-vigged consensus, and flat-favorite betting beats it on ROI. That is
the prior. The experiment is worth running for *why* and *by how much*, not for the sign.

---

## 2. What the evidence says about the draft

### 2.1 Survives, and is better than the published work

- **Two-stage blind→priced split.** No prior work does this. KellyBench and WC2026 both showed the
  model the price throughout, so when they lost they could not say whether the forecast or the
  sizing was at fault. This is betsim's real contribution — lead with it.
- **Kelly shadow on the *same* probabilities.** The counterfactual "what if this exact forecast had
  been sized by rule" is what makes the decomposition work. Keep the identical-caps rule.
- **Zero bets always valid.** KellyBench *required* one bet per matchday, which manufactures losses.
  The human quant's +5.1% came from 39 bets across a season — selectivity is where the skill was.
  Make "share of slates with zero bets" a headline behavioral outcome, not a footnote.
- **Forward-only live betting.** Removes contamination by construction. KellyBench had to block
  network access to simulate it.
- **Tiers assigned by code, validator rejects rather than clips, append-only ledger.** All sound.

### 2.2 Must change

**(a) One seed is not enough.** The single largest hole. Fix: k independent Stage 1 draws and k
independent Stage 2 decisions per slate, each Stage 2 with its own bankroll (`llm_s1`…`llm_sk`).
Costs k× LLM tokens and **zero extra odds credits** — the same snapshot feeds all of them.
Recommend k=5; k=3 if budget is tight. This also yields a free forecast-dispersion measure
(disagreement among draws) which is itself a usable uncertainty signal.

**(b) No non-LLM statistical baseline.** `fav` and `random` are floor baselines; `kelly` is a sizing
counterfactual on the LLM's own numbers. None of them answers "could a simple model have done
better?" Add an **Elo arm** (home-ice advantage, updated after each game, seeded from the prior
season) sized by the same quarter-Kelly rule and the same caps.

**(c) RQ1 is already answered in the literature.** "Do blind forecasts carry information beyond the
market?" — closed-book LLM forecasts are clearly worse than the market; only *open-book* forecasting
reached parity (0.497 vs 0.498). Since Stage 1 is deliberately closed-book, the answer is
near-certainly no. Reframe to: **how large is the gap, with a CI; is it concentrated in a tier or
on underdogs; and does post-hoc recalibration close it?**

**(d) De-vig method is the benchmark, and the draft picked the weakest one.** Multiplicative de-vig
distributes the margin proportionally and does not correct favorite-longshot bias — which is
precisely the axis the tier system is built on. The de-vigged consensus *is* the yardstick for
the headline Brier comparison, so a biased yardstick biases the headline. Implement multiplicative,
power and **Shin**; report Shin as primary with the other two as sensitivity.

**(e) "Fixed sampling params" is unimplementable.** On `claude-opus-5`, `claude-sonnet-5`,
`claude-opus-4-8`/`4-7` and `claude-fable-5-1`, `temperature`/`top_p`/`top_k` are **removed and
return 400**, and `budget_tokens` likewise. Replace with: log `model_id`, `output_config.effort`,
the `thinking` config and `max_tokens`. Use **structured outputs** (`output_config.format` with
`messages.parse()`) to guarantee schema-valid JSON instead of the draft's parse-retry-log loop.
Handle `stop_reason: "refusal"` as a distinct logged outcome — gambling-adjacent prompts make this
non-hypothetical, and a refusal must not crash a nightly run.

**(f) CLV's null is not zero.** Bets placed at the slate snapshot are taken hours before close, and
early prices are systematically softer — so some positive CLV accrues from *timing*, not skill.
The `random` arm bets at the same snapshots with no skill, so **its CLV distribution is the correct
null**. Compare the LLM arm's CLV to the random arm's, not to zero. This falls out of the existing
design for free and should be stated explicitly.

**(g) Brier scale depends on market width.** Under the sum-over-outcomes convention the draft
specifies, a uniform 2-way forecast scores 0.500 and a uniform 3-way scores 0.667. NHL numbers are
therefore **not comparable** to the 3-way soccer literature (market Brier 0.469, LLM 0.497–0.546).
State the convention in the report and never mix widths in one table.

---

## 3. Closed decisions

The draft's four open decisions, settled with evidence.

### Sport for v1: **NHL** (`icehockey_nhl`), adding **NBA** (`basketball_nba`) later

| | NHL | NBA | NFL | Soccer |
|---|---|---|---|---|
| 2026-27 season opens | **Sept 29, 2026** (8 days out) | Oct 20, 2026 | underway | underway |
| Regular-season games | 1,312 | 1,230 | 272 | varies |
| Market width | 2-way | 2-way | 2-way + tie push | **3-way (draw)** |
| Free official data API | yes, no key | yes | — | paid/limited |

NHL wins on four counts: it opens in 8 days, giving a **clean season boundary to pre-register
against**; ~7 games a night sustains volume; the 2-way market avoids the draw, which is where
22 of 25 KellyBench seeds miscalibrated; and `api-web.nhle.com` serves standings, schedule,
results and rosters free with no key. Add NBA on Oct 20 once the pipeline has run a week.
Soccer stays in v2 — 3-way markets, multi-league fixture assembly, and the draw confound all at once.

### Designated bookmaker: ~~Pinnacle, region `eu`~~ -> **DraftKings, region `us`**

> **Correction, 2026-09-21 (verified against the live API).** The designated
> bookmaker is **DraftKings, region `us`**, not Pinnacle/`eu`. Two things only a
> live call revealed:
>
> 1. **Pinnacle is not served on this account.** The request succeeds and returns
>    every game with no bookmaker entry at all, so the choice was unavailable
>    rather than merely worse.
> 2. **The EU region is the wrong market.** Twelve of its seventeen NHL books
>    quote **3-way h2h on regulation time**, with a Draw -- a different market
>    from the 2-way moneyline this experiment is built on. A 3-way bet on a team
>    *loses* when that team wins in overtime, so mixing the two would have
>    corrupted the de-vigged consensus, tier assignment and settlement at once.
>    `parse_odds` now filters by market width and the CLI reports what it drops.
>
> Every US-region book is 2-way. DraftKings is the only one covering all 33 games
> of a live slate, at 4.28% median margin. Sharper books exist there --
> betonlineag and lowvig at 3.17% -- but each covers about a fifth of the slate,
> which is useless for a designated book. The cost is a softer closing line than
> Pinnacle's, so CLV is a slightly weaker benchmark than planned; it remains the
> best available and stays the primary metric.


Pinnacle runs 2–3% margins against a typical 5%+, welcomes sharp money, and its closing line is the
industry-standard CLV benchmark. Since CLV is now the primary metric, the benchmark must be the
sharpest available line — a soft book's closing number measures much less. Cost is unchanged
(h2h × 1 region = 1 credit). Keep storing every bookmaker returned for the consensus.

### Model under test: **Claude Opus 5** (`claude-opus-5`), `effort: "high"`

1M context, $5/$25 per MTok. Use adaptive thinking (`thinking: {type: "adaptive"}`) — on Opus 5 this
is the default and `budget_tokens` is rejected. If budget allows a second model, add
`claude-sonnet-5` ($2/$10) as a cheaper comparison rather than an older Opus. An `effort: "low"`
vs `"high"` contrast is a cheap and genuinely interesting second factor: does more thinking improve
*calibration*, or only confidence?

### Context sources: NHL official API, free

`api-web.nhle.com/v1/` — `standings/now`, `schedule/`, `score/`, `club-stats/`, rosters and scratches.
No key, no quota. Assemble verbatim into `contexts.payload_json` and hash it. Nothing derived from odds.

---

## 4. Revised experiment design

### Arms

| Arm | Probabilities | Sizing | Purpose |
|---|---|---|---|
| `llm_s1`…`llm_sk` | — (Stage 2 decides) | Stage 2, post-validation | The subject. k independent seeds. |
| `kelly_s1`…`kelly_sk` | `p_blind` (seed i) | quarter-Kelly, same caps | Isolates **sizing** skill, per seed. |
| `kelly_revised` | `p_revised` | quarter-Kelly, same caps | Does seeing the price *improve* the forecast, or just anchor it? Free — `p_revised` is already collected. |
| `kelly_cal` | Platt-scaled `p_blind` | quarter-Kelly, same caps | Is the forecast salvageable by recalibration? Fitted after a 200-game burn-in, refit weekly, **strictly out-of-sample**. |
| `elo` | Elo + home ice | quarter-Kelly, same caps | Non-LLM statistical baseline. |
| `fav` | — | flat 10 units on the favorite | The baseline that beat all four WC2026 agents. |
| `random` | — | flat 10 units, seeded uniform | Floor, **and the null distribution for CLV**. |
| *(no-bet)* | — | — | Constant 1,000 units. |

`fav` and `random` are exempt from **both** the exposure cap and the tier caps — they are flat-stake
references, not managed bankrolls. State this explicitly; the draft only exempted the exposure cap.

### Metrics, in priority order

1. **CLV** — `d_taken / d_close − 1` at Pinnacle, per arm and tier, compared against the `random`
   arm's CLV distribution. Settles fastest; best-evidenced skill signal.
2. **Calibration of Stage 1** — reliability curve in 10 bins, Brier and log loss vs the Shin-de-vigged
   consensus, on **all** games rather than only those bet. Expect overconfidence concentrated on
   high-probability outcomes; that is the documented failure mode and the thing `kelly_cal` tests.
3. **Sizing decomposition** — `llm_si` vs `kelly_si` on identical probabilities. If both lose, the
   forecast is the problem; if `kelly_si` wins, the sizing is.
4. **Behavior under drawdown** — stake % of balance after losing vs winning days, tier mix vs
   drawdown depth, zero-bet rate, rejection rate by reason. KellyBench's most damaging findings were
   behavioral (static strategies, ignored self-critique, phantom "season complete" declarations),
   so instrument for them deliberately.
5. **ROI, max drawdown, time-to-ruin** — reported with bootstrap 95% CIs resampled by day, and
   reported *as underpowered* until the bet count justifies otherwise.

### Research questions, restated

1. How large is the gap between blind LLM forecasts and the de-vigged market, with a CI — and is it
   concentrated by tier or by favorite/underdog?
2. Given its own forecasts, does the model size better or worse than quarter-Kelly?
3. Does seeing the price improve the forecast (`kelly_revised` vs `kelly`) or merely anchor it
   (`p_revised` distance to market)?
4. Does post-hoc recalibration rescue the forecast (`kelly_cal` vs `kelly`)?
5. How does the model behave in drawdowns, and does it adapt at all across a season?
6. Does any arm beat `fav` and `elo`?

---

## 5. Spec defects to fix before M0 ships

| # | Defect | Fix |
|---|---|---|
| 1 | Tier bounds not half-open — 60% and 40% each match two tiers | `safe: p ≥ 0.60`; `medium: 0.40 ≤ p < 0.60`; `risky: p < 0.40` |
| 2 | `stake_units` is a float; no rounding rule to minor units | Round **down** to whole minor units, as Kelly already does |
| 3 | `fav`/`random` exempt from exposure cap; tier cap unstated | Exempt from both |
| 4 | **Exposure-cap tie-break order unspecified for the LLM arm** — results become order-dependent | Process bets in the order the model returned them; reject those that breach. This also measures whether the model self-manages exposure. |
| 5 | **Bust semantics unspecified** | Bust is absorbing: the arm stops betting, the ledger freezes, the run continues, and time-to-ruin is reported |
| 6 | Push/void on h2h unspecified per sport | NHL and NBA cannot tie (OT/shootout) → no push. NFL can tie → push, stake refunded. Encode in settlement tests. |
| 7 | Scores calls missing from the credit budget | `/scores` with `daysFrom` costs **2 credits**; `/events` and `/sports` are **free** |
| 8 | "Fixed sampling params" — parameters no longer exist | Log `effort`, `thinking`, `max_tokens`, `model_id`; k draws instead of seeds |
| 9 | No `stop_reason: "refusal"` handling | Log as a distinct outcome; consider server-side fallbacks; never crash the run |
| 10 | `p_revised` collected but unused | Now drives `kelly_revised` |
| 11 | Brier convention not stated | Sum-over-outcomes; never mix 2-way and 3-way in one table |

---

## 6. Build plan

Milestones sized so that **M0–M5 land before NHL opening night on Sept 29**.

**M0 — skeleton and pure logic (days 1–2).**
`uv init --package` (src layout, Python 3.13+), SQLite schema, and the pure functions with unit
tests: American↔decimal conversion, **three de-vig methods** (multiplicative, power, Shin), tier
assignment on half-open intervals, quarter-Kelly, the validator, settlement (including NHL OT/SO and
the NFL tie-push case), and the metrics. All deterministic, all tested, no network.

**M1 — Odds API client (day 3).**
httpx client, fixture recording into `tests/fixtures/`, credit accounting from `x-requests-remaining`
recorded per run, hard stop before the quota reaches zero. Use the **free** `/events` endpoint for
scheduling so only the odds calls cost.

**M2 — context builder and Elo (day 4).**
NHL official API ingestion; context assembled and stored verbatim with a hash. Elo with home-ice
advantage, seeded from 2025-26 results, updated after each game. Elo is a pure function — test it.

**M3 — Stage 1 (day 5).**
`claude-opus-5`, structured outputs via `output_config.format` + `messages.parse()` against the
Pydantic model, k independent draws per game, every call logged with input, hash, params and raw
response. Refusal handling.

**M4 — Stage 2, validator, arms (days 6–7).**
k independent Stage 2 decisions with separate bankrolls; validator; ledger; and the `kelly_*`,
`elo`, `fav`, `random` arms computed from the same snapshot.

**M5 — closing snapshots and settlement (day 7).**
Closing odds within 15 minutes of puck drop; `/scores` with `daysFrom` for settlement; void handling.

**M6 — report, calibration, pre-registration (day 8, and ongoing).**
Metrics tables with bootstrap CIs, reliability curves, balance paths. `kelly_cal` fitting switches on
after the 200-game burn-in. **Write `PREREGISTRATION.md`, commit and tag it before the first live
bet** — the AI World Cup authors' first recommendation, and it costs an hour.

**Sept 29:** go live on NHL opening night.
**Oct 20:** add NBA if credit headroom allows.
**Later:** web-search Stage 1 as its own arm (logging every query — this is the configuration that
reached market parity in the literature, so it is the most promising extension), spreads and totals
via expected margin, parlays, multi-model tournament.

---

## 7. Budget

**Odds API.** Free tier is 500 credits/month. Per sport per day: 1 slate + ~3 closing windows +
2 settlement = **6 credits**, about **180/month**. NHL alone fits comfortably. NHL + NBA ≈ 360/month
fits but leaves no room for error — move to the $30/month 20K tier when NBA comes on.

**LLM.** At k=5 on `claude-opus-5` ($5/$25 per MTok), roughly 7 games/night:
Stage 1 is 35 calls/night, Stage 2 is 5 — on the order of **$3–4 per night**, or **$500–700 for a
full NHL season**. k=3 brings that to roughly $320. `claude-sonnet-5` would be about 40% of the cost.
Prompt caching helps Stage 2 (large stable prefix of rules and caps) more than Stage 1. The Batch API
halves cost but its latency is not guaranteed inside the 60-minute snapshot window — keep Stage 2
synchronous; Stage 1 could batch if the slate runs early enough.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| **Underpowered result.** The most likely failure: a season ends with wide CIs and no conclusion. | CLV and Brier as primary (they settle fastest); k seeds for a variance estimate; volume sport; pre-register what counts as an answer. |
| **Model refuses gambling-adjacent prompts** mid-season, silently zeroing a night. | Handle `stop_reason: "refusal"`; log it as data; alert on consecutive refusals. |
| **Quota exhaustion** kills closing snapshots, destroying the primary metric. | Hard stop with headroom; `/events` is free; alert at 20% remaining. |
| **Credit/season mismatch** — starting mid-season loses the clean boundary. | Sept 29 start is 8 days out; this is the reason for the tight M0–M5 schedule. |
| **Silent pipeline drift** (stale context, wrong bookmaker, clock skew). | Snapshot age enforced in the validator; automated pipeline checks (AI World Cup recommendation #8); UTC everywhere. |
| **Over-interpreting one path.** | k seeds, bootstrap CIs by day, and an explicit statement of the prior: expect to lose to the margin. |

---

## 9. Sources

- KellyBench — https://arxiv.org/abs/2604.27865 · https://www.gr.inc/releases/introducing-kellybench
- WC2026-Agents — https://arxiv.org/abs/2607.17765 · https://github.com/graphuofm/FIFA2026LLM
- LLM-SoccerArena — https://arxiv.org/html/2607.24573
- AI World Cup 2026 — https://arxiv.org/html/2608.03416
- ForecastBench — https://faculty.wharton.upenn.edu/wp-content/uploads/2026/02/ForecastBench_A_Dynamic_.pdf
- AIA Forecaster — https://arxiv.org/pdf/2511.07678
- The Odds API docs — https://the-odds-api.com/liveapi/guides/v4/
- NHL API reference — https://github.com/Zmalski/NHL-API-Reference
- 2026-27 NHL schedule — https://www.nhl.com/news/nhl-releases-2026-27-regular-season-schedule
- 2026-27 NBA schedule — https://www.nba.com (opening night Oct 20, 2026)
- De-vig method comparison — https://betherosports.com/blog/devigging-methods-explained
- CLV as skill metric — https://tradematesports.medium.com/closing-line-the-most-important-metric-in-sports-trading-58e56cdb4458
