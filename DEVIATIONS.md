# Deviations from the pre-registration

`PREREGISTRATION.md` is fixed and is not edited. Anything that departs from it
is recorded here, with the date and the reason, so the record stays honest.

---

## 2026-09-21 — two exploratory arms added (before the first live bet)

**What changed.** Two arms were added alongside the pre-registered set. No
pre-registered arm was changed, removed or re-tuned, and no threshold moved.

| Arm | What it tests |
|---|---|
| `llm_anchored_s1..k` | Stage 2 shown the de-vigged consensus and required to name what the price is missing before staking. Does *thinking like a bettor* beat *forecasting like an analyst*? |
| `best_price` | Flat stake on the same selection as `fav`, taken at the best price across all bookmakers rather than the designated one. Isolates line shopping. |

**Why.** The rehearsal showed the blind arm computing the no-vig market itself
from the prices it was shown, then treating its residual disagreement with that
market as an edge — and staking the underdog at 3.25. That is the mechanism by
which a model with less information than the market loses money: absent
knowledge of the confirmed goaltender or late scratches, its estimate regresses
toward even, and betting that regression against a sharp line means backing
longshots, which the favourite-longshot bias already overprices.

`llm_anchored` tests whether instructing the model to treat the price as its
prior changes that. On the single rehearsal slate it placed **zero** bets where
the blind arm placed two. Whether that restraint is correct is unknown until
games settle, and one slate is not evidence.

`best_price` costs nothing to run and needs the forecaster to be good at
nothing. Measured over 66 outcomes across 8 books on a live slate, taking the
best price gained **+1.17%** on average against the designated book — real, but
against a 4.28% median margin it closes about a quarter of the gap.

**Status.** Exploratory. Neither arm is part of the pre-registered hypotheses
and neither will be reported as a confirmatory result. The predictions and
success thresholds in `PREREGISTRATION.md` are unchanged.

**Timing.** Added before opening night on 2026-09-29 and before any live bet, so
no pre-registered arm has yet placed a wager that these could have influenced.
