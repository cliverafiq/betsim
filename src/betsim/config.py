"""Experiment constants.

Anything here that changes mid-experiment invalidates comparability, so these are
frozen before opening night and recorded in ``PREREGISTRATION.md``.
"""

from __future__ import annotations

from datetime import timedelta

from betsim.money import MINOR_PER_UNIT

# --- bankroll ---------------------------------------------------------------
EXPOSURE_CAP_PCT = 0.20
KELLY_FRACTION = 0.25
FLAT_STAKE_MINOR = 10 * MINOR_PER_UNIT  # `fav` and `random` arms

# --- timing -----------------------------------------------------------------
# Stage 2 must decide within 60 minutes of the snapshot it was shown. This is
# about price staleness and CLV integrity, not forecast quality -- forecast
# horizon has been shown to matter very little to accuracy.
MAX_SNAPSHOT_AGE = timedelta(minutes=60)
CLOSING_WINDOW = timedelta(minutes=15)

# --- market -----------------------------------------------------------------
# Chosen from what the account can actually see, measured on a live 33-game
# slate rather than assumed. Pinnacle was the plan's designated book -- 2-3%
# margins, the industry-standard CLV benchmark -- but it is not served on this
# account: it returns games with no bookmaker entry at all.
#
# The EU region is also the wrong place to look. Twelve of its seventeen NHL
# books quote **3-way** h2h on regulation time, with a Draw, which is a
# different market from the 2-way moneyline this experiment is built on.
#
# Every US-region book is 2-way, and DraftKings is the only one covering all 33
# games (median margin 4.28%). Sharper books exist there -- betonlineag and
# lowvig at 3.17% -- but each covers only about a fifth of the slate, which is
# useless for a designated book. The cost of this substitution is a softer
# closing line than Pinnacle's, so CLV is a slightly weaker benchmark than
# planned; it is still the best available and remains the primary metric.
DESIGNATED_BOOKMAKER = "draftkings"
REGION = "us"
MARKET = "h2h"
PRIMARY_DEVIG_METHOD = "shin"

# --- models under test ------------------------------------------------------
MODEL_ID = "claude-opus-5"
EFFORT = "high"
# k independent draws per slate. A single live forward path is one seed, and
# KellyBench's per-model seed spread was +34.1% to -32.9% on identical data.
K_SEEDS = 5
MAX_TOKENS = 4096

# USD per million tokens (input, output), for budgeting only -- the authoritative
# spend is whatever the API reports back in usage.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost_usd(input_tokens: int, output_tokens: int, model: str = MODEL_ID) -> float:
    """Rough spend for one call. Thinking tokens are billed as output."""
    rate_in, rate_out = MODEL_PRICING.get(model, MODEL_PRICING[MODEL_ID])
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


# --- calibration ------------------------------------------------------------
CALIBRATION_BURN_IN_GAMES = 200
CALIBRATION_REFIT_DAYS = 7

# --- reporting --------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 10_000
CALIBRATION_BINS = 10
