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
# Pinnacle runs 2-3% margins against a typical 5%+, and its closing line is the
# industry-standard CLV benchmark. Since CLV is the primary metric, the
# benchmark must be the sharpest line available.
DESIGNATED_BOOKMAKER = "pinnacle"
REGION = "eu"
MARKET = "h2h"
PRIMARY_DEVIG_METHOD = "shin"

# --- models under test ------------------------------------------------------
MODEL_ID = "claude-opus-5"
EFFORT = "high"
# k independent draws per slate. A single live forward path is one seed, and
# KellyBench's per-model seed spread was +34.1% to -32.9% on identical data.
K_SEEDS = 5

# --- calibration ------------------------------------------------------------
CALIBRATION_BURN_IN_GAMES = 200
CALIBRATION_REFIT_DAYS = 7

# --- reporting --------------------------------------------------------------
BOOTSTRAP_RESAMPLES = 10_000
CALIBRATION_BINS = 10
