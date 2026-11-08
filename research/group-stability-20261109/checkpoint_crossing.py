"""Apply the recorded checkpoint policy to the standalone P101 proof."""

import sys

import crossing_regimes

sys.modules["regime_recovery"] = crossing_regimes

import checkpoint  # noqa: E402

checkpoint.main()
