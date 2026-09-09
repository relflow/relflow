"""Request five order statistics from one randomly ordered numeric bag.

This is the preferred user-facing schema: only raw values and a visible rank
request are supplied.  The same trained model must recover endpoints and
interior ranks, remain invariant to complete-item permutation, collapse when
the request is hidden, and lose skill when rank labels are cycled.

Example
-------
The rank selects an observed sorted position even though the items arrive in
random order:

```yaml
input:
  items: [{value: 4}, {value: 1}, {value: 9}, {value: 3}, {value: 7},
          {value: 2}, {value: 8}, {value: 6}, {value: 5}]
  rank: median
expected_output:
  answer: 5
control:
  permute_complete_items: true
  hide_rank_from_five_request_copies: true
  cycle_rank_labels_without_changing_targets: true
expected_control_output:
  permuted_answer: 5
  hidden_copy_predictions: identical
  cycled_rank_skill: degraded
```
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.aggregation.order_statistics.support import (
    RANKS,
    SHAPES,
    column,
    cycle_rank,
    diagnostics,
    fit,
    hide_rank,
    model,
    overall_score,
    permute_items,
    prediction,
    records,
    rmse,
    scores,
    validate_request_families,
)

pytestmark = pytest.mark.proof


def test_visible_rank_selects_order_statistic_from_random_item_order() -> None:
    """Require endpoints, quartiles, median, and causal controls together."""

    seed = 3700
    lit.seed_everything(seed, workers=True)
    train = records(bags=1024, seed=3701)
    validate = records(bags=192, seed=3702)
    test = records(bags=384, seed=3703)
    configured = model()
    trainer = fit(configured, train, validate, seed=seed, steps=800)

    intact_prediction = prediction(configured, test)
    by_rank = scores(train=train, test=test, predicted=intact_prediction, keys=("rank",))
    by_shape = scores(train=train, test=test, predicted=intact_prediction, keys=("shape",))
    intact = overall_score(train=train, test=test, predicted=intact_prediction)

    permuted_prediction = prediction(configured, permute_items(test, seed=3704))
    target_scale = float(np.std(column(test, "answer").astype(np.float64)))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale

    cycled_prediction = prediction(configured, cycle_rank(test))
    cycled = overall_score(train=train, test=test, predicted=cycled_prediction)

    hidden_prediction = prediction(configured, hide_rank(test))
    hidden = scores(train=train, test=test, predicted=hidden_prediction, keys=("rank",))
    hidden_family_spread = float(np.max(np.ptp(hidden_prediction.reshape(-1, len(RANKS)), axis=1)))

    details = "\n".join(
        (
            diagnostics("intact overall", intact),
            diagnostics("intact by requested rank", by_rank),
            diagnostics("intact by bag shape", by_shape),
            diagnostics("cycled-rank overall", cycled),
            diagnostics("hidden rank by original request", hidden),
            f"steps={trainer.global_step}",
            f"complete-item permutation drift={permutation_drift:.4f} target SD",
            f"hidden within-bag prediction spread={hidden_family_spread:.3e}",
        )
    )
    assert len(train) == 1024 * len(RANKS) and len(test) == 384 * len(RANKS), details
    assert set(column(test, "rank")) == set(RANKS), details
    assert set(column(test, "shape")) == set(SHAPES), details
    train_bags = validate_request_families(train)
    validate_bags = validate_request_families(validate)
    test_bags = validate_request_families(test)
    assert train_bags.isdisjoint(validate_bags | test_bags), details
    assert validate_bags.isdisjoint(test_bags), details
    calibration = np.asarray(
        [
            target_scale,
            intact.rmse,
            intact.baseline_rmse,
            intact.nrmse,
            cycled.rmse,
            cycled.baseline_rmse,
            cycled.nrmse,
            permutation_drift,
            hidden_family_spread,
            *(score.baseline_rmse for score in by_rank.values()),
            *(score.baseline_rmse for score in by_shape.values()),
        ]
    )
    assert np.isfinite(calibration).all(), details
    assert target_scale > 0.80, details
    assert min(score.baseline_rmse for score in by_rank.values()) > 0.60, details
    assert min(score.baseline_rmse for score in by_shape.values()) > 0.80, details
    assert hidden_family_spread <= 1e-6, details

    endpoint_passes = max(by_rank[(rank,)].nrmse for rank in ("minimum", "maximum")) < 0.30
    interior_passes = max(by_rank[(rank,)].nrmse for rank in ("q25", "median", "q75")) < 0.35
    shape_passes = all(by_shape[(shape,)].nrmse < 0.45 for shape in SHAPES)
    controls_pass = (
        permutation_drift < 0.10
        and cycled.nrmse >= 0.80
        and cycled.nrmse >= intact.nrmse + 0.35
        and all(hidden[(rank,)].nrmse >= 0.65 for rank in ("minimum", "maximum"))
        and sum(score.nrmse >= 0.65 for score in hidden.values()) >= 3
    )
    assert endpoint_passes and interior_passes and shape_passes and controls_pass, details
