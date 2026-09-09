"""Raw sibling value and weight fields should support a weighted sum.

This is the desired user-facing path: put independently drawn ``value`` and
``weight`` on each item and let the learned architecture infer both their
same-coordinate product and the sum. A within-row weight permutation is the
causal check that the model used the pairing rather than either marginal.

Example
-------
Each value must bind to the weight beside it before reduction:

```yaml
input:
  items:
    - {value: 0.8, weight: 0.5}
    - {value: -0.4, weight: 1.0}
    - {value: 0.2, weight: 1.4}
    - {value: 0.5, weight: 0.2}
    - {value: -0.5, weight: 0.4}
    - {value: 0.9, weight: 0.3}
expected_output:
  weighted_sum: 0.45
control:
  reorder_complete_items: true
  swap_only_the_first_two_weights: true
expected_control_output:
  reordered_prediction: approximately_unchanged
  swapped_weights:
    retained_label: 0.45
    mathematical_answer_after_swap: 1.05
    dataset_nrmse: increases
```
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.aggregation.weighted_aggregation.support import (
    ITEMS,
    column,
    data,
    diagnostics,
    model,
    permute_items,
    permute_weights,
    prediction,
    rmse,
    scale_weights,
    score,
    trainer,
    weighted_records,
)

pytestmark = pytest.mark.proof


def test_raw_item_values_and_weights_form_weighted_sum() -> None:
    """Learn products and their sum without a derived contribution field."""

    seed = 3304
    lit.seed_everything(seed, workers=True)
    train = weighted_records(rows=1024, length=ITEMS, seed=3305, source="raw")
    validate = weighted_records(rows=256, length=ITEMS, seed=3306, source="raw")
    test = weighted_records(rows=512, length=ITEMS, seed=3307, source="raw")
    configured = model(source="raw")
    fit = trainer(steps=900)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    intact_prediction = prediction(configured, test, "weighted_sum")
    intact = score(
        train=train,
        test=test,
        predicted=intact_prediction,
        target="weighted_sum",
    )
    corrupted_table = permute_weights(test, seed=3308)
    corrupted = score(
        train=train,
        test=test,
        predicted=prediction(configured, corrupted_table, "weighted_sum"),
        target="weighted_sum",
    )
    permuted_prediction = prediction(configured, permute_items(test, seed=3309), "weighted_sum")
    factor = 0.75
    scaled_table = scale_weights(test, factor=factor)
    scaled_prediction = prediction(configured, scaled_table, "weighted_sum")
    scaled = score(
        train=scale_weights(train, factor=factor),
        test=scaled_table,
        predicted=scaled_prediction,
        target="weighted_sum",
    )
    target_scale = float(np.std(column(test, "weighted_sum")))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    scaling_error = rmse(factor * intact_prediction, scaled_prediction) / (factor * target_scale)
    details = "\n".join(
        (
            diagnostics("intact raw value/weight", intact),
            diagnostics("permuted item-local weights", corrupted),
            diagnostics("uniformly scaled weights", scaled),
            f"joint-permutation drift={permutation_drift:.4f} target SD",
            f"positive-weight scaling error={scaling_error:.4f} scaled-target SD",
        )
    )
    assert intact.nrmse < 0.30, details
    assert corrupted.nrmse >= intact.nrmse + 0.35, details
    assert scaled.nrmse < 0.35, details
    assert permutation_drift < 0.08, details
    assert scaling_error < 0.15, details
