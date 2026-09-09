"""A weighted mean should remain identifiable when collection length varies.

Unlike a weighted sum, this target is normalized by its own weight mass, so
variable cardinality is not information the answer intrinsically needs. Broad
positive weights make ignoring or misaligning the weights a poor shortcut.

Example
-------
The numerator and denominator change together under the two invariances:

```yaml
input:
  items:
    - {value: 0.8, weight: 0.5}
    - {value: -0.4, weight: 1.25}
    - {value: 0.2, weight: 2.0}
expected_output:
  weighted_mean: 0.08
control:
  scale_all_weights_by_0.75: true
  duplicate_every_complete_item: true
  swap_only_the_first_two_weights: true
expected_control_output:
  scaled_prediction: approximately_unchanged
  duplicated_prediction: approximately_unchanged
  swapped_weights:
    retained_label: 0.08
    mathematical_answer_after_swap: 0.32
    dataset_nrmse: increases
```
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pytest

from proofs.aggregation.weighted_aggregation.support import (
    ITEMS,
    column,
    data,
    diagnostics,
    duplicate_items,
    model,
    permute_items,
    permute_weights,
    prediction,
    rmse,
    scale_weights,
    score,
    trainer,
    variable_weighted_mean_records,
)

pytestmark = pytest.mark.proof


def test_raw_weighted_mean_across_variable_cardinality() -> None:
    """Learn a normalized weighted mean from two through six raw pairs."""

    seed = 3313
    lit.seed_everything(seed, workers=True)
    train = variable_weighted_mean_records(rows=1536, seed=3314)
    validate = variable_weighted_mean_records(rows=384, seed=3315)
    test = variable_weighted_mean_records(rows=768, seed=3316)
    configured = model(source="raw", target="weighted_mean")
    fit = trainer(steps=1_100)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    intact_prediction = prediction(configured, test, "weighted_mean")
    intact = score(
        train=train,
        test=test,
        predicted=intact_prediction,
        target="weighted_mean",
    )
    corrupted_table = permute_weights(test, seed=3317)
    corrupted = score(
        train=train,
        test=test,
        predicted=prediction(configured, corrupted_table, "weighted_mean"),
        target="weighted_mean",
    )
    permuted_prediction = prediction(configured, permute_items(test, seed=3318), "weighted_mean")
    scaled_prediction = prediction(configured, scale_weights(test, factor=0.75), "weighted_mean")
    original_short = pa.Table.from_pylist(
        [row for row in test.to_pylist() if len(row["items"]) <= ITEMS // 2],
        schema=test.schema,
    )
    duplicated_prediction = prediction(
        configured,
        duplicate_items(test, maximum_original=ITEMS // 2),
        "weighted_mean",
    )
    original_short_prediction = prediction(configured, original_short, "weighted_mean")
    target_scale = float(np.std(column(test, "weighted_mean")))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    scaling_drift = rmse(intact_prediction, scaled_prediction) / target_scale
    duplication_drift = rmse(original_short_prediction, duplicated_prediction) / target_scale
    details = "\n".join(
        (
            diagnostics("intact variable-cardinality weighted mean", intact),
            diagnostics("permuted item-local weights", corrupted),
            f"joint-permutation drift={permutation_drift:.4f} target SD",
            f"positive-weight scaling drift={scaling_drift:.4f} target SD",
            f"complete-item duplication drift={duplication_drift:.4f} target SD",
        )
    )
    assert intact.nrmse < 0.40, details
    assert corrupted.nrmse >= intact.nrmse + 0.25, details
    assert permutation_drift < 0.10, details
    assert scaling_drift < 0.10, details
    assert duplication_drift < 0.10, details
