"""Supplied products isolate category selection and fixed-width summation.

This diagnostic intentionally exposes ``contribution = value * weight``.  It
is useful for localizing a failure, but is not the desired end-user schema.

Example
-------

```yaml
input:
  selected_group: A
  items:
    - {group: B, contribution: 0.40}
    - {group: A, contribution: 0.60}
    - {group: C, contribution: -0.20}
    - {group: A, contribution: -0.10}
    - {group: B, contribution: 0.30}
    - {group: C, contribution: 0.80}
expected_output: {answer: 0.50}
control:
  kind: rotate_item_labels
  mapping: {A: B, B: C, C: A}
expected_control_output:
  visibly_implied_answer: 0.60
  retained_test_label: 0.50
  dataset_nrmse: increases
```

Jointly permuting complete items still implies ``0.50``. Rotating labels
``A -> B -> C -> A`` makes the visible A contributions imply ``0.60`` while
the negative control retains the original ``0.50`` label, so its score should
worsen.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.aggregation.grouped_weighted_aggregation.support import (
    column,
    data,
    diagnostics,
    model,
    permute_items,
    prediction,
    records,
    rmse,
    rotate_group_labels,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_selected_group_sums_supplied_item_contributions() -> None:
    """Learn filtering and reduction after item-local products are supplied."""

    seed = 3600
    lit.seed_everything(seed, workers=True)
    train = records(bags=384, seed=3601, source="contribution")
    validate = records(bags=96, seed=3602, source="contribution")
    test = records(bags=192, seed=3603, source="contribution")
    configured = model(source="contribution")
    fit = trainer(steps=800)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    intact_prediction = prediction(configured, test)
    intact = score(train=train, test=test, predicted=intact_prediction)
    corrupted = score(
        train=train,
        test=test,
        predicted=prediction(configured, rotate_group_labels(test)),
    )
    permuted_prediction = prediction(configured, permute_items(test, seed=3604))
    target_scale = float(np.std(column(test, "answer")))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    details = "\n".join(
        (
            diagnostics("intact supplied contributions", intact),
            diagnostics("rotated group labels", corrupted),
            f"joint-permutation drift={permutation_drift:.4f} target SD",
        )
    )
    assert all(np.isfinite(value) for value in (intact.nrmse, corrupted.nrmse, permutation_drift)), details
    assert intact.nrmse < 1.60, details
    assert corrupted.nrmse < 1.80, details
    assert intact.nrmse < 0.50, details
    assert corrupted.nrmse >= intact.nrmse + 0.20, details
    assert permutation_drift < 0.10, details
