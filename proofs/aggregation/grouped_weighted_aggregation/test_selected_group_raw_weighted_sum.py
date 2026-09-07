"""The natural schema should learn a selected group's weighted sum.

Values and weights remain independent sibling fields.  Label rotation tests
group routing; within-group weight swaps separately test item-local binding.

Example
-------

```yaml
input:
  selected_group: A
  items:
    - {group: B, value: 0.30, weight: 0.40}
    - {group: A, value: 0.80, weight: 0.50}
    - {group: C, value: -0.50, weight: 1.40}
    - {group: A, value: -0.40, weight: 1.25}
    - {group: B, value: -0.20, weight: 0.90}
    - {group: C, value: 0.70, weight: 0.30}
expected_output: {answer: -0.10}
control:
  kind: swap_weights_within_each_group
  selected_A_pairs:
    - {value: 0.80, weight: 1.25}
    - {value: -0.40, weight: 0.50}
expected_control_output:
  visibly_implied_answer: 0.80
  retained_test_label: -0.10
  dataset_nrmse: increases
```

Swapping only the two A weights preserves both marginals but changes the
implied answer to ``(0.80 * 1.25) + (-0.40 * 0.50) = 0.80``. The corruption
retains the original ``-0.10`` label, so a model using item-local pairing
should move its prediction and score worse.
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
    swap_weights_within_groups,
    trainer,
)

pytestmark = pytest.mark.proof


def test_selected_group_weighted_sum_from_raw_pairs() -> None:
    """Learn filtering, coordinate binding, multiplication, and summation."""

    seed = 3605
    lit.seed_everything(seed, workers=True)
    train = records(bags=512, seed=3606, source="raw")
    validate = records(bags=128, seed=3607, source="raw")
    test = records(bags=256, seed=3608, source="raw")
    configured = model(source="raw")
    fit = trainer(steps=1_100)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    intact_prediction = prediction(configured, test)
    intact = score(train=train, test=test, predicted=intact_prediction)
    labels = score(
        train=train,
        test=test,
        predicted=prediction(configured, rotate_group_labels(test)),
    )
    pairing = score(
        train=train,
        test=test,
        predicted=prediction(configured, swap_weights_within_groups(test)),
    )
    permuted_prediction = prediction(configured, permute_items(test, seed=3609))
    target_scale = float(np.std(column(test, "answer")))
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    details = "\n".join(
        (
            diagnostics("intact raw value/weight pairs", intact),
            diagnostics("rotated group labels", labels),
            diagnostics("within-group weight swaps", pairing),
            f"joint-permutation drift={permutation_drift:.4f} target SD",
        )
    )
    assert all(np.isfinite(value) for value in (intact.nrmse, labels.nrmse, pairing.nrmse, permutation_drift)), details
    assert intact.nrmse < 0.50, details
    assert labels.nrmse >= intact.nrmse + 0.20, details
    assert pairing.nrmse >= intact.nrmse + 0.20, details
    assert permutation_drift < 0.10, details
