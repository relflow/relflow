"""Learned Attention can interpolate a sum over seen cardinalities.

Independent item values prevent count alone or one selected value from solving
the task.  Complete-item permutation is label preserving.  This accuracy rung
does not establish sum algebra: the strict unseen-length proof separately
checks equal-value cardinality and duplication scaling.

Example
-------

```yaml
input:
  items:
    - {amount: 0.50}
    - {amount: -0.20}
    - {amount: 0.90}
expected_output: {total: 1.20}
control:
  kind: complete_item_permutation
  items:
    - {amount: 0.90}
    - {amount: 0.50}
    - {amount: -0.20}
expected_control_output: {total: 1.20}
```

The second row checks that item order does not materially change the learned
prediction.
"""

from __future__ import annotations

import lightning.pytorch as lit
import pytest

from proofs.aggregation.cardinality_generalization.support import (
    column,
    data,
    diagnostics,
    model,
    permute,
    prediction,
    random_records,
    rmse,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_attention_sum_interpolates_across_seen_cardinalities() -> None:
    """Check ordinary in-range accuracy and approximate order invariance."""

    seed = 3609
    lit.seed_everything(seed, workers=True)
    train = random_records(rows=1536, seed=3610)
    validate = random_records(rows=384, seed=3611)
    test = random_records(rows=768, seed=3612)
    configured = model(reduction="attention", include_count=False, target="total")
    fit = trainer(steps=1_100)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    intact_prediction = prediction(configured, test, "total")
    measured = score(train=train, test=test, predicted=intact_prediction, target="total")
    permuted_prediction = prediction(configured, permute(test, seed=3614), "total")
    target_scale = float(column(test, "total").std())
    permutation_drift = rmse(intact_prediction, permuted_prediction) / target_scale
    details = "\n".join(
        (
            diagnostics("Attention in-range sum", measured),
            f"steps={fit.global_step}",
            f"item-permutation drift={permutation_drift:.4f} target SD",
        )
    )
    assert measured.nrmse < 0.35, details
    assert permutation_drift < 0.12, details
