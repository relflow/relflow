"""Compose category filtering and operation selection through the natural schema.

Example
-------
One interleaved bag is expanded into separate group-operation requests:

```yaml
input:
  items:
    - {group: A, value: 0.1}
    - {group: B, value: -0.1}
    - {group: C, value: 0.0}
    - {group: A, value: 0.4}
    - {group: B, value: 0.3}
    - {group: C, value: 0.5}
    - {group: A, value: -1.0}
    - {group: B, value: -0.9}
    - {group: C, value: -1.2}
    - {group: A, value: 1.2}
    - {group: B, value: 1.3}
    - {group: C, value: 1.0}
  selected_group: B
  operation: max
expected_output:
  answer: 1.3
expected_output_by_operation:
  sum: 0.6
  mean: 0.15
  min: -0.9
  max: 1.3
```
"""

import lightning.pytorch as lit
import numpy as np
import pytest

import relflow as rf
from proofs.relational.category_conditioned_reduction.support import (
    diagnostics,
    fit,
    grouped_table,
    model,
    predict,
    scores,
)

pytestmark = pytest.mark.proof


def test_category_conditioned_group_and_operation_composition() -> None:
    """Compose a requested group filter with four requested reductions."""

    lit.seed_everything(16, workers=True)
    train = grouped_table(bags=512, items_per_group=4, seed=160)
    validate = grouped_table(bags=64, items_per_group=4, seed=161)
    test = grouped_table(bags=128, items_per_group=4, seed=162)

    configured = model(
        length=12,
        item_reduction=None,
        root_reduction=rf.Attention(n_outputs=12, n_layers=2),
    )
    fit(configured, train, validate, steps=1_200)
    result = scores(
        train=train,
        test=test,
        predicted=predict(configured, test),
        keys=("operation", "selected_group"),
    )

    assert all(np.isfinite(score.nrmse) and score.nrmse <= 1.60 for score in result.values()), diagnostics(result)
    assert all(score.nrmse <= 0.30 for score in result.values()), diagnostics(result)
