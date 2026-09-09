"""Test the smallest category-filtering rung with a fixed mean operation.

Example
-------
Items are interleaved; only the requested category should contribute:

```yaml
input:
  items:
    - {group: A, value: 1.0}
    - {group: B, value: -0.4}
    - {group: C, value: 0.5}
    - {group: A, value: 1.2}
    - {group: B, value: -0.2}
    - {group: C, value: 0.7}
  selected_group: B
  operation: mean
expected_output:
  answer: -0.3  # mean([-0.4, -0.2])
control:
  change: permute group labels while retaining values and answer
expected_control_output:
  behavior: prediction error should rise because group/value pairing was destroyed
```
"""

import lightning.pytorch as lit
import pytest

import relflow as rf
from proofs.relational.category_conditioned_reduction.support import (
    diagnostics,
    filtered_mean_table,
    fit,
    model,
    permute_item_groups,
    predict,
    scores,
)

pytestmark = pytest.mark.proof


def test_category_conditioned_mean_reduction() -> None:
    """The smallest filtering rung selects one of three interleaved groups."""

    lit.seed_everything(16, workers=True)
    train = filtered_mean_table(bags=512, items_per_group=2, seed=163)
    validate = filtered_mean_table(bags=64, items_per_group=2, seed=164)
    test = filtered_mean_table(bags=128, items_per_group=2, seed=165)

    configured = model(
        length=6,
        item_reduction=rf.Attention(n_layers=2),
        root_reduction=rf.Attention(n_layers=2),
    )
    fit(configured, train, validate, steps=800)
    predicted = predict(configured, test)
    result = scores(
        train=train,
        test=test,
        predicted=predicted,
        keys=("selected_group",),
    )
    corrupted = scores(
        train=train,
        test=test,
        predicted=predict(configured, permute_item_groups(test, seed=166)),
        keys=("selected_group",),
    )

    details = f"intact:\n{diagnostics(result)}\ncorrupted labels:\n{diagnostics(corrupted)}"
    assert all(score.nrmse <= 0.55 for score in result.values()), details
    assert all(corrupted[cell].nrmse >= result[cell].nrmse + 0.25 for cell in result), details
