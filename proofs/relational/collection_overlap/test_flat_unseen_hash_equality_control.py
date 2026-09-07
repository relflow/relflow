"""Confirm that unseen Hash equality works before testing collection routing.

Example
-------
This removes branches and tests the underlying unseen-ID comparison directly:

```yaml
input:
  positive: {left_id: owl, right_id: owl}
  negative: {left_id: owl, right_id: yak}
expected_output:
  positive: {equal: true}
  negative: {equal: false}
control:
  positive_after_break: {left_id: owl, right_id: eel}
  retained_label: true
expected_control_output:
  mean_positive_probability: lower
  aggregate_auc: chance
```

If this control fails, a collection-overlap failure cannot be blamed only on
cross-branch routing.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.relational.collection_overlap.support import (
    auc,
    flat_pair_model,
    flat_pair_records,
    labels,
    probabilities,
    train,
)

pytestmark = pytest.mark.proof


def test_flat_unseen_hash_equality_primitive_control() -> None:
    """Learn one unseen-ID equality and reject an all-unequal intervention."""

    seed = 3701
    train_rows = flat_pair_records(rows=3072, seed=3702, namespace="train")
    validate_rows = flat_pair_records(rows=768, seed=3703, namespace="validate")
    test = flat_pair_records(rows=1024, seed=3704, namespace="test")
    broken = flat_pair_records(rows=1024, seed=3704, namespace="test", break_equal=True)
    lit.seed_everything(seed, workers=True)
    configured = train(flat_pair_model(), train_rows, validate_rows, seed=seed, steps=600)

    target = labels(test, target="equal")
    intact_probability = probabilities(configured, test, address="pair/equal")
    broken_probability = probabilities(configured, broken, address="pair/equal")
    equality_auc = auc(target, intact_probability)
    broken_auc = auc(target, broken_probability)
    drop = float(np.mean(intact_probability[target] - broken_probability[target]))
    details = (
        f"flat Hash equality: intact_auc={equality_auc:.4f}, broken_auc={broken_auc:.4f}, positive_drop={drop:.4f}"
    )

    assert np.isfinite([equality_auc, broken_auc, drop]).all(), details
    assert equality_auc >= 0.95, details
    assert 0.35 <= broken_auc <= 0.65, details
    assert drop >= 0.25, details
