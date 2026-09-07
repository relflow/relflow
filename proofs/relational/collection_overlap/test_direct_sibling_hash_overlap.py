"""Record the current boundary for direct sibling-collection overlap.

The natural schema exposes every identity token, but the ordinary scalar
decoder does not reliably compare every left member with every right member.
This expected-pass proof deliberately asserts the observed chance behavior so
a future implementation cannot silently claim success without changing its
contract and evidence.

Example
-------
The shared unseen identity can occur at any position in either branch:

```yaml
input:
  left:
    - {entity_id: fox}
    - {entity_id: owl}
    - {entity_id: lynx}
  right:
    - {entity_id: yak}
    - {entity_id: owl}
expected_output:
  has_overlap: true
observed_current_model:
  has_overlap: chance-level probability
control:
  transformation: rename all IDs and independently permute both sides
  expected_overlap: unchanged
expected_control_output:
  has_overlap: true
  observed_current_model: still chance-level
```

For production, compute deterministic overlap before the model. Do not infer
from ``reduction=None`` that the target performs an all-pairs comparison. The
flat equality control beside this file proves that unseen ``Hash`` equality
itself remains learnable; ``RELATIONS.md`` specifies a possible future feature.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.relational.collection_overlap.support import (
    MAX_STEPS,
    TEST_ROWS,
    TRAIN_ROWS,
    VALIDATE_ROWS,
    break_overlaps,
    collection_model,
    collection_records,
    diagnostics,
    labels,
    overlap_truth,
    rename_and_permute,
    route_score,
    train,
)

pytestmark = pytest.mark.proof


def test_direct_sibling_branches_record_overlap_limitation() -> None:
    """Keep the unsupported collection-comparison behavior explicit."""

    seed = 3711
    train_rows = collection_records(rows=TRAIN_ROWS, seed=3712, namespace="train")
    validate_rows = collection_records(rows=VALIDATE_ROWS, seed=3713, namespace="validate")
    test = collection_records(rows=TEST_ROWS, seed=3714, namespace="test")
    invariant = rename_and_permute(test, seed=3715)
    broken = break_overlaps(test)

    target = labels(test, target="has_overlap")
    assert target.mean() == 0.5
    assert np.array_equal(target, overlap_truth(test))
    assert np.array_equal(target, overlap_truth(invariant))
    assert not overlap_truth(broken).any()

    lit.seed_everything(seed, workers=True)
    configured = train(
        collection_model(),
        train_rows,
        validate_rows,
        seed=seed,
        steps=MAX_STEPS,
    )
    score = route_score(configured, test, invariant, broken)
    details = diagnostics("ordinary direct sibling Hash overlap", score)
    calibration = np.asarray(
        [
            score.intact_auc,
            score.invariant_auc,
            score.broken_auc,
            score.invariant_drift,
            score.positive_break_drop,
        ]
    )
    assert np.isfinite(calibration).all(), details
    assert 0.35 <= score.intact_auc <= 0.65, details
    assert 0.35 <= score.invariant_auc <= 0.65, details
    assert 0.35 <= score.broken_auc <= 0.65, details
    assert score.invariant_drift <= 0.15, details
    assert abs(score.positive_break_drop) <= 0.10, details
