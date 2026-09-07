"""Aligned sibling fields should support covariance.

The desired schema places ``x`` and ``y`` on each repeated item. A negative
control circularly shifts only ``y`` within every held-out row; this preserves
both univariate marginals exactly but breaks the association with the original
covariance target.

Example
-------

```yaml
input:
  items:
    - {x: -1, y: -1.4}
    - {x: 1, y: 0.2}
    - {x: -1, y: -0.2}
    - {x: 1, y: 1.4}
    - {x: -1, y: -1.4}
    - {x: 1, y: 0.2}
    - {x: -1, y: -0.2}
    - {x: 1, y: 1.4}
expected_output: {covariance: 0.8}
control:
  kind: circular_shift_y_only
  items:
    - {x: -1, y: 1.4}
    - {x: 1, y: -1.4}
    - {x: -1, y: 0.2}
    - {x: 1, y: -0.2}
    - {x: -1, y: 1.4}
    - {x: 1, y: -1.4}
    - {x: -1, y: 0.2}
    - {x: 1, y: -0.2}
expected_control_output:
  implied_covariance: -0.8
  retained_test_label: 0.8
  dataset_nrmse: increases
```

The negative control deliberately retains the original ``0.80`` target. A
model using pair alignment should therefore move its prediction and score
worse after the shift, even though each field's values are unchanged.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.aggregation.distribution_statistics.support import (
    covariance_records,
    data,
    diagnostics,
    model,
    prediction,
    score,
    shuffle_y,
    trainer,
)

pytestmark = pytest.mark.proof


def test_aligned_item_pairs_reveal_covariance() -> None:
    """Learn the centered cross-moment and respond to pairing corruption."""

    seed = 3508
    lit.seed_everything(seed, workers=True)
    train = covariance_records(rows=1536, seed=3509)
    validate = covariance_records(rows=384, seed=3510)
    test = covariance_records(rows=768, seed=3511)
    configured = model(source="paired", target="covariance")
    fit = trainer(steps=1_100)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    intact = score(
        train=train,
        test=test,
        predicted=prediction(configured, test, "covariance"),
        target="covariance",
    )
    shuffled = shuffle_y(test, seed=3512)
    corrupted = score(
        train=train,
        test=shuffled,
        predicted=prediction(configured, shuffled, "covariance"),
        target="covariance",
    )
    details = "\n".join(
        (
            diagnostics("intact aligned covariance", intact),
            diagnostics("shuffled-y negative control", corrupted),
        )
    )
    calibration = np.asarray(
        [
            intact.rmse,
            intact.baseline_rmse,
            intact.nrmse,
            corrupted.rmse,
            corrupted.baseline_rmse,
            corrupted.nrmse,
        ]
    )
    assert np.isfinite(calibration).all(), details
    assert intact.baseline_rmse > 0.35, details
    assert np.isclose(intact.baseline_rmse, corrupted.baseline_rmse, atol=1e-12, rtol=0.0), details
    assert intact.nrmse < 0.35, details
    assert corrupted.nrmse >= intact.nrmse + 0.35, details
