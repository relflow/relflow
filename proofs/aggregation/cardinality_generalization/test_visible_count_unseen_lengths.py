"""The explicit-count control should extrapolate more cleanly than hidden count.

With item attention disabled, Mean produces a shared encoded-value summary and
``item_count`` exposes the missing cardinality. Testing lengths seven through
ten after training on one through six asks whether the ordinary root decoder
can extend that learned product.

Example
-------

```yaml
training_cardinality: {minimum: 1, maximum: 6}
input:
  item_count: 8
  items:
    - {amount: 0.50}
    - {amount: 0.50}
    - {amount: 0.50}
    - {amount: 0.50}
    - {amount: 0.50}
    - {amount: 0.50}
    - {amount: 0.50}
    - {amount: 0.50}
expected_output: {total: 4.00}
control:
  kind: duplicate_complete_bag
  before:
    item_count: 4
    items: [{amount: 0.25}, {amount: 0.25}, {amount: 0.25}, {amount: 0.25}]
  after:
    item_count: 8
    items:
      - {amount: 0.25}
      - {amount: 0.25}
      - {amount: 0.25}
      - {amount: 0.25}
      - {amount: 0.25}
      - {amount: 0.25}
      - {amount: 0.25}
      - {amount: 0.25}
expected_control_output: {before_total: 1.00, after_total: 2.00}
```

The explicit count gives the decoder the multiplicity that Mean discards.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.aggregation.cardinality_generalization.support import (
    column,
    data,
    diagnostics,
    duplicate,
    model,
    prediction,
    random_records,
    rmse,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_visible_count_sum_extrapolates_to_unseen_lengths() -> None:
    """Require count-aware sum accuracy and duplication scaling out of range."""

    seed = 3621
    lit.seed_everything(seed, workers=True)
    train = random_records(rows=1024, seed=3622, repeated=True, include_count=True)
    validate = random_records(rows=256, seed=3623, repeated=True, include_count=True)
    unseen = random_records(
        rows=512,
        seed=3624,
        minimum=7,
        maximum=10,
        repeated=True,
        include_count=True,
    )
    short = random_records(rows=256, seed=3625, maximum=5, repeated=True, include_count=True)
    configured = model(reduction="mean", include_count=True, target="total")
    fit = trainer(steps=700)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    unseen_score = score(
        train=train,
        test=unseen,
        predicted=prediction(configured, unseen, "total"),
        target="total",
    )
    short_prediction = prediction(configured, short, "total")
    duplicated_prediction = prediction(configured, duplicate(short, include_count=True), "total")
    duplication_error = rmse(2.0 * short_prediction, duplicated_prediction) / float(np.std(column(unseen, "total")))
    details = "\n".join(
        (
            diagnostics("visible-count unseen lengths 7..10", unseen_score),
            f"steps={fit.global_step}",
            f"duplication equivariance error={duplication_error:.4f} unseen target SD",
        )
    )
    assert np.isfinite([unseen_score.nrmse, duplication_error]).all(), details
    assert unseen_score.nrmse < 0.40 and duplication_error < 0.20, details
