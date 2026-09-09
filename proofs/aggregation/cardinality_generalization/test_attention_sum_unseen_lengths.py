"""Characterize sum extrapolation beyond every trained collection length.

The schema capacity is twelve, but training contains at most six items.  The
test set contains seven through ten.  This distinction matters: capacity is a
shape bound, not evidence that a learned normalized reduction acquired sum
semantics for unseen cardinalities.

Example
-------

```yaml
training_cardinality: {minimum: 1, maximum: 6}
input:
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
  before: [{amount: 0.30}, {amount: -0.10}]
  after: [{amount: 0.30}, {amount: -0.10}, {amount: 0.30}, {amount: -0.10}]
expected_control_output: {before_total: 0.20, after_total: 0.40}
```

The proof requires both unseen-length accuracy and the doubling response; mere
capacity for twelve items does not guarantee either behavior.
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
    equal_value_probes,
    model,
    prediction,
    random_records,
    rmse,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_attention_sum_extrapolates_beyond_trained_cardinality() -> None:
    """Require unseen-length accuracy and exact complete-bag scaling together."""

    seed = 3615
    lit.seed_everything(seed, workers=True)
    train = random_records(rows=1536, seed=3616)
    validate = random_records(rows=384, seed=3617)
    in_range = random_records(rows=512, seed=3618)
    unseen = random_records(rows=512, seed=3619, minimum=7, maximum=10)
    short = random_records(rows=256, seed=3620, maximum=5)
    configured = model(reduction="attention", include_count=False, target="total")
    fit = trainer(steps=1_100)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    in_range_score = score(
        train=train,
        test=in_range,
        predicted=prediction(configured, in_range, "total"),
        target="total",
    )
    unseen_score = score(
        train=train,
        test=unseen,
        predicted=prediction(configured, unseen, "total"),
        target="total",
    )
    short_prediction = prediction(configured, short, "total")
    duplicated_prediction = prediction(configured, duplicate(short, include_count=False), "total")
    duplication_error = rmse(2.0 * short_prediction, duplicated_prediction) / float(np.std(column(unseen, "total")))
    probes = equal_value_probes(value=0.65, lengths=(1, 3, 6))
    probe_prediction = prediction(configured, probes, "total")
    probe_target = column(probes, "total")
    probe_error = rmse(probe_target, probe_prediction) / float(np.std(column(unseen, "total")))
    details = "\n".join(
        (
            diagnostics("Attention seen lengths 1..6", in_range_score),
            diagnostics("Attention unseen lengths 7..10", unseen_score),
            f"steps={fit.global_step}",
            f"duplication equivariance error={duplication_error:.4f} unseen target SD",
            f"equal-value targets={probe_target.tolist()}",
            f"equal-value predictions={probe_prediction.tolist()}",
            f"equal-value cardinality error={probe_error:.4f} unseen target SD",
        )
    )
    assert np.isfinite([in_range_score.nrmse, unseen_score.nrmse, duplication_error, probe_error]).all(), details
    assert (
        in_range_score.nrmse < 0.35 and unseen_score.nrmse < 0.30 and duplication_error < 0.20 and probe_error < 0.20
    ), details
