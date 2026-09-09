"""A visible item count diagnoses cardinality lost by Mean.

Every item in a row has the same random value. With item attention disabled,
Mean therefore produces the same encoded-value summary at every cardinality,
while the ordinary root ``item_count`` field supplies the only changing factor
needed for sum. This is a practical fallback and an isolation control, not the
desired long-term need for users to hand-author counts.

Example
-------

```yaml
input:
  rows:
    - item_count: 1
      items: [{amount: 0.70}]
    - item_count: 6
      items:
        - {amount: 0.70}
        - {amount: 0.70}
        - {amount: 0.70}
        - {amount: 0.70}
        - {amount: 0.70}
        - {amount: 0.70}
expected_output:
  rows: [{total: 0.70}, {total: 4.20}]
footgun_without_item_count:
  branch_mean_summary: identical_for_both_rows
  distinguishable_sum_targets: false
```

Without ``item_count``, the two Mean branch summaries are identical and a
decoder cannot distinguish these sum targets.
"""

from __future__ import annotations

import lightning.pytorch as lit
import pytest

from proofs.aggregation.cardinality_generalization.support import (
    TRAIN_MAX,
    column,
    data,
    diagnostics,
    equal_value_probes,
    model,
    prediction,
    random_records,
    rmse,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_visible_count_restores_in_range_sum_after_mean() -> None:
    """Learn value times visible cardinality and distinguish matched probes."""

    seed = 3605
    lit.seed_everything(seed, workers=True)
    train = random_records(rows=1024, seed=3606, repeated=True, include_count=True)
    validate = random_records(rows=256, seed=3607, repeated=True, include_count=True)
    test = random_records(rows=512, seed=3608, repeated=True, include_count=True)
    configured = model(reduction="mean", include_count=True, target="total")
    fit = trainer(steps=700)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    predicted = prediction(configured, test, "total")
    measured = score(train=train, test=test, predicted=predicted, target="total")
    probes = equal_value_probes(value=0.7, lengths=(1, TRAIN_MAX), include_count=True)
    probe_prediction = prediction(configured, probes, "total")
    probe_target = column(probes, "total")
    probe_error = rmse(probe_target, probe_prediction)
    details = "\n".join(
        (
            diagnostics("visible-count in-range sum", measured),
            f"steps={fit.global_step}",
            f"equal-value targets={probe_target.tolist()}",
            f"equal-value predictions={probe_prediction.tolist()}, rmse={probe_error:.4f}",
        )
    )
    assert measured.nrmse < 0.25, details
    assert probe_prediction[1] - probe_prediction[0] > 2.5, details
    assert probe_error < 0.35, details
