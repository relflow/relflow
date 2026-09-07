"""Mean is a weighted-sum footgun when item cardinality varies.

The model gets the hard part for free: every item already contains its
``value * weight`` contribution. With item attention disabled, Mean averages
identical encoded contribution tokens and therefore erases their multiplicity
in this configuration. The decoder should learn the corresponding raw average,
but cannot recover both sums from that unchanged summary.

Example
-------
Mean preserves the supplied contribution but not how often it occurred:

```yaml
input:
  one_item: {items: [{contribution: 0.75}]}
  six_items:
    items: [{contribution: 0.75}, {contribution: 0.75}, {contribution: 0.75},
            {contribution: 0.75}, {contribution: 0.75}, {contribution: 0.75}]
expected_output:
  one_item: {mean_contribution: 0.75, weighted_sum: 0.75}
  six_items: {mean_contribution: 0.75, weighted_sum: 4.50}
expected_control_output:
  mean_reducer_summary: identical
  learned_sum_predictions: identical
```

The matching predictions are the intended footgun demonstration, not correct
sum answers.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pytest

from proofs.aggregation.weighted_aggregation.support import (
    ITEMS,
    data,
    diagnostics,
    mean_model,
    prediction,
    repeated_contribution_records,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_mean_learns_contribution_average_but_erases_sum_count() -> None:
    """Learn the raw average while a no-attention Mean discards multiplicity."""

    seed = 3309
    lit.seed_everything(seed, workers=True)
    train = repeated_contribution_records(rows=1024, seed=3310)
    validate = repeated_contribution_records(rows=256, seed=3311)
    test = repeated_contribution_records(rows=512, seed=3312)
    configured = mean_model()
    fit = trainer(steps=300)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    mean_score = score(
        train=train,
        test=test,
        predicted=prediction(configured, test, "mean_contribution"),
        target="mean_contribution",
    )
    paired = pa.Table.from_pylist(
        [
            {"items": [{"contribution": 0.75}]},
            {"items": [{"contribution": 0.75}] * ITEMS},
        ]
    )
    paired_sum = prediction(configured, paired, "weighted_sum")
    details = (
        f"{diagnostics('Mean contribution average', mean_score)}\n"
        f"one-versus-{ITEMS} identical-contribution sum predictions={paired_sum.tolist()}"
    )
    assert mean_score.nrmse < 0.25, details
    assert np.allclose(paired_sum[0], paired_sum[1], atol=1e-6, rtol=0.0), details
