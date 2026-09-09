"""Mean is invariant here with item attention disabled.

This is the positive recipe for average targets and the negative recipe for
sums. The reducer averages encoded Number tokens rather than raw values; with
no earlier item interaction, both interventions retain that encoded-token
average while duplication changes total mass.

Example
-------

```yaml
input:
  items:
    - {amount: 0.20}
    - {amount: 0.80}
expected_output: {mean_amount: 0.50}
control:
  kind: duplicate_complete_bag
  items:
    - {amount: 0.20}
    - {amount: 0.80}
    - {amount: 0.20}
    - {amount: 0.80}
expected_control_output:
  mean_amount: 0.50
footgun_if_target_were_sum:
  original_total: 1.00
  duplicated_total: 2.00
  branch_summaries_are_indistinguishable: true
```

Duplication invariance is correct for the tested mean target, but it is the
footgun that makes Mean insufficient for a sum target without visible count.
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
    permute,
    prediction,
    random_records,
    rmse,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_mean_is_duplicate_invariant_for_variable_length_bags() -> None:
    """Learn average and retain it exactly under two multiset interventions."""

    seed = 3600
    lit.seed_everything(seed, workers=True)
    train = random_records(rows=768, seed=3601)
    validate = random_records(rows=192, seed=3602)
    test = random_records(rows=384, seed=3603, maximum=3)
    configured = model(reduction="mean", include_count=False, target="mean_amount")
    fit = trainer(steps=350)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    intact_prediction = prediction(configured, test, "mean_amount")
    measured = score(train=train, test=test, predicted=intact_prediction, target="mean_amount")
    permutation_prediction = prediction(configured, permute(test, seed=3604), "mean_amount")
    duplication_prediction = prediction(configured, duplicate(test, include_count=False), "mean_amount")
    scale = float(np.std(column(test, "mean_amount")))
    permutation_drift = rmse(intact_prediction, permutation_prediction) / scale
    duplication_drift = rmse(intact_prediction, duplication_prediction) / scale
    details = "\n".join(
        (
            diagnostics("variable-length mean", measured),
            f"steps={fit.global_step}",
            f"item-permutation drift={permutation_drift:.8f} target SD",
            f"whole-bag duplication drift={duplication_drift:.8f} target SD",
        )
    )
    assert measured.nrmse < 0.25, details
    assert permutation_drift < 1e-6, details
    assert duplication_drift < 1e-6, details
