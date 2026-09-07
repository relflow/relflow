"""Interleaved categories should select peers before aggregate broadcast.

The only production-like inputs are each item's category and number.  Category
corruption, independent group translations, and complete-record permutations
separately test peer selection, relative arithmetic, and set equivariance.

Example
-------
```yaml
input:
  items:
    - {group: A, value: 2.0, deviation: null}
    - {group: B, value: 1.0, deviation: null}
    - {group: C, value: -3.0, deviation: null}
    - {group: A, value: 4.0, deviation: null}
    - {group: C, value: -1.0, deviation: null}
    - {group: B, value: -1.0, deviation: null}
expected_output:
  deviations: [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
  rule: value minus the mean of values in the same group
control:
  group_translation: add a different constant to every group's values
  group_corruption: rotate only group labels while retaining values and targets
expected_control_output:
  translated_deviations: [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
  corrupted_group_behavior: prediction error increases
```
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.relational.peer_relative_inference.support import (
    corrupt_groups,
    data,
    diagnostics,
    group_translation,
    grouped_records,
    implied_deviation,
    model,
    permute_items,
    prediction,
    rmse,
    score,
    target,
    trainer,
    validate_score,
)

pytestmark = pytest.mark.proof


def test_group_mean_is_selected_and_routed_back_to_each_item() -> None:
    """Infer grouped deviations from randomly interleaved raw item records."""

    seed = 3720
    lit.seed_everything(seed, workers=True)
    train = grouped_records(rows=2048, seed=3721)
    validate = grouped_records(rows=512, seed=3722)
    test = grouped_records(rows=768, seed=3723)
    configured = model(grouped=True, source="raw")
    fit = trainer(steps=900)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    predicted = prediction(configured, test)
    measured = score(train=train, test=test, predicted=predicted)
    translated = group_translation(test, seed=3724)
    translated_prediction = prediction(configured, translated)
    translated_score = score(train=train, test=translated, predicted=translated_prediction)
    corrupted = corrupt_groups(test)
    corrupted_score = score(train=train, test=corrupted, predicted=prediction(configured, corrupted))
    permuted, order = permute_items(test, seed=3725)
    permutation_drift = rmse(prediction(configured, permuted), predicted[order]) / measured.baseline_rmse

    assert abs(float(target(train).mean())) < 1e-12
    assert rmse(target(translated), target(test)) == 0.0
    oracle_corruption = rmse(implied_deviation(corrupted), target(test)) / measured.baseline_rmse
    assert oracle_corruption > 0.75, f"weak group corruption: oracle nRMSE={oracle_corruption:.3f}"
    assert sorted(target(permuted).tolist()) == sorted(target(test).tolist())
    validate_score("grouped", measured)
    validate_score("group translation", translated_score)
    validate_score("corrupted labels", corrupted_score)
    assert np.isfinite(oracle_corruption), f"oracle corruption is not finite: {oracle_corruption}"
    assert np.isfinite(permutation_drift), f"permutation drift is not finite: {permutation_drift}"

    details = (
        f"{diagnostics('grouped', measured)}\n"
        f"{diagnostics('group translation', translated_score)}\n"
        f"{diagnostics('corrupted labels', corrupted_score)}\n"
        f"oracle corruption={oracle_corruption:.3f}\n"
        f"permutation drift={permutation_drift:.3f}"
    )
    assert measured.nrmse < 0.40, details
    assert translated_score.nrmse < 0.50, details
    assert corrupted_score.nrmse >= measured.nrmse + 0.25, details
    assert permutation_drift < 0.20, details
