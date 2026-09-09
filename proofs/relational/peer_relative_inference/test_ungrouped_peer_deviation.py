"""Raw peer values should support an aggregate broadcast to every coordinate.

There is no group selection in this rung.  Wide random row translations make a
local-value shortcut fail, while common-translation invariance and complete
record permutation test the intended mean-subtraction behavior directly.

Example
-------
```yaml
input:
  items:
    - {value: 1.5, deviation: null}
    - {value: 1.9, deviation: null}
    - {value: 2.3, deviation: null}
    - {value: 2.7, deviation: null}
    - {value: 3.1, deviation: null}
    - {value: 3.5, deviation: null}
expected_output:
  row_mean: 2.5
  deviations: [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
control:
  translated_values: [3.5, 3.9, 4.3, 4.7, 5.1, 5.5]
  permutation: reorder whole item records
expected_control_output:
  translated_deviations: [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
  permutation: deviations reorder with their items
```
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.relational.peer_relative_inference.support import (
    common_translation,
    data,
    diagnostics,
    model,
    permute_items,
    prediction,
    rmse,
    score,
    target,
    trainer,
    ungrouped_records,
    validate_score,
)

pytestmark = pytest.mark.proof


def test_ungrouped_mean_is_routed_back_to_each_item() -> None:
    """Infer each deviation from raw peers without a supplied collection mean."""

    seed = 3710
    lit.seed_everything(seed, workers=True)
    train = ungrouped_records(rows=1536, seed=3711)
    validate = ungrouped_records(rows=384, seed=3712)
    test = ungrouped_records(rows=768, seed=3713)
    configured = model(grouped=False, source="raw")
    fit = trainer(steps=900)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    predicted = prediction(configured, test)
    measured = score(train=train, test=test, predicted=predicted)
    translated = common_translation(test, seed=3714)
    translated_prediction = prediction(configured, translated)
    translated_score = score(train=train, test=translated, predicted=translated_prediction)
    permuted, order = permute_items(test, seed=3715)
    permuted_prediction = prediction(configured, permuted)
    permutation_drift = rmse(permuted_prediction, predicted[order]) / measured.baseline_rmse

    assert abs(float(target(train).mean())) < 1e-12
    assert target(translated).shape == target(test).shape
    assert rmse(target(translated), target(test)) == 0.0
    assert sorted(target(permuted).tolist()) == sorted(target(test).tolist())
    validate_score("ungrouped", measured)
    validate_score("common translation", translated_score)
    assert np.isfinite(permutation_drift), f"permutation drift is not finite: {permutation_drift}"

    details = (
        f"{diagnostics('ungrouped', measured)}\n"
        f"{diagnostics('common translation', translated_score)}\n"
        f"permutation drift={permutation_drift:.3f}"
    )
    assert measured.nrmse < 0.35, details
    assert translated_score.nrmse < 0.45, details
    assert permutation_drift < 0.20, details
