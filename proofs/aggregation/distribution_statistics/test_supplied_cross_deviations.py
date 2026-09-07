"""Supplied centered products isolate covariance reduction.

This positive control provides each item's
``(x - mean(x)) * (y - mean(y))`` value. It is intentionally diagnostic: a
pass here beside a failing raw covariance run would identify missing
sibling-field interaction rather than an inability to average a cross-moment;
the adjacent raw route currently passes as well.

Example
-------

```yaml
illustrative_source:
  x: [-1, 1, -1, 1, -1, 1, -1, 1]
  y: [-1.4, 0.2, -0.2, 1.4, -1.4, 0.2, -0.2, 1.4]
input:
  items:
    - {cross_deviation: 1.4}
    - {cross_deviation: 0.2}
    - {cross_deviation: 0.2}
    - {cross_deviation: 1.4}
    - {cross_deviation: 1.4}
    - {cross_deviation: 0.2}
    - {cross_deviation: 0.2}
    - {cross_deviation: 1.4}
expected_output: {covariance: 0.8}
```

This control receives ``input.items``, not the raw illustrative ``x`` and
``y`` values.
"""

from __future__ import annotations

import lightning.pytorch as lit
import pytest

from proofs.aggregation.distribution_statistics.support import (
    covariance_sufficient_records,
    data,
    diagnostics,
    model,
    prediction,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_supplied_cross_deviations_reduce_to_covariance() -> None:
    """Mean learns covariance once centered products are supplied."""

    seed = 3513
    lit.seed_everything(seed, workers=True)
    train = covariance_sufficient_records(rows=768, seed=3514)
    validate = covariance_sufficient_records(rows=192, seed=3515)
    test = covariance_sufficient_records(rows=384, seed=3516)
    configured = model(source="cross_deviation", target="covariance")
    fit = trainer(steps=350)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    measured = score(
        train=train,
        test=test,
        predicted=prediction(configured, test, "covariance"),
        target="covariance",
    )
    assert measured.nrmse < 0.20, diagnostics("supplied cross-deviation covariance", measured)
