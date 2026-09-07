"""Supplied squared deviations isolate collection averaging.

This deliberately spoon-fed control is useful for diagnosis, not the desired
user schema. Passing it shows that a failure on raw variance belongs to the
learned item transform or centering interaction, rather than to averaging
already sufficient per-item statistics.

Example
-------

```yaml
illustrative_source:
  values: [-1, 1, -1, 1, -1, 1, -1, 1]
  mean: 0
input:
  items:
    - {squared_deviation: 1.0}
    - {squared_deviation: 1.0}
    - {squared_deviation: 1.0}
    - {squared_deviation: 1.0}
    - {squared_deviation: 1.0}
    - {squared_deviation: 1.0}
    - {squared_deviation: 1.0}
    - {squared_deviation: 1.0}
expected_output: {variance: 1.0}
```

The proof receives the squared deviations, so it tests averaging but does not
prove that RelFlow learned centering or squaring from raw values.
"""

from __future__ import annotations

import lightning.pytorch as lit
import pytest

from proofs.aggregation.distribution_statistics.support import (
    data,
    diagnostics,
    dispersion_records,
    model,
    prediction,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_supplied_squared_deviations_reduce_to_variance() -> None:
    """Mean learns population variance once squared deviations are supplied."""

    seed = 3500
    lit.seed_everything(seed, workers=True)
    train = dispersion_records(rows=768, seed=3501, source="squared_deviation")
    validate = dispersion_records(rows=192, seed=3502, source="squared_deviation")
    test = dispersion_records(rows=384, seed=3503, source="squared_deviation")
    configured = model(source="squared_deviation", target="variance")
    fit = trainer(steps=350)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    measured = score(
        train=train,
        test=test,
        predicted=prediction(configured, test, "variance"),
        target="variance",
    )
    assert measured.nrmse < 0.20, diagnostics("supplied squared-deviation variance", measured)
