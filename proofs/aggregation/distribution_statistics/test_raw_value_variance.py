"""A raw repeated number should reveal spread beyond its mean.

Locations and spreads are independent, and every generated row has deviations
whose mean is exactly zero. The held-out paired intervention gives two rows the
same mean and shape but changes only scale, ruling out a mean-only solution.

Example
-------

```yaml
input:
  rows:
    - items: [{value: 0.10}, {value: 0.50}, {value: 0.10}, {value: 0.50},
              {value: 0.10}, {value: 0.50}, {value: 0.10}, {value: 0.50}]
    - items: [{value: -0.80}, {value: 1.40}, {value: -0.80}, {value: 1.40},
              {value: -0.80}, {value: 1.40}, {value: -0.80}, {value: 1.40}]
expected_output:
  rows: [{variance: 0.04}, {variance: 1.21}]
shared_mean: 0.30
```

A mean-only representation would make both rows look identical and fail this
separation.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pytest

from proofs.aggregation.distribution_statistics.support import (
    ITEMS,
    data,
    diagnostics,
    dispersion_records,
    model,
    prediction,
    score,
    trainer,
)

pytestmark = pytest.mark.proof


def test_raw_values_reveal_variance_for_matched_means() -> None:
    """Learn a second central moment without receiving squared deviations."""

    seed = 3504
    lit.seed_everything(seed, workers=True)
    train = dispersion_records(rows=1280, seed=3505, source="raw")
    validate = dispersion_records(rows=320, seed=3506, source="raw")
    test = dispersion_records(rows=640, seed=3507, source="raw")
    configured = model(source="raw", target="variance")
    fit = trainer(steps=900)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    measured = score(
        train=train,
        test=test,
        predicted=prediction(configured, test, "variance"),
        target="variance",
    )
    basis = np.linspace(-1.0, 1.0, ITEMS)
    basis -= basis.mean()
    basis /= np.sqrt(np.mean(np.square(basis)))
    matched = pa.Table.from_pylist(
        [{"items": [{"value": float(0.3 + scale * value)} for value in basis]} for scale in (0.20, 1.10)]
    )
    paired_prediction = prediction(configured, matched, "variance")
    details = (
        f"{diagnostics('raw matched-mean variance', measured)}\n"
        f"matched-mean predictions(low={paired_prediction[0]:.4f}, "
        f"high={paired_prediction[1]:.4f})"
    )
    assert measured.nrmse < 0.30, details
    assert paired_prediction[1] - paired_prediction[0] > 0.65, details
