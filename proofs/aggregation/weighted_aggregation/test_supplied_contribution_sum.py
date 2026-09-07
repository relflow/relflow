"""A supplied item-local product isolates fixed-width sum reduction.

This positive control deliberately derives ``contribution = value * weight``
in synthetic data. Passing here means a raw-case failure should be localized
to multiplication or value/weight alignment rather than basic summation.

Example
-------
This control starts after multiplication has already happened:

```yaml
input:
  items:
    - {contribution: 0.4}
    - {contribution: -0.4}
    - {contribution: 0.3}
    - {contribution: 0.1}
    - {contribution: -0.2}
    - {contribution: 0.3}
expected_output:
  weighted_sum: 0.5
```

If this passes while the raw-pair proof fails, investigate pair binding or
multiplication rather than sum reduction.
"""

from __future__ import annotations

import lightning.pytorch as lit
import pytest

from proofs.aggregation.weighted_aggregation.support import (
    ITEMS,
    data,
    diagnostics,
    model,
    prediction,
    score,
    trainer,
    weighted_records,
)

pytestmark = pytest.mark.proof


def test_supplied_item_contributions_can_be_summed() -> None:
    """Attention learns a fixed-width sum after products are supplied."""

    seed = 3300
    lit.seed_everything(seed, workers=True)
    train = weighted_records(rows=768, length=ITEMS, seed=3301, source="contribution")
    validate = weighted_records(rows=192, length=ITEMS, seed=3302, source="contribution")
    test = weighted_records(rows=384, length=ITEMS, seed=3303, source="contribution")
    configured = model(source="contribution")
    fit = trainer(steps=500)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    measured = score(
        train=train,
        test=test,
        predicted=prediction(configured, test, "weighted_sum"),
        target="weighted_sum",
    )
    assert measured.nrmse < 0.25, diagnostics("supplied contribution weighted sum", measured)
