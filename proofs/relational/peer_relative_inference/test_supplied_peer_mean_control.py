"""A supplied peer mean should make coordinate-local subtraction easy.

This is deliberately diagnostic, not the recommended user schema.  It removes
aggregation and peer selection so a failure would implicate repeated decoding
or local sibling arithmetic before any collection-level reasoning is tested.

Example
-------
```yaml
input:
  items:
    - {value: 1.5, peer_mean: 2.5, deviation: null}
    - {value: 1.9, peer_mean: 2.5, deviation: null}
    - {value: 2.3, peer_mean: 2.5, deviation: null}
    - {value: 2.7, peer_mean: 2.5, deviation: null}
    - {value: 3.1, peer_mean: 2.5, deviation: null}
    - {value: 3.5, peer_mean: 2.5, deviation: null}
expected_output:
  deviations: [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
  rule: value minus the supplied same-coordinate peer_mean
control:
  removed_work: aggregation and peer selection
expected_control_output:
  meaning: success isolates local subtraction and repeated writeback
```
"""

from __future__ import annotations

import lightning.pytorch as lit
import pytest

from proofs.relational.peer_relative_inference.support import (
    data,
    diagnostics,
    model,
    prediction,
    score,
    trainer,
    ungrouped_records,
    validate_score,
)

pytestmark = pytest.mark.proof


def test_supplied_peer_mean_localizes_subtraction_and_writeback() -> None:
    """Subtract a visible same-coordinate mean and decode each item."""

    seed = 3700
    lit.seed_everything(seed, workers=True)
    train = ungrouped_records(rows=1024, seed=3701, supplied_mean=True)
    validate = ungrouped_records(rows=256, seed=3702, supplied_mean=True)
    test = ungrouped_records(rows=512, seed=3703, supplied_mean=True)
    configured = model(grouped=False, source="supplied_mean")
    fit = trainer(steps=600)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    measured = score(train=train, test=test, predicted=prediction(configured, test))
    validate_score("supplied peer mean", measured)
    assert measured.nrmse < 0.25, diagnostics("supplied peer mean", measured)
