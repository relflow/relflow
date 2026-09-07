"""One learned summary can retain context needed by coordinate queries.

This variant uses exactly the raw grouped problem from the token-preserving
case, but reduces the coordinate-contextualized ``group`` and ``value`` tokens
to one learned output. Each repeated target still has its aligned visible
siblings as query context, while the branch output carries shared collection
context.

Example
-------
```yaml
input:
  configuration:
    items_reduction: Attention(n_outputs=1)
  items:
    - {group: A, value: 2.0, deviation: null}
    - {group: B, value: 1.0, deviation: null}
    - {group: C, value: -3.0, deviation: null}
    - {group: A, value: 4.0, deviation: null}
    - {group: C, value: -1.0, deviation: null}
    - {group: B, value: -1.0, deviation: null}
expected_output:
  deviations: [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
control:
  change: rotate only group labels while retaining values and targets
expected_control_output:
  behavior: error rises when the group/value relationship is broken
```
"""

from __future__ import annotations

import lightning.pytorch as lit
import pytest

from proofs.relational.peer_relative_inference.support import (
    corrupt_groups,
    data,
    diagnostics,
    grouped_records,
    implied_deviation,
    model,
    prediction,
    rmse,
    score,
    target,
    trainer,
    validate_score,
)

pytestmark = pytest.mark.proof


def test_single_query_summary_preserves_group_membership() -> None:
    """A shared summary and aligned queries recover peer-relative values."""

    seed = 3720
    lit.seed_everything(seed, workers=True)
    train = grouped_records(rows=2048, seed=3721)
    validate = grouped_records(rows=512, seed=3722)
    test = grouped_records(rows=768, seed=3723)
    configured = model(grouped=True, source="raw", compress_siblings=True)
    fit = trainer(steps=1200)
    fit.fit(configured, datamodule=data(configured, train, validate, seed=seed))

    measured = score(train=train, test=test, predicted=prediction(configured, test))
    corrupted = corrupt_groups(test)
    corrupted_score = score(train=train, test=corrupted, predicted=prediction(configured, corrupted))
    validate_score("single-query grouped", measured)
    validate_score("single-query corrupted labels", corrupted_score)
    oracle_corruption = rmse(implied_deviation(corrupted), target(test)) / measured.baseline_rmse
    assert oracle_corruption > 0.75, f"weak group corruption: oracle nRMSE={oracle_corruption:.3f}"

    details = (
        f"{diagnostics('single-query grouped', measured)}\n"
        f"{diagnostics('single-query corrupted labels', corrupted_score)}\n"
        f"oracle corruption={oracle_corruption:.3f}"
    )
    assert measured.nrmse <= 0.20, details
    assert corrupted_score.nrmse >= 0.90, details
    assert corrupted_score.nrmse >= measured.nrmse + 0.75, details
