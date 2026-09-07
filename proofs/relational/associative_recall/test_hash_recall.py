"""Test same-branch associative recall by an unseen Hash key.

Example
-------
Source and query records are shuffled together in one branch:

```yaml
input:
  memory:
    - {role: source, entity_id: K1, value: 0.4}
    - {role: query, entity_id: K2, value: null}
    - {role: source, entity_id: K2, value: -0.8}
    - {role: query, entity_id: K1, value: null}
expected_output:
  query_values:
    - {entity_id: K2, value: -0.8}
    - {entity_id: K1, value: 0.4}
control:
  change: rotate query identities while retaining target values
expected_control_output:
  behavior: identity lookup accuracy should collapse
```
"""

import numpy as np
import pytest

from proofs.relational.associative_recall.support import PAIR_COUNT, TEST_ROWS, normalized_rmse, records, train

pytestmark = pytest.mark.proof


def test_same_branch_hash_associative_recall() -> None:
    configured = train(route="preserved")
    test = records(
        rows=TEST_ROWS,
        pairs=PAIR_COUNT,
        seed=20,
        namespace="test",
    )
    broken = records(
        rows=TEST_ROWS,
        pairs=PAIR_COUNT,
        seed=20,
        namespace="test",
        broken_identity=True,
    )

    recall_nrmse = normalized_rmse(configured, test)
    broken_nrmse = normalized_rmse(configured, broken)

    assert np.isfinite([recall_nrmse, broken_nrmse]).all()
    assert recall_nrmse <= 0.90, f"same-branch partial recall regressed beyond calibration: {recall_nrmse:.4f}"
    details = (
        f"same-branch identity lookup contract failed: recall={recall_nrmse:.4f} (expected <= 0.25), "
        f"broken={broken_nrmse:.4f} (expected >= 0.80), gap={broken_nrmse - recall_nrmse:.4f} "
        f"(expected >= 0.50)"
    )
    assert recall_nrmse <= 0.25, details
    assert broken_nrmse >= 0.80, details
    assert broken_nrmse - recall_nrmse >= 0.50, details
