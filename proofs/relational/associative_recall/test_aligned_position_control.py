"""Show that aligned copying can succeed without consulting identity.

Example
-------
The source and query halves occupy the same relative positions:

```yaml
input:
  memory:
    - {role: source, entity_id: K1, value: 0.4}
    - {role: source, entity_id: K2, value: -0.8}
    - {role: query, entity_id: K1, value: null}
    - {role: query, entity_id: K2, value: null}
expected_output:
  query_values:
    - {entity_id: K1, value: 0.4}
    - {entity_id: K2, value: -0.8}
control:
  query_entity_ids: [K2, K1]
  retained_target_values: [0.4, -0.8]
expected_control_output:
  query_values: [0.4, -0.8]
  meaning: unchanged output demonstrates positional copying, not identity recall
```
"""

import pytest

from proofs.relational.associative_recall.support import PAIR_COUNT, TEST_ROWS, normalized_rmse, records, train

pytestmark = pytest.mark.proof


def test_same_branch_aligned_position_copy_control() -> None:
    configured = train(aligned=True, route="compressed")
    test = records(
        rows=TEST_ROWS,
        pairs=PAIR_COUNT,
        seed=20,
        namespace="test",
        aligned=True,
    )
    broken = records(
        rows=TEST_ROWS,
        pairs=PAIR_COUNT,
        seed=20,
        namespace="test",
        broken_identity=True,
        aligned=True,
    )
    copy_nrmse = normalized_rmse(configured, test)
    broken_nrmse = normalized_rmse(configured, broken)

    assert copy_nrmse <= 0.25, f"aligned same-branch position-copy nRMSE={copy_nrmse:.4f}, expected <= 0.25"
    assert broken_nrmse <= 0.25, (
        f"aligned position copy incorrectly depended on identity: broken-key nRMSE={broken_nrmse:.4f}"
    )
    assert abs(copy_nrmse - broken_nrmse) <= 0.10, (
        f"changing keys altered an explicitly positional control: valid={copy_nrmse:.4f}, broken={broken_nrmse:.4f}"
    )
