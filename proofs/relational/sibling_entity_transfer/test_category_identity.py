"""Guard persistent-Category keyed transfer between sibling branches.

Example
-------
```yaml
input:
  configuration:
    root_reduction: null
    source_reduction: null
    target_reduction: null
  source:
    - {entity_id: entity-A, value: 0.7}
    - {entity_id: entity-B, value: -0.2}
  target:
    - {entity_id: entity-B, value: null}
    - {entity_id: entity-A, value: null}
expected_output:
  target:
    - {entity_id: entity-B, value: -0.2}
    - {entity_id: entity-A, value: 0.7}
control:
  compressed_configuration:
    root_reduction: Attention()
    source_reduction: Attention()
    target_reduction: Attention()
  broken_target:
    - {entity_id: entity-A, retained_target_value: -0.2}
    - {entity_id: entity-B, retained_target_value: 0.7}
expected_control_output:
  compressed_configuration: remains accurate in this bounded persistent-Category proof
  broken_identity: lookup accuracy collapses
```
"""

import numpy as np
import pytest

from proofs.relational.sibling_entity_transfer.support import PAIR_COUNT, TEST_ROWS, normalized_rmse, records, train

pytestmark = pytest.mark.proof


def test_sibling_branch_persistent_category_entity_value_transfer() -> None:
    compressed = train(identity="category", route="compressed")
    configured = train(identity="category", route="preserved")
    test = records(
        rows=TEST_ROWS,
        pairs=PAIR_COUNT,
        seed=20,
        namespace="test",
        identity="category",
    )
    broken = records(
        rows=TEST_ROWS,
        pairs=PAIR_COUNT,
        seed=20,
        namespace="test",
        identity="category",
        broken_identity=True,
    )
    transfer_nrmse = normalized_rmse(configured, test)
    broken_nrmse = normalized_rmse(configured, broken)
    compressed_nrmse = normalized_rmse(compressed, test)

    assert np.isfinite([compressed_nrmse, transfer_nrmse, broken_nrmse]).all()
    assert compressed_nrmse <= 0.25, (
        "persistent Category identities should remain routable through fixed-width compression: "
        f"compressed={compressed_nrmse:.4f}"
    )
    assert transfer_nrmse <= 1.10, f"persistent-Category transfer regressed beyond its baseline: {transfer_nrmse:.4f}"
    details = (
        f"persistent-Category sibling contract failed: compressed={compressed_nrmse:.4f}, "
        f"transfer={transfer_nrmse:.4f} "
        f"(expected <= 0.25), broken={broken_nrmse:.4f} (expected >= 0.80), "
        f"gap={broken_nrmse - transfer_nrmse:.4f} (expected >= 0.50)"
    )
    assert transfer_nrmse <= 0.25, details
    assert broken_nrmse >= 0.80, details
    assert broken_nrmse - transfer_nrmse >= 0.50, details
