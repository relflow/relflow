"""Guard unseen-Hash keyed transfer between sibling branches.

Example
-------
```yaml
input:
  configuration:
    root_reduction: null
    source_reduction: null
    target_reduction: null
  source:
    - {entity_id: row-42-K1, value: 0.7}
    - {entity_id: row-42-K2, value: -0.2}
  target:
    - {entity_id: row-42-K2, value: null}
    - {entity_id: row-42-K1, value: null}
expected_output:
  target:
    - {entity_id: row-42-K2, value: -0.2}
    - {entity_id: row-42-K1, value: 0.7}
control:
  compressed_configuration:
    root_reduction: Attention()
    source_reduction: Attention()
    target_reduction: Attention()
  broken_target:
    - {entity_id: row-42-K1, retained_target_value: -0.2}
    - {entity_id: row-42-K2, retained_target_value: 0.7}
expected_control_output:
  compressed_configuration: worse than preserving all unseen-Hash tokens
  broken_identity: lookup accuracy collapses
```
"""

import numpy as np
import pytest

from proofs.relational.sibling_entity_transfer.support import PAIR_COUNT, TEST_ROWS, normalized_rmse, records, train

pytestmark = pytest.mark.proof


def test_sibling_branch_hash_entity_value_transfer() -> None:
    compressed = train(route="compressed")
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
    transfer_nrmse = normalized_rmse(configured, test)
    broken_nrmse = normalized_rmse(configured, broken)
    compressed_nrmse = normalized_rmse(compressed, test)

    assert np.isfinite([compressed_nrmse, transfer_nrmse, broken_nrmse]).all()
    assert transfer_nrmse <= compressed_nrmse - 0.15, (
        f"preserving Hash entity tokens did not improve generic transport: "
        f"compressed={compressed_nrmse:.4f}, preserved={transfer_nrmse:.4f}"
    )
    assert transfer_nrmse <= 1.10, f"unseen-Hash transfer regressed beyond its baseline: {transfer_nrmse:.4f}"
    details = (
        f"unseen-Hash sibling contract failed: compressed={compressed_nrmse:.4f}, "
        f"transfer={transfer_nrmse:.4f} (expected <= 0.25), "
        f"broken={broken_nrmse:.4f} (expected >= 0.80), gap={broken_nrmse - transfer_nrmse:.4f} "
        f"(expected >= 0.50)"
    )
    assert transfer_nrmse <= 0.25, details
    assert broken_nrmse >= 0.80, details
    assert broken_nrmse - transfer_nrmse >= 0.50, details
