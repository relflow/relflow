"""Test visibly selected reductions on fixed-length bags.

Example
-------
The same eight-value bag is emitted once for each requested operation:

```yaml
input:
  rows:
    - {operation: sum, items: &bag [-1.0, 0.1, 0.2, 0.4, 0.0, 0.3, 0.5, 1.2]}
    - {operation: mean, items: *bag}
    - {operation: min, items: *bag}
    - {operation: max, items: *bag}
expected_output:
  rows:
    - {answer: 1.7000}
    - {answer: 0.2125}
    - {answer: -1.0000}
    - {answer: 1.2000}
control:
  apply_to_each_row:
    permuted_items: [0.3, -1.0, 1.2, 0.2, 0.5, 0.0, 0.4, 0.1]
    hidden_operation: null
expected_control_output:
  permutation: answers remain unchanged
  hidden_operation: identical requests receive one repeated prediction and cannot satisfy all targets
```
"""

import lightning.pytorch as lit
import numpy as np
import pytest

from proofs.relational.operation_conditioned_reduction.support import (
    OPERATIONS,
    diagnostics,
    fit,
    hide_operation,
    model,
    operation_table,
    permute_items,
    predict,
    scores,
)

pytestmark = pytest.mark.proof


def test_operation_conditioned_reduction_at_fixed_length() -> None:
    """A fixed-length bag supports four visibly requested numeric reductions."""

    lit.seed_everything(15, workers=True)
    train = operation_table(bags=768, length=8, seed=150)
    validate = operation_table(bags=96, length=8, seed=151)
    test = operation_table(bags=192, length=8, seed=152)

    configured = model(length=8)
    fit(configured, train, validate, steps=800)
    intact_prediction = predict(configured, test)
    result = scores(
        train=train,
        test=test,
        predicted=intact_prediction,
        keys=("operation",),
    )
    permuted_prediction = predict(configured, permute_items(test, seed=153))
    permuted = scores(
        train=train,
        test=test,
        predicted=permuted_prediction,
        keys=("operation",),
    )
    hidden_prediction = predict(configured, hide_operation(test))
    hidden = scores(
        train=train,
        test=test,
        predicted=hidden_prediction,
        keys=("operation",),
    )

    details = (
        f"intact:\n{diagnostics(result)}\npermuted items:\n{diagnostics(permuted)}"
        f"\nhidden operation:\n{diagnostics(hidden)}"
    )
    target_scale = float(np.std(test["answer"].to_numpy(zero_copy_only=False)))
    permutation_delta = float(np.sqrt(np.mean(np.square(intact_prediction - permuted_prediction)))) / target_scale
    assert all(score.nrmse <= 0.20 for score in result.values()), details
    assert all(score.nrmse <= 0.25 for score in permuted.values()), details
    assert permutation_delta <= 0.05, f"{details}\npermutation prediction delta={permutation_delta:.4f} target SD"
    assert np.max(np.ptp(hidden_prediction.reshape(-1, len(OPERATIONS)), axis=1)) <= 1e-6, details
    assert sum(score.nrmse >= 0.75 for score in hidden.values()) >= 3, details
