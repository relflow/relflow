"""Attention can learn a one-level fixed-width sum when all values are visible.

This is the positive control for the harder nested hierarchy case. It proves
that failure to recover nested session statistics is not merely an inability
to regress a numerical sum at all.

Example
-------
All six visible amounts contribute to one fixed-width total:

```yaml
input:
  transactions:
    - {amount: 0.8}
    - {amount: -0.4}
    - {amount: 0.2}
    - {amount: 0.5}
    - {amount: -0.5}
    - {amount: 0.9}
expected_output:
  global_total: 1.5
control:
  action: reorder_complete_transactions
expected_control_output:
  global_total: 1.5
```
"""

from __future__ import annotations

import lightning.pytorch as lit
import pytest
import torch

import relflow as rf
from proofs.structure.hierarchical_statistics.support import (
    TRANSACTIONS,
    column,
    data,
    prediction,
    rmse,
    sum_records,
    trainer,
)

pytestmark = pytest.mark.proof


def test_attention_learns_fixed_width_global_sum() -> None:
    """A fixed-width branch retains enough information to learn its total."""
    seed = 2600
    lit.seed_everything(seed, workers=True)

    train = sum_records(rows=512, seed=11)
    validate = sum_records(rows=128, seed=22)
    test = sum_records(rows=256, seed=33)
    model = rf.Model(
        d_model=32,
        n_layers=2,
        n_heads=4,
        reduction=rf.Attention(n_layers=2),
        batch_size=64,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        transactions=rf.Branch(
            length=TRANSACTIONS,
            n_layers=2,
            reduction=rf.Attention(n_layers=2),
            amount=rf.Number,
        ),
        global_total=rf.Number(mask=True, objective="mse"),
    )
    fit = trainer(400)
    fit.fit(model=model, datamodule=data(model, train, validate, seed))

    train_target = column(train, "global_total")
    validate_target = column(validate, "global_total")
    test_target = column(test, "global_total")
    validate_prediction = prediction(model, validate, "transactions", "global_total")
    test_prediction = prediction(model, test, "transactions", "global_total")
    constant_rmse = rmse(test_target, float(train_target.mean()))
    validate_rmse = rmse(validate_target, validate_prediction)
    test_rmse = rmse(test_target, test_prediction)
    nrmse = test_rmse / constant_rmse

    diagnostics = (
        f"hierarchical_statistics one-level control seed={seed}, steps={fit.global_step}, "
        f"rmse(train_mean={constant_rmse:.4f}, validate={validate_rmse:.4f}, "
        f"test={test_rmse:.4f}, nrmse={nrmse:.4f}), target_std={test_target.std():.4f}"
    )
    assert nrmse < 0.25, diagnostics
