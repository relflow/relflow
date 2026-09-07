"""Mean is the right reduction for an average, but cannot also preserve count.

This paired positive and negative case is intentionally a user guide: the
average target should converge, while one and six repetitions of the same
value must remain indistinguishable to a downstream total decoder.

Example
-------
With branch attention disabled, equal repeated tokens have the same Mean:

```yaml
input:
  one_item: {transactions: [{amount: 1.25}]}
  six_items:
    transactions: [{amount: 1.25}, {amount: 1.25}, {amount: 1.25},
                   {amount: 1.25}, {amount: 1.25}, {amount: 1.25}]
expected_output:
  one_item: {mean_amount: 1.25, global_total: 1.25}
  six_items: {mean_amount: 1.25, global_total: 7.50}
expected_control_output:
  mean_reducer_summary: identical
  learned_total_predictions: identical
```

The final equality demonstrates why Mean is unsafe when the parent needs a
count or sum.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from proofs.structure.hierarchical_statistics.support import (
    TRANSACTIONS,
    column,
    data,
    mean_records,
    prediction,
    rmse,
    trainer,
)

pytestmark = pytest.mark.proof


def test_mean_learns_average_but_erases_cardinality() -> None:
    """Mean keeps an average while making repeated equal inputs count-blind."""
    seed = 2599
    lit.seed_everything(seed, workers=True)

    train = mean_records(rows=1024, seed=1)
    validate = mean_records(rows=256, seed=2)
    test = mean_records(rows=512, seed=3)
    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        transactions=rf.Branch(
            length=TRANSACTIONS,
            attention="none",
            reduction=rf.Mean(),
            amount=rf.Number,
        ),
        mean_amount=rf.Number(mask=True, objective="mse"),
        global_total=rf.Number(mask=True, objective="mse"),
    )
    fit = trainer(300)
    fit.fit(model=model, datamodule=data(model, train, validate, seed))

    test_mean = column(test, "mean_amount")
    predicted_mean = prediction(model, test, "transactions", "mean_amount")
    mean_baseline = rmse(test_mean, float(column(train, "mean_amount").mean()))
    mean_nrmse = rmse(test_mean, predicted_mean) / mean_baseline

    paired = pa.Table.from_pylist(
        [
            {"transactions": [{"amount": 1.25}]},
            {"transactions": [{"amount": 1.25}] * TRANSACTIONS},
        ]
    )
    paired_total = prediction(model, paired, "transactions", "global_total")

    diagnostics = (
        f"hierarchical_statistics Mean seed={seed}, steps={fit.global_step}, "
        f"mean_nrmse={mean_nrmse:.4f}, same-value total predictions={paired_total.tolist()}"
    )
    assert mean_nrmse < 0.25, diagnostics
    assert np.allclose(paired_total[0], paired_total[1], atol=1e-6, rtol=0.0), diagnostics
