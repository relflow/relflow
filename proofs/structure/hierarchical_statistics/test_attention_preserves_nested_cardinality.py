"""Retain nested cardinality through one-output Attention reductions.

Use learned Attention when an unordered hierarchy needs a compact summary,
but verify the required statistic at every level. This route relies on the
reducer's additive mass lane; normalized attention alone cannot distinguish
repeated identical values with different counts. Matched observations have
identical flattened amounts but different session boundaries. See
``support.py`` for the complete guide, remaining work, and promotion criteria.

Example
-------
Both rows contain the same six values; only their session boundaries differ:

```yaml
input:
  even_sessions:
    sessions:
      - {transactions: [{amount: 0.5}, {amount: 0.5}]}
      - {transactions: [{amount: 0.5}, {amount: 0.5}]}
      - {transactions: [{amount: 0.5}, {amount: 0.5}]}
  uneven_sessions:
    sessions:
      - {transactions: [{amount: 0.5}]}
      - {transactions: [{amount: 0.5}, {amount: 0.5}, {amount: 0.5}, {amount: 0.5}]}
      - {transactions: [{amount: 0.5}]}
expected_output:
  even_sessions: {largest_session_total: 1.0}
  uneven_sessions: {largest_session_total: 2.0}
control:
  flatten_either_row:
    - {amount: 0.5}
    - {amount: 0.5}
    - {amount: 0.5}
    - {amount: 0.5}
    - {amount: 0.5}
    - {amount: 0.5}
expected_control_output:
  rows_are_indistinguishable_without_session_boundaries: true
```

The reducer must retain local cardinality before the parent chooses the
largest session.

Run with::

    uv run pytest -n 0 proofs/structure/hierarchical_statistics/test_attention_preserves_nested_cardinality.py -q
"""

import lightning.pytorch as lit
import numpy as np
import pytest
import torch

import relflow as rf
from proofs.structure.hierarchical_statistics.support import (
    column,
    data,
    flatten,
    prediction,
    records,
    rmse,
    trainer,
)

pytestmark = pytest.mark.proof


def test_attention_preserves_cardinality_for_parent_maximum() -> None:
    seed = 2601
    lit.seed_everything(seed, workers=True)
    train = records(pairs=128, seed=44)
    validate = records(pairs=32, seed=55)
    test = records(pairs=64, seed=66)

    flat = flatten(test)["transactions"].to_pylist()
    if not all(flat[index] == flat[index + 1] for index in range(0, len(flat), 2)):
        raise RuntimeError("hierarchical-statistics generator produced a regrouping pair with different flat values")

    model = rf.Model(
        d_model=24,
        n_layers=1,
        n_heads=4,
        reduction=rf.Attention(),
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        sessions=rf.Branch(
            length=3,
            reduction=rf.Attention(),
            transactions=rf.Branch(
                length=4,
                reduction=rf.Attention(),
                amount=rf.Number,
            ),
        ),
        largest_session_total=rf.Number(mask=True, objective="mse"),
    )
    fit = trainer(80)
    fit.fit(model=model, datamodule=data(model, train, validate, seed))

    test_target = column(test, "largest_session_total")
    test_prediction = prediction(model, test, "sessions", "largest_session_total")
    pair_targets = test_target.reshape(-1, 2)
    flat_oracle = np.repeat(pair_targets.mean(axis=1), 2)
    flat_oracle_rmse = rmse(test_target, flat_oracle)
    test_rmse = rmse(test_target, test_prediction)
    predicted_pair_delta = np.abs(np.diff(test_prediction.reshape(-1, 2), axis=1)).mean()
    true_pair_delta = np.abs(np.diff(pair_targets, axis=1)).mean()
    diagnostics = (
        f"hierarchical Attention seed={seed}, steps={fit.global_step}, "
        f"rmse(test={test_rmse:.4f}, flat_oracle={flat_oracle_rmse:.4f}), "
        f"mean_pair_delta(predicted={predicted_pair_delta:.6f}, true={true_pair_delta:.4f})"
    )

    assert np.isfinite(test_rmse) and test_rmse < flat_oracle_rmse * 0.75, diagnostics
    assert predicted_pair_delta >= true_pair_delta * 0.75, diagnostics
