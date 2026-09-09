"""Prove that an explicit null is distinguishable from a real zero.

Claim
-----
A Number's validity state carries information independently of its numeric
content.

Use
---
Keep Arrow nulls intact when missingness may be predictive and let RelFlow
encode the Number state.

Avoid
-----
Do not pre-fill nulls with a real zero before encoding. That erases the
distinction this proof establishes and destroys its only predictive signal.

Why
---
RelFlow represents validity separately from continuous content, so null and
zero are observably different even when every present value is zero.

Example
-------
In the signal data, validity is the only useful feature:

```yaml
input:
  present_row: {measurement: 0.0}
  missing_row: {measurement: null}
expected_output:
  present_row: {target: false}
  missing_row: {target: true}
control:
  action: replace_null_with_0.0
expected_control_output:
  inputs_are_identical_while_targets_differ: true
  held_out_performance: chance
```

Protocol and gate
-----------------
Train on 1,024 balanced rows, validate on 512, and test on 2,048. A paired
control keeps zero content but makes validity independent of the target. Both
models train for eight deterministic CPU epochs. Evaluate the trained signal
model once more after replacing every null with zero.

The signal must reach AUC >= 0.99 and accuracy >= 0.98. Independent-validity
and null-to-zero controls must remain within [0.42, 0.58] AUC and [0.45, 0.55]
accuracy, with a signal/control AUC gap of at least 0.40.

Status and current evidence
---------------------------
This proof is provisional. All intended variants pass for one seed: Arrow null
state predicts the target above both gates despite every present Number being
zero; independent validity remains at chance; and replacing nulls with real
zeros returns the trained model to chance.

Further work
------------
Measure AUC and accuracy distributions for all three variants across seeds,
and freeze chance bands only after that stability run.

Promotion criteria
------------------
Pass three paired core seeds, then calibrate the gates with at least ten seeds
without weakening either negative control.

Run with::

    uv run pytest -n 0 proofs/state/missingness/test_null_versus_zero.py -q
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf

pytestmark = pytest.mark.proof


def records(*, rows: int, seed: int, signal: bool) -> pa.Table:
    rng = np.random.default_rng(seed)
    target = np.tile(np.array([False, True]), (rows + 1) // 2)[:rows]
    rng.shuffle(target)
    state = target if signal else rng.integers(0, 2, size=rows).astype(bool)
    measurement = pa.array([None if missing else 0.0 for missing in state], type=pa.float64())
    return pa.table({"measurement": measurement, "target": target})


def score(model: rf.Model, table: pa.Table) -> tuple[float, float]:
    data = rf.ArrowDataModule(
        model=model,
        test=table,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return (
        float(metrics["state.target/test.auc.content"]),
        float(metrics["state.target/test.accuracy@0.5.content"]),
    )


def fit_and_test(*, signal: bool, seed: int = 11) -> tuple[rf.Model, pa.Table, float, float]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        name="state",
        d_model=16,
        n_layers=1,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=5e-3),
        measurement=rf.Number,
        target=rf.Boolean(mask=True),
    )
    test = records(rows=2048, seed=seed + 3, signal=signal)
    data = rf.ArrowDataModule(
        model=model,
        train=records(rows=1024, seed=seed + 1, signal=signal),
        validate=records(rows=512, seed=seed + 2, signal=signal),
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=8,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)
    auc, accuracy = score(model, test)
    return model, test, auc, accuracy


def test_number_state_preserves_null_versus_zero() -> None:
    model, test, state_auc, state_accuracy = fit_and_test(signal=True)
    _, _, control_auc, control_accuracy = fit_and_test(signal=False)
    measurement_index = test.schema.get_field_index("measurement")
    prefilled = test.set_column(
        measurement_index,
        "measurement",
        pa.array(np.zeros(len(test)), type=pa.float64()),
    )
    prefilled_auc, prefilled_accuracy = score(model, prefilled)

    assert state_auc >= 0.99, f"null state did not predict the held-out target: AUC={state_auc:.4f}"
    assert state_accuracy >= 0.98, (
        f"null state did not reach the fixed-threshold accuracy gate: accuracy={state_accuracy:.4f}"
    )
    assert 0.42 <= control_auc <= 0.58, f"independent state control escaped chance: AUC={control_auc:.4f}"
    assert 0.45 <= control_accuracy <= 0.55, (
        f"independent state control escaped chance at threshold 0.5: accuracy={control_accuracy:.4f}"
    )
    assert 0.42 <= prefilled_auc <= 0.58, f"null-to-zero control escaped chance: AUC={prefilled_auc:.4f}"
    assert 0.45 <= prefilled_accuracy <= 0.55, (
        f"null-to-zero control escaped chance at threshold 0.5: accuracy={prefilled_accuracy:.4f}"
    )
    assert state_auc - control_auc >= 0.40, (
        f"state/content boundary was not identifiable: state={state_auc:.4f}, control={control_auc:.4f}"
    )
