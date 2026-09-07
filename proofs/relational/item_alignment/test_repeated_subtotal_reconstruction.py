"""Verify that repeated subtotal reconstruction stays item-aligned.

Example
-------
Each masked subtotal belongs to the quantity and price at the same coordinate:

```yaml
input:
  items:
    - {quantity: 1.5, unit_price: 1.2, subtotal: null}
    - {quantity: 0.5, unit_price: -1.0, subtotal: null}
expected_output:
  items:
    - {subtotal: 1.8}
    - {subtotal: -0.5}
control:
  change: permute only target subtotals
  retained_input_items: true
  target_subtotals: [-0.5, 1.8]
expected_control_output:
  dataset_nrmse: increases by at least 0.50
```

Empty and short rows also verify that only real coordinates are marked inferred;
padding remains part of the fixed output schema but is not a prediction.
"""

import lightning.pytorch as lit
import pytest
import torch

import relflow as rf
from proofs.relational.item_alignment.support import LENGTH, actual, predicted, records, requests, rmse

pytestmark = pytest.mark.proof


def test_repeated_subtotal_reconstruction_stays_item_aligned() -> None:
    seed = 31
    lit.seed_everything(seed, workers=True)
    train = records(rows=4096, seed=seed + 1)
    validate = records(rows=1024, seed=seed + 2)
    test = records(rows=2048, seed=seed + 3)
    broken = records(rows=2048, seed=seed + 3, permute_targets=True)
    assert requests(test).equals(requests(broken))

    model = rf.Model(
        name="order",
        d_model=48,
        n_layers=1,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.Adam(module.parameters(), lr=1e-3),
        items=rf.Branch(
            length=LENGTH,
            overflow="error",
            n_layers=2,
            quantity=rf.Number,
            unit_price=rf.Number,
            subtotal=rf.Number(mask=True, objective="mse"),
        ),
    )
    data = rf.ArrowDataModule(
        model=model,
        train=train,
        validate=validate,
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=25,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)

    target = actual(test)
    estimate, output_rows = predicted(model, test)
    broken_target = actual(broken)
    baseline = rmse(target, float(actual(train).mean()))
    aligned_nrmse = rmse(target, estimate) / baseline
    broken_nrmse = rmse(broken_target, estimate) / baseline

    assert all(len(row) == LENGTH for row in output_rows), "branch predictions did not retain their fixed schema length"
    input_lengths = [len(row) for row in test["items"].to_pylist()]
    assert [[bool(coordinate["inferred"]) for coordinate in row] for row in output_rows] == [
        [True] * length + [False] * (LENGTH - length) for length in input_lengths
    ], "inferred prediction coordinates did not match valued and padded branch positions"
    assert aligned_nrmse <= 0.25, f"item-aligned subtotal nRMSE={aligned_nrmse:.4f}, expected <= 0.25"
    assert broken_nrmse >= aligned_nrmse + 0.50, (
        f"permuting targets did not remove coordinate-level skill: "
        f"aligned={aligned_nrmse:.4f}, permuted={broken_nrmse:.4f}"
    )
