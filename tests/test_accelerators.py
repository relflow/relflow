"""Real accelerator execution complements CPU hardware-selection unit tests."""

import math

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch

import relflow as rf


@pytest.mark.integration
@pytest.mark.parametrize(
    "accelerator",
    [
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_accelerator_fit_and_prediction_use_real_parameters(tmp_path, accelerator):
    class Placement(lit.Callback):
        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            assert pl_module.device.type == accelerator
            assert batch.tensors["/amount"].content.device.type == accelerator

    model = rf.Model.xs(d_model=8, embed=True, amount=rf.Number, label=rf.Boolean(mask=True))
    rows = pa.table({"amount": [1.0, 3.0], "label": [False, True]})
    data = rf.ArrowDataModule(model, train=rows, validate=rows, shuffle=False)
    before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}

    model.fit(
        data,
        batch_size=2,
        accelerator=accelerator,
        devices=1,
        fused=accelerator != "mps",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        early_stopping=False,
        rollback=False,
        enable_progress_bar=False,
        callbacks=[Placement()],
        default_root_dir=tmp_path,
    )

    assert model.trainer.global_step == 1
    assert model.trainer.precision in {"bf16-mixed", "32-true"}
    assert any(not torch.equal(parameter.detach().cpu(), before[name]) for name, parameter in model.named_parameters())
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    model.to(accelerator)
    result = model.predict(rows)
    assert model.device.type == accelerator
    assert result.num_rows == 2
    assert all(
        math.isfinite(value)
        for row in result["representations"].to_pylist()
        for representation in row
        for value in representation["embedding"]
    )
