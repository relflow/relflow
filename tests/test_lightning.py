import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
from lightning.pytorch.callbacks import BatchSizeFinder
from lightning.pytorch.tuner import Tuner

import relflow as rf


def model(merchant=rf.Category):
    configured = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        amount=rf.Number,
        merchant=merchant,
        label=rf.Boolean(mask=True),
    )
    configured.optimizer = rf.adamw(learning_rate=1e-3, fused=None)
    return configured


def data(configured):
    sample = pa.table(
        {"amount": [float(i) for i in range(32)], "merchant": ["alpha", "beta"] * 16, "label": [False, True] * 16}
    )
    return rf.ArrowDataModule(model=configured, train=sample, validate=sample, shuffle=False)


def test_batch_size_is_one_writable_model_owned_value():
    configured = model()
    datamodule = data(configured)
    datamodule.batch_size = 4
    assert configured.batch_size == 4
    assert next(iter(datamodule.train_dataloader())).source.num_rows == 4
    configured.batch_size = 8
    assert datamodule.batch_size == 8
    assert next(iter(datamodule.train_dataloader())).source.num_rows == 8


def test_checkpoint_schema_loads_without_custom_pickle_globals(tmp_path):
    configured = model()
    path = tmp_path / "model.ckpt"
    configured.save(path)
    checkpoint = torch.load(path, weights_only=True)
    schema = rf.Schema.model_validate(checkpoint["schema"])
    assert schema.model_dump(mode="python") == configured.schema.model_dump(mode="python")


@pytest.mark.parametrize("accelerator", ["cpu", "cuda", "mps", "auto"])
@pytest.mark.parametrize("kind", ["category", "set", "cluster"])
def test_native_batch_size_callback_restores_state_then_trains_bf16(tmp_path, accelerator, kind):
    if accelerator == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if accelerator == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if accelerator == "auto" and torch.cuda.device_count() > 1:
        pytest.skip("Lightning's batch-size finder requires a single device")
    field = {"category": rf.Category, "set": rf.Set, "cluster": rf.Cluster}[kind]
    options = {"bounds": 2, "revive_temperature": 0.0} if kind == "cluster" else {}
    configured = model(field(mask=rf.Mask(reconstruct=True), **options))
    initial = {name: value.clone() for name, value in configured.named_parameters()}
    finder = BatchSizeFinder(mode="binsearch", init_val=2, max_trials=2, max_val=8, margin=0)
    restored = []

    class CheckRestore(lit.Callback):
        def on_train_start(self, trainer, pl_module):
            assert rf.Number.normalization(pl_module, "/amount")["count"] == 0
            assert field.vocabulary(pl_module, "/merchant") == ()
            parameters = {id(parameter) for parameter in pl_module.parameters()}
            optimized = {id(parameter) for group in trainer.optimizers[0].param_groups for parameter in group["params"]}
            assert optimized == parameters
            for name, parameter in pl_module.named_parameters():
                torch.testing.assert_close(parameter.detach().cpu(), initial[name])
            restored.append(True)

    trainer = lit.Trainer(
        accelerator=accelerator,
        devices="auto" if accelerator == "auto" else 1,
        precision="bf16-mixed",
        callbacks=[finder, CheckRestore()],
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(configured, datamodule=data(configured))

    assert restored == [True]
    assert configured.batch_size == finder.optimal_batch_size == 4
    assert trainer.global_step == 1
    assert rf.Number.normalization(configured, "/amount")["count"] == 4
    assert field.vocabulary(configured, "/merchant") == ("alpha", "beta")
    assert any(
        not torch.equal(parameter.detach().cpu(), initial[name]) for name, parameter in configured.named_parameters()
    )
    assert configured.nodes["/amount"].embedder.normalizer.mean.dtype == torch.float32
    assert not list(tmp_path.glob(".scale_batch_size_*.ckpt"))


def test_native_tuner_can_prepare_a_separate_training_run(tmp_path):
    configured = model()
    datamodule = data(configured)
    tuner_trainer = lit.Trainer(
        accelerator="cpu",
        devices=1,
        precision="bf16-mixed",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    batch_size = Tuner(tuner_trainer).scale_batch_size(
        configured, datamodule=datamodule, mode="binsearch", init_val=2, max_trials=2, max_val=8, margin=0
    )
    assert configured.batch_size == batch_size == 4
    assert rf.Number.normalization(configured, "/amount")["count"] == 0

    trainer = lit.Trainer(
        accelerator="cpu",
        devices="auto",
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(configured, datamodule=datamodule)
    assert trainer.global_step == 1
    assert rf.Number.normalization(configured, "/amount")["count"] == 4
