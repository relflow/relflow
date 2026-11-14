import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
from lightning.pytorch.callbacks import BatchSizeFinder
from lightning.pytorch.tuner import Tuner

import relflow as rf


def model(merchant=rf.Category):
    return rf.Model.xs(
        d_model=8,
        amount=rf.Number,
        merchant=merchant,
        label=rf.Boolean(mask=True),
        optimizer=rf.adamw(learning_rate=1e-3, fused=None),
    )


def data(configured=None):
    sample = pa.table(
        {"amount": [float(i) for i in range(32)], "merchant": ["alpha", "beta"] * 16, "label": [False, True] * 16}
    )
    return rf.ArrowDataModule(
        model=configured, train=sample, validate=sample, test=sample, predict=sample, shuffle=False
    )


@pytest.mark.integration
@pytest.mark.parametrize("loop", ["fit", "validate", "test", "predict"])
def test_native_trainer_binds_before_callbacks_and_loaders(tmp_path, loop):
    configured = model()
    configured.batch_size = 4
    datamodule = data()
    assert datamodule.model is None
    observed = []

    class CheckBinding(lit.Callback):
        def setup(self, trainer, pl_module, stage):
            assert trainer.datamodule is datamodule
            assert datamodule.model is pl_module is configured
            assert datamodule.schema is configured.schema
            assert datamodule.batch_size == 4
            observed.append(stage)

    trainer = lit.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        callbacks=[CheckBinding()],
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        limit_test_batches=1,
        limit_predict_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    result = getattr(trainer, loop)(configured, datamodule=datamodule)

    assert observed == [loop]
    assert datamodule.model is configured
    if loop == "fit":
        assert trainer.global_step == 1
    elif loop == "predict":
        assert len(result) == 1
        assert isinstance(result[0], pa.Table)
        assert result[0].num_rows == 4
    else:
        assert len(result) == 1
        assert f"loss/{loop}" in result[0]


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


@pytest.mark.integration
@pytest.mark.parametrize("accelerator", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("kind", ["category", "set", "group"])
def test_native_batch_size_callback_restores_state_then_trains_bf16(tmp_path, accelerator, kind):
    if accelerator == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if accelerator == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    field = {"category": rf.Category, "set": rf.Set, "group": rf.Group}[kind]
    options = {"n_groups": 2} if kind == "group" else {}
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
        devices=1,
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
    datamodule = data()
    assert datamodule.model is None
    trainer.fit(configured, datamodule=datamodule)

    assert datamodule.model is configured
    assert restored == [True]
    assert configured.batch_size == finder.optimal_batch_size == 4
    assert trainer.global_step == 1
    assert rf.Number.normalization(configured, "/amount")["count"] == 4
    assert field.vocabulary(configured, "/merchant") == ("alpha", "beta")
    assert any(
        not torch.equal(parameter.detach().cpu(), initial[name]) for name, parameter in configured.named_parameters()
    )
    # Statistics stay float64 except on MPS, which has no float64.
    expected = torch.float32 if trainer.strategy.root_device.type == "mps" else torch.float64
    assert configured.nodes["/amount"].embedder.normalizers["raw"].mean.dtype == expected
    assert not list(tmp_path.glob(".scale_batch_size_*.ckpt"))


@pytest.mark.integration
@pytest.mark.parametrize("bound", [False, True])
def test_native_tuner_can_prepare_a_separate_training_run(tmp_path, bound):
    configured = model()
    datamodule = data(configured if bound else None)
    assert (datamodule.model is configured) is bound
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
    assert datamodule.model is configured
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


@pytest.mark.integration
def test_external_trainer_trains_validates_and_predicts_deterministically(tmp_path):
    lit.seed_everything(0, workers=True)
    configured = rf.Model.xs(
        d_model=8,
        label=rf.Category(mask=True),
        ett=rf.Branch.chronological(
            length=4,
            direction=rf.Direction.bidirectional,
            reduction=rf.Mean(),
            x=rf.Number,
            load=rf.Vector(n_dim=2),
        ),
    )
    configured.optimizer = rf.adamw(learning_rate=1e-3, fused=None)
    loads = ([1.0, float("nan")], [0.5, 0.25], [0.0, 1.0], [float("nan"), 2.0])
    records = [
        {"label": "a" if i % 2 else "b", "ett": [{"x": float(i + j), "load": loads[j]} for j in range(4)]}
        for i in range(8)
    ]
    row = pa.struct([("x", pa.float64()), ("load", pa.list_(pa.float64(), 2))])  # dense fixed_size_list<2>
    sample = pa.Table.from_pylist(records, schema=pa.schema([("label", pa.string()), ("ett", pa.list_(row))]))
    datamodule = rf.ArrowDataModule(
        train=sample,
        validate=sample,
        predict=sample,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
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
    assert "loss/validate" in trainer.validate(configured, datamodule=datamodule)[0]
    first, second = (pa.concat_tables(trainer.predict(configured, datamodule=datamodule)) for _ in range(2))
    assert first.equals(second)
