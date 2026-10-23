import csv
from collections import Counter
from datetime import timedelta
from pathlib import Path

import lightning.pytorch as lit
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

import relflow as rf
from relflow.architecture.training import hardware


class DistributedState(lit.Callback):
    """Record completed updates and restored state from each spawned rank."""

    def __init__(self, path):
        self.path = path
        self.counts = Counter()

    def setup(self, trainer, pl_module, stage):
        assert trainer.datamodule.model is pl_module

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.counts["train"] += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.counts["validate"] += 1

    def teardown(self, trainer, pl_module, stage):
        torch.save(
            {
                "counts": dict(self.counts),
                "global_step": trainer.global_step,
                "loss": trainer.callback_metrics["loss/train_epoch"],
                "normalizer": rf.Number.normalization(pl_module, "/amount"),
                "best": trainer.checkpoint_callback.best_model_path,
            },
            self.path / f"rank-{trainer.global_rank}.pt",
        )


@pytest.fixture
def model():
    return rf.Model.xs(d_model=8, batch_size=2, amount=rf.Number, label=rf.Boolean(mask=True))


@pytest.fixture
def data(model):
    table = pa.table({"amount": [float(i) for i in range(48)], "label": [False, True] * 24, "id": list(range(48))})
    return rf.ArrowDataModule(
        model=model, train=table, validate=table, test=table, predict=table, shuffle=False, retain=("id",)
    )


@pytest.fixture
def options(tmp_path):
    return dict(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
    )


class Events(lit.Callback):
    def __init__(self):
        self.counts = Counter()
        self.validation_sizes = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.counts["train"] += 1

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.counts["validate"] += 1

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.counts["test"] += 1

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.counts["predict"] += 1

    def on_validation_epoch_start(self, trainer, pl_module):
        self.before_validation = self.counts["validate"]

    def on_validation_epoch_end(self, trainer, pl_module):
        self.validation_sizes.append(self.counts["validate"] - self.before_validation)


def test_fit_stops_and_restores_the_best_full_model_state(model, data, options):
    class Scores(Events):
        def on_validation_epoch_end(self, trainer, pl_module):
            super().on_validation_epoch_end(trainer, pl_module)
            score = [3.0, 1.0, 2.0][trainer.current_epoch]
            pl_module.log("quality", torch.tensor(score, device=pl_module.device), on_epoch=True)
            if trainer.current_epoch == 1:
                self.best = {key: value.clone() for key, value in pl_module.state_dict().items()}

    callback = Scores()
    assert (
        model.fit(
            data,
            monitor="quality",
            patience=1,
            max_epochs=6,
            limit_train_batches=2,
            limit_val_batches=1,
            callbacks=[callback],
            **options,
        )
        is model
    )
    assert callback.counts["train"] == 6
    assert callback.validation_sizes == [1, 1, 1]
    assert rf.Number.normalization(model, "/amount")["count"] == 8
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, callback.best[key])
    checkpoint = model.trainer.checkpoint_callback
    saved = torch.load(checkpoint.best_model_path, weights_only=False)
    assert saved["epoch"] == 1
    assert saved["optimizer_states"]
    assert "schema" in saved
    assert saved["callbacks"]
    assert saved["state_dict"]["nodes./amount.embedder.normalizers.raw.count"] == 8


def test_callbacks_and_prediction_overloads_keep_every_row(model, data, options):
    callback = Events()
    model.fit(data, max_epochs=1, limit_train_batches=1, limit_val_batches=1, callbacks=[callback], **options)
    validated = model.validate(data, limit_val_batches=2, callbacks=[callback], verbose=False, **options)
    tested = model.test(data, limit_test_batches=3, callbacks=[callback], verbose=False, **options)
    predicted = model.predict(datamodule=data, callbacks=[callback], **options)
    assert "loss/validate" in validated[0]
    assert "loss/test" in tested[0]
    assert callback.counts == {"train": 1, "validate": 3, "test": 3, "predict": 24}
    assert predicted.num_rows == 48
    assert predicted["inputs"].combine_chunks().field("id").to_pylist() == list(range(48))
    direct = model.predict(pa.table({"amount": [1.0], "id": [7]}), retain=("id",))
    assert direct.num_rows == 1
    assert model.predict(data, limit_predict_batches=2, **options).num_rows == 4


@pytest.mark.parametrize("loop", ["fit", "validate", "test", "predict"])
def test_each_loop_can_first_bind_a_data_module(model, options, loop):
    table = pa.table({"amount": [1.0, 2.0, 3.0, 4.0], "label": [False, True, False, True], "id": [0, 1, 2, 3]})
    data = rf.ArrowDataModule(train=table, validate=table, test=table, predict=table, retain=("id",))
    assert data.model is None

    class Bound(Events):
        def setup(self, trainer, pl_module, stage):
            assert trainer.datamodule.model is pl_module

    callback = Bound()
    settings = {**options, "batch_size": 256, "callbacks": [callback]}
    if loop == "fit":
        result = model.fit(data, max_epochs=1, limit_train_batches=1, limit_val_batches=1, **settings)
        assert result is model
        assert callback.counts == {"train": 1, "validate": 1}
    elif loop == "validate":
        result = model.validate(data, limit_val_batches=1, verbose=False, **settings)
        assert "loss/validate" in result[0]
        assert callback.counts == {"validate": 1}
    elif loop == "test":
        result = model.test(data, limit_test_batches=1, verbose=False, **settings)
        assert "loss/test" in result[0]
        assert callback.counts == {"test": 1}
    else:
        result = model.predict(data, **settings)
        assert result.num_rows == 4
        assert result["inputs"].combine_chunks().field("id").to_pylist() == [0, 1, 2, 3]
        assert callback.counts == {"predict": 1}

    assert data.model is model
    assert data.batch_size == 256
    assert data.options[rf.Strata.train].shuffle_rows == 8192


def test_lazy_binding_keeps_model_ownership_across_loops(model, options):
    table = pa.table({"amount": [1.0, 2.0], "label": [False, True]})
    data = rf.ArrowDataModule(validate=table, predict=table)
    model.validate(data, limit_val_batches=1, verbose=False, **options)
    other = rf.Model.xs(d_model=8, amount=rf.Number, label=rf.Boolean(mask=True))
    with pytest.raises(ValueError, match="different model"):
        other.predict(data, **options)
    assert data.model is model


def test_invalid_loop_configuration_does_not_bind_data(model, options):
    data = rf.ArrowDataModule(predict=pa.table({"amount": [1.0]}))
    with pytest.raises(ValueError, match="validate split"):
        model.validate(data, **options)
    assert data.model is None
    with pytest.raises(TypeError):
        model.predict(data, invented_option=True, **options)
    assert data.model is None


def test_custom_lightning_data_module_can_own_lazy_binding(model, options):
    table = pa.table({"amount": [1.0, 2.0], "label": [False, True]})

    class Deferred(lit.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.model = None
            self.bindings = 0

        def bind(self, configured):
            self.model = configured
            self.bindings += 1

        def val_dataloader(self):
            return rf.ArrowDataModule(self.model, validate=table).val_dataloader()

    data = Deferred()
    result = model.validate(data, limit_val_batches=1, verbose=False, **options)
    assert "loss/validate" in result[0]
    assert data.model is model
    assert data.bindings == 1
    model.validate(data, limit_val_batches=1, verbose=False, **options)
    assert data.bindings == 1
    explicit = Deferred()
    explicit.model = model
    model.validate(explicit, limit_val_batches=1, verbose=False, **options)
    assert explicit.bindings == 0
    other = rf.Model.xs(d_model=8, amount=rf.Number, label=rf.Boolean(mask=True))
    with pytest.raises(ValueError, match="different model"):
        other.validate(data, **options)
    assert data.model is model
    assert data.bindings == 1


def test_custom_model_owned_data_module_retains_ownership_guard(model, options):
    class Owned(lit.LightningDataModule):
        def __init__(self):
            super().__init__()
            self.model = model

    other = rf.Model.xs(d_model=8, amount=rf.Number, label=rf.Boolean(mask=True))
    with pytest.raises(ValueError, match="different model"):
        other.validate(Owned(), **options)


def test_streamed_prediction_does_not_collect_tables(model, data, options, tmp_path):
    writer = rf.Writer(tmp_path / "predictions")
    assert model.predict(datamodule=data, callbacks=[writer], return_predictions=False, **options) is None
    assert pq.read_table(tmp_path / "predictions").num_rows == 48
    assert writer.writer is None
    assert not model.trainer.predict_loop.predictions


def test_loops_use_complete_splits_by_default(model, data, options):
    callback = Events()
    model.fit(data, max_epochs=1, callbacks=[callback], **options)
    assert model.trainer.log_every_n_steps == 50
    assert callback.counts == {"train": 24, "validate": 24}
    model.validate(data, callbacks=[callback], verbose=False, **options)
    model.test(data, callbacks=[callback], verbose=False, **options)
    assert callback.counts == {"train": 24, "validate": 48, "test": 24}


def test_explicit_limits_still_bound_each_loop(model, data, options):
    callback = Events()
    model.fit(data, max_epochs=1, limit_train_batches=2, limit_val_batches=3, callbacks=[callback], **options)
    assert callback.counts["train"] == 2
    assert callback.validation_sizes == [3]
    model.test(data, limit_test_batches=4, callbacks=[callback], verbose=False, **options)
    assert callback.counts["test"] == 4


@pytest.mark.parametrize("report_interval", [1, 3])
def test_logging_cadence_keeps_validation_selection_and_recovery_independent(
    model, data, options, tmp_path, report_interval
):
    from lightning.pytorch.callbacks import ModelCheckpoint

    class Scores(Events):
        def on_validation_epoch_end(self, trainer, pl_module):
            super().on_validation_epoch_end(trainer, pl_module)
            score = [3.0, 1.0, 2.0][trainer.global_step // 2 - 1]
            pl_module.log("quality", torch.tensor(score, device=pl_module.device), on_epoch=True)
            if trainer.global_step == 4:
                self.best = {key: value.clone() for key, value in pl_module.state_dict().items()}

    scores = Scores()
    logger = lit.loggers.CSVLogger(tmp_path / "logs")

    def schedule(module, optimizer):
        return {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5), "interval": "step"}

    recovery = ModelCheckpoint(
        dirpath=tmp_path / "recovery",
        filename="step-{step}",
        monitor=None,
        save_top_k=1,
        save_last=True,
        every_n_train_steps=3,
    )
    model.fit(
        data,
        monitor="quality",
        patience=1,
        max_epochs=3,
        limit_train_batches=6,
        limit_val_batches=1,
        val_check_interval=2,
        check_val_every_n_epoch=None,
        log_every_n_steps=report_interval,
        logger=logger,
        scheduler=schedule,
        callbacks=[scores, recovery],
        **options,
    )
    assert model.trainer.global_step == 6
    with (Path(logger.log_dir) / "metrics.csv").open() as stream:
        reports = [row for row in csv.DictReader(stream) if row["loss/train"]]
    assert [int(row["step"]) for row in reports] == list(range(report_interval - 1, 6, report_interval))
    assert scores.validation_sizes == [1, 1, 1]
    best = next(
        callback for callback in model.trainer.checkpoint_callbacks if isinstance(callback, rf.RollbackCheckpoint)
    )
    assert "step-4" in best.best_model_path
    selected = torch.load(best.best_model_path, weights_only=False)
    latest = torch.load(recovery.last_model_path, weights_only=False)
    assert selected["global_step"] == 4
    assert latest["global_step"] == 6
    assert latest["optimizer_states"]
    assert selected["lr_schedulers"][0]["last_epoch"] == 4
    assert latest["lr_schedulers"][0]["last_epoch"] == 6
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, scores.best[name])


def test_custom_policies_replace_defaults_and_train_only_requires_a_policy_choice(model, options, tmp_path):
    table = pa.table({"amount": [1.0, 2.0], "label": [True, False]})
    data = rf.ArrowDataModule(model, train=table)
    with pytest.raises(ValueError, match="supply a validate split"):
        model.fit(data, max_epochs=1, **options)
    stop = rf.EarlyStopping(monitor="loss/train_epoch", patience=2)
    rollback = rf.RollbackCheckpoint(dirpath=tmp_path / "custom", monitor="loss/train_epoch", mode="min")
    model.fit(data, early_stopping=stop, rollback=rollback, max_epochs=1, **options)
    assert [item for item in model.trainer.callbacks if isinstance(item, rf.EarlyStopping)] == [stop]
    assert [item for item in model.trainer.callbacks if isinstance(item, rf.RollbackCheckpoint)] == [rollback]


def test_input_ownership_and_option_conflicts_fail_before_execution(model, data, options):
    other = rf.Model.xs(d_model=8, amount=rf.Number, label=rf.Boolean(mask=True))
    with pytest.raises(ValueError, match="different model"):
        other.fit(data, **options)
    with pytest.raises(TypeError, match="LightningDataModule"):
        model.validate(pa.table({"amount": [1.0]}), **options)
    model.optimizer = rf.adamw(1e-3)
    with pytest.raises(ValueError, match="conflict"):
        model.fit(data, learning_rate=1e-2, **options)
    with pytest.raises(TypeError, match="not both"):
        model.predict(pa.table({"amount": [1.0]}), datamodule=data)
    with pytest.raises(ValueError, match="prediction writer"):
        model.predict(datamodule=data, return_predictions=False, **options)
    with pytest.raises(TypeError):
        model.validate(data, invented_option=True, **options)


def test_inline_adamw_retains_the_default_recipe(model, data, options):
    model.fit(
        data,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        early_stopping=False,
        rollback=False,
        **options,
    )
    optimizer = model.trainer.optimizers[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 1e-3
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["fused"] is True
    assert model.optimizer is None
    assert model.scheduler is None


@pytest.mark.parametrize("fused", [None, False])
@pytest.mark.parametrize("fragments", [("attention",), ()])
def test_inline_adamw_options_control_the_live_optimizer(model, data, options, fused, fragments):
    model.fit(
        data,
        learning_rate=0.003,
        weight_decay=0.17,
        betas=(0.8, 0.91),
        eps=1e-6,
        fused=fused,
        decay_bias=True,
        decay_1d=True,
        no_decay_name_fragments=fragments,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        early_stopping=False,
        rollback=False,
        **options,
    )
    optimizer = model.trainer.optimizers[0]
    assert optimizer.defaults["lr"] == 0.003
    assert optimizer.defaults["weight_decay"] == 0.17
    assert optimizer.defaults["betas"] == (0.8, 0.91)
    assert optimizer.defaults["eps"] == 1e-6
    assert optimizer.defaults["fused"] is fused
    names = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == names.keys()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            expected = 0.0 if any(fragment in names[id(parameter)].lower() for fragment in fragments) else 0.17
            assert group["weight_decay"] == expected
    assert optimizer.state
    assert all(state["step"] == 1 for state in optimizer.state.values())
    assert model.optimizer is None


def test_a_new_fit_binds_the_live_graph_after_rollback_and_extension(model, data, options):
    model.fit(
        data,
        learning_rate=1e-3,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        **options,
    )
    previous = model.trainer.optimizers[0]
    model.extend(rf.where("address") == rf.Address(), id=rf.Number)
    live = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    model.fit(
        data,
        learning_rate=2e-4,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        early_stopping=False,
        rollback=False,
        **options,
    )
    optimizer = model.trainer.optimizers[0]
    assert optimizer is not previous
    assert optimizer.defaults["lr"] == 2e-4
    assert {id(parameter) for group in optimizer.param_groups for parameter in group["params"]} == live
    assert optimizer.state
    assert all(state["step"] == 1 for state in optimizer.state.values())
    assert model.optimizer is None


def test_a_fit_scheduler_does_not_leak_into_the_next_fit(model, data, options):
    def schedule(module, optimizer):
        return {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5), "interval": "step"}

    model.fit(
        data,
        learning_rate=0.01,
        scheduler=schedule,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        early_stopping=False,
        rollback=False,
        **options,
    )
    assert model.trainer.lr_scheduler_configs[0].scheduler.last_epoch == 1
    assert model.trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.005)
    assert model.scheduler is None
    model.fit(
        data,
        learning_rate=0.003,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        early_stopping=False,
        rollback=False,
        **options,
    )
    assert model.trainer.lr_scheduler_configs == []
    assert model.trainer.optimizers[0].param_groups[0]["lr"] == 0.003
    assert model.optimizer is None
    assert model.scheduler is None


def test_failed_fit_restores_the_models_explicit_configuration(model, data, options):
    previous_optimizer = rf.adamw(learning_rate=0.007)

    def previous_scheduler(module, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    def broken_scheduler(module, optimizer):
        raise RuntimeError("scheduler setup failed")

    model.optimizer = previous_optimizer
    model.scheduler = previous_scheduler
    with pytest.raises(RuntimeError, match="scheduler setup failed"):
        model.fit(
            data,
            optimizer=rf.adamw(learning_rate=0.002),
            scheduler=broken_scheduler,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            early_stopping=False,
            rollback=False,
            **options,
        )
    assert model.optimizer is previous_optimizer
    assert model.scheduler is previous_scheduler


@pytest.mark.parametrize("configuration", ["argument", "model"])
@pytest.mark.parametrize(
    "inline",
    [
        {"learning_rate": 0.003},
        {"weight_decay": 0.0},
        {"betas": (0.8, 0.9)},
        {"eps": 1e-6},
        {"fused": None},
        {"fused": False},
        {"decay_bias": False},
        {"decay_1d": False},
        {"no_decay_name_fragments": ()},
    ],
)
def test_custom_optimizer_configuration_rejects_inline_adamw_options(model, data, options, configuration, inline):
    optimizer = rf.adamw(learning_rate=1e-3)
    supplied = {"optimizer": optimizer} if configuration == "argument" else {}
    if configuration == "model":
        model.optimizer = optimizer
    with pytest.raises(ValueError, match="conflict"):
        model.fit(data, **supplied, **inline, **options)


def test_an_optimizer_factory_cannot_return_parameters_from_a_previous_graph(model, data, options):
    previous = rf.adamw(learning_rate=1e-3)(model)
    model.update(rf.where("address") == rf.Address(), description="Rebuilt payment model")
    with pytest.raises(ValueError, match="optimizer"):
        model.fit(
            data,
            optimizer=lambda module: previous,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            early_stopping=False,
            rollback=False,
            **options,
        )
    assert model.optimizer is None


def test_a_scheduler_factory_must_bind_the_optimizer_for_that_fit(model, data, options):
    previous = rf.adamw(learning_rate=1e-3)(model)
    scheduler = torch.optim.lr_scheduler.StepLR(previous, step_size=1)
    with pytest.raises(ValueError, match="scheduler.*optimizer"):
        model.fit(
            data,
            scheduler=lambda module, optimizer: {"scheduler": scheduler, "interval": "step"},
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            early_stopping=False,
            rollback=False,
            **options,
        )
    assert model.optimizer is None
    assert model.scheduler is None


def test_automatic_hardware_handles_unused_ddp_parameters_and_checks_all_selected_gpu_capabilities(monkeypatch):
    from lightning.pytorch.accelerators import CUDAAccelerator, MPSAccelerator
    from lightning.pytorch.strategies import DDPStrategy

    monkeypatch.setattr(MPSAccelerator, "is_available", lambda: False)
    monkeypatch.setattr(CUDAAccelerator, "is_available", lambda: True)
    monkeypatch.setattr(CUDAAccelerator, "auto_device_count", staticmethod(lambda: 2))
    monkeypatch.setattr(CUDAAccelerator, "parse_devices", staticmethod(lambda devices: list(range(devices))))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8 if device.index == 0 else 7, 0))
    options = {}
    hardware(options)
    assert options == {
        "accelerator": "cuda",
        "devices": 2,
        "strategy": "ddp_find_unused_parameters_true",
        "precision": "32-true",
    }
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (8, 0))
    options = {}
    hardware(options)
    assert options["precision"] == "bf16-mixed"
    options = {"devices": 1}
    hardware(options)
    assert options["strategy"] == "auto"
    options = {"devices": 1, "num_nodes": 2, "strategy": "auto"}
    hardware(options)
    assert options["strategy"] == "ddp_find_unused_parameters_true"
    options = {"precision": "64-true", "strategy": "ddp_spawn"}
    hardware(options)
    assert options["precision"] == "64-true"
    assert options["strategy"] == "ddp_spawn"
    strategy = DDPStrategy(find_unused_parameters=False)
    options = {"strategy": strategy}
    hardware(options)
    assert options["strategy"] is strategy


def test_resume_restores_optimizer_progress_from_a_full_checkpoint(model, data, options):
    def schedule(module, optimizer):
        return {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5), "interval": "step"}

    model.fit(
        data,
        learning_rate=1e-3,
        scheduler=schedule,
        max_epochs=1,
        limit_train_batches=2,
        limit_val_batches=1,
        **options,
    )
    checkpoint = model.trainer.checkpoint_callback.last_model_path
    assert torch.load(checkpoint, weights_only=False)["batch_size"] == 2
    model.fit(
        data,
        ckpt_path=checkpoint,
        batch_size=3,
        learning_rate=0.02,
        scheduler=schedule,
        max_epochs=2,
        limit_train_batches=2,
        limit_val_batches=1,
        early_stopping=False,
        rollback=False,
        **options,
    )
    assert model.trainer.global_step == 4
    assert rf.Number.normalization(model, "/amount")["count"] == 10
    assert model.batch_size == data.batch_size == 3
    assert all(state["step"] == 4 for state in model.trainer.optimizers[0].state.values())
    assert model.trainer.lr_scheduler_configs[0].scheduler.last_epoch == 4
    assert model.trainer.optimizers[0].param_groups[0]["lr"] == pytest.approx(1e-3 * 0.5**4)
    assert model.optimizer is None
    assert model.scheduler is None


@pytest.mark.parametrize("accelerator", ["cpu", "auto"])
def test_default_devices_precision_and_optimizer_run_on_the_selected_backend(model, data, options, accelerator):
    if accelerator == "auto" and torch.cuda.device_count() > 1:
        pytest.skip("multi-device execution is exercised by the dedicated DDP test")
    options.pop("precision")
    options.pop("devices")
    options["accelerator"] = accelerator
    model.fit(data, max_epochs=1, limit_train_batches=1, limit_val_batches=1, **options)
    assert model.trainer.precision in ("bf16-mixed", "32-true")
    if accelerator == "cpu":
        assert model.trainer.precision == "bf16-mixed"
    assert rf.Number.normalization(model, "/amount")["count"] == 2


def test_arbitrary_data_module_evaluates_complete_validation_loaders(model, data, options):
    class Validation(lit.LightningDataModule):
        def val_dataloader(self):
            return [data.val_dataloader(), data.val_dataloader()]

    callback = Events()
    result = model.validate(Validation(), callbacks=[callback], verbose=False, **options)
    assert len(result) == 2
    assert callback.counts["validate"] == 48


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="requires CPU Gloo"
)
@pytest.mark.parametrize("lazy", [False, True])
def test_distributed_logging_and_rollback_complete_on_every_rank(model, data, options, tmp_path, lazy):
    from lightning.pytorch.strategies import DDPStrategy

    if lazy:
        data = rf.ArrowDataModule(
            train=data.sources[rf.Strata.train], validate=data.sources[rf.Strata.validate], shuffle=False
        )
    options["devices"] = 2
    logger = lit.loggers.CSVLogger(tmp_path / "logs")
    model.fit(
        data,
        max_epochs=1,
        val_check_interval=4,
        check_val_every_n_epoch=None,
        limit_val_batches=1,
        log_every_n_steps=3,
        logger=logger,
        callbacks=[DistributedState(tmp_path)],
        strategy=DDPStrategy(start_method="spawn", process_group_backend="gloo", timeout=timedelta(seconds=45)),
        **options,
    )
    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    assert ranks[0]["global_step"] == ranks[1]["global_step"] == 12
    assert all(torch.isfinite(rank["loss"]) for rank in ranks)
    assert ranks[0]["counts"] == ranks[1]["counts"]
    assert ranks[0]["counts"] == {"train": 12, "validate": 3}
    assert ranks[0]["best"] == ranks[1]["best"]
    assert ranks[0]["normalizer"] == ranks[1]["normalizer"]
    with (tmp_path / "logs/lightning_logs/version_0/metrics.csv").open() as stream:
        reports = [row for row in csv.DictReader(stream) if row["loss/train"]]
    assert [int(row["step"]) for row in reports] == [2, 5, 8, 11]
    assert all(float(row["throughput/train"]) > 0.0 for row in reports)


def test_explicit_precision_plugin_overrides_automatic_precision(model, data, options):
    from lightning.pytorch.plugins.precision import MixedPrecision

    options.pop("precision")
    plugin = MixedPrecision("bf16-mixed", "cpu")
    model.validate(data, plugins=[plugin], limit_val_batches=1, verbose=False, **options)
    assert model.trainer.precision_plugin is plugin


def test_fast_dev_run_skips_automatic_selection_policies(model, data, options):
    model.fit(data, fast_dev_run=True, **options)
    assert model.trainer.global_step == 1
    assert not any(
        isinstance(callback, (rf.EarlyStopping, rf.RollbackCheckpoint)) for callback in model.trainer.callbacks
    )


class ModuleOptimizer(torch.nn.Module):
    """Keep a recipe buffer outside the model's learned architecture."""

    def __init__(self, learning_rate):
        super().__init__()
        self.register_buffer("learning_rate", torch.tensor(learning_rate, dtype=torch.float64))

    def forward(self, model):
        return torch.optim.AdamW(model.parameters(), lr=float(self.learning_rate), fused=False)


class ModuleScheduler(torch.nn.Module):
    """Configure a scheduler without contributing buffers to the model."""

    def __init__(self, gamma):
        super().__init__()
        self.register_buffer("gamma", torch.tensor(gamma, dtype=torch.float64))

    def forward(self, model, optimizer):
        return {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=float(self.gamma)),
            "interval": "step",
        }


def test_module_factories_remain_outside_the_graph_across_fit_calls(model, data, options):
    expected_state = set(model.state_dict())

    class Ownership(lit.Callback):
        def on_fit_start(self, trainer, pl_module):
            assert "optimizer" not in pl_module._modules
            assert "scheduler" not in pl_module._modules
            assert set(pl_module.state_dict()) == expected_state

    model.fit(
        data,
        optimizer=ModuleOptimizer(0.01),
        scheduler=ModuleScheduler(0.5),
        callbacks=[Ownership()],
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        early_stopping=False,
        rollback=False,
        **options,
    )
    previous = model.trainer.optimizers[0]
    assert previous.param_groups[0]["lr"] == pytest.approx(0.005)
    assert model.optimizer is None
    assert model.scheduler is None
    assert set(model.state_dict()) == expected_state
    model.fit(
        data,
        learning_rate=0.003,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        early_stopping=False,
        rollback=False,
        **options,
    )
    assert model.trainer.optimizers[0] is not previous
    assert model.trainer.optimizers[0].param_groups[0]["lr"] == 0.003
    assert model.trainer.lr_scheduler_configs == []
    assert model.optimizer is None
    assert model.scheduler is None
    assert "optimizer" not in model._modules
    assert "scheduler" not in model._modules
    assert set(model.state_dict()) == expected_state


def test_constructor_module_factories_do_not_become_checkpointed_architecture(tmp_path):
    optimizer = ModuleOptimizer(0.01)
    scheduler = ModuleScheduler(0.5)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        amount=rf.Number,
        label=rf.Boolean(mask=True),
        optimizer=optimizer,
        scheduler=scheduler,
    )
    configured = model.configure_optimizers()
    active = configured["optimizer"]
    assert active.param_groups[0]["lr"] == 0.01
    assert configured["lr_scheduler"]["scheduler"].optimizer is active
    assert {id(parameter) for group in active.param_groups for parameter in group["params"]} == {
        id(parameter) for parameter in model.parameters()
    }
    assert model.optimizer is optimizer
    assert model.scheduler is scheduler
    assert "optimizer" not in model._modules
    assert "scheduler" not in model._modules
    assert all(not key.startswith(("optimizer.", "scheduler.")) for key in model.state_dict())
    checkpoint = model.save(tmp_path / "recipes.ckpt")
    restored = rf.Model.load(checkpoint)
    assert restored.optimizer is None
    assert restored.scheduler is None
    assert set(restored.state_dict()) == set(model.state_dict())


def test_loop_batch_size_controls_each_arrow_split_and_retains_partial_batches(model, options):
    class BatchRows(lit.Callback):
        def __init__(self):
            self.rows = {stage: [] for stage in ("train", "validate", "test", "predict")}

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.rows["train"].append(batch.source.num_rows)

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.rows["validate"].append(batch.source.num_rows)

        def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.rows["test"].append(batch.source.num_rows)

        def on_predict_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.rows["predict"].append(batch.source.num_rows)

    identifiers = [f"payment-{index}" for index in range(7)]
    table = pa.table(
        {
            "amount": [12.50, 42.75, 18.20, 65.00, 7.99, 120.00, 31.40],
            "label": [False, True, False, False, True, False, True],
            "payment_id": identifiers,
        }
    )
    data = rf.ArrowDataModule(
        model, train=table, validate=table, test=table, predict=table, shuffle=False, retain=("payment_id",)
    )
    callback = BatchRows()
    model.fit(
        data,
        batch_size=3,
        max_epochs=1,
        early_stopping=False,
        rollback=False,
        callbacks=[callback],
        **options,
    )
    assert callback.rows["train"] == [3, 3, 1]
    assert callback.rows["validate"] == [3, 3, 1]
    assert model.trainer.global_step == 3
    assert data.batch_size == model.batch_size == 3

    callback.rows["validate"].clear()
    model.validate(data, batch_size=4, callbacks=[callback], verbose=False, **options)
    assert callback.rows["validate"] == [4, 3]
    assert data.batch_size == model.batch_size == 4
    model.test(data, batch_size=5, callbacks=[callback], verbose=False, **options)
    assert callback.rows["test"] == [5, 2]
    assert data.batch_size == model.batch_size == 5
    result = model.predict(datamodule=data, batch_size=6, callbacks=[callback], **options)
    assert callback.rows["predict"] == [6, 1]
    assert result["inputs"].combine_chunks().field("payment_id").to_pylist() == identifiers
    assert result.num_rows == 7
    assert data.batch_size == model.batch_size == 6

    inherited = model.predict(data, batch_size=None, callbacks=[callback], **options)
    assert callback.rows["predict"] == [6, 1, 6, 1]
    assert inherited.equals(result)
    callback.rows["validate"].clear()
    model.validate(data, callbacks=[callback], verbose=False, **options)
    assert callback.rows["validate"] == [6, 1]
    assert data.batch_size == model.batch_size == 6


@pytest.mark.parametrize("loop", ["fit", "validate", "test", "predict"])
@pytest.mark.parametrize("batch_size", [True, False, 0, -3, 1.5, "4"])
def test_public_loop_batch_size_rejects_invalid_overrides_without_changing_the_current_size(
    model, data, options, loop, batch_size
):
    with pytest.raises(ValueError, match="batch_size.*positive integer"):
        getattr(model, loop)(data, batch_size=batch_size, **options)
    assert model.batch_size == data.batch_size == 2


@pytest.mark.parametrize("entrypoint", ["constructor", "property"])
@pytest.mark.parametrize("batch_size", [True, False, 0, -3, 1.5, "4", None])
def test_model_batch_size_requires_a_positive_integer(model, entrypoint, batch_size):
    with pytest.raises((ValueError, BeartypeCallHintParamViolation), match="batch_size"):
        if entrypoint == "constructor":
            rf.Model(model.schema, batch_size=batch_size)
        else:
            model.batch_size = batch_size
    assert model.batch_size == 2
