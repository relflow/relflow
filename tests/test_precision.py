"""BF16 fused optimization keeps real clipping and caller-owned precision choices."""

from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch
from lightning.pytorch.plugins.io import TorchCheckpointIO
from lightning.pytorch.plugins.precision import MixedPrecision, Precision
from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy
from lightning.pytorch.utilities import GradClipAlgorithmType

import relflow as rf
from relflow.architecture import training
from relflow.structs.enums import Strata


class LargeGradient(rf.Model):
    """Observe clipping after Lightning applies it to deliberately large gradients."""

    def training_step(self, batch, batch_idx):
        result = super().training_step(batch, batch_idx)
        result["loss"] = result["loss"] * 10_000
        return result

    def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
        before = torch.cat(
            [parameter.grad.detach().flatten() for parameter in self.parameters() if parameter.grad is not None]
        )
        super().configure_gradient_clipping(optimizer, gradient_clip_val, gradient_clip_algorithm)
        after = torch.cat(
            [parameter.grad.detach().flatten() for parameter in self.parameters() if parameter.grad is not None]
        )
        self.gradient_before_norm = torch.linalg.vector_norm(before)
        self.gradient_after_norm = torch.linalg.vector_norm(after)
        self.gradient_before_max = before.abs().max()
        self.gradient_after_max = after.abs().max()


class CustomMixedPrecision(MixedPrecision):
    """Represent a caller's independently owned mixed-precision policy."""


def configured() -> tuple[rf.Model, rf.ArrowDataModule]:
    model = LargeGradient(
        d_model=8, n_layers=1, n_heads=2, batch_size=2, amount=rf.Number, target=rf.Boolean(mask=True)
    )
    table = pa.table({"amount": [1.0, 3.0], "target": [False, True]})
    datamodule = rf.ArrowDataModule(
        model=model,
        train=table,
        validate=table,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        shuffle=False,
    )
    return model, datamodule


@pytest.mark.parametrize("precision", ["auto", "bf16-mixed"])
@pytest.mark.parametrize("algorithm", ["norm", "value"])
def test_bf16_fused_adamw_clips_large_gradients_and_updates_weights(precision, algorithm):
    model, datamodule = configured()
    model.optimizer = rf.adamw(1e-3, fused=True)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    model.fit(
        datamodule,
        accelerator="cpu",
        devices=1,
        precision=precision,
        gradient_clip_val=0.25,
        gradient_clip_algorithm=algorithm,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        early_stopping=False,
        rollback=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
    )

    assert isinstance(model.trainer.precision_plugin, training.BFloatPrecision)
    assert model.trainer.precision_plugin.scaler is None
    assert model.trainer.precision == "bf16-mixed"
    assert model.trainer.gradient_clip_val == 0.25
    assert model.trainer.global_step == 1
    assert model.trainer.optimizers[0].defaults["fused"] is True
    if algorithm == "norm":
        assert model.gradient_before_norm > 0.25
        assert model.gradient_after_norm == pytest.approx(0.25, abs=1e-6)
    else:
        assert model.gradient_before_max > 0.25
        assert model.gradient_after_max == pytest.approx(0.25, abs=1e-6)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(not torch.equal(parameter.detach(), before[name]) for name, parameter in model.named_parameters())


@pytest.mark.parametrize("container", ["single", "list", "tuple", "iterator"])
@pytest.mark.parametrize("kind", ["ordinary", "mixed", "custom"])
def test_explicit_precision_plugins_are_preserved(container, kind):
    model, datamodule = configured()
    plugin = (
        Precision()
        if kind == "ordinary"
        else (MixedPrecision if kind == "mixed" else CustomMixedPrecision)("bf16-mixed", "cpu")
    )
    supplied = {
        "single": plugin,
        "list": [plugin],
        "tuple": (plugin,),
        "iterator": iter([plugin]),
    }[container]
    runner = training.trainer(
        model,
        datamodule,
        strata=Strata.validate,
        callbacks=None,
        options={"accelerator": "cpu", "devices": 1, "plugins": supplied},
    )

    assert runner.precision_plugin is plugin


@pytest.mark.parametrize("kind", ["ordinary", "mixed", "custom"])
def test_strategy_owned_precision_is_preserved(kind):
    model, datamodule = configured()
    plugin = (
        Precision()
        if kind == "ordinary"
        else (MixedPrecision if kind == "mixed" else CustomMixedPrecision)("bf16-mixed", "cpu")
    )
    strategy = SingleDeviceStrategy(device=torch.device("cpu"), precision_plugin=plugin)
    runner = training.trainer(
        model,
        datamodule,
        strata=Strata.validate,
        callbacks=None,
        options={"accelerator": "cpu", "devices": 1, "strategy": strategy},
    )

    assert runner.strategy is strategy
    assert runner.precision_plugin is plugin


@pytest.mark.parametrize("kind", ["single", "ddp"])
def test_strategy_without_owned_precision_receives_bf16_clipping_support(kind):
    model, datamodule = configured()
    strategy = SingleDeviceStrategy(device=torch.device("cpu")) if kind == "single" else DDPStrategy()
    runner = training.trainer(
        model,
        datamodule,
        strata=Strata.validate,
        callbacks=None,
        options={"accelerator": "cpu", "devices": 1, "strategy": strategy},
    )

    assert runner.strategy is strategy
    assert isinstance(runner.precision_plugin, training.BFloatPrecision)


def test_checkpoint_plugin_is_retained_alongside_default_bf16_precision():
    model, datamodule = configured()
    checkpoint = TorchCheckpointIO()
    runner = training.trainer(
        model,
        datamodule,
        strata=Strata.validate,
        callbacks=None,
        options={"accelerator": "cpu", "devices": 1, "plugins": [checkpoint]},
    )

    assert runner.strategy.checkpoint_io is checkpoint
    assert isinstance(runner.precision_plugin, training.BFloatPrecision)


@pytest.mark.parametrize("kind", ["fsdp", "custom"])
def test_framework_selected_specialized_precision_is_retained(monkeypatch, kind):
    model, datamodule = configured()
    plugin = FSDPPrecision("bf16-mixed") if kind == "fsdp" else CustomMixedPrecision("bf16-mixed", "cpu")
    strategy = SingleDeviceStrategy(device=torch.device("cpu"), precision_plugin=plugin)
    selected = SimpleNamespace(precision_plugin=plugin, strategy=strategy)
    monkeypatch.setattr(training.lit, "Trainer", lambda **options: selected)

    runner = training.trainer(
        model,
        datamodule,
        strata=Strata.validate,
        callbacks=None,
        options={"accelerator": "cpu", "devices": 1},
    )

    assert runner.precision_plugin is plugin
    assert runner.strategy.precision_plugin is plugin


def test_fp16_precision_retains_the_fused_optimizer_unscaling_guard():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([30.0, 40.0])
    optimizer = torch.optim.AdamW([parameter], fused=True)
    plugin = MixedPrecision("16-mixed", "cpu", scaler=torch.amp.GradScaler("cpu"))

    with pytest.raises(RuntimeError, match="performs unscaling"):
        plugin.clip_gradients(optimizer, 0.25, GradClipAlgorithmType.NORM)
    torch.testing.assert_close(parameter.grad, torch.tensor([30.0, 40.0]))


def test_zero_clipping_preserves_unscaled_bf16_gradients():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    parameter.grad = torch.tensor([30.0, 40.0])
    optimizer = torch.optim.AdamW([parameter], fused=True)
    plugin = training.BFloatPrecision("cpu")
    plugin.clip_gradients(optimizer, 0.0)

    assert plugin.scaler is None
    assert plugin.state_dict() == {}
    torch.testing.assert_close(parameter.grad, torch.tensor([30.0, 40.0]))
