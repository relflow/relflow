"""EMA preserves training state while selecting and exporting averaged parameters."""

from types import SimpleNamespace

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

import relflow as rf


class Scalar(lit.LightningModule):
    """A deterministic optimizer trajectory with resources that must not be averaged."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.frozen = torch.nn.Parameter(torch.tensor(7.0), requires_grad=False)
        self.register_buffer("count", torch.tensor(0))
        self.validated = []
        self.tested = []

    def configure_optimizers(self):
        return torch.optim.AdamW([self.weight], lr=0.1, weight_decay=0)

    def training_step(self, batch, batch_idx):
        self.count.add_(batch.numel())
        return (self.weight - batch).square().mean()

    def validation_step(self, batch, batch_idx):
        self.validated.append(self.weight.item())
        self.log("quality", (self.weight - batch).square().mean())

    def test_step(self, batch, batch_idx):
        self.tested.append(self.weight.item())

    def predict_step(self, batch, batch_idx):
        return self.weight.detach().expand_as(batch).clone()


def trainer(tmp_path, *, callbacks=(), **options):
    return lit.Trainer(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=any(isinstance(callback, ModelCheckpoint) for callback in callbacks),
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
        callbacks=list(callbacks),
        **options,
    )


@pytest.mark.parametrize("decay", [-1.0, 1.0, float("inf"), float("nan"), True])
def test_decay_requires_a_finite_probability(decay):
    with pytest.raises(ValueError, match="decay"):
        rf.EMA(decay=decay)


def test_optimizer_updates_drive_ema_not_accumulated_batches(tmp_path):
    model = Scalar()
    ema = rf.EMA(decay=0.99)
    observations = []

    class Audit(lit.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            observations.append((ema.updates, pl_module.weight.detach().clone()))

    run = trainer(tmp_path, callbacks=[ema, Audit()], max_epochs=1, accumulate_grad_batches=2)
    run.fit(model, DataLoader(torch.tensor([1.0, 2.0, 3.0, 4.0]), batch_size=1))

    assert [updates for updates, weight in observations] == [0, 1, 1, 2]
    assert ema.updates == run.global_step == 2
    # The short-run warmup uses both consumed snapshots, rather than assigning
    # 99% of a two-update result to the first optimizer step.
    torch.testing.assert_close(model.weight, (observations[1][1] + observations[3][1]) / 2)
    assert set(ema.average) == {"weight"}
    assert model.frozen.item() == 7
    assert model.count.item() == 4
    assert not ema.handles and not ema.active


def test_checkpoint_exports_evaluated_weights_and_resume_restores_live_optimizer_state(tmp_path):
    inputs = DataLoader(torch.tensor([1.0, 2.0]), batch_size=1)
    ema = rf.EMA(decay=0.5)
    model = Scalar()
    checkpoint = ModelCheckpoint(dirpath=tmp_path / "checkpoints", monitor="quality", save_last=True)
    run = trainer(tmp_path, callbacks=[ema, checkpoint], max_epochs=2)
    run.fit(model, inputs, inputs)
    saved = torch.load(checkpoint.last_model_path, map_location="cpu", weights_only=False)
    state = saved["callbacks"][ema.state_key]
    assert state["updates"] == 4
    assert set(state["average"]) == set(state["live"]) == {"weight"}
    assert not torch.equal(state["live"]["weight"], saved["state_dict"]["weight"])
    assert saved["state_dict"]["weight"].item() == pytest.approx(model.validated[-1])
    assert saved["state_dict"]["count"].item() == 4
    assert saved["state_dict"]["frozen"].item() == 7

    inference = Scalar()
    inference.load_state_dict(saved["state_dict"])
    evaluate = trainer(tmp_path)
    predictions = evaluate.predict(inference, inputs)
    evaluate.test(inference, inputs)
    assert all(torch.allclose(output, saved["state_dict"]["weight"].expand_as(output)) for output in predictions)
    assert inference.tested == pytest.approx([model.validated[-1]] * 2)

    # A separate complete trajectory is the ground truth for full continuation.
    baseline = Scalar()
    reference = trainer(tmp_path / "reference", callbacks=[rf.EMA(decay=0.5)], max_epochs=3)
    reference.fit(baseline, inputs, inputs)

    resumed = Scalar()
    resumed_ema = rf.EMA(decay=0.5)
    observed = []

    class Resume(lit.Callback):
        def on_train_start(self, trainer, pl_module):
            torch.testing.assert_close(pl_module.weight, state["live"]["weight"])
            expected = saved["optimizer_states"][0]["state"][0]
            actual = trainer.optimizers[0].state[pl_module.weight]
            for name, value in expected.items():
                torch.testing.assert_close(actual[name], value)
            observed.append(True)

    continued = trainer(tmp_path / "resume", callbacks=[resumed_ema, Resume()], max_epochs=3)
    continued.fit(resumed, inputs, inputs, ckpt_path=checkpoint.last_model_path)
    assert observed == [True]
    assert continued.global_step == reference.global_step == resumed_ema.updates == 6
    torch.testing.assert_close(resumed.weight, baseline.weight, rtol=0, atol=0)
    for name, value in reference.optimizers[0].state[baseline.weight].items():
        torch.testing.assert_close(continued.optimizers[0].state[resumed.weight][name], value, rtol=0, atol=0)


def test_checkpoint_saved_after_fit_keeps_live_resume_snapshot(tmp_path):
    inputs = DataLoader(torch.tensor([1.0, 2.0]), batch_size=1)
    ema = rf.EMA(decay=0.5)
    model = Scalar()
    run = trainer(tmp_path, callbacks=[ema], max_epochs=1)
    run.fit(model, inputs)
    path = tmp_path / "finished.ckpt"
    run.save_checkpoint(path)
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert not torch.equal(saved["callbacks"][ema.state_key]["live"]["weight"], saved["state_dict"]["weight"])


def test_row_growth_restarts_the_replaced_parameter_average():
    module = Scalar()
    module.weight = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
    optimizer = torch.optim.SGD([module.weight], lr=0.1)
    ema = rf.EMA(decay=0.5)
    ema.update(module, optimizer)
    # A replacement may remap existing rows, so no averaged prefix is assumed.
    module.weight = torch.nn.Parameter(torch.tensor([[3.0], [4.0], [30.0], [40.0]]))
    ema.update(module, optimizer)
    torch.testing.assert_close(ema.average["weight"], torch.tensor([[3.0], [4.0], [30.0], [40.0]]))
    assert module.count.item() == 0 and module.frozen.item() == 7


def test_reordered_replacement_resets_only_that_parameter():
    module = Scalar()
    module.weight = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
    module.other = torch.nn.Parameter(torch.tensor(2.0))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    ema = rf.EMA(decay=0.5)
    ema.update(module, optimizer)
    module.weight = torch.nn.Parameter(torch.tensor([[2.0], [1.0]]))
    with torch.no_grad():
        module.other.fill_(4.0)
    ema.update(module, optimizer)
    torch.testing.assert_close(ema.average["weight"], torch.tensor([[2.0], [1.0]]))
    assert ema.average["other"].item() == 3
    with torch.no_grad():
        module.weight.add_(2.0)
    ema.update(module, optimizer)
    torch.testing.assert_close(ema.average["weight"], torch.tensor([[3.0], [2.0]]))


def test_evaluation_and_checkpoint_reset_replacement_before_an_optimizer_step():
    module = Scalar()
    module.weight = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    ema = rf.EMA(decay=0.5)
    ema.fitting = True
    ema.update(module, optimizer)
    module.weight = torch.nn.Parameter(torch.tensor([[2.0], [9.0], [1.0]]))
    expected = module.weight.detach().clone()
    checkpoint = {"state_dict": module.state_dict(), "callbacks": {ema.state_key: ema.state_dict()}}
    ema.on_save_checkpoint(None, module, checkpoint)
    torch.testing.assert_close(checkpoint["state_dict"]["weight"], expected)
    torch.testing.assert_close(checkpoint["callbacks"][ema.state_key]["live"]["weight"], expected)
    ema.begin(None, module)
    torch.testing.assert_close(module.weight, expected)
    ema.restore(None, module)
    torch.testing.assert_close(module.weight, expected)


def test_replacement_during_evaluation_keeps_its_new_row_layout():
    module = Scalar()
    module.weight = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    ema = rf.EMA(decay=0.5)
    ema.fitting = True
    ema.update(module, optimizer)
    ema.begin(None, module)
    module.weight = torch.nn.Parameter(torch.tensor([[2.0], [9.0], [1.0]]))
    expected = module.weight.detach().clone()
    ema.restore(None, module)
    torch.testing.assert_close(module.weight, expected)
    torch.testing.assert_close(ema.average["weight"], expected)


def test_fused_optimizer_overflow_does_not_update_average():
    module = Scalar()
    optimizer = torch.optim.SGD([module.weight], lr=0.1)
    optimizer.found_inf = torch.tensor(1.0)
    ema = rf.EMA()
    ema.update(module, optimizer)
    assert not ema.average and ema.updates == 0


def test_incompatible_snapshot_does_not_partially_overwrite_parameters():
    model = Scalar()
    ema = rf.EMA()
    with pytest.raises(ValueError, match="absent"):
        ema.copy(model, {"weight": torch.tensor(8.0), "missing": torch.tensor(1.0)})
    assert model.weight.item() == 0


def test_validation_restores_live_parameters_and_exception_removes_hooks(tmp_path):
    model = Scalar()
    ema = rf.EMA(decay=0.5)
    live = []

    class Fail(lit.Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            live.append(pl_module.weight.detach().clone())

        def on_validation_epoch_start(self, trainer, pl_module):
            torch.testing.assert_close(pl_module.weight, ema.average["weight"])
            raise RuntimeError("deliberate validation failure")

    run = trainer(tmp_path, callbacks=[ema, Fail()], max_epochs=1)
    inputs = DataLoader(torch.tensor([1.0, 2.0]), batch_size=1)
    with pytest.raises(RuntimeError, match="deliberate validation failure"):
        run.fit(model, inputs, inputs)
    torch.testing.assert_close(model.weight, live[-1])
    assert not ema.active and not ema.handles
    assert not ema.live


def test_relflow_growth_and_rollback_export_the_selected_averaged_model(tmp_path):
    model = rf.Model.xs(
        d_model=8,
        n_layers=1,
        attention=None,
        reduction=rf.Mean(),
        batch_size=2,
        category=rf.Category(p_unavailable=0),
        amount=rf.Number,
        answer=rf.Number(mask=True),
    )
    table = pa.table({"category": list("abcdefgh"), "amount": list(range(8)), "answer": list(range(8))})
    data = rf.ArrowDataModule(model=model, train=table, validate=table, shuffle=False, num_workers=0)
    ema = rf.EMA(decay=0.5)
    evaluated = []

    class Audit(lit.Callback):
        def on_validation_epoch_start(self, trainer, pl_module):
            assert ema.active
            assert all(
                torch.equal(value, dict(pl_module.named_parameters())[name]) for name, value in ema.average.items()
            )
            evaluated.append(model.predict(table)["predictions"])

        def on_validation_end(self, trainer, pl_module):
            assert not ema.active

    model.fit(
        data,
        callbacks=[ema, Audit()],
        early_stopping=False,
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        checkpoint_dir=tmp_path / "selected",
        default_root_dir=tmp_path,
        fused=False,
    )
    selected = model.trainer.checkpoint_callback
    saved = torch.load(selected.best_model_path, map_location="cpu", weights_only=False)
    restored = rf.Model.load(selected.best_model_path)
    assert rf.Category.vocabulary(model, "/category") == tuple("abcdefgh")
    assert rf.Category.counts(model, "/category") == dict.fromkeys("abcdefgh", saved["epoch"] + 1)
    assert rf.Number.normalization(model, "/amount") == rf.Number.normalization(restored, "/amount")
    expected = evaluated[saved["epoch"]]
    assert restored.predict(table)["predictions"].equals(expected)
    assert model.predict(table)["predictions"].equals(expected)
    assert not ema.active and not ema.handles


def test_weights_only_callbacks_are_rejected_before_training(tmp_path):
    checkpoint = ModelCheckpoint(save_weights_only=True)
    with pytest.raises(ValueError, match="full checkpoint"):
        rf.EMA().setup(SimpleNamespace(callbacks=[checkpoint]), Scalar(), "fit")


def test_relflow_full_resume_restores_live_rows_before_admitting_new_categories(tmp_path):
    def build():
        return rf.Model.xs(
            d_model=8,
            n_layers=1,
            attention=None,
            reduction=rf.Mean(),
            batch_size=2,
            category=rf.Category(p_unavailable=0),
            amount=rf.Number,
            answer=rf.Number(mask=True),
        )

    def data(model, labels):
        table = pa.table({"category": list(labels), "amount": list(range(4)), "answer": list(range(4))})
        return rf.ArrowDataModule(model=model, train=table, shuffle=False, num_workers=0)

    options = dict(
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        default_root_dir=tmp_path,
        early_stopping=False,
        rollback=False,
        fused=False,
    )
    model = build()
    ema = rf.EMA(decay=0.5)
    model.fit(data(model, "abcd"), callbacks=[ema], max_epochs=1, **options)
    path = tmp_path / "grown.ckpt"
    model.trainer.save_checkpoint(path)
    saved = torch.load(path, weights_only=False, map_location="cpu")
    state = saved["callbacks"][ema.state_key]
    restored = rf.Model.load(path)
    resumed_ema = rf.EMA(decay=0.5)

    class Audit(lit.Callback):
        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == resumed_ema.updates == 2
            for name, parameter in pl_module.named_parameters():
                if name in state["live"]:
                    torch.testing.assert_close(parameter, state["live"][name], rtol=0, atol=0)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            for name, parameter in pl_module.named_parameters():
                if name in resumed_ema.average:
                    assert resumed_ema.average[name].shape == parameter.shape

    restored.fit(data(restored, "efgh"), callbacks=[resumed_ema, Audit()], max_epochs=2, ckpt_path=path, **options)
    assert restored.trainer.global_step == resumed_ema.updates == 4
    assert rf.Category.vocabulary(restored, "/category") == tuple("abcdefgh")
    assert rf.Category.counts(restored, "/category") == dict.fromkeys("abcdefgh", 1)
    assert rf.Number.normalization(restored, "/amount")["count"] == 8


def test_update_follows_the_lerp_recurrence_without_aliasing_parameters():
    module = torch.nn.Linear(3, 2)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    ema = rf.EMA(decay=0.5)

    ema.update(module, optimizer)
    first = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.add_(1.0)
    ema.update(module, optimizer)

    assert ema.updates == 2
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(ema.average[name], 0.5 * first[name] + 0.5 * parameter.detach())
        assert ema.average[name].data_ptr() != parameter.data_ptr()
        assert not ema.average[name].requires_grad
