import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch

import relflow as rf


class RecordingLogger(lit.loggers.Logger):
    def __init__(self):
        super().__init__()
        self.records = []

    @property
    def name(self):
        return "training"

    @property
    def version(self):
        return 0

    def log_hyperparams(self, params):
        pass

    def log_metrics(self, metrics, step=None):
        self.records.append((dict(metrics), step))


class FixedLosses(rf.Model):
    def training_step(self, batch, batch_idx):
        loss = next(self.parameters()).sum() * 0 + batch_idx + 1.0
        self.track(("loss", "train"), value=loss)
        self.track(("/label", "train.loss.content"), value=loss * 2)
        self.track(("loss", "train_epoch"), value=loss, on_step=False, on_epoch=True, logger=False)
        return {"loss": loss}


def test_track_logs_only_steps_and_keeps_training_loss_internal_with_accumulation(tmp_path):
    model = FixedLosses(d_model=8, n_layers=1, n_heads=2, batch_size=2, label=rf.Boolean(mask=True))
    rows = pa.table({"label": [True, False, True, False, True, False, True]})
    data = rf.ArrowDataModule(model, train=rows)
    logger = RecordingLogger()
    model.fit(
        data,
        early_stopping=False,
        rollback=False,
        max_epochs=1,
        accumulate_grad_batches=2,
        log_every_n_steps=1,
        logger=logger,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    reports = [(metrics, step) for metrics, step in logger.records if "loss/train" in metrics]
    assert [step for _, step in reports] == [0, 1]
    assert [metrics["loss/train"] for metrics, _ in reports] == [2.0, 4.0]
    assert [metrics[".label/train.loss.content"] for metrics, _ in reports] == [4.0, 8.0]
    assert all(metrics["throughput/train"] > 0.0 for metrics, _ in reports)
    assert [step for metrics, step in logger.records if "throughput/train" in metrics] == [0, 1]
    assert not any(name.endswith(("_step", "_epoch")) for metrics, _ in logger.records for name in metrics)
    assert model.trainer.callback_metrics["loss/train_epoch"] == 2.5
    assert model.trainer.global_step == 2


def test_logging_interval_samples_steps_and_zero_keeps_training_loss_internal(tmp_path):
    model = FixedLosses(d_model=8, n_layers=1, n_heads=2, batch_size=2, label=rf.Boolean(mask=True))
    rows = pa.table({"label": [True, False] * 5})
    data = rf.ArrowDataModule(model, train=rows)
    logger = RecordingLogger()
    options = dict(
        early_stopping=False,
        rollback=False,
        max_epochs=1,
        logger=logger,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    model.fit(data, log_every_n_steps=3, **options)
    reports = [(metrics, step) for metrics, step in logger.records if "loss/train" in metrics]
    assert [step for _, step in reports] == [2]
    assert [metrics["loss/train"] for metrics, _ in reports] == [3.0]
    assert reports[0][0]["throughput/train"] > 0.0
    assert [step for metrics, step in logger.records if "throughput/train" in metrics] == [2]
    assert not any("loss/train_epoch" in metrics for metrics, _ in logger.records)
    assert model.trainer.callback_metrics["loss/train_epoch"] == 3.0
    logger.records.clear()
    model.fit(data, log_every_n_steps=0, **options)
    assert logger.records == []
    assert model.trainer.callback_metrics["loss/train_epoch"] == 3.0
    assert model.trainer.global_step == 5


def test_validation_keeps_its_metric_name_and_default_checkpoint_monitor(tmp_path):
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, batch_size=2, label=rf.Boolean(mask=True))
    rows = pa.table({"label": [True, False] * 3})
    data = rf.ArrowDataModule(model, train=rows, validate=rows)
    logger = RecordingLogger()
    model.fit(
        data,
        max_epochs=1,
        log_every_n_steps=1,
        logger=logger,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    assert any("loss/train" in metrics for metrics, _ in logger.records)
    training = [metrics for metrics, _ in logger.records if "loss/train" in metrics]
    assert all(0.0 <= metrics[".label/train.accuracy@0.5.content"] <= 1.0 for metrics in training)
    assert all(0.0 <= metrics[".label/train.auc.content"] <= 1.0 for metrics in training)
    validations = [metrics for metrics, _ in logger.records if "loss/validate" in metrics]
    assert len(validations) == 1
    assert not any("loss/validate_step" in metrics or "loss/validate_epoch" in metrics for metrics, _ in logger.records)
    assert len([metrics for metrics, _ in logger.records if "throughput/validate" in metrics]) == 3
    assert "throughput/validate" not in validations[0]
    assert model.trainer.checkpoint_callback.monitor == "loss/validate"
    assert model.trainer.checkpoint_callback.best_model_score.item() == pytest.approx(validations[0]["loss/validate"])


@pytest.mark.parametrize(
    ("field", "labels"),
    [
        (rf.Category(mask=True), ["a", "b", "b", "a"]),
        (rf.Enum(values=("a", "b"), mask=True), ["a", "b", "b", "a"]),
        (rf.Set(mask=True), [["a"], ["b"], ["b"], ["a"]]),
        (rf.Cluster(bounds=2, mask=True), ["a", "b", "b", "a"]),
    ],
)
def test_vocabulary_scores_log_per_batch_and_validation_keeps_pass_totals(tmp_path, field, labels):
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, batch_size=2, amount=rf.Number, label=field)
    rows = pa.table({"amount": [1.0, 2.0, 3.0, 4.0], "label": labels})
    data = rf.ArrowDataModule(model, train=rows, validate=rows, shuffle=False)
    logger = RecordingLogger()
    model.fit(
        data,
        max_epochs=1,
        log_every_n_steps=1,
        logger=logger,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=tmp_path,
    )
    training = [(metrics, step) for metrics, step in logger.records if "loss/train" in metrics]
    assert [step for _, step in training] == [0, 1]
    assert all(metrics[".label/train.targets.known"] == 2 for metrics, _ in training)
    assert all(metrics[".label/train.coverage.content"] == 1.0 for metrics, _ in training)
    assert not any(name.endswith(("_step", "_epoch")) for metrics, _ in logger.records for name in metrics)
    validations = [metrics for metrics, _ in logger.records if "loss/validate" in metrics]
    assert len(validations) == 1
    assert validations[0][".label/validate.targets.known"] == 4
    assert validations[0][".label/validate.coverage.content"] == 1.0
    assert model.trainer.checkpoint_callback.best_model_score.item() == pytest.approx(validations[0]["loss/validate"])
    assert torch.isfinite(model.trainer.callback_metrics["loss/train_epoch"])
