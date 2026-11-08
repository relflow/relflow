import math

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch

import relflow as rf

pytestmark = pytest.mark.integration


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


@pytest.fixture
def fit_options(tmp_path):
    return {
        "batch_size": 2,
        "max_epochs": 1,
        "accelerator": "cpu",
        "devices": 1,
        "precision": "32-true",
        "enable_progress_bar": False,
        "enable_model_summary": False,
        "default_root_dir": tmp_path,
    }


def test_track_logs_only_steps_and_keeps_training_loss_internal_with_accumulation(fit_options):
    model = FixedLosses.xs(d_model=8, label=rf.Boolean(mask=True))
    rows = pa.table({"label": [True, False, True, False, True, False, True]})
    data = rf.ArrowDataModule(train=rows)
    logger = RecordingLogger()
    model.fit(
        data,
        early_stopping=False,
        rollback=False,
        accumulate_grad_batches=2,
        log_every_n_steps=1,
        logger=logger,
        **fit_options,
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


def test_logging_interval_samples_steps_and_zero_keeps_training_loss_internal(fit_options):
    model = FixedLosses.xs(d_model=8, label=rf.Boolean(mask=True))
    rows = pa.table({"label": [True, False] * 5})
    data = rf.ArrowDataModule(train=rows)
    logger = RecordingLogger()
    options = dict(
        early_stopping=False,
        rollback=False,
        logger=logger,
        **fit_options,
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


def test_validation_keeps_its_metric_name_and_default_checkpoint_monitor(fit_options):
    model = rf.Model.xs(d_model=8, label=rf.Boolean(mask=True))
    rows = pa.table({"label": [True, False] * 3})
    data = rf.ArrowDataModule(train=rows, validate=rows)
    logger = RecordingLogger()
    model.fit(
        data,
        log_every_n_steps=1,
        logger=logger,
        num_sanity_val_steps=0,
        **fit_options,
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
    ("field", "labels", "has_content_scores"),
    [
        (rf.Category(mask=True), ["a", "b", "b", "a"], True),
        (rf.Enum(values=("a", "b"), mask=True), ["a", "b", "b", "a"], True),
        (rf.Set(mask=True), [["a"], ["b"], ["b"], ["a"]], True),
        (rf.Group, ["a", "b", "b", "a"], False),
    ],
)
def test_vocabulary_fields_log_available_scores(fit_options, field, labels, has_content_scores):
    if field is rf.Group:
        field = rf.Group(n_groups=2, mask=True)
    model = rf.Model.xs(d_model=8, batch_size=2, amount=rf.Number, label=field)
    rows = pa.table({"amount": [1.0, 2.0, 3.0, 4.0], "label": labels})
    data = rf.ArrowDataModule(train=rows, validate=rows, shuffle=False)
    logger = RecordingLogger()
    model.fit(
        data,
        log_every_n_steps=1,
        logger=logger,
        num_sanity_val_steps=0,
        early_stopping=False,
        rollback=False,
        **fit_options,
    )
    training = [(metrics, step) for metrics, step in logger.records if "loss/train" in metrics]
    assert [step for _, step in training] == [0, 1]
    assert all(".label/train.loss.state" in metrics for metrics, _ in training)
    if has_content_scores:
        assert all(metrics[".label/train.targets.known"] == 2 for metrics, _ in training)
        assert all(metrics[".label/train.coverage.content"] == 1.0 for metrics, _ in training)
    else:
        assert all(".label/train.targets.known" not in metrics for metrics, _ in training)
        assert all(".label/train.coverage.content" not in metrics for metrics, _ in training)
    assert not any(name.endswith(("_step", "_epoch")) for metrics, _ in logger.records for name in metrics)
    validations = [metrics for metrics, _ in logger.records if "loss/validate" in metrics]
    assert len(validations) == 1
    assert math.isfinite(validations[0][".label/validate.loss.state"])
    if has_content_scores:
        assert validations[0][".label/validate.targets.known"] == 4
        assert validations[0][".label/validate.coverage.content"] == 1.0
    else:
        assert ".label/validate.targets.known" not in validations[0]
        assert ".label/validate.coverage.content" not in validations[0]
        assert ".label/validate.loss.content" not in validations[0]
    assert torch.isfinite(model.trainer.callback_metrics["loss/train_epoch"])


@pytest.mark.parametrize("masked", [False, True], ids=["input", "target"])
@pytest.mark.parametrize("native", [False, True], ids=["model-fit", "native-trainer"])
@pytest.mark.parametrize("accumulate", [1, 2])
def test_group_distribution_logs_only_at_training_batch_end(fit_options, monkeypatch, masked, native, accumulate):
    model = rf.Model.xs(
        d_model=8,
        batch_size=2,
        label=rf.Group(n_groups=3, mask=masked, p_unavailable=0.0),
        target=rf.Number(mask=True),
    )
    for token, assignment in zip(["a", "b", "c", "d"], [0, 1, 2, 2], strict=True):
        rf.Group.assign(model, "/label", token, assignment)
    rows = pa.table({"label": ["a", "b", "c", "d"], "target": [1.0, 2.0, 3.0, 4.0]})
    data = rf.ArrowDataModule(
        train=rows,
        validate=rows,
        test=rows,
        predict=rows,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    logger = RecordingLogger()
    calls = []
    log = model.log

    def record(**options):
        if options["name"].startswith(".label/train.vocabulary.") and not options["name"].endswith(".size"):
            calls.append((model._current_fx_name, model.trainer.global_step, dict(options)))
        log(**options)

    monkeypatch.setattr(model, "log", record)
    options = dict(
        accumulate_grad_batches=accumulate,
        log_every_n_steps=1,
        logger=logger,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        **fit_options,
    )
    if native:
        model.optimizer = rf.adamw(learning_rate=1e-3, fused=False)
        trainer = lit.Trainer(**{name: value for name, value in options.items() if name != "batch_size"})
        trainer.fit(model, datamodule=data)
    else:
        model.fit(data, early_stopping=False, rollback=False, **options)
        trainer = model.trainer
    trainer.validate(model, datamodule=data, verbose=False)
    trainer.test(model, datamodule=data, verbose=False)
    trainer.predict(model, datamodule=data)

    expected = {
        ".label/train.vocabulary.imbalance": 0.5,
        ".label/train.vocabulary.majority_class": 0.5,
        ".label/train.vocabulary.standard_deviation": math.sqrt(1 / 72),
    }
    assert len(calls) == 6
    assert [step for _, step, _ in calls] == ([1, 1, 1, 2, 2, 2] if accumulate == 1 else [0, 0, 0, 1, 1, 1])
    for hook, _, options in calls:
        assert hook == "on_train_batch_end"
        assert options["on_step"] is True
        assert options["on_epoch"] is False
        assert options["value"].item() == pytest.approx(expected[options["name"]], abs=1e-7)
    reports = [(metrics, step) for metrics, step in logger.records if expected.keys() & metrics.keys()]
    assert [step for _, step in reports] == ([0, 1] if accumulate == 1 else [0])
    for metrics, _ in reports:
        assert {name: metrics[name] for name in expected} == pytest.approx(expected)
    assert not any(
        name.endswith(("vocabulary.imbalance", "vocabulary.majority_class", "vocabulary.standard_deviation"))
        and name not in expected
        for metrics, _ in logger.records
        for name in metrics
    )
