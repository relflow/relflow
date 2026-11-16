"""Fit presets select run defaults without changing the model's architecture."""

from types import SimpleNamespace

import lightning.pytorch as lit
import pyarrow as pa
import pytest
from lightning.pytorch.callbacks import EarlyStopping

import relflow as rf
from relflow.architecture import training
from relflow.presets import registry


@pytest.fixture
def runs(monkeypatch):
    captured = []

    def trainer(model, data, **options):
        run = dict(options)
        captured.append(run)

        def fit(module, **arguments):
            run["optimizer"] = module.optimizer(module) if callable(module.optimizer) else module.optimizer
            run["arguments"] = arguments

        return SimpleNamespace(fit=fit)

    monkeypatch.setattr(training, "trainer", trainer)
    return captured


@pytest.fixture
def model():
    return rf.Model(preset="small", d_model=16, amount=rf.Number, label=rf.Boolean(mask=True))


@pytest.mark.parametrize("name", ["small", "medium"])
def test_fit_uses_the_selected_training_recipe_and_preserves_architecture(model, runs, name):
    schema = model.schema.model_dump()
    profile = model.preset
    model.fit(lit.LightningDataModule(), preset=name)
    recipe = rf.presets.get(name).fit
    run = runs[-1]
    assert run["batch_size"] == recipe.batch_size
    assert run["options"] == {"max_epochs": 8, "gradient_clip_val": 0.25}
    assert run["optimizer"].defaults["lr"] == recipe.learning_rate
    assert run["optimizer"].defaults["weight_decay"] == recipe.weight_decay
    assert len(run["callbacks"]) == 1
    assert isinstance(run["callbacks"][0], rf.EMA)
    assert run["callbacks"][0].decay == 0.995
    assert model.schema.model_dump() == schema
    assert model.preset is profile
    assert model.optimizer is None


def test_explicit_options_override_the_preset_including_disabled_averaging(model, runs):
    model.fit(
        lit.LightningDataModule(),
        preset="small",
        batch_size=3,
        learning_rate=0.02,
        weight_decay=0.0,
        ema_decay=None,
        max_steps=17,
        gradient_clip_val=0.0,
        early_stopping=True,
        rollback=False,
    )
    run = runs[-1]
    assert run["batch_size"] == 3
    assert run["optimizer"].defaults["lr"] == 0.02
    assert run["optimizer"].defaults["weight_decay"] == 0.0
    assert "max_epochs" not in run["options"]
    assert run["options"]["max_steps"] == 17
    assert run["options"]["gradient_clip_val"] == 0.0
    assert any(isinstance(callback, EarlyStopping) for callback in run["callbacks"])
    assert not any(isinstance(callback, rf.EMA) for callback in run["callbacks"])


def test_an_explicit_optimizer_replaces_preset_optimizer_defaults(model, runs):
    model.fit(lit.LightningDataModule(), preset="medium", optimizer=rf.adamw(0.02, weight_decay=0.3))
    assert runs[-1]["optimizer"].defaults["lr"] == 0.02
    assert runs[-1]["optimizer"].defaults["weight_decay"] == 0.3
    with pytest.raises(ValueError, match="conflict"):
        model.fit(lit.LightningDataModule(), preset="small", optimizer=rf.adamw(1e-3), learning_rate=0.1)


def test_an_explicit_ema_callback_replaces_the_preset_default(model, runs):
    average = rf.EMA(decay=0.9)
    model.fit(lit.LightningDataModule(), preset="small", callbacks=[average])
    assert runs[-1]["callbacks"] == [average]
    with pytest.raises(ValueError, match="ema_decay conflicts"):
        model.fit(lit.LightningDataModule(), preset="small", callbacks=[average], ema_decay=None)


def test_presets_must_exist_and_declare_fit_defaults(model, runs, monkeypatch):
    with pytest.raises(ValueError, match="unknown preset"):
        model.fit(lit.LightningDataModule(), preset="xs")
    profile = rf.presets.SMALL.model_copy(update={"name": "architecture-only", "fit": None})
    monkeypatch.setattr(registry, "PRESETS", dict(registry.PRESETS))
    rf.presets.register(profile)
    with pytest.raises(ValueError, match="no fit configuration"):
        model.fit(lit.LightningDataModule(), preset=profile.name)
    assert not runs


def test_constructor_preset_does_not_implicitly_select_a_fit_recipe(model, runs):
    model.fit(lit.LightningDataModule())
    run = runs[-1]
    assert run["batch_size"] is None
    assert run["optimizer"].defaults["lr"] == 1e-3
    assert run["optimizer"].defaults["weight_decay"] == 0.01
    assert "max_epochs" not in run["options"]
    assert not any(isinstance(callback, rf.EMA) for callback in run["callbacks"])
    assert any(isinstance(callback, EarlyStopping) for callback in run["callbacks"])


@pytest.mark.integration
def test_preset_fit_trains_and_updates_ema_on_a_native_loop(tmp_path):
    model = rf.Model(preset="small", d_model=8, value=rf.Number, label=rf.Boolean(mask=True))
    table = pa.table({"value": [1.0, 2.0], "label": [False, True]})
    data = rf.ArrowDataModule(train=table, num_workers=0, persistent_workers=False, pin_memory=False)
    model.fit(
        data,
        preset="small",
        batch_size=2,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        fused=False,
        default_root_dir=tmp_path,
        enable_progress_bar=False,
    )
    assert model.trainer.global_step == 1
    average = next(callback for callback in model.trainer.callbacks if isinstance(callback, rf.EMA))
    assert average.updates == 1
    assert average.finished
