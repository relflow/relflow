"""Input metadata stays bounded and independent of learned training resources."""

import copy
import json
import pickle

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture.runtime import profile, step
from relflow.data.datasets.arrow import ship
from relflow.data.iterables import encode
from relflow.logging import logger


def captured(monkeypatch):
    records = []
    monkeypatch.setattr(logger.logger, "handle", records.append)
    return records


def batches(model, values, strata=rf.Strata.predict):
    return encode(
        pa.table({"value": values}), model.schema, strata, model.interprocess_encoding_context, drift=model.drift
    )


def test_numeric_shift_warns_once_and_preserves_training_state(monkeypatch):
    records = captured(monkeypatch)
    model = rf.Model.xs(drift=True, value=rf.Number)
    model.encode(pa.table({"value": np.linspace(0, 1, 512)}), strata="train")
    tensors = copy.deepcopy(model.state_dict())
    for split in ("validate", "test", "predict"):
        model.encode(pa.table({"value": np.linspace(10, 11, 512)}), strata=split)
        model.encode(pa.table({"value": np.linspace(10, 11, 512)}), strata=split)
    warnings = [record.relflow_context for record in records if record.levelname == "WARNING"]
    assert {(r["strata"], r["feature"]) for r in warnings} == {(s, "value") for s in ("validate", "test", "predict")}
    assert len(warnings) == 3
    assert all(r["address"] == "/value" and r["reference_count"] == 512 and r["distance"] > 0.9 for r in warnings)
    for key, value in tensors.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(model.state_dict()[key], value, equal_nan=True)
    summary = model.distributions.summary()["/value"]
    assert summary["train"]["value"]["count"] == 512
    assert summary["test"]["value"]["count"] == 1024
    json.dumps(summary, allow_nan=False)


def test_matching_small_and_missing_references_are_quiet(monkeypatch):
    records = captured(monkeypatch)
    model = rf.Model.xs(drift=True, value=rf.Number)
    model.encode(pa.table({"value": np.zeros(512)}))
    model.encode(pa.table({"value": np.zeros(512)}), strata="train")
    model.distributions.clear(strata="predict")
    model.encode(pa.table({"value": np.ones(10)}))
    assert not [r for r in records if r.levelname == "WARNING"]


def test_same_distribution_across_independent_samples_is_quiet(monkeypatch):
    records = captured(monkeypatch)
    rng = np.random.default_rng(7)
    model = rf.Model.xs(drift=True, value=rf.Number)
    model.encode(pa.table({"value": rng.normal(size=4096)}), strata="train")
    model.encode(pa.table({"value": rng.normal(size=4096)}))
    assert not [r for r in records if r.levelname == "WARNING"]


@pytest.mark.parametrize("option", [False, rf.Drift(threshold=0.95)])
def test_expected_difference_can_be_disabled_or_tolerated(monkeypatch, option):
    records = captured(monkeypatch)
    model = rf.Model.xs(drift=True, value=rf.Number(drift=option))
    model.encode(pa.table({"value": np.zeros(512)}), strata="train")
    model.encode(pa.table({"value": np.ones(512)}))
    assert not [r for r in records if r.levelname == "WARNING"]
    if option is False:
        assert model.distributions.summary() == {}


@pytest.mark.parametrize(
    "kwargs",
    [{"threshold": 0}, {"threshold": 1}, {"threshold": float("nan")}, {"min_samples": True}, {"min_samples": 1}],
)
def test_policy_validation(kwargs):
    with pytest.raises(ValueError):
        rf.Drift(**kwargs)


def test_nulls_nonfinite_and_padding_are_distinct(monkeypatch):
    records = captured(monkeypatch)
    model = rf.Model.xs(drift=True, items=rf.Branch(length=4, value=rf.Number))
    for split, value in [("train", 1.0), ("test", None)]:
        model.encode(pa.Table.from_pylist([{"items": [{"value": value}]}] * 512), strata=split)
    states = model.distributions.summary()["/items/value"]["test"]["state"]
    assert states["count"] == 512
    assert states["counts"] == [0, 512, 0, 0, 0]
    assert any(r.relflow_context.get("feature") == "state" for r in records if r.levelname == "WARNING")
    model = rf.Model.xs(drift=True, value=rf.Number)
    model.encode(pa.table({"value": [1.0, float("nan"), float("inf"), -float("inf")]}))
    assert model.distributions.summary()["/value"]["predict"]["nonfinite"]["counts"] == [1, 1, 1, 1]


@pytest.mark.parametrize(
    "spec, values, expected",
    [
        (rf.Boolean, [True, False], {"state", "value"}),
        (rf.Category, ["A", "B"], {"state", "label"}),
        (rf.Enum(values=("A", "B")), ["A", "B"], {"state", "label"}),
        (rf.Cluster(bounds=4), ["A", "B"], {"state", "label"}),
        (rf.Hash, ["A", "B"], {"state", "label"}),
        (rf.Set, [["A", "A"], ["B"]], {"state", "size", "member"}),
        (rf.Vector(n_dim=2), [[1.0, 2.0], [3.0, 4.0]], {"state", "mean", "norm", "finite"}),
        (rf.Period("month_of_year"), ["2026-01-01", "2026-02-01"], {"state", "month_of_year.sin", "month_of_year.cos"}),
        (rf.Quantile, [1.0, 2.0], {"state", "value", "nonfinite"}),
    ],
)
def test_builtins_offer_explicit_profile_components(spec, values, expected):
    model = rf.Model.xs(drift=True, value=spec)
    batch = batches(model, values)
    assert set(batch.profiles["/value"]) == expected
    assert model.distributions.profiles == {}  # Encoding workers never own metadata.
    profile(model, batch, rf.Strata.predict)
    assert model.distributions.summary()["/value"]["predict"]["state"]["count"] == 2


def test_folded_categories_and_deduplicated_sets():
    model = rf.Model.xs(drift=True, value=rf.Category(strip=True, cased=False))
    a = batches(model, [" A ", "B"]).profiles["/value"]["label"]
    b = batches(model, ["a", "b"]).profiles["/value"]["label"]
    assert a.compare(b)[0] == 0
    model = rf.Model.xs(drift=True, value=rf.Set(strip=True, cased=False))
    batch = batches(model, [[" A ", "a", None], [], ["B"]])
    assert batch.profiles["/value"]["member"].count == 2
    assert batch.profiles["/value"]["size"].summary()["quantiles"][-1] == 1


@pytest.mark.integration
def test_profile_survives_shipping_transfer_and_pickle():
    model = rf.Model.xs(drift=True, value=rf.Number(embed=True))
    batch = batches(model, [1.0, 2.0])
    moved = model.transfer_batch_to_device(pickle.loads(pickle.dumps(ship(batch))), torch.device("cpu"), 0)
    assert moved.profiles["/value"]["value"].summary() == batch.profiles["/value"]["value"].summary()
    step(model, moved, 0, strata=rf.Strata.predict)
    assert model.distributions.summary()["/value"]["predict"]["value"]["count"] == 2


def test_checkpoint_restores_reference_and_schema_policy(tmp_path):
    model = rf.Model.xs(drift=True, value=rf.Number(drift=rf.Drift(threshold=0.3)))
    model.encode(pa.table({"value": np.arange(512.0)}), strata="train")
    path = tmp_path / "model.ckpt"
    model.save(path)
    loaded = rf.Model.load(path)
    assert loaded.drift is True
    assert loaded.schema.requests["/value"].drift.threshold == 0.3
    assert loaded.distributions.summary() == model.distributions.summary()
    loaded.distributions.clear(strata="train")
    assert "train" in model.distributions.summary()["/value"]


def test_profile_size_is_bounded_and_clear_rearms_warnings(monkeypatch):
    records = captured(monkeypatch)
    model = rf.Model.xs(drift=True, value=rf.Number)
    for index in range(8):
        model.encode(pa.table({"value": np.arange(index * 1024.0, (index + 1) * 1024.0)}), strata="train")
    digest = model.distributions.profiles["/value"]["train"]["value"].digest
    assert digest.count == 8192 and digest.centroids.num_rows <= digest.compression + 3
    for _ in range(2):
        model.encode(pa.table({"value": np.full(512, -100.0)}))
        model.distributions.clear(strata="predict")
    assert len([r for r in records if r.levelname == "WARNING"]) == 2


def test_custom_extension_profiles_use_the_public_registration_boundary(monkeypatch):
    # Reuse a cheap codec; the registered request and feature remain third-party owned.
    extension = rf.Extension(name="distribution_test", types=(float | int,))

    class Request(rf.Number):
        type: str = "distribution_test"

    extension.register(rf.Component.Request, Request)
    extension.register(rf.Component.TensorField, rf.TENSORFIELDS["number"].TensorField)
    extension.register(rf.Component.Embedder, rf.TENSORFIELDS["number"].Embedder)
    extension.register(rf.Component.Decoder, rf.TENSORFIELDS["number"].Decoder)
    calls = []

    @extension.register(rf.Component.profile)
    def observe(field, *, request):
        calls.append(request.type)
        return {"custom": rf.Distribution.numeric(field.values)}

    try:
        schema = rf.Schema.from_tree(d_model=8, n_layers=1, n_heads=2, value=Request())
        batch = encode(pa.table({"value": [1.0, 2.0]}), schema, rf.Strata.predict, {}, drift=True)
        assert set(batch.profiles["/value"]) == {"state", "custom"}
        assert calls == ["distribution_test"]
    finally:
        del rf.TENSORFIELDS["distribution_test"]


@pytest.mark.parametrize("spec", [rf.Boolean, rf.Category, rf.Set, rf.Number, rf.Vector(n_dim=2)])
def test_empty_profiles_do_not_require_a_physical_value_type(spec):
    model = rf.Model.xs(drift=True, value=spec)
    model.encode(pa.table({"value": pa.array([], type=pa.null())}))
    summary = model.distributions.summary()["/value"]["predict"]
    assert summary["state"]["count"] == 0


def test_missing_prediction_targets_are_expected():
    model = rf.Model.xs(drift=True, value=rf.Number, label=rf.Boolean(mask=True))
    model.encode(pa.table({"value": np.zeros(512), "label": np.ones(512, dtype=bool)}), strata="train")
    model.encode(pa.table({"value": np.zeros(512)}))
    assert "predict" not in model.distributions.summary()["/label"]


def test_changed_field_discards_its_reference():
    model = rf.Model.xs(drift=True, value=rf.Category)
    model.encode(pa.table({"value": ["A", "B"]}), strata="train")
    model.update(rf.where("address") == "/value", cased=False)
    assert model.distributions.summary() == {}


@pytest.mark.integration
@pytest.mark.parametrize("workers", [0, 1])
@pytest.mark.parametrize("enabled", [False, True])
def test_real_trainer_commits_each_batch_once(tmp_path, workers, enabled):
    import lightning.pytorch as lit

    model = rf.Model.xs(drift=enabled, value=rf.Number, label=rf.Boolean(mask=True), optimizer=rf.adamw(1e-3))
    train = pa.table({"value": np.zeros(128), "label": np.zeros(128, dtype=bool)})
    validation = pa.table({"value": np.ones(128), "label": np.zeros(128, dtype=bool)})
    data = rf.ArrowDataModule(
        model=model, train=train, validate=validation, num_workers=workers, persistent_workers=False, pin_memory=False
    )
    model.batch_size = 128
    trainer = lit.Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(model, datamodule=data)
    if not enabled:
        assert model.distributions.summary() == {}
        return
    summary = model.distributions.summary()["/value"]
    assert summary["train"]["value"]["count"] == 128
    assert summary["validate"]["value"]["count"] == 128
    assert ("/value", "validate", "value") in model.distributions.warned


def test_categorical_encoding_change_discards_incompatible_history():
    from relflow.distributions import Distributions

    monitor = Distributions()
    address = rf.Address("/value")
    policy = {address: rf.Drift()}
    for split in (rf.Strata.train, rf.Strata.test):
        monitor.update(
            {address: {"label": rf.Distribution.categorical(np.array([0, 1]), bins=2, encoding="old")}},
            strata=split,
            policies=policy,
        )
    monitor.update(
        {address: {"label": rf.Distribution.categorical(np.array([0, 1]), bins=2, encoding="new")}},
        strata=rf.Strata.predict,
        policies=policy,
    )
    assert monitor.summary()["/value"]["train"] == {}
    assert monitor.summary()["/value"]["predict"]["label"]["count"] == 2


def test_numeric_shape_change_with_equal_means_is_detected(monkeypatch):
    records = captured(monkeypatch)
    model = rf.Model.xs(drift=True, value=rf.Number)
    model.encode(pa.table({"value": np.zeros(2048)}), strata="train")
    model.encode(pa.table({"value": np.tile([-10.0, 10.0], 1024)}))
    assert any(r.relflow_context.get("feature") == "value" for r in records if r.levelname == "WARNING")


def test_text_length_profile_does_not_load_an_encoder():
    from relflow.tensorfields.shared.distributions import text

    field = rf.RaggedField(
        values=pa.array(["a", "🦊a"]),
        state=pa.array([0, 0], type=pa.int8()),
        placement=pa.array([0, 1], type=pa.int64()),
        shape=(2, 1),
    )
    assert text(field, request=rf.Text())["characters"].summary()["quantiles"][-1] == 2


def test_categorical_shift_does_not_admit_evaluation_labels(monkeypatch):
    records = captured(monkeypatch)
    model = rf.Model.xs(drift=True, value=rf.Category)
    model.encode(pa.table({"value": ["training"] * 512}), strata="train")
    vocabulary = rf.Category.vocabulary(model, "/value")
    model.encode(pa.table({"value": ["evaluation"] * 512}))
    assert rf.Category.vocabulary(model, "/value") == vocabulary
    assert any(r.relflow_context.get("feature") == "label" for r in records if r.levelname == "WARNING")


def test_chunked_prediction_counts_processed_inputs_once():
    model = rf.Model.xs(drift=True, value=rf.Number(embed=True))
    output = model.predict(pa.table({"value": np.arange(9.0)}), batch_size=4)
    assert output.num_rows == 9
    assert model.distributions.summary()["/value"]["predict"]["value"]["count"] == 9


@pytest.mark.parametrize(
    "result, error",
    [(None, TypeError), ({"value": 1}, TypeError), ({"state": rf.Distribution.numeric(pa.array([1.0]))}, ValueError)],
)
def test_profiler_contract_is_checked_at_the_extension_boundary(monkeypatch, result, error):
    monkeypatch.setitem(rf.TENSORFIELDS["number"].components, rf.Component.profile, lambda field, *, request: result)
    model = rf.Model.xs(drift=True, value=rf.Number)
    with pytest.raises(error, match="profiler at '/value'"):
        model.encode(pa.table({"value": [1.0]}))


def test_numeric_labels_preserve_equality_and_large_integer_identity():
    model = rf.Model.xs(drift=True, value=rf.Category)
    integers = batches(model, [0, 1, 2]).profiles["/value"]["label"]
    floats = batches(model, [-0.0, 1.0, 2.0]).profiles["/value"]["label"]
    assert integers.compare(floats)[0] == 0
    first = batches(model, [2**60]).profiles["/value"]["label"]
    second = batches(model, [2**60 + 1]).profiles["/value"]["label"]
    assert first.compare(second)[0] == 1


def test_vector_validity_matches_float32_codec():
    model = rf.Model.xs(drift=True, value=rf.Vector(n_dim=2))
    profile = batches(model, [[1e300, 0.0], [3.0, 4.0]]).profiles["/value"]
    assert profile["finite"].summary()["counts"] == [1, 1]
    assert profile["norm"].summary()["quantiles"] == [5.0] * 5


def test_set_profile_rebases_sliced_arrow_offsets():
    from relflow.tensorfields.shared.distributions import sets

    values = pa.array([["a", "b"], ["c"], ["d", "e"], []]).slice(2, 2)
    field = rf.RaggedField(
        values=values, state=pa.array([0, 0], type=pa.int8()), placement=pa.array([0, 1], type=pa.int64()), shape=(2, 1)
    )
    profiles = sets(field, request=rf.Set())
    assert profiles["member"].count == 2
    assert profiles["size"].count == 2
    assert profiles["size"].summary()["quantiles"][-1] == 2


@pytest.mark.parametrize("constructor", [rf.Model, rf.Model.xs, rf.Model.sm, rf.Model.md, rf.Model.lg, rf.Model.xl])
def test_model_wide_monitoring_requires_explicit_opt_in(constructor):
    options = dict(d_model=8, n_layers=1, n_heads=2, value=rf.Number(drift=rf.Drift(threshold=0.3)))
    model = constructor(**options)
    assert model.drift is False
    source = pa.table({"value": np.arange(256.0)})
    for strata in rf.Strata:
        model.encode(source, strata=strata)
    assert model.distributions.summary() == {}
    enabled = constructor(drift=True, **options)
    enabled.encode(source, strata="train")
    assert enabled.distributions.summary()["/value"]["train"]["value"]["count"] == 256


def test_disabled_model_skips_all_profile_work_for_direct_and_loader_paths(monkeypatch):
    model = rf.Model.xs(value=rf.Number(drift=True, embed=True))
    source = pa.table({"value": np.arange(256.0)})

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled model called a distribution profiler")

    monkeypatch.setattr(rf.Distribution, "categorical", forbidden)
    monkeypatch.setattr(rf.Distribution, "numeric", forbidden)
    monkeypatch.setitem(rf.TENSORFIELDS["number"].components, rf.Component.profile, forbidden)
    for strata in rf.Strata:
        model.encode(source, strata=strata)
    assert model.predict(source, batch_size=128).num_rows == 256
    data = rf.ArrowDataModule(model=model, train=source, validate=source, test=source, predict=source, num_workers=0)
    for strata in rf.Strata:
        assert all(not batch.profiles for batch in data.dataloader(strata))
    model.reset_contracts()
    assert model.distributions.state_dict() == rf.Distributions().state_dict()


def test_disabling_model_blocks_precomputed_profiles_and_preserves_history():
    model = rf.Model.xs(drift=True, value=rf.Number)
    model.encode(pa.table({"value": np.arange(256.0)}), strata="train")
    batch = batches(model, np.arange(256.0) + 1000)
    history = model.distributions.state_dict()
    model.drift = False
    profile(model, batch, rf.Strata.predict)
    model.encode(pa.table({"value": np.arange(256.0) + 1000}))
    assert model.distributions.state_dict() == history
    assert not batches(model, [1.0, 2.0]).profiles


@pytest.mark.parametrize("enabled", [False, True])
def test_global_drift_setting_round_trips_through_checkpoint(tmp_path, enabled):
    model = rf.Model.xs(drift=enabled, value=rf.Number)
    model.encode(pa.table({"value": np.arange(256.0)}), strata="train")
    path = model.save(tmp_path / "model.ckpt")
    saved = torch.load(path, weights_only=True)
    assert saved["drift"] is enabled
    loaded = rf.Model.load(path)
    assert loaded.drift is enabled
    assert loaded.distributions.state_dict() == model.distributions.state_dict()


def test_old_checkpoint_does_not_implicitly_enable_monitoring(tmp_path):
    model = rf.Model.xs(drift=True, value=rf.Number)
    path = model.save(tmp_path / "model.ckpt")
    saved = torch.load(path, weights_only=True)
    saved.pop("drift")
    saved.pop("distributions")
    torch.save(saved, path)
    loaded = rf.Model.load(path)
    assert loaded.drift is False
    loaded.encode(pa.table({"value": np.arange(256.0)}), strata="train")
    assert loaded.distributions.summary() == {}


def test_failed_checkpoint_restore_retains_global_drift_setting(tmp_path):
    source = rf.Model.xs(drift=True, value=rf.Number)
    path = source.save(tmp_path / "source.ckpt")
    saved = torch.load(path, weights_only=True)
    saved["state_dict"] = {}
    target = rf.Model.xs(value=rf.Number)
    with pytest.raises(RuntimeError, match="Missing key"):
        target.restore_checkpoint_state(saved)
    assert target.drift is False


@pytest.mark.parametrize("preset", ["", "preset: xs\n"])
def test_yaml_template_accepts_global_drift_and_explicit_override(tmp_path, preset):
    path = tmp_path / "model.yaml"
    path.write_text(preset + "d_model: 8\nn_layers: 1\nn_heads: 2\ndrift: true\nfields:\n  value: {type: number}\n")
    assert rf.Model.from_yaml(path).drift is True
    assert rf.Model.from_yaml(path, drift=False).drift is False


def test_drift_named_input_remains_available_through_fields_mapping():
    model = rf.Model.xs(drift=True, fields={"drift": rf.Number})
    model.encode(pa.table({"drift": [1.0, 2.0]}))
    assert model.drift is True and "/drift" in model.distributions.profiles


@pytest.mark.parametrize("value", [None, 1, 0, "true", rf.Drift()])
def test_global_drift_rejects_non_boolean_values(value):
    model = rf.Model.xs(value=rf.Number)
    with pytest.raises(TypeError, match="drift must be a boolean"):
        model.drift = value


@pytest.mark.integration
@pytest.mark.parametrize("enabled", [False, True])
def test_loader_worker_receives_global_opt_in(enabled):
    model = rf.Model.xs(drift=enabled, value=rf.Number)
    data = rf.ArrowDataModule(
        model=model,
        predict=pa.table({"value": np.arange(16.0)}),
        num_workers=1,
        persistent_workers=False,
        pin_memory=False,
    )
    batches = list(data.predict_dataloader())
    assert batches and all(bool(batch.profiles) is enabled for batch in batches)
    assert model.distributions.summary() == {}


def test_default_prediction_does_not_copy_pristine_fields_for_profiling(monkeypatch):
    from relflow.data import iterables

    model = rf.Model.xs(value=rf.Number(embed=True))

    def forbidden(*args, **kwargs):
        raise AssertionError("default prediction copied a pristine field solely for profiling")

    monkeypatch.setattr(iterables, "replace", forbidden)
    model.encode(pa.table({"value": [1.0, 2.0]}))
    assert model.predict(pa.table({"value": [1.0, 2.0]})).num_rows == 2
