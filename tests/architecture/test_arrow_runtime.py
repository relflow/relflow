import polars as pl
import pyarrow as pa
import pytest

import relflow as rf
from relflow.architecture.runtime import ModelRuntime
from relflow.data.arrow import Encoded
from relflow.structs.enums import Component, Strata, TensorKey
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS


def model(*, nested: bool = False, embed: bool = False) -> rf.Model:
    fields = (
        {
            "items": rf.Branch(
                length=3,
                value=rf.Number,
                label=rf.Boolean(mask=True),
            )
        }
        if nested
        else {"value": rf.Number, "label": rf.Boolean(mask=True)}
    )
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        embed=embed,
        **fields,
    )


def test_predict_returns_canonical_arrow_table_with_retained_inputs():
    configured = model()
    source = pa.table({"request_id": ["a", "b"], "value": [1.0, 2.0]})

    result = configured.predict(source, retain=("request_id",))

    assert isinstance(result, pa.Table)
    assert result.column_names == ["inputs", "predictions"]
    assert len(result) == 2
    assert result["inputs"].combine_chunks().type == pa.struct([("request_id", pa.string())])
    prediction_type = result["predictions"].combine_chunks().type
    assert prediction_type.get_field_index("/label") == 0
    assert result["inputs"].to_pylist() == [{"request_id": "a"}, {"request_id": "b"}]


def test_lightning_predict_step_uses_the_datamodule_retain_plan():
    configured = model()
    data = rf.ArrowDataModule(
        model=configured,
        predict=pa.table({"request_id": ["a"], "value": [1.0]}),
        retain=("request_id",),
        shuffle=False,
    )
    batch = next(iter(data.predict_dataloader()))

    result = configured.predict_step(batch, 0)

    assert isinstance(result, pa.Table)
    assert result["inputs"].to_pylist() == [{"request_id": "a"}]


def test_training_dataloader_carries_pristine_observations_into_step():
    configured = model()
    data = rf.ArrowDataModule(
        model=configured,
        train=pa.table({"value": [1.0, 100.0], "label": [False, True]}),
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    batch = next(iter(data.train_dataloader()))
    assert isinstance(batch, Encoded)
    assert rf.Number.normalization(configured, "/value")["count"] == 0

    result = configured.training_step(batch, 0)
    statistics = rf.Number.normalization(configured, "/value")

    assert result is not None
    assert statistics["count"] == 2
    assert statistics["mean"] == pytest.approx(50.5)
    assert statistics["variance"] == pytest.approx(2450.25)


def test_predict_adapts_a_small_python_record_sequence_once():
    result = model().predict([{"value": 1.0}, {"value": 2.0}])

    assert isinstance(result, pa.Table)
    assert len(result) == 2


@pytest.mark.parametrize("kind", ["table", "record_batch", "mappings"])
@pytest.mark.parametrize("batch_size,expected", [(None, [5]), (2, [2, 2, 1]), (3, [3, 2])])
def test_predict_bounds_preparation_and_forward_without_reordering(
    monkeypatch: pytest.MonkeyPatch, kind: str, batch_size: int | None, expected: list[int]
):
    configured = model()
    source = pa.table({"request_id": ["a", "b", "c", "d", "e"], "value": [1.0, 2.0, 3.0, 4.0, 5.0]})
    inputs = source if kind == "table" else source.to_batches()[0] if kind == "record_batch" else source.to_pylist()
    prepare = ModelRuntime.prepare
    prepared: list[int] = []
    forwarded: list[int] = []

    def record_prepare(module, batch, **options):
        prepared.append(batch.num_rows)
        return prepare(module, batch, **options)

    def record_forward(module, args):
        forwarded.append(args[0][Address("value")].state.shape[0])

    monkeypatch.setattr(ModelRuntime, "prepare", staticmethod(record_prepare))
    configured.register_forward_pre_hook(record_forward)

    result = configured.predict(inputs, batch_size=batch_size, retain=("request_id",))

    assert prepared == expected
    assert forwarded == expected
    assert configured.batch_size == (2 if batch_size is None else batch_size)
    assert result["inputs"].to_pylist() == [{"request_id": name} for name in source["request_id"].to_pylist()]
    assert result["predictions"].type.get_field_index("/label") == 0


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5, "2"])
def test_predict_rejects_invalid_batch_size_before_preprocessing_or_forward(batch_size):
    configured = model()
    calls: list[str] = []

    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("preprocess")
        return frame

    def record_forward(module, args):
        calls.append("forward")

    configured.register_forward_pre_hook(record_forward)
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        configured.predict(pa.table({"value": [1.0]}), batch_size=batch_size, preprocess=prepare)

    assert calls == []
    assert configured.batch_size == 2


def test_predict_batches_after_global_preprocessor_expansion_and_postprocesses_once(monkeypatch: pytest.MonkeyPatch):
    configured = model()
    calls = {"pre": 0, "post": 0}
    prepared: list[int] = []
    prepare = ModelRuntime.prepare

    def record_prepare(module, batch, **options):
        prepared.append(batch.num_rows)
        return prepare(module, batch, **options)

    @rf.preprocess(scope="dataset")
    def expand(frame: pl.DataFrame) -> list[pl.DataFrame]:
        calls["pre"] += 1
        selected = frame.filter(pl.col("value") > 0).sort("value", descending=True)
        return [selected, selected.with_columns(pl.col("request_id") + "-copy")]

    @rf.postprocess
    def enumerate_rows(frame: pl.DataFrame) -> pl.DataFrame:
        calls["post"] += 1
        return frame.select(request_id=pl.col("inputs").struct.field("request_id")).with_row_index("row")

    monkeypatch.setattr(ModelRuntime, "prepare", staticmethod(record_prepare))
    result = configured.predict(
        pa.table({"request_id": ["a", "b", "c", "d"], "value": [0.0, 1.0, 2.0, 3.0]}),
        batch_size=4,
        preprocess=expand,
        postprocess=enumerate_rows,
        retain=("request_id",),
    )

    assert calls == {"pre": 1, "post": 1}
    assert prepared == [4, 2]
    assert result.to_pydict() == {
        "row": [0, 1, 2, 3, 4, 5],
        "request_id": ["d", "c", "b", "d-copy", "c-copy", "b-copy"],
    }


def test_predict_rejects_fieldless_mapping_rows():
    configured = rf.Model(
        label=rf.Boolean(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    with pytest.raises(TypeError, match="without fields"):
        configured.predict([{}, {}])


def test_python_prediction_ingress_preserves_keys_introduced_after_the_first_row():
    result = model().predict(
        [{"value": 1.0}, {"value": 2.0, "request_id": "second"}],
        retain="*",
    )

    assert result["inputs"].to_pylist() == [
        {"value": 1.0, "request_id": None},
        {"value": 2.0, "request_id": "second"},
    ]


def test_predict_requires_typed_arrow_for_an_empty_sequence():
    with pytest.raises(ValueError, match="empty Python prediction sequence has no Arrow schema"):
        model().predict([])


@pytest.mark.parametrize("batch_size", [None, 2])
def test_typed_empty_prediction_compiles_the_exact_output_schema_without_forward(batch_size: int | None):
    source = pa.table(
        {
            "request_id": pa.array([], type=pa.large_string()),
            "value": pa.array([], type=pa.float64()),
        }
    )

    configured = model()

    def reject_forward(module, args):
        pytest.fail("typed empty observations must not enter forward")

    configured.register_forward_pre_hook(reject_forward)
    result = configured.predict(source, batch_size=batch_size, retain=("request_id",))

    assert len(result) == 0
    assert result["inputs"].type == pa.struct([pa.field("request_id", pa.large_string())])
    predictions = result["predictions"].type
    assert pa.types.is_struct(predictions)
    assert predictions.get_field_index("/label") == 0


@pytest.mark.parametrize("batch_size", [None, 2])
def test_preprocessor_may_filter_every_prediction_row_without_erasing_schema(batch_size: int | None):
    calls = {"pre": 0, "post": 0}

    @rf.preprocess
    def discard(frame: pl.DataFrame) -> pl.DataFrame:
        calls["pre"] += 1
        return frame.filter(pl.lit(False))

    @rf.postprocess
    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        calls["post"] += 1
        assert frame.height == 0
        return frame

    configured = model()

    def reject_forward(module, args):
        pytest.fail("filtered empty observations must not enter forward")

    configured.register_forward_pre_hook(reject_forward)
    result = configured.predict(
        pa.table({"request_id": ["a"], "value": [1.0]}),
        batch_size=batch_size,
        preprocess=discard,
        postprocess=finish,
        retain=("request_id",),
    )

    assert calls == {"pre": 1, "post": 1}
    assert len(result) == 0
    assert result["inputs"].type == pa.struct([pa.field("request_id", pa.large_string())])
    assert result["predictions"].type.get_field_index("/label") == 0


def test_predict_preserves_nested_model_axes_as_fixed_lists():
    configured = model(nested=True)
    source = pa.Table.from_pylist(
        [
            {"items": [{"value": 1.0}, {"value": 2.0}]},
            {"items": [{"value": 3.0}]},
        ]
    )

    result = configured.predict(source)
    predictions = result["predictions"].combine_chunks()
    field = predictions.type.field("/items/label")

    assert pa.types.is_fixed_size_list(field.type)
    assert field.type.list_size == 3
    assert all(len(row["/items/label"]) == 3 for row in predictions.to_pylist())


def test_predict_batches_nested_outputs_with_retained_row_identity():
    configured = model(nested=True)
    source = pa.Table.from_pylist(
        [
            {"request_id": "a", "items": [{"value": 1.0}, {"value": 2.0}]},
            {"request_id": "b", "items": []},
            {"request_id": "c", "items": [{"value": 3.0}]},
            {"request_id": "d", "items": [{"value": 4.0}, {"value": 5.0}, {"value": 6.0}]},
            {"request_id": "e", "items": [{"value": 7.0}]},
        ]
    )

    result = configured.predict(source, batch_size=2, retain=("request_id",))
    predictions = result["predictions"].combine_chunks()

    assert result["inputs"].to_pylist() == [{"request_id": name} for name in ["a", "b", "c", "d", "e"]]
    assert predictions.type.field("/items/label").type.list_size == 3
    assert all(len(row["/items/label"]) == 3 for row in predictions.to_pylist())


def test_embedding_only_address_has_no_state_or_inferred_fields():
    configured = model(embed=True)

    result = configured.predict(pa.table({"value": [1.0]}))
    root = result["predictions"].combine_chunks().type.field("/").type

    assert [field.name for field in root] == ["embedding"]
    assert root.field("embedding").type == pa.list_(pa.float32(), 8)


def test_root_embedding_exposes_each_configured_reduction_output():
    configured = rf.Model(
        value=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=2,
        embed=True,
        reduction=rf.Attention(n_outputs=3),
    )

    result = configured.predict(pa.table({"value": [1.0, 2.0]}))
    root = result["predictions"].combine_chunks().type.field("/").type

    assert pa.types.is_fixed_size_list(root)
    assert root.list_size == 3
    assert root.value_type.field("embedding").type == pa.list_(pa.float32(), 8)


def test_none_reduction_root_embedding_exposes_every_input_token():
    configured = rf.Model(
        first=rf.Number,
        second=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=2,
        embed=True,
        attention=None,
        reduction=None,
    )

    result = configured.predict(pa.table({"first": [1.0], "second": [2.0]}))
    root = result["predictions"].combine_chunks().type.field("/").type

    assert pa.types.is_fixed_size_list(root)
    assert root.list_size == 2


def test_nested_branch_embedding_preserves_parent_and_reduction_axes():
    configured = rf.Model(
        parents=rf.Branch(
            length=3, children=rf.Branch(length=2, embed=True, reduction=rf.Attention(n_outputs=2), value=rf.Number)
        ),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    source = pa.Table.from_pylist(
        [
            {
                "parents": [
                    {"children": [{"value": 1.0}, {"value": 2.0}]},
                    {"children": [{"value": 3.0}]},
                ]
            }
        ]
    )

    result = configured.predict(source)
    children = result["predictions"].combine_chunks().type.field("/parents/children").type

    assert pa.types.is_fixed_size_list(children)
    assert children.list_size == 3
    assert pa.types.is_fixed_size_list(children.value_type)
    assert children.value_type.list_size == 2


def test_empty_retain_uses_typed_null_without_changing_row_count():
    result = model().predict(pa.table({"value": [1.0, 2.0, 3.0]}))

    assert result["inputs"].type == pa.null()
    assert result["inputs"].null_count == 3
    assert len(result) == 3


def test_reconstruction_without_public_extension_output_keeps_a_typed_null_prediction_column():
    configured = rf.Model(
        value=rf.Number,
        identifier=rf.Hash(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    result = configured.predict(pa.table({"value": [1.0, 2.0]}))

    assert result["predictions"].type == pa.null()
    assert result["predictions"].null_count == 2


def test_predict_runs_polars_preprocessor_and_postprocessor_once():
    calls = {"pre": 0, "post": 0}

    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        calls["pre"] += 1
        return frame.with_columns(pl.col("value") * 2)

    @rf.postprocess
    def compact(frame: pl.DataFrame) -> pl.DataFrame:
        calls["post"] += 1
        return frame.select(prepared=pl.col("inputs").struct.field("value"))

    result = model().predict(
        pa.table({"value": [1.0, 2.0]}),
        preprocess=prepare,
        postprocess=compact,
        retain=("value",),
    )

    assert calls == {"pre": 1, "post": 1}
    assert result.to_pydict() == {"prepared": [2.0, 4.0]}


def test_predict_runs_postprocessors_in_order():
    calls = []

    @rf.postprocess
    def project(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("project")
        return frame.select(prepared=pl.col("inputs").struct.field("value"))

    @rf.postprocess
    def double(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("double")
        return frame.select(doubled=pl.col("prepared") * 2)

    result = model().predict(
        pa.table({"value": [1.0, 2.0]}),
        postprocess=[project, double],
        retain=("value",),
    )

    assert calls == ["project", "double"]
    assert result.to_pydict() == {"doubled": [2.0, 4.0]}


@pytest.mark.parametrize("batch_size", [None, 2])
def test_predict_snapshots_postprocessors_before_preprocessing(batch_size: int | None):
    calls: list[str] = []

    @rf.postprocess
    def first(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("first")
        return frame

    @rf.postprocess
    def late(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("late")
        return frame

    configured = [first]

    @rf.preprocess
    def mutate(frame: pl.DataFrame) -> pl.DataFrame:
        configured.append(late)
        return frame

    model().predict(
        pa.table({"value": [1.0, 2.0, 3.0]}),
        batch_size=batch_size,
        preprocess=mutate,
        postprocess=configured,
    )

    assert configured == [first, late]
    assert calls == ["first"]


def test_predict_restores_training_mode_after_inference():
    configured = model()
    configured.train()

    configured.predict(pa.table({"value": [1.0]}))

    assert configured.training


@pytest.mark.parametrize("training", [False, True])
def test_predict_restores_mode_when_a_later_batch_fails(training: bool):
    configured = model()
    configured.train(training)
    forwarded: list[int] = []
    postprocessed: list[int] = []

    def fail_later(module, args):
        assert not module.training
        forwarded.append(args[0][Address("value")].state.shape[0])
        if len(forwarded) == 2:
            raise RuntimeError("second prediction batch failed")

    @rf.postprocess
    def finish(frame: pl.DataFrame) -> pl.DataFrame:
        postprocessed.append(frame.height)
        return frame

    configured.register_forward_pre_hook(fail_later)
    with pytest.raises(RuntimeError, match="second prediction batch failed"):
        configured.predict(pa.table({"value": [1.0, 2.0, 3.0, 4.0, 5.0]}), batch_size=2, postprocess=finish)

    assert forwarded == [2, 2]
    assert configured.training is training
    assert postprocessed == []


def test_predict_rejects_missing_or_duplicate_retain_columns():
    configured = model()
    source = pa.table({"value": [1.0]})

    with pytest.raises(KeyError, match="absent after preprocessing"):
        configured.predict(source, retain=("missing",))
    with pytest.raises(ValueError, match="must be unique"):
        configured.predict(source, retain=("value", "value"))


@pytest.mark.parametrize("batch_size,writes", [(None, 1), (2, 3)])
def test_output_plan_declares_once_and_passes_that_type_to_writer(
    monkeypatch: pytest.MonkeyPatch, batch_size: int | None, writes: int
):
    extension = TENSORFIELDS["boolean"]
    declare = extension.output
    render = extension.write
    calls = {"output": 0, "write": 0}
    declarations: dict[Address, pa.StructType] = {}

    def output(module, address):
        calls["output"] += 1
        datatype = declare(module=module, address=address)
        declarations[address] = datatype
        return datatype

    def write(module, prediction, datatype):
        calls["write"] += 1
        assert datatype is declarations[prediction.address]
        return render(module=module, prediction=prediction, datatype=datatype)

    monkeypatch.setitem(extension.components, Component.output, output)
    monkeypatch.setitem(extension.components, Component.write, write)

    model().predict(pa.table({"value": [1.0, 2.0, 3.0, 4.0, 5.0]}), batch_size=batch_size)

    assert calls == {"output": 1, "write": writes}


def test_write_rejects_an_ordinary_unplanned_forward_address():
    configured = model()
    encoded = ModelRuntime.prepare(
        configured,
        pa.table({"value": [1.0]}),
        preprocess=None,
        strata=Strata.predict,
    )
    predictions = configured(encoded.tensors, strata=Strata.predict)
    target = predictions[0]
    extra = Prediction(
        address=Address("/value"),
        payload=target.payload.clone(),
        batch_size=target.batch_size,
    )

    with pytest.raises(ValueError, match="unplanned prediction address.*/value"):
        configured.write([*predictions, extra], source=encoded.source)


def test_output_plan_rejects_runtime_embedding_width_drift():
    configured = model(embed=True)
    encoded = ModelRuntime.prepare(
        configured,
        pa.table({"value": [1.0]}),
        preprocess=None,
        strata=Strata.predict,
    )
    predictions = configured(encoded.tensors, strata=Strata.predict)
    root = next(prediction for prediction in predictions if prediction.address == Address("/"))
    root.payload[TensorKey.embedding] = root.payload[TensorKey.embedding][..., :-1]

    with pytest.raises(ValueError, match="must end with model width 8"):
        configured.write(predictions, source=encoded.source)
