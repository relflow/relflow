from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pytest
from tensordict import TensorDict
from torch.utils.data import IterableDataset

import relflow as rf
from relflow.data.arrow import Encoded
from relflow.data.datasets import arrow
from relflow.data.datasets.custom import adapt
from relflow.structs.enums import Strata


def model(batch_size: int = 2) -> rf.Model:
    return rf.Model.xs(d_model=8, batch_size=batch_size, id=rf.Number)


class Records(IterableDataset):
    def __init__(self, values):
        self.values = values

    def __iter__(self):
        yield from self.values


def collect(dataset: arrow.ArrowDataset, monkeypatch: pytest.MonkeyPatch) -> list[pa.Table]:
    monkeypatch.setattr(
        arrow,
        "encode",
        lambda *, batch, **kwargs: Encoded(
            tensors=TensorDict({}, batch_size=[len(batch)]),
            source=batch,
        ),
    )
    return [item.source for item in dataset]


def test_public_surface_has_five_modules_and_no_source_specific_datasets():
    import relflow.data.datasets as datasets

    assert rf.ArrowDataModule is arrow.ArrowDataModule
    assert datasets.__all__ == [
        "ArrowDataModule",
        "ArrowInput",
        "ArrowSource",
        "ArrowStream",
        "ArrowUnit",
        "CustomDataModule",
        "DuckDBDataModule",
        "PolarsDataModule",
        "Retain",
        "StratumConfig",
        "SyntheticDataModule",
    ]
    assert callable(rf.source)
    assert not hasattr(rf, "parquet")
    assert not hasattr(rf, "PartitionedSource")
    assert not hasattr(rf, "StreamingDataModule")
    assert not hasattr(datasets, "PolarsBatchDataset")
    assert not hasattr(datasets, "CustomBatchDataset")


@pytest.mark.parametrize(
    "module_type,source",
    [
        (rf.ArrowDataModule, pa.table({"id": [1, 2]})),
        (rf.PolarsDataModule, pl.DataFrame({"id": [1, 2]})),
        (rf.CustomDataModule, Records([{"id": 1}, {"id": 2}])),
        (rf.SyntheticDataModule, lambda: iter([{"id": 1}, {"id": 2}])),
        (rf.DuckDBDataModule, "SELECT id FROM (VALUES (1), (2)) observations(id) ORDER BY id"),
    ],
)
def test_all_data_modules_allow_deferred_model_binding(module_type, source):
    module = module_type(validate=source, shuffle=False)
    configured = model()

    assert module.model is None
    module.bind(configured)

    assert module.model is configured
    assert module.schema is configured.schema
    assert module.encoding_context == configured.interprocess_encoding_context
    batches = list(module.val_dataloader())
    assert pa.concat_tables([batch.source for batch in batches])["id"].to_pylist() == [1, 2]


@pytest.mark.parametrize("setting", ["schema", "batch_size", "encoding_context"])
def test_unbound_model_settings_raise_an_actionable_error(setting):
    module = rf.ArrowDataModule(model=None, train=pa.table({"id": [1]}))

    with pytest.raises(RuntimeError, match=r"not bound.*model\.fit"):
        getattr(module, setting)


def test_unbound_loader_access_does_not_open_the_source():
    calls = 0

    def source():
        nonlocal calls
        calls += 1
        yield pa.table({"id": [1]})

    module = rf.ArrowDataModule(train=source)

    with pytest.raises(RuntimeError, match="not bound"):
        module.train_dataloader()
    with pytest.raises(RuntimeError, match="not bound"):
        module.batch_size = 2

    assert calls == 0
    assert module.val_dataloader() is None
    module.bind(model())
    assert calls == 0
    assert len(list(module.train_dataloader())) == 1
    assert calls == 1


def test_binding_is_idempotent_and_rejects_a_different_model():
    configured = model()
    module = rf.ArrowDataModule(configured, train=pa.table({"id": [1]}))
    original = module.options.copy()

    module.bind(configured)
    with pytest.raises(ValueError, match="different model"):
        module.bind(model())

    assert module.model is configured
    assert module.options == original


def test_binding_requires_a_relflow_model():
    module = rf.ArrowDataModule(train=pa.table({"id": [1]}))

    with pytest.raises(TypeError, match="requires a relflow.Model"):
        module.bind(object())

    assert module.model is None


def test_lazy_binding_resolves_model_dependent_defaults_and_preserves_overrides():
    source = pa.table({"id": [1]})
    module = rf.ArrowDataModule(train=source, validate=source, shuffle_rows={"validate": 17})
    configured = model(batch_size=256)
    explicit = rf.ArrowDataModule(configured, train=source, validate=source, shuffle_rows={"validate": 17})

    module.bind(configured)

    assert module.options == explicit.options
    assert module.options[Strata.train].shuffle_rows == 8192
    assert module.options[Strata.validate].shuffle_rows == 17
    configured.batch_size = 1
    module.bind(configured)
    assert module.options[Strata.train].shuffle_rows == 8192


def test_native_trainer_setup_uses_the_same_binding_guard():
    module = rf.ArrowDataModule(train=pa.table({"id": [1]}))
    configured = model()
    module.trainer = SimpleNamespace(lightning_module=configured)

    module.setup("fit")

    assert module.model is configured
    module.trainer = SimpleNamespace(lightning_module=model())
    with pytest.raises(ValueError, match="different model"):
        module.setup("validate")


@pytest.mark.parametrize(
    "source",
    [
        pa.table({"id": [1, 2]}),
        pa.record_batch({"id": [1, 2]}),
    ],
)
def test_arrow_module_accepts_table_and_record_batch(source):
    module = rf.ArrowDataModule(model=model(), train=source, shuffle=False)

    assert module.sources == {Strata.train: source}
    assert isinstance(module.train_dataloader().dataset, arrow.ArrowDataset)
    assert module.val_dataloader() is None


def test_arrow_module_accepts_datasets_and_rejects_python_or_one_shot_sources():
    with pytest.raises(TypeError, match="CustomDataModule or SyntheticDataModule"):
        rf.ArrowDataModule(model=model(), train={"id": 1})

    reader = pa.RecordBatchReader.from_batches(pa.schema([("id", pa.int64())]), [])
    with pytest.raises(TypeError, match="one-shot RecordBatchReader"):
        rf.ArrowDataModule(model=model(), train=reader)

    dataset = ds.dataset(pa.table({"id": [1]}))
    module = rf.ArrowDataModule(model=model(), train=dataset)
    assert module.sources[Strata.train] is dataset
    with pytest.raises(TypeError, match="configured Scanner"):
        rf.ArrowDataModule(model=model(), train=dataset.scanner())


def test_empty_arrow_factory_must_declare_its_schema():
    module = rf.ArrowDataModule(model=model(), validate=lambda: iter(()), shuffle=False)

    with pytest.raises(ValueError, match="empty Arrow factory"):
        list(module.val_dataloader().dataset)


def test_zero_batch_reader_preserves_its_declared_schema(monkeypatch: pytest.MonkeyPatch):
    schema = pa.schema(
        [pa.field("id", pa.int64(), metadata={b"unit": b"stable"})],
        metadata={b"source": b"reader"},
    )

    def source():
        return pa.RecordBatchReader.from_batches(schema, [])

    module = rf.ArrowDataModule(model=model(), validate=source, shuffle=False)

    assert collect(module.val_dataloader().dataset, monkeypatch) == []
    assert module.schemas[Strata.validate].source.equals(schema, check_metadata=True)
    assert module.schemas[Strata.validate].processed.equals(schema, check_metadata=True)


def test_source_schema_lock_survives_repeated_dataloader_creation(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    def source():
        nonlocal calls
        calls += 1
        dtype = pa.int64() if calls == 1 else pa.string()
        value = 1 if calls == 1 else "1"
        yield pa.record_batch({"id": pa.array([value], type=dtype)})

    module = rf.ArrowDataModule(model=model(), validate=source, shuffle=False)

    collect(module.val_dataloader().dataset, monkeypatch)
    with pytest.raises(TypeError, match="Arrow source schema changed"):
        collect(module.val_dataloader().dataset, monkeypatch)


def test_processed_schema_lock_survives_repeated_dataloader_creation(monkeypatch: pytest.MonkeyPatch):
    calls = 0

    @rf.preprocess
    def change(frame: pl.DataFrame) -> pl.DataFrame:
        nonlocal calls
        calls += 1
        if calls == 1:
            return frame
        return frame.with_columns(pl.col("id").cast(pl.String))

    module = rf.ArrowDataModule(
        model=model(),
        validate=pa.table({"id": [1]}),
        preprocessor=change,
        shuffle=False,
    )

    collect(module.val_dataloader().dataset, monkeypatch)
    with pytest.raises(TypeError, match="preprocessor schema changed"):
        collect(module.val_dataloader().dataset, monkeypatch)


def test_arrow_pipeline_batches_without_materializing_rows(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"id": list(range(5)), "nested": [[1, 2], [], [3], [4, 5], None]})
    module = rf.ArrowDataModule(model=model(), validate=source, shuffle=False)
    dataset = module.val_dataloader().dataset

    batches = collect(dataset, monkeypatch)

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert pa.concat_tables(batches).equals(source)


@pytest.mark.parametrize(
    ("drop_last", "expected_sizes", "expected_values"),
    [
        (False, [3, 2], set(range(20))),
        (True, [3], set(range(12))),
    ],
)
def test_distributed_batches_are_disjoint_and_have_equal_tails(drop_last, expected_sizes, expected_values):
    source = arrow.convert(pa.table({"id": list(range(23))}))
    ranks = [
        list(
            arrow.distribute(
                (source,),
                size=3,
                global_rank=global_rank,
                world_size=4,
                drop_last=drop_last,
            )
        )
        for global_rank in range(4)
    ]

    assert [[len(batch) for batch in batches] for batches in ranks] == [expected_sizes] * 4
    values = [{value for batch in batches for value in batch["id"].to_pylist()} for batches in ranks]
    assert set().union(*values) == expected_values
    assert sum(map(len, values)) == len(set().union(*values))


def test_distributed_ownership_is_independent_of_arrow_batch_boundaries():
    table = pa.table({"id": list(range(24))})

    def assignments(sizes):
        offset = 0
        batches = []
        for size in sizes:
            batches.append(arrow.convert(table.slice(offset, size)))
            offset += size
        return [
            [
                value
                for batch in arrow.distribute(
                    batches,
                    size=3,
                    global_rank=global_rank,
                    world_size=4,
                    drop_last=False,
                )
                for value in batch["id"].to_pylist()
            ]
            for global_rank in range(4)
        ]

    assert assignments([24]) == assignments([1, 7, 3, 13])


def test_arrow_dataset_uses_distributed_rank_and_world_size(monkeypatch: pytest.MonkeyPatch):
    configured = rf.Model.xs(d_model=8, batch_size=2, id=rf.Category)
    module = rf.ArrowDataModule(
        model=configured,
        validate=pa.table({"id": ["a", "b", "c", "d", "e"]}),
        shuffle=False,
    )
    monkeypatch.setattr(arrow, "rank", lambda: 1)
    monkeypatch.setattr(arrow, "world_size", lambda: 2)

    dataset = module.val_dataloader().dataset
    batches = collect(dataset, monkeypatch)

    assert [batch["id"].to_pylist() for batch in batches] == [["b", "d"]]
    assert dataset.distributed_rank == 1
    assert dataset.distributed_world_size == 2


def test_arrow_workers_take_complete_batches_from_the_parent_rank(monkeypatch: pytest.MonkeyPatch):
    module = rf.ArrowDataModule(
        model=model(),
        validate=pa.table({"id": list(range(24))}),
        shuffle=False,
        num_workers=3,
    )
    monkeypatch.setattr(arrow, "rank", lambda: 1)
    monkeypatch.setattr(arrow, "world_size", lambda: 2)
    dataset = module.val_dataloader().dataset
    monkeypatch.setattr(arrow, "get_worker_info", lambda: SimpleNamespace(id=1, num_workers=3))

    batches = collect(dataset, monkeypatch)

    assert [batch["id"].to_pylist() for batch in batches] == [[5, 7], [17, 19]]


@pytest.mark.parametrize(
    ("rows", "workers", "drop_last", "expected"),
    [(5, 2, False, list(range(5))), (1, 3, False, [0]), (5, 3, True, list(range(4)))],
)
def test_arrow_workers_preserve_single_rank_tail_rows(rows, workers, drop_last, expected, monkeypatch):
    module = rf.ArrowDataModule(
        model=model(),
        predict=pa.table({"id": list(range(rows))}),
        num_workers=workers,
        drop_last=drop_last,
    )
    dataset = module.predict_dataloader().dataset
    values = []
    for worker in range(workers):
        monkeypatch.setattr(arrow, "get_worker_info", lambda: SimpleNamespace(id=worker, num_workers=workers))
        values.extend(value for batch in collect(dataset, monkeypatch) for value in batch["id"].to_pylist())

    assert sorted(values) == expected


@pytest.mark.parametrize("workers", [1, 3])
def test_arrow_worker_count_preserves_distributed_rank_ownership_and_equal_tails(workers, monkeypatch):
    module = rf.ArrowDataModule(
        model=model(),
        validate=pa.table({"id": list(range(23))}),
        num_workers=workers,
    )
    monkeypatch.setattr(arrow, "world_size", lambda: 2)
    for rank in range(2):
        monkeypatch.setattr(arrow, "rank", lambda: rank)
        dataset = module.val_dataloader().dataset
        batches = []
        for worker in range(workers):
            monkeypatch.setattr(arrow, "get_worker_info", lambda: SimpleNamespace(id=worker, num_workers=workers))
            batches.extend(collect(dataset, monkeypatch))

        assert sorted(len(batch) for batch in batches) == [1, 2, 2, 2, 2, 2]
        assert sorted(value for batch in batches for value in batch["id"].to_pylist()) == list(range(rank, 22, 2))


def test_arrow_dataset_source_scans_through_the_shared_pipeline(monkeypatch: pytest.MonkeyPatch):
    source = ds.dataset(pa.table({"id": list(range(5))}))
    module = rf.ArrowDataModule(model=model(), validate=source, shuffle=False)

    batches = collect(module.val_dataloader().dataset, monkeypatch)

    assert [len(batch) for batch in batches] == [2, 2, 1]
    assert pa.concat_tables(batches)["id"].to_pylist() == list(range(5))


def test_dataset_scope_preprocessor_receives_one_logical_split(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    @rf.preprocess(scope="dataset")
    def inspect(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(len(frame))
        return frame

    source = ds.dataset(pa.table({"id": list(range(9))}))
    module = rf.ArrowDataModule(model=model(), validate=source, preprocessor=inspect, shuffle=False)

    collect(module.val_dataloader().dataset, monkeypatch)

    assert calls == [9]


def test_preprocessor_pipeline_runs_in_order_and_snapshots_configuration(monkeypatch: pytest.MonkeyPatch):
    calls: list[str] = []

    @rf.preprocess
    def derive(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("derive")
        return frame.with_columns(amount=pl.col("id") + 1)

    @rf.preprocess
    def replace(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append("replace")
        return frame.select(id=pl.col("amount") * 2)

    configured = [derive, replace]
    module = rf.ArrowDataModule(
        model=model(),
        validate=pa.table({"id": [1, 2, 3]}),
        preprocessor=configured,
        shuffle=False,
    )
    configured.reverse()

    batches = collect(module.val_dataloader().dataset, monkeypatch)

    assert calls == ["derive", "replace"]
    assert pa.concat_tables(batches)["id"].to_pylist() == [4, 6, 8]


def test_preprocessor_mapping_accepts_an_ordered_pipeline_per_split(monkeypatch: pytest.MonkeyPatch):
    calls: list[tuple[Strata, str]] = []

    @rf.preprocess
    def first(frame: pl.DataFrame, *, strata: Strata) -> pl.DataFrame:
        calls.append((strata, "first"))
        return frame

    @rf.preprocess
    def second(frame: pl.DataFrame, *, strata: Strata) -> pl.DataFrame:
        calls.append((strata, "second"))
        return frame

    source = pa.table({"id": [1, 2]})
    module = rf.ArrowDataModule(
        model=model(),
        train=source,
        validate=source,
        preprocessor={
            "train": [first, second],
            "validate": (second,),
        },
        shuffle=False,
    )

    collect(module.train_dataloader().dataset, monkeypatch)
    collect(module.val_dataloader().dataset, monkeypatch)

    assert calls == [
        (Strata.train, "first"),
        (Strata.train, "second"),
        (Strata.validate, "second"),
    ]


def test_preprocessor_pipeline_flat_maps_each_stage():
    calls: list[int] = []

    @rf.preprocess
    def split(frame: pl.DataFrame):
        yield frame.slice(0, 1)
        yield frame.slice(1)

    @rf.preprocess
    def inspect(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(len(frame))
        return frame

    source = arrow.convert(pa.table({"id": [1, 2, 3]}))
    outputs = list(
        arrow.process(
            (source,),
            preprocessor=(split, inspect),
            strata=Strata.validate,
            schema=None,
            encoding_context={},
        )
    )

    assert calls == [1, 2]
    assert [batch["id"].to_pylist() for batch in outputs] == [[1], [2, 3]]


def test_dataset_scope_materializes_at_its_ordered_pipeline_stage():
    calls: list[tuple[str, int]] = []

    @rf.preprocess
    def before(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(("before", len(frame)))
        return frame.with_columns(ready=pl.lit(True))

    @rf.preprocess(scope="dataset")
    def global_order(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(("global", len(frame)))
        return frame

    @rf.preprocess
    def after(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(("after", len(frame)))
        return frame

    sources = (
        arrow.convert(pa.table({"id": [1, 2]})),
        arrow.convert(pa.table({"id": [3, 4, 5]})),
    )
    outputs = list(
        arrow.process(
            sources,
            preprocessor=(before, global_order, after),
            strata=Strata.validate,
            schema=None,
            encoding_context={},
        )
    )

    assert calls == [("before", 2), ("before", 3), ("global", 5), ("after", 5)]
    assert len(outputs) == 1
    assert outputs[0]["id"].to_pylist() == [1, 2, 3, 4, 5]


def test_callable_source_rejects_dataset_scope_anywhere_in_pipeline():
    @rf.preprocess
    def local(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    @rf.preprocess(scope="dataset")
    def global_order(frame: pl.DataFrame) -> pl.DataFrame:
        return frame

    def source():
        yield pa.record_batch({"id": [1]})

    with pytest.raises(NotImplementedError, match="dataset-scoped preprocessing"):
        rf.ArrowDataModule(model=model(), validate=source, preprocessor=(local, global_order), shuffle=False)


def test_training_shuffle_is_reproducible_and_changes_by_epoch(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"id": list(range(32))})
    first = rf.ArrowDataModule(model=model(8), train=source, seed=7).train_dataloader().dataset
    second = rf.ArrowDataModule(model=model(8), train=source, seed=7).train_dataloader().dataset

    first_epoch = pa.concat_tables(collect(first, monkeypatch))["id"].to_pylist()
    repeated_epoch = pa.concat_tables(collect(second, monkeypatch))["id"].to_pylist()
    next_epoch = pa.concat_tables(collect(first, monkeypatch))["id"].to_pylist()

    assert first_epoch == repeated_epoch
    assert next_epoch != first_epoch
    assert sorted(first_epoch) == list(range(32))
    assert sorted(next_epoch) == list(range(32))


def test_training_epoch_survives_dataloader_recreation(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"id": list(range(32))})
    module = rf.ArrowDataModule(model=model(8), train=source, seed=7)

    first = pa.concat_tables(collect(module.train_dataloader().dataset, monkeypatch))["id"].to_pylist()
    second = pa.concat_tables(collect(module.train_dataloader().dataset, monkeypatch))["id"].to_pylist()

    assert first != second
    assert sorted(first) == sorted(second) == list(range(32))


def test_evaluation_sampling_and_shuffle_are_stable_across_iterations(monkeypatch: pytest.MonkeyPatch):
    source = pa.table({"id": list(range(128))})
    dataset = (
        rf.ArrowDataModule(
            model=model(16),
            validate=source,
            seed=11,
            shuffle={"validate": True},
            sample={"validate": 0.5},
        )
        .val_dataloader()
        .dataset
    )

    first = pa.concat_tables(collect(dataset, monkeypatch))["id"].to_pylist()
    second = pa.concat_tables(collect(dataset, monkeypatch))["id"].to_pylist()

    assert first == second


def test_stream_epoch_limit_selects_before_shuffling(monkeypatch: pytest.MonkeyPatch):
    consumed: list[int] = []

    def source():
        for value in range(16):
            consumed.append(value)
            yield pa.record_batch({"id": [value]})

    dataset = (
        rf.ArrowDataModule(
            model=model(2),
            train=source,
            seed=19,
            shuffle_rows=4,
            epoch_size=2,
        )
        .train_dataloader()
        .dataset
    )

    selected = pa.concat_tables(collect(dataset, monkeypatch))["id"].to_pylist()

    assert len(selected) == 2
    assert sorted(selected) == [0, 1]
    assert consumed == [0, 1]


def test_stream_randomization_is_independent_of_record_batch_boundaries(monkeypatch: pytest.MonkeyPatch):
    table = pa.table({"id": list(range(64))})

    def source(sizes):
        def factory():
            offset = 0
            for size in sizes:
                yield table.slice(offset, size).to_batches()[0]
                offset += size

        return factory

    options = dict(seed=23, shuffle_rows=7, sample=0.7)
    left = rf.ArrowDataModule(model=model(5), train=source([1] * 64), **options).train_dataloader().dataset
    right = rf.ArrowDataModule(model=model(5), train=source([17, 3, 29, 15]), **options).train_dataloader().dataset

    left_ids = pa.concat_tables(collect(left, monkeypatch))["id"].to_pylist()
    right_ids = pa.concat_tables(collect(right, monkeypatch))["id"].to_pylist()

    assert left_ids == right_ids


def test_one_row_stream_shuffle_window_preserves_order(monkeypatch: pytest.MonkeyPatch):
    def source():
        yield pa.record_batch({"id": list(range(8))})

    dataset = (
        rf.ArrowDataModule(
            model=model(3),
            train=source,
            shuffle=True,
            shuffle_rows=1,
        )
        .train_dataloader()
        .dataset
    )

    selected = pa.concat_tables(collect(dataset, monkeypatch))["id"].to_pylist()

    assert selected == list(range(8))


def test_arrow_preprocessor_runs_before_model_batching(monkeypatch: pytest.MonkeyPatch):
    calls: list[int] = []

    @rf.preprocess
    def positive(frame: pl.DataFrame) -> pl.DataFrame:
        calls.append(len(frame))
        return frame.filter(pl.col("id") >= 0)

    source = pa.table({"id": [-2, -1, 0, 1, 2]})
    module = rf.ArrowDataModule(model=model(), validate=source, preprocessor=positive, shuffle=False)
    batches = collect(module.val_dataloader().dataset, monkeypatch)

    assert calls == [5]
    assert [batch["id"].to_pylist() for batch in batches] == [[0, 1], [2]]


def test_polars_is_a_one_time_conversion_adapter(monkeypatch: pytest.MonkeyPatch):
    calls = 0
    original = pl.DataFrame.to_arrow

    def track(frame, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original(frame, *args, **kwargs)

    monkeypatch.setattr(pl.DataFrame, "to_arrow", track)
    frame = pl.DataFrame({"id": [1, 2]})
    module = rf.PolarsDataModule(model=model(), train=frame, validate=frame, shuffle=False)

    assert calls == 2
    assert all(isinstance(source, pa.Table) for source in module.sources.values())
    module.train_dataloader()
    module.val_dataloader()
    assert calls == 2


def test_polars_requires_eager_frames():
    with pytest.raises(TypeError, match="collected"):
        rf.PolarsDataModule(model=model(), train=pl.DataFrame({"id": [1]}).lazy())


def test_custom_adapter_bounds_mapping_conversion_and_restarts():
    dataset = Records([{"id": index} for index in range(5)])
    module = rf.CustomDataModule(model=model(), train=dataset, ingress_rows=2, shuffle=False)
    source = module.sources[Strata.train]

    first = list(source())
    second = list(source())

    assert [batch.num_rows for batch in first] == [2, 2, 1]
    assert [batch.num_rows for batch in second] == [2, 2, 1]
    assert all(isinstance(batch, pa.RecordBatch) for batch in first)
    assert first[0].schema == second[0].schema


def test_custom_inference_uses_the_union_of_first_chunk_mapping_keys():
    source = adapt(
        lambda: iter([{"id": 1}, {"id": 2, "late_in_chunk": "kept"}]),
        schema=None,
        rows=2,
        name="train",
    )

    batch = list(source())[0]

    assert batch.schema.names == ["id", "late_in_chunk"]
    assert batch.to_pylist() == [
        {"id": 1, "late_in_chunk": None},
        {"id": 2, "late_in_chunk": "kept"},
    ]


def test_custom_schema_errors_explain_arrow_schema_remedy():
    empty = adapt(lambda: iter(()), schema=None, rows=2, name="train")
    nulls = adapt(lambda: iter([{"id": None}]), schema=None, rows=2, name="train")
    late = adapt(lambda: iter([{"id": 1}, {"id": 2, "late": True}]), schema=None, rows=1, name="train")

    with pytest.raises(ValueError, match="provide arrow_schema"):
        list(empty())
    with pytest.raises(TypeError, match="all-null.*provide arrow_schema"):
        list(nulls())
    with pytest.raises(TypeError, match="introduced field.*arrow_schema"):
        list(late())


def test_explicit_custom_schema_supports_empty_sources():
    schema = pa.schema([("id", pa.int64())])
    source = adapt(lambda: iter(()), schema=schema, rows=2, name="validate")

    batches = list(source())

    assert len(batches) == 1
    assert batches[0].num_rows == 0
    assert batches[0].schema == schema


def test_synthetic_generator_is_called_for_every_source_iteration():
    calls = 0

    def generate() -> Iterator[dict[str, int]]:
        nonlocal calls
        calls += 1
        yield {"id": calls}

    module = rf.SyntheticDataModule(model=model(), train=generate, shuffle=False)
    source = module.sources[Strata.train]

    assert list(source())[0]["id"].to_pylist() == [1]
    assert list(source())[0]["id"].to_pylist() == [2]


def test_parallel_configuration_validation():
    source = pa.table({"id": [1, 2]})
    module = rf.ArrowDataModule(
        model=model(),
        train=source,
        num_workers=1,
        persistent_workers=True,
        prefetch_factor=3,
    )

    loader = module.train_dataloader()

    assert loader.num_workers == 1
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 3
    assert loader.multiprocessing_context.get_start_method() == "spawn"

    with pytest.raises(ValueError, match="requires num_workers"):
        rf.ArrowDataModule(model=model(), train=source, persistent_workers=True)
    with pytest.raises(ValueError, match="prefetch_factor"):
        rf.ArrowDataModule(model=model(), train=source, prefetch_factor=0)


def test_recreated_loaders_use_current_model_batch_size_schema_and_context():
    configured_model = model(2)
    module = rf.ArrowDataModule(
        model=configured_model,
        validate=pa.table({"id": list(range(5)), "merchant": ["a", "b", "a", "b", "a"]}),
        shuffle=False,
    )
    first = list(module.val_dataloader())
    assert [len(batch.source) for batch in first] == [2, 2, 1]
    assert all("/merchant" not in batch.tensors for batch in first)

    module.batch_size = 3
    configured_model.extend(rf.where("address") == "/", merchant=rf.Category)
    batches = list(module.val_dataloader())

    assert module.batch_size == configured_model.batch_size == 3
    assert [len(batch.source) for batch in batches] == [3, 2]
    assert all("/merchant" in batch.tensors for batch in batches)
    assert all(batch.tensors["/merchant"].present.all() for batch in batches)


def test_retain_mapping_matches_configured_splits_exactly():
    source = pa.table({"id": [1, 2], "request_id": ["a", "b"]})
    module = rf.ArrowDataModule(
        model=model(),
        train=source,
        validate=source,
        retain={"train": ("request_id",), "validate": "*"},
    )

    assert all(batch.retain == ("request_id",) for batch in module.train_dataloader())
    assert all(batch.retain == "*" for batch in module.val_dataloader())

    with pytest.raises(ValueError, match="match configured splits exactly"):
        rf.ArrowDataModule(model=model(), train=source, validate=source, retain={"train": ()})
    with pytest.raises(ValueError, match="match configured splits exactly"):
        rf.ArrowDataModule(model=model(), train=source, retain={"train": (), "validate": ()})


def test_split_overrides_stay_isolated_when_loaders_are_recreated():
    @rf.preprocess
    def shift(frame: pl.DataFrame, *, strata: Strata) -> pl.DataFrame:
        offset = 100 if strata == Strata.train else 200
        return frame.with_columns(id=pl.col("id") + offset)

    source = pa.table({"id": list(range(7)), "request_id": [str(index) for index in range(7)]})
    module = rf.ArrowDataModule(
        model=model(),
        train=source,
        validate=source,
        test=source,
        predict=source,
        preprocessor={"train": shift, "validate": shift},
        shuffle=False,
        epoch_size={"train": 5, "validate": 3},
        drop_last={"train": True},
        num_workers={"predict": 1},
        persistent_workers={"predict": True},
        pin_memory={"predict": True},
        prefetch_factor={"predict": 3},
        retain={"train": ("request_id",), "validate": "*", "test": (), "predict": ("request_id",)},
    )
    expected = {
        Strata.train: ([100, 101, 102, 103], ("request_id",)),
        Strata.validate: ([200, 201, 202], "*"),
        Strata.test: (list(range(7)), ()),
        Strata.predict: (list(range(7)), ("request_id",)),
    }

    for _ in range(2):
        for strata, (values, retained) in expected.items():
            loader = module.dataloader(strata)
            prediction = strata == Strata.predict
            assert loader.num_workers == int(prediction)
            assert loader.persistent_workers is prediction
            assert loader.prefetch_factor == (3 if prediction else None)
            assert loader.pin_memory is False
            batches = list(loader.dataset)
            assert pa.concat_tables([batch.source for batch in batches])["id"].to_pylist() == values
            assert all(batch.retain == retained for batch in batches)


@pytest.mark.parametrize(
    "option,value,error",
    [
        ("preprocessor", 42, TypeError),
        ("shuffle", "yes", TypeError),
        ("sample", True, TypeError),
        ("sample", 0.0, ValueError),
        ("epoch_size", False, ValueError),
        ("shuffle_rows", 0, ValueError),
        ("drop_last", 1, TypeError),
        ("num_workers", -1, ValueError),
        ("persistent_workers", True, ValueError),
        ("pin_memory", 1, TypeError),
        ("prefetch_factor", 0, ValueError),
    ],
)
def test_invalid_overrides_are_rejected_even_for_unused_splits(option, value, error):
    with pytest.raises(error, match=option):
        rf.ArrowDataModule(model=model(), train=pa.table({"id": [1]}), **{option: {"validate": value}})
