from __future__ import annotations

import os
import pickle
import re
from datetime import timedelta
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.dataset as ds
import pyarrow.feather as feather
import pyarrow.fs as fs
import pyarrow.orc as orc
import pyarrow.parquet as pq
import pytest
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

import relflow as rf
from relflow.data.datasets import arrow
from relflow.data.sources import Source, matches


def model(batch_size=2):
    return rf.Model(id=rf.Number, d_model=8, n_layers=1, n_heads=4, batch_size=batch_size)


@rf.preprocess
def prepare(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.filter(pl.col("id") % 2 == 0).with_columns(id=pl.col("id") + 100)


def distributed_source(rank: int, directory: str, missing: bool) -> None:
    root = Path(directory)
    distributed.init_process_group(
        "gloo", init_method=(root / "rendezvous").as_uri(), rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        # Only rank zero's description is discoverable. Rank one must use the
        # broadcast manifest rather than independently discovering its own path.
        path = root / ("missing.parquet" if missing or rank == 1 else "train.parquet")
        data = rf.ArrowDataModule(model(), validate=path, num_workers=1)
        if missing:
            with pytest.raises(FileNotFoundError, match="validate source"):
                data.val_dataloader()
        else:
            actual = [v for batch in data.val_dataloader() for v in batch.source["id"].to_pylist()]
            assert actual == list(range(rank, 8, 2))
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("suffix", ["parquet", "csv", "jsonl", "ndjson", "arrow", "ipc", "feather", "orc"])
def test_file_formats_share_the_arrow_pipeline(tmp_path, suffix):
    path = tmp_path / f"train.{suffix}"
    expected = pa.table({"id": [1, 2, 3], "brand": ["a", "b", "c"]})
    if suffix == "parquet":
        pq.write_table(expected, path)
    elif suffix == "csv":
        csv.write_csv(expected, path)
    elif suffix in {"jsonl", "ndjson"}:
        path.write_text('{"id":1,"brand":"a"}\n{"id":2,"brand":"b"}\n{"id":3,"brand":"c"}\n')
    elif suffix == "orc":
        orc.write_table(expected, path)
    else:
        feather.write_feather(expected, path)

    data = rf.ArrowDataModule(model(), validate=path)
    tables = [item.source for item in data.val_dataloader()]
    assert pa.concat_tables(tables).equals(expected)
    assert [len(item) for item in tables] == [2, 1]


def test_construction_is_lazy_and_preparation_shares_a_manifest(tmp_path, monkeypatch):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "train.parquet")
    discover = Source.discover
    calls = []

    def record(source):
        calls.append(source)
        return discover(source)

    monkeypatch.setattr(Source, "discover", record)
    source = rf.source(tmp_path)
    data = rf.ArrowDataModule(model(), train=source)
    assert not calls
    assert source.manifest is None
    assert source.dataset is None
    loader = data.train_dataloader()
    assert len(calls) == 1
    assert loader.dataset.source.manifest is not None
    assert loader.dataset.source.dataset is None
    data.train_dataloader()
    assert len(calls) == 1


def test_pickle_preserves_manifest_and_drops_live_resources(tmp_path):
    pq.write_table(pa.table({"id": [1, 2]}), tmp_path / "train.parquet")
    source = rf.source(tmp_path)
    expected = list(source())
    assert source.dataset is not None
    restored = pickle.loads(pickle.dumps(source))
    assert restored.manifest == source.manifest
    assert restored.dataset is None
    assert restored.process is None
    assert list(restored()) == expected


def test_source_discards_resources_inherited_from_another_process(tmp_path, monkeypatch):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "train.parquet")
    source = rf.source(tmp_path)
    list(source())
    original = source.dataset
    source.process = os.getpid() + 1
    list(source())
    assert source.dataset is not original


@pytest.mark.parametrize(
    ("path", "pattern", "expected"),
    [
        ("/a/train.csv", "/a/*.csv", True),
        ("/a/nested/train.csv", "/a/*.csv", False),
        ("/a/train.csv", "/a/**/*.csv", True),
        ("/a/nested/train.csv", "/a/**/*.csv", True),
        ("/a/train-1.csv", "/a/train-?.csv", True),
        ("/a/train-12.csv", "/a/train-?.csv", False),
        ("/a/train-1.csv", "/a/train-[0-9].csv", True),
    ],
)
def test_glob_segments(path, pattern, expected):
    assert matches(path, pattern) is expected


def test_globs_and_regexes_select_sorted_files(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    for path, value in [
        (tmp_path / "TRAIN-00002.parquet", 2),
        (tmp_path / "train-00001.parquet", 1),
        (nested / "train-00003.parquet", 3),
        (tmp_path / "validation-00000.parquet", 4),
    ]:
        pq.write_table(pa.table({"id": [value]}), path)
    source = rf.source(tmp_path, match=re.compile(r"train-\d{5}\.parquet", re.IGNORECASE))
    assert sorted(value for batch in source() for value in batch["id"].to_pylist()) == [1, 2, 3]
    assert [item.path for item in source.manifest.files] == sorted(item.path for item in source.manifest.files)
    shallow = rf.source(str(tmp_path / "train-*.parquet"))
    assert [value for batch in shallow() for value in batch["id"].to_pylist()] == [1]
    recursive = rf.source(str(tmp_path / "**/train-*.parquet"))
    assert sorted(value for batch in recursive() for value in batch["id"].to_pylist()) == [1, 3]


def test_explicit_files_are_sorted_deduplicated_and_uri_normalized(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.parquet"
    pq.write_table(pa.table({"id": [1]}), a)
    pq.write_table(pa.table({"id": [2]}), b)
    source = rf.source([b.as_uri(), a.as_uri(), b.as_uri()])
    assert [value for batch in source() for value in batch["id"].to_pylist()] == [1, 2]


def test_relative_path_is_fixed_at_construction(tmp_path, monkeypatch):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "train.parquet")
    monkeypatch.chdir(tmp_path)
    source = rf.source("train.parquet")
    monkeypatch.chdir(tmp_path.parent)
    assert list(source())[0]["id"].to_pylist() == [1]


def test_file_uri_question_mark_glob(tmp_path):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "train-1.parquet")
    assert list(rf.source(tmp_path.as_uri() + "/train-?.parquet")())[0]["id"].to_pylist() == [1]


def test_filesystem_factory_is_lazy_and_remote_selection_uses_its_namespace(tmp_path):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "train-1.parquet")
    calls = []

    def filesystem():
        calls.append(True)
        return fs.SubTreeFileSystem(str(tmp_path), fs.LocalFileSystem())

    source = rf.source("train-*.parquet", filesystem=filesystem)
    assert not calls
    assert list(source())[0]["id"].to_pylist() == [1]
    assert calls
    assert source.manifest.files[0].path == "train-1.parquet"


def test_filesystem_uri_scopes_relative_paths(tmp_path):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "a.parquet")
    source = rf.source("a.parquet", filesystem=tmp_path.as_uri())
    assert list(source())[0]["id"].to_pylist() == [1]


def test_sidecars_are_ignored_but_unknown_files_are_not(tmp_path):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "train.parquet")
    (tmp_path / "_SUCCESS").write_text("")
    (tmp_path / ".metadata").write_text("")
    assert list(rf.source(tmp_path)())[0]["id"].to_pylist() == [1]
    (tmp_path / "unknown.bin").write_text("data")
    with pytest.raises(ValueError, match="mixed or unknown.*format"):
        list(rf.source(tmp_path)())


def test_mixed_formats_require_explicit_selection(tmp_path):
    table = pa.table({"id": [1]})
    pq.write_table(table, tmp_path / "train.parquet")
    csv.write_csv(table, tmp_path / "train.csv")
    with pytest.raises(ValueError, match="mixed or unknown"):
        list(rf.source(tmp_path)())


def test_corrupt_file_is_not_silently_omitted(tmp_path):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "a.parquet")
    (tmp_path / "b.parquet").write_text("broken")
    data = rf.ArrowDataModule(model(), train=tmp_path)
    with pytest.raises(pa.ArrowInvalid, match="train source.*b.parquet") as caught:
        data.train_dataloader()
    assert caught.value.__cause__ is not None


def test_incompatible_shard_schema_is_rejected_before_iteration(tmp_path):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"id": ["two"]}), tmp_path / "b.parquet")
    data = rf.ArrowDataModule(model(), train=tmp_path)
    with pytest.raises(TypeError, match="train source.*schema changed.*b.parquet"):
        data.train_dataloader()


def test_explicit_schema_preserves_csv_codes_and_handles_null_shards(tmp_path):
    (tmp_path / "a.csv").write_text("brand,id\n001,1\n")
    (tmp_path / "b.csv").write_text("brand,id\n,2\n")
    schema = pa.schema([("brand", pa.string()), ("id", pa.int64())])
    source = rf.source(tmp_path, schema=schema)
    actual = pa.Table.from_batches(list(source()))
    assert actual.schema == schema
    assert actual["brand"].to_pylist()[0] == "001"


def test_missing_declared_field_is_not_synthesized(tmp_path):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "a.parquet")
    source = rf.source(tmp_path, schema=pa.schema([("id", pa.int64()), ("absent", pa.string())]))
    with pytest.raises(TypeError, match="a.parquet.*missing declared fields.*absent"):
        list(source())


def test_arrow_format_parsing_options(tmp_path):
    (tmp_path / "a.csv").write_text("brand;id\n001;1\n")
    source = rf.source(
        tmp_path,
        format=ds.CsvFileFormat(
            parse_options=csv.ParseOptions(delimiter=";"),
            convert_options=csv.ConvertOptions(column_types={"brand": pa.string()}),
        ),
    )
    assert list(source())[0].to_pydict() == {"brand": ["001"], "id": [1]}


def test_explicit_format_reads_extensionless_file(tmp_path):
    path = tmp_path / "observations"
    pq.write_table(pa.table({"id": [1]}), path)
    assert list(rf.source(path, format="parquet")())[0]["id"].to_pylist() == [1]


def test_hive_partition_columns_survive_scanning(tmp_path):
    for year in (2024, 2025):
        folder = tmp_path / f"year={year}"
        folder.mkdir()
        pq.write_table(pa.table({"id": [year - 2023]}), folder / "train.parquet")
    source = rf.source(tmp_path, partitioning="hive")
    assert pa.Table.from_batches(list(source())).to_pydict() == {"id": [1, 2], "year": [2024, 2025]}


def test_empty_file_keeps_declared_schema(tmp_path):
    schema = pa.schema([pa.field("id", pa.int64())], metadata={b"source": b"empty"})
    pq.write_table(pa.Table.from_batches([], schema), tmp_path / "empty.parquet")
    batches = list(rf.source(tmp_path)())
    assert len(batches) == 1 and batches[0].num_rows == 0
    assert batches[0].schema == schema
    assert list(rf.ArrowDataModule(model(), validate=tmp_path).val_dataloader()) == []


def test_missing_source_and_empty_selection_are_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="train source.*existing"):
        rf.ArrowDataModule(model(), train=tmp_path / "missing.parquet").train_dataloader()
    with pytest.raises(FileNotFoundError, match="no files selected"):
        list(rf.source(tmp_path, match="nothing")())


def test_manifest_ignores_added_files_and_detects_replacement(tmp_path):
    path = tmp_path / "a.parquet"
    pq.write_table(pa.table({"id": [1]}), path)
    source = rf.source(tmp_path)
    assert list(source())[0]["id"].to_pylist() == [1]
    pq.write_table(pa.table({"id": [2]}), tmp_path / "b.parquet")
    assert list(source())[0]["id"].to_pylist() == [1]
    pq.write_table(pa.table({"id": [3, 4]}), path)
    with pytest.raises(RuntimeError, match="a.parquet.*changed.*new source"):
        list(source())


def test_data_modules_have_independent_source_lifecycles(tmp_path):
    pq.write_table(pa.table({"id": [1]}), tmp_path / "a.parquet")
    source = rf.source(tmp_path)
    first = rf.ArrowDataModule(model(), validate=source)
    first.val_dataloader()
    pq.write_table(pa.table({"id": [2]}), tmp_path / "b.parquet")
    second = rf.ArrowDataModule(model(), validate=source)
    assert [v for b in first.val_dataloader() for v in b.source["id"].to_pylist()] == [1]
    assert [v for b in second.val_dataloader() for v in b.source["id"].to_pylist()] == [1, 2]


@pytest.mark.parametrize("workers", [0, 1, 2])
def test_files_with_unequal_fragments_and_filtering_need_no_epoch_quota(tmp_path, workers):
    pq.write_table(pa.table({"id": [0]}), tmp_path / "a.parquet")
    pq.write_table(pa.table({"id": list(range(1, 16))}), tmp_path / "b.parquet", row_group_size=3)
    data = rf.ArrowDataModule(model(), train=tmp_path, preprocessor=prepare, num_workers=workers, shuffle=False)
    values = [v for b in data.train_dataloader() for v in b.source["id"].to_pylist()]
    assert sorted(values) == list(range(100, 116, 2))


def test_persistent_file_workers_restart_and_keep_the_frozen_selection(tmp_path):
    pq.write_table(pa.table({"id": list(range(32))}), tmp_path / "a.parquet")
    data = rf.ArrowDataModule(model(4), train=tmp_path, num_workers=2, persistent_workers=True)
    loader = data.train_dataloader()
    try:
        first = [v for b in loader for v in b.source["id"].to_pylist()]
        pq.write_table(pa.table({"id": [1000]}), tmp_path / "b.parquet")
        second = [v for b in loader for v in b.source["id"].to_pylist()]
        assert sorted(first) == sorted(second) == list(range(32))
        assert first != second
    finally:
        loader._iterator._shutdown_workers()


def test_global_epoch_cap_need_not_be_divisible_by_workers(tmp_path):
    pq.write_table(pa.table({"id": list(range(16))}), tmp_path / "a.parquet")
    data = rf.ArrowDataModule(model(2), train=tmp_path, epoch_size=6, num_workers=2, shuffle=False)
    batches = list(data.train_dataloader())
    assert sorted(v for b in batches for v in b.source["id"].to_pylist()) == list(range(6))


def test_discovery_error_is_broadcast_before_workers_start(tmp_path, monkeypatch):
    results = []

    def broadcast(result):
        results.append(result)
        return result

    monkeypatch.setattr(arrow, "broadcast_object", broadcast)
    data = rf.ArrowDataModule(model(), train=tmp_path / "missing.parquet", num_workers=2)
    with pytest.raises(FileNotFoundError):
        data.train_dataloader()
    assert isinstance(results[0], FileNotFoundError)


@pytest.mark.parametrize("missing", [False, True])
def test_ranks_share_manifest_and_discovery_errors(tmp_path, missing):
    pq.write_table(pa.table({"id": list(range(8))}), tmp_path / "train.parquet")
    multiprocessing.spawn(distributed_source, args=(str(tmp_path), missing), nprocs=2, join=True)


def test_factory_generator_closes_on_early_stop_even_when_provider_retains_it():
    closed = []
    streams = []

    def batches():
        try:
            yield pa.table({"id": [1]})
            yield pa.table({"id": [2]})
        finally:
            closed.append(True)

    def factory():
        stream = batches()
        streams.append(stream)
        return stream

    scanned = arrow.scan(factory)
    assert next(scanned)["id"].to_pylist() == [1]
    scanned.close()
    assert closed == [True]


@pytest.mark.parametrize("location", [[], [1], b"bytes", ""])
def test_invalid_locations_fail_at_construction(location):
    with pytest.raises(TypeError, match="path"):
        rf.source(location)


def test_live_filesystem_and_bytes_patterns_are_rejected():
    with pytest.raises(TypeError, match="filesystem.*URI or a factory"):
        rf.source("train.parquet", filesystem=fs.LocalFileSystem())
    with pytest.raises(TypeError, match="text pattern"):
        rf.source("train.parquet", match=re.compile(b"train"))
