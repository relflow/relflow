"""Native Arrow S3 readers must release their clients before interpreter exit."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import weakref
from datetime import timedelta
from functools import partial

import pyarrow as pa
import pyarrow.csv as csv
import pyarrow.feather as feather
import pyarrow.fs as fs
import pyarrow.orc as orc
import pyarrow.parquet as pq
import pytest

import relflow as rf
from relflow.data import sources
from relflow.data.sources import Source
from tests.data.s3 import S3Fixture


def parquet(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


@pytest.fixture
def s3():
    table = pa.table({"id": [1, 2, 3], "items": [[{"amount": 2.5}]] * 3})
    with S3Fixture(
        {
            "records/part-00.parquet": parquet(table.slice(0, 2)),
            "records/nested/part-01.parquet": parquet(table.slice(2)),
            "records/_ignored.parquet": parquet(table),
            "records/.hidden/part-02.parquet": parquet(table),
            "records/notes.txt": b"ignored",
        }
    ) as fixture:
        yield fixture


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("diagnostics/records/", [3, 1, 2]),
        ("diagnostics/records/**/*.parquet", [3, 1, 2]),
        ("diagnostics/records/part-??.parquet", [1, 2]),
        ("diagnostics/records/part-[0-9][0-9].parquet", [1, 2]),
        ("diagnostics/records/part-00.parquet", [1, 2]),
        (["diagnostics/records/part-00.parquet"] * 2, [1, 2]),
    ],
)
def test_native_s3_selection_preserves_sorted_nested_arrow_rows(s3, path, expected):
    source = rf.source(path, match=r".*\.parquet", filesystem=partial(fs.S3FileSystem, **s3.options))
    assert source.manifest is None
    assert not s3.requests
    result = pa.Table.from_batches(list(source()))
    assert result["id"].to_pylist() == expected
    assert result["items"].to_pylist() == [[{"amount": 2.5}]] * len(expected)
    assert all(item.modified is not None for item in source.manifest.files)
    assert any(method == "HEAD" for method, _, _ in s3.requests)


@pytest.mark.parametrize("suffix", ["parquet", "csv", "jsonl", "ndjson", "arrow", "ipc", "feather", "orc"])
def test_native_s3_formats_stream_through_the_arrow_pipeline(suffix):
    expected = pa.table({"id": [1, 2], "brand": ["a", "b"]})
    sink = pa.BufferOutputStream()
    if suffix == "parquet":
        pq.write_table(expected, sink)
    elif suffix == "csv":
        csv.write_csv(expected, sink)
    elif suffix in {"jsonl", "ndjson"}:
        sink.write(b'{"id":1,"brand":"a"}\n{"id":2,"brand":"b"}\n')
    elif suffix == "orc":
        orc.write_table(expected, sink)
    else:
        feather.write_feather(expected, sink)
    with S3Fixture({f"records/input.{suffix}": sink.getvalue().to_pybytes()}) as fixture:
        source = rf.source(
            f"diagnostics/records/input.{suffix}", filesystem=partial(fs.S3FileSystem, **fixture.options)
        )
        assert pa.Table.from_batches(list(source())).equals(expected)


@pytest.mark.parametrize("mutation", ["timestamp", "size", "deleted"])
def test_native_s3_snapshot_checks_remain_effective(s3, mutation):
    source = rf.source("diagnostics/records", match=r".*\.parquet", filesystem=partial(fs.S3FileSystem, **s3.options))
    list(source())
    key = "records/part-00.parquet"
    if mutation == "timestamp":
        s3.modified[key] += timedelta(seconds=1)
    elif mutation == "size":
        s3.objects[key] += b"mutation"
    else:
        del s3.objects[key]
    with pytest.raises(RuntimeError, match="changed; create a new source"):
        list(source())


@pytest.mark.parametrize("stage", ["discover", "scan"])
def test_native_s3_clients_are_released_even_when_error_tracebacks_are_retained(s3, monkeypatch, stage):
    filesystems = []
    connect = Source.connect

    def record(source):
        filesystem, paths = connect(source)
        filesystems.append(weakref.ref(filesystem))
        return filesystem, paths

    monkeypatch.setattr(Source, "connect", record)
    path = "diagnostics/missing" if stage == "discover" else "diagnostics/records"
    source = rf.source(path, match=r".*\.parquet", filesystem=partial(fs.S3FileSystem, **s3.options))
    if stage == "scan":
        source.manifest = source.discover()
        s3.deny_reads = True
    with pytest.raises(OSError) as caught:
        list(source())
    assert caught.value.__traceback__ is not None
    assert caught.value.__cause__ is not None
    assert stage in " ".join(caught.value.__cause__.__notes__)
    assert all(item() is None for item in filesystems)


def test_native_s3_pickle_preserves_the_manifest_without_live_resources(s3):
    source = rf.source("diagnostics/records", match=r".*\.parquet", filesystem=partial(fs.S3FileSystem, **s3.options))
    expected = pa.Table.from_batches(list(source()))
    restored = pickle.loads(pickle.dumps(source))
    assert restored.manifest == source.manifest
    assert pa.Table.from_batches(list(restored())).equals(expected)


@pytest.mark.parametrize("stop", ["preprocess", "epoch"])
def test_native_s3_data_pipeline_closes_scans_after_preprocessing_failure_or_epoch_limit(s3, monkeypatch, stop):
    filesystems = []
    datasets = []
    connect = Source.connect
    create = sources.ds.dataset

    def record_connect(source):
        filesystem, paths = connect(source)
        filesystems.append(weakref.ref(filesystem))
        return filesystem, paths

    def record_dataset(*args, **kwargs):
        dataset = create(*args, **kwargs)
        datasets.append(weakref.ref(dataset))
        return dataset

    monkeypatch.setattr(Source, "connect", record_connect)
    monkeypatch.setattr(sources.ds, "dataset", record_dataset)

    @rf.preprocess
    def prepare(frame):
        if stop == "preprocess":
            raise ValueError("injected preprocessing failure")
        return frame

    model = rf.Model(id=rf.Number, d_model=8, n_layers=1, n_heads=4, batch_size=2)
    source = rf.source("diagnostics/records", match=r".*\.parquet", filesystem=partial(fs.S3FileSystem, **s3.options))
    data = rf.ArrowDataModule(model, predict=source, preprocessor=prepare, epoch_size=1 if stop == "epoch" else None)
    stream = iter(data.predict_dataloader())
    if stop == "preprocess":
        with pytest.raises(ValueError, match="injected preprocessing failure") as caught:
            next(stream)
        assert caught.value.__traceback__ is not None
    else:
        assert sum(batch.source.num_rows for batch in stream) == 1
    assert filesystems and datasets
    assert all(item() is None for item in filesystems)
    assert all(item() is None for item in datasets)


PREDICT = r"""
import json
import sys
from functools import partial
import lightning.pytorch as lit
import polars as pl
import pyarrow.fs as fs
import pyarrow.parquet as pq
import relflow as rf

options, mode, output = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
model = rf.Model(id=rf.Number, d_model=8, n_layers=1, n_heads=4, batch_size=2, embed=True)
source = rf.source('diagnostics/records', match=r'.*\.parquet', filesystem=partial(fs.S3FileSystem, **options))

@rf.preprocess
def prepare(frame: pl.DataFrame) -> pl.DataFrame:
    if mode == 'preprocess':
        raise ValueError('injected preprocessing failure')
    return frame

data = rf.ArrowDataModule(model=model, predict=source, preprocessor=prepare, retain='*')
writer = rf.Writer(output)
trainer = lit.Trainer(accelerator='cpu', devices=1, logger=False, enable_checkpointing=False,
                      enable_progress_bar=False, callbacks=[writer])
trainer.predict(model, datamodule=data, return_predictions=False)
result = pq.read_table(output + '/rank-0.parquet')
assert result.num_rows == 3
assert 'predictions' in result.column_names
assert sorted(result['inputs'].combine_chunks().field('id').to_pylist()) == [1, 2, 3]
print('committed 3 predictions', flush=True)
# Keep the data module reachable through interpreter exit.
"""


@pytest.mark.parametrize("mode", ["success", "read", "preprocess"])
def test_native_s3_prediction_process_exits_naturally_after_success_and_failure(s3, tmp_path, mode):
    s3.deny_reads = mode == "read"
    result = subprocess.run(
        [sys.executable, "-c", PREDICT, json.dumps(s3.options), mode, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if mode == "success":
        assert result.returncode == 0, result.stderr
        assert "committed 3 predictions" in result.stdout
        assert pq.read_table(tmp_path / "rank-0.parquet").num_rows == 3
    else:
        assert result.returncode != 0
        assert "Traceback (most recent call last)" in result.stderr
        assert "committed" not in result.stdout
        if mode == "read":
            assert any(token in result.stderr.lower() for token in ("403", "forbidden", "denied")), result.stderr
        else:
            assert "injected preprocessing failure" in result.stderr
