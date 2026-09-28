from __future__ import annotations

import pickle
import sys
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import relflow as rf
from relflow.data.datasets import arrow
from relflow.structs.enums import Strata

duckdb = pytest.importorskip("duckdb")


def model(batch_size: int = 2) -> rf.Model:
    return rf.Model(id=rf.Number, d_model=8, n_layers=1, n_heads=4, batch_size=batch_size)


@pytest.fixture
def database(tmp_path: Path) -> Path:
    path = tmp_path / "observations.duckdb"
    with duckdb.connect(str(path)) as connection:
        connection.execute(
            """
            CREATE TABLE observations AS
            SELECT id, [{'amount': id::DOUBLE}, {'amount': id + 0.5}] AS line_items
            FROM range(12) AS rows(id)
            """
        )
    return path


def test_named_splits_bind_parameters_and_preserve_nested_arrow_values(database: Path):
    configured = rf.Model(
        id=rf.Number,
        line_items=rf.Branch(length=2, amount=rf.Number),
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
    )
    data = rf.DuckDBDataModule(
        model=configured,
        database=database,
        train="SELECT * FROM observations WHERE id < ? ORDER BY id",
        validate="WITH held AS (SELECT * FROM observations WHERE id = 4) SELECT * FROM held ORDER BY id",
        test="SELECT * FROM observations WHERE id >= $minimum ORDER BY id",
        predict="SELECT * FROM observations WHERE id = $chosen ORDER BY id",
        parameters={"train": [3], "test": {"minimum": 10}, Strata.predict: {"chosen": 7}},
        shuffle=False,
    )

    for loader, expected in (
        (data.train_dataloader(), [0, 1, 2]),
        (data.val_dataloader(), [4]),
        (data.test_dataloader(), [10, 11]),
        (data.predict_dataloader(), [7]),
    ):
        batches = list(loader)
        table = pa.concat_tables([batch.source for batch in batches])
        assert table["id"].to_pylist() == expected
        assert pa.types.is_list(table.schema.field("line_items").type)
        assert pa.types.is_struct(table.schema.field("line_items").type.value_type)
        assert table["line_items"].to_pylist() == [
            [{"amount": float(value)}, {"amount": value + 0.5}] for value in expected
        ]
        assert all(set(batch.tensors.keys()) == {"/id", "/line_items/amount"} for batch in batches)


def test_arrow_ingress_chunks_are_rebatched_for_the_model(database: Path):
    data = rf.DuckDBDataModule(
        model=model(4),
        database=database,
        validate="SELECT * FROM observations WHERE id < 10 ORDER BY id",
        ingress_rows=3,
    )

    chunks = list(data.sources[Strata.validate]())
    assert all(isinstance(chunk, pa.RecordBatch) for chunk in chunks)
    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 1]
    batches = list(data.val_dataloader())
    assert [len(batch.source) for batch in batches] == [4, 4, 2]
    assert [value for batch in batches for value in batch.source["id"].to_pylist()] == list(range(10))


def test_empty_query_preserves_its_declared_schema(database: Path):
    data = rf.DuckDBDataModule(model=model(), database=database, validate="SELECT * FROM observations WHERE false")

    chunks = list(data.sources[Strata.validate]())
    assert len(chunks) == 1
    assert len(chunks[0]) == 0
    assert chunks[0].schema.field("id").type == pa.int64()
    assert pa.types.is_list(chunks[0].schema.field("line_items").type)
    assert list(data.val_dataloader()) == []
    assert data.schemas[Strata.validate].source.equals(chunks[0].schema, check_metadata=True)
    assert data.schemas[Strata.validate].processed.equals(chunks[0].schema, check_metadata=True)


def test_sources_restart_after_pickling_and_reopen_the_database(database: Path):
    data = rf.DuckDBDataModule(
        model=model(), database=database, validate="SELECT id FROM observations ORDER BY id", ingress_rows=5
    )
    original = data.sources[Strata.validate]
    restored = pickle.loads(pickle.dumps(original))

    for source in (original, restored):
        assert [value for batch in source() for value in batch["id"].to_pylist()] == list(range(12))
    with duckdb.connect(str(database)) as connection:
        connection.execute("INSERT INTO observations VALUES (12, [])")
    for source in (original, restored):
        assert [value for batch in source() for value in batch["id"].to_pylist()] == list(range(13))


def test_connections_open_lazily_and_close_after_early_stop(database: Path, monkeypatch: pytest.MonkeyPatch):
    connect = duckdb.connect
    connections = []

    def record(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(duckdb, "connect", record)
    data = rf.DuckDBDataModule(
        model=model(), database=database, validate="SELECT id FROM observations ORDER BY id", ingress_rows=1
    )
    loader = data.val_dataloader()
    assert connections == []
    scanned = arrow.scan(loader.dataset.source)
    assert next(scanned)["id"].to_pylist() == [0]
    scanned.close()

    assert len(connections) == 1
    with pytest.raises(duckdb.ConnectionException, match="closed"):
        connections[0].execute("SELECT 1")
    with connect(str(database)) as connection:
        connection.execute("INSERT INTO observations VALUES (12, [])")


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM observations",
        "CREATE TABLE unexpected AS SELECT 1 AS id",
        "SELECT id FROM observations; DELETE FROM observations",
    ],
)
def test_only_one_select_statement_is_allowed(database: Path, query: str):
    data = rf.DuckDBDataModule(model=model(), database=database, validate=query)

    with pytest.raises(ValueError, match="SELECT"):
        list(data.val_dataloader())
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT count(*) FROM observations").fetchone() == (12,)
        assert connection.execute("SHOW TABLES").fetchall() == [("observations",)]


def test_database_connections_are_read_only_and_missing_files_are_not_created(database: Path, tmp_path: Path):
    data = rf.DuckDBDataModule(
        model=model(),
        database=database,
        validate="SELECT 1 AS id, current_setting('access_mode') AS access_mode",
    )
    assert next(iter(data.val_dataloader())).source["access_mode"].to_pylist() == ["read_only"]

    missing = tmp_path / "missing.duckdb"
    data = rf.DuckDBDataModule(model=model(), database=missing, validate="SELECT 1 AS id")
    with pytest.raises(duckdb.IOException):
        list(data.val_dataloader())
    assert not missing.exists()


def test_in_memory_queries_read_parameterized_files(tmp_path: Path):
    path = tmp_path / "observations.parquet"
    pq.write_table(pa.table({"id": [3, 1, 2]}), path)
    data = rf.DuckDBDataModule(
        model=model(),
        validate="SELECT * FROM read_parquet(?) ORDER BY id",
        parameters={"validate": [str(path)]},
    )

    for _ in range(2):
        assert [value for batch in data.val_dataloader() for value in batch.source["id"].to_pylist()] == [1, 2, 3]


def test_preprocessing_retention_and_epoch_limit_use_the_arrow_pipeline(database: Path):
    @rf.preprocess
    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(pl.col("measurement") % 2 == 0).rename({"measurement": "id"})

    data = rf.DuckDBDataModule(
        model=model(),
        database=database,
        validate="SELECT id AS measurement, 'row-' || id AS request_id FROM observations ORDER BY id",
        preprocessor={"validate": prepare},
        retain={"validate": ("request_id",)},
        epoch_size={"validate": 3},
        ingress_rows=3,
    )
    batches = list(data.val_dataloader())

    assert [len(batch.source) for batch in batches] == [2, 1]
    assert pa.concat_tables([batch.source for batch in batches]).to_pydict() == {
        "id": [0, 2, 4],
        "request_id": ["row-0", "row-2", "row-4"],
    }
    assert all(batch.retain == ("request_id",) for batch in batches)


def test_spawned_persistent_workers_restart_without_duplicate_rows(database: Path):
    data = rf.DuckDBDataModule(
        model=model(),
        database=database,
        train="SELECT id FROM observations ORDER BY id",
        ingress_rows=5,
        num_workers=2,
        persistent_workers=True,
        seed=19,
    )
    loader = data.train_dataloader()
    loader.timeout = 30

    try:
        assert loader.multiprocessing_context.get_start_method() == "spawn"
        first = [value for batch in loader for value in batch.source["id"].to_pylist()]
        second = [value for batch in loader for value in batch.source["id"].to_pylist()]
        assert sorted(first) == sorted(second) == list(range(12))
        assert first != second
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


@pytest.mark.parametrize(("config", "expected"), [(None, 1), ({"threads": 2}, 2)])
def test_connection_configuration_controls_duckdb_execution(config, expected: int):
    data = rf.DuckDBDataModule(model=model(), validate="SELECT current_setting('threads') AS id", config=config)
    assert next(iter(data.val_dataloader())).source["id"].to_pylist() == [expected]


def test_live_connections_and_parameters_for_unconfigured_splits_are_rejected():
    with duckdb.connect() as connection:
        with pytest.raises(TypeError, match="database"):
            rf.DuckDBDataModule(model=model(), database=connection, train="SELECT 1 AS id")
    with pytest.raises(ValueError, match="parameters"):
        rf.DuckDBDataModule(model=model(), train="SELECT 1 AS id", parameters={"validate": [2]})


def test_missing_optional_dependency_reports_the_install_extra(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "duckdb", None)
    data = rf.DuckDBDataModule(model=model(), validate="SELECT 1 AS id")

    with pytest.raises(ImportError, match=r"relflow\[duckdb\]"):
        list(data.val_dataloader())
