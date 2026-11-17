"""Public predictions preserve records while embeddings retain explicit coordinates."""

import typing

import lightning.pytorch
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from polars.testing import assert_frame_equal

import relflow as rf
from relflow.architecture import output
from relflow.architecture.output import plan
from relflow.structs.grain import GrainConfig


def configured():
    return rf.Model(
        preset="small",
        d_model=8,
        embed=True,
        reduction=rf.Attention(n_outputs=1),
        accounts=rf.Branch(
            length=4,
            embed=True,
            reduction=rf.Attention(n_outputs=4),
            amount=rf.Number,
            approved=rf.Boolean(mask=True, embed=True),
            transactions=rf.Branch(
                length=3,
                embed=True,
                reduction=rf.Attention(n_outputs=4),
                amount=rf.Number,
                disputed=rf.Boolean(mask=True, embed=True),
                score=rf.Number(mask=True),
            ),
        ),
    ).eval()


def observations():
    transaction = pa.struct([("amount", pa.float64())])
    account = pa.struct([("amount", pa.float64()), ("transactions", pa.list_(transaction))])
    return pa.Table.from_pylist(
        [
            {"id": "missing", "accounts": None},
            {"id": "empty", "accounts": []},
            {
                "id": "mixed",
                "accounts": [None, {"amount": 2.0, "transactions": [None, {"amount": 3.0}]}, {"transactions": None}],
            },
        ],
        schema=pa.schema([("id", pa.string()), ("accounts", pa.list_(account))]),
    )


def test_nested_records_share_collection_boundaries_and_preserve_structural_holes():
    model = configured()
    source = observations()
    output = model.predict(source, retain=("id",))
    rows = output.to_dicts()

    assert output.columns == ["inputs", "predictions", "representations"]
    assert rows[0]["predictions"] == {"accounts": None}
    assert rows[1]["predictions"] == {"accounts": []}
    accounts = rows[2]["predictions"]["accounts"]
    assert len(accounts) == 3
    assert accounts[0] is None
    assert set(accounts[1]) == {"approved", "transactions"}
    assert accounts[1]["transactions"][0] is None
    assert set(accounts[1]["transactions"][1]) == {"disputed", "score"}
    assert accounts[2]["transactions"] is None
    assert accounts[2]["approved"]["inferred"] is True
    assert "embedding" not in accounts[1]["approved"]
    account_type = output["predictions"].struct.field("accounts").dtype
    assert isinstance(account_type, pl.List)
    assert isinstance(
        output["predictions"].struct.field("accounts").explode().struct.field("transactions").dtype, pl.List
    )

    chunked = model.predict(source, batch_size=1, retain=("id",))
    assert chunked.schema == output.schema
    assert chunked["predictions"].to_list()[:2] == [row["predictions"] for row in rows[:2]]
    assert chunked["predictions"].to_list()[2]["accounts"][0] is None


def test_representations_have_schema_paths_local_ordinals_and_unique_output_indices():
    model = configured()
    source = observations()
    rows = model.predict(source).to_dicts()
    chunked = model.predict(source, batch_size=1).to_dicts()
    for first, second in zip(rows, chunked, strict=True):
        assert [(value["path"], value["parents"], value["output_index"]) for value in first["representations"]] == [
            (value["path"], value["parents"], value["output_index"]) for value in second["representations"]
        ]
    expected = [
        ((), (), 0),
        *((("accounts",), (), index) for index in range(4)),
        (("accounts", "approved"), (1,), 0),
        (("accounts", "approved"), (2,), 0),
        *((("accounts", "transactions"), (1,), index) for index in range(4)),
        *((("accounts", "transactions"), (2,), index) for index in range(4)),
        (("accounts", "transactions", "disputed"), (1, 1), 0),
    ]
    actual = [
        (tuple(value["path"]), tuple(value["parents"]), value["output_index"]) for value in rows[2]["representations"]
    ]
    assert actual == expected
    assert len(set(actual)) == len(actual)
    for row in rows:
        assert row["representations"][0]["path"] == []
        for value in row["representations"]:
            vector = torch.tensor(value["embedding"])
            assert torch.linalg.vector_norm(vector).item() == pytest.approx(1.0, abs=1e-6) or not vector.any()


@pytest.mark.parametrize("selection", [slice(None), slice(0, 2), slice(0, 0)])
def test_nested_output_schema_and_values_roundtrip_without_caller_supplied_schema(selection, tmp_path):
    model = configured()
    source = observations()[selection]
    output = model.predict(source)
    assert output.schema == model.predict(observations()).schema
    ipc_path = tmp_path / "predictions.arrow"
    output.write_ipc(ipc_path)
    assert_frame_equal(pl.read_ipc(ipc_path), output, check_exact=True)
    path = tmp_path / "predictions.parquet"
    output.write_parquet(path)
    assert_frame_equal(pl.read_parquet(path), output, check_exact=True)


def test_embeddings_only_omit_prediction_and_input_sentinel_columns():
    model = rf.Model(preset="small", d_model=8, embed=True, amount=rf.Number)
    source = pa.table({"amount": [1.0]})
    output = model.predict(source)
    assert output.columns == ["representations"]
    assert output.to_dicts()[0]["representations"][0]["path"] == []
    empty = model.predict(source.slice(0, 0))
    assert empty.schema == output.schema
    assert empty.height == 0


def test_model_without_selected_output_rejects_prediction_plan():
    model = rf.Model(preset="small", d_model=8, amount=rf.Number)
    with pytest.raises(ValueError, match="no public prediction outputs"):
        plan(model)


def test_third_party_rich_results_keep_nested_declared_types_and_metadata(tmp_path):
    from relflow.structs.enums import TensorKey
    from relflow.tensorfields.base import TENSORFIELDS
    from relflow.tensorfields.output import struct, variable
    from tests.architecture.test_extension_contract import build_extension

    extension, Request = build_extension()
    candidate = pa.struct([pa.field("score", pa.float32(), metadata={b"unit": b"ratio"})])
    collection = pa.list_(pa.field("element", candidate, metadata={b"role": b"candidate"}))
    datatype = pa.struct(
        [
            pa.field("candidates", collection, nullable=False, metadata={b"owner": b"extension"}),
            pa.field("embedding", pa.float64(), nullable=False),
        ]
    )

    @extension.register(rf.Component.output)
    def declare(module, address):
        return datatype

    @extension.register(rf.Component.write)
    def write(module, prediction, datatype):
        count = prediction.payload[TensorKey.state].numel() // 5
        values = struct({"score": pa.array([0.5] * count, type=pa.float32())}, candidate)
        return struct(
            {
                "candidates": variable(values, torch.ones(count, dtype=torch.int64), datatype=collection),
                "embedding": pa.array([0.25] * count, type=pa.float64()),
            },
            datatype,
        )

    try:
        model = rf.Model(
            preset="small", d_model=8, events=rf.Branch(length=3, amount=rf.Number, note=Request(mask=True))
        )
        source = pa.Table.from_pylist([{"events": [None, {"amount": 1.0}]}])
        data = rf.ArrowDataModule(model=model, predict=source)
        output = model.predict_step(next(iter(data.predict_dataloader())), 0)
        rows = output.to_pylist()
        public = model.predict(source)
        assert isinstance(public, pl.DataFrame)
        assert public.to_dicts() == rows
        assert rows[0]["predictions"]["events"][0] is None
        assert rows[0]["predictions"]["events"][1]["note"]["candidates"] == [{"score": 0.5}]
        assert rows[0]["predictions"]["events"][1]["note"]["embedding"] == 0.25
        assert output.column_names == ["predictions"]
        note = output.schema.field("predictions").type.field("events").type.value_type.field("note").type
        assert note.field("candidates").equals(datatype.field("candidates"), check_metadata=True)
        path = tmp_path / "custom.parquet"
        pq.write_table(output, path)
        assert pq.read_table(path).equals(output, check_metadata=True)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_custom_output_rejects_noncanonical_list_names_at_planning():
    from relflow.tensorfields.base import TENSORFIELDS
    from tests.architecture.test_extension_contract import build_extension

    extension, Request = build_extension()

    @extension.register(rf.Component.output)
    def declare(module, address):
        return pa.struct([pa.field("candidates", pa.list_(pa.float32()))])

    try:
        model = rf.Model(preset="small", d_model=8, note=Request(mask=True))
        with pytest.raises(ValueError, match="/note.*list children named 'element'.*'item'"):
            plan(model)
    finally:
        TENSORFIELDS.pop(extension.name, None)


@pytest.mark.integration
def test_forecast_model_trains_and_writes_horizon_rows():
    lightning.pytorch.seed_everything(7, workers=True)
    model = rf.Model(
        preset="small",
        d_model=8,
        batch_size=4,
        events=rf.Branch(
            length=4,
            overflow="tail",
            direction=rf.Direction.causal,
            mask=rf.Mask(select=rf.Forecast(horizon=2, targets=("value",)), reconstruct=True),
            value=rf.Number,
            load=rf.Number,
        ),
    )
    table = pa.table({"events": [[{"value": float(i + j), "load": float(i)} for i in range(9)] for j in range(8)]})
    datamodule = rf.ArrowDataModule(
        model=model, train=table, validate=table, num_workers=0, persistent_workers=False, pin_memory=False
    )
    model.fit(datamodule, max_epochs=1, limit_train_batches=1, limit_val_batches=1)
    events = model.predict(table.slice(0, 2))["predictions"].struct.field("events").to_list()
    assert [len(record) for record in events] == [6, 6]
    assert [[row["value"]["inferred"] for row in record] for record in events] == [[False] * 4 + [True] * 2] * 2
    assert all("load" not in row for record in events for row in record)


def test_a_hidden_covariate_in_a_user_future_row_does_not_change_the_prediction():
    lightning.pytorch.seed_everything(7, workers=True)
    model = rf.Model(
        preset="small",
        d_model=8,
        batch_size=4,
        events=rf.Branch(
            length=4,
            overflow="tail",
            direction=rf.Direction.causal,
            mask=rf.Mask(select=rf.Forecast(horizon=1, targets=("value",), known=("temperature",)), reconstruct=True),
            value=rf.Number,
            load=rf.Number,
            temperature=rf.Number,
        ),
    )

    def forecast(load: float, temperature: float) -> list:
        past = [{"value": float(i), "load": float(i), "temperature": float(i)} for i in range(4)]
        future = {"value": None, "load": load, "temperature": temperature}
        return model.predict(pa.table({"events": [[*past, future]]}))["predictions"].to_list()

    assert forecast(load=1.0, temperature=5.0) == forecast(load=-1e6, temperature=5.0)
    assert forecast(load=1.0, temperature=5.0) != forecast(load=1.0, temperature=-1e6)


def test_tokenwise_table_covers_every_grain():
    assert set(output.TOKENWISE) == set(typing.get_args(typing.get_args(GrainConfig)[0]))


def test_steps_branch_without_reduction_writes_one_representation_per_row():
    model = rf.Model(
        preset="small",
        d_model=8,
        ett=rf.Branch.chronological(length=4, reduction=None, embed=True, x=rf.Number, y=rf.Number),
    )
    rows = [
        {"ett": [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 1.0}, {"x": 3.0, "y": 0.0}]},
        {"ett": [{"x": 4.0, "y": 4.0}]},
    ]
    records = model.predict(pa.Table.from_pylist(rows))["representations"].to_list()
    assert [[r["parents"] for r in row] for row in records] == [[[0], [1], [2]], [[0]]]
    assert all(r["output_index"] == 0 and r["path"] == ["ett"] for row in records for r in row)


def test_coordinates_branch_without_reduction_keeps_one_representation_per_token():
    model = rf.Model(
        preset="small",
        d_model=8,
        ett=rf.Branch(length=4, reduction=None, embed=True, x=rf.Number, y=rf.Number),
    )
    rows = [{"ett": [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 1.0}, {"x": 3.0, "y": 0.0}]}]
    first = model.predict(pa.Table.from_pylist(rows))["representations"].to_list()[0]
    assert [r["parents"] for r in first] == [[]] * 6
    assert [r["output_index"] for r in first] == [0, 1, 2, 3, 4, 5]


def test_nested_steps_rows_carry_parent_and_row_ordinals():
    model = rf.Model(
        preset="small",
        d_model=8,
        visits=rf.Branch(
            length=2,
            reduction=rf.Mean(),
            ett=rf.Branch.chronological(length=3, reduction=None, embed=True, x=rf.Number),
        ),
    )
    rows = [{"visits": [{"ett": [{"x": 1.0}, {"x": 2.0}]}, {"ett": [{"x": 3.0}]}]}]
    first = model.predict(pa.Table.from_pylist(rows))["representations"].to_list()[0]
    assert [r["parents"] for r in first] == [[0, 0], [0, 1], [1, 0]]


@pytest.mark.parametrize("mixer", [rf.Conv(), rf.Transformer()])
@pytest.mark.parametrize("ragged", [False, True], ids=["dense", "packed"])
def test_changing_a_future_row_never_changes_earlier_row_representations(mixer, ragged):
    torch.manual_seed(0)
    model = rf.Model(
        preset="small",
        d_model=8,
        dropout=0.0,
        ett=rf.Branch.chronological(length=6, mixer=mixer, reduction=None, embed=True, x=rf.Number, y=rf.Number),
    )

    def table(future: float) -> pa.Table:
        first = [{"x": float(i), "y": float(-i)} for i in range(6)]
        first[4] = {"x": future, "y": future}
        second = first[:3] if ragged else first
        return pa.Table.from_pylist([{"ett": first}, {"ett": second}])

    before, after = (model.predict(table(value))["representations"].to_list()[0] for value in (4.0, 99.0))
    for row in range(4):
        torch.testing.assert_close(
            torch.tensor(after[row]["embedding"]), torch.tensor(before[row]["embedding"]), rtol=0, atol=1e-6
        )
    assert before[4]["embedding"] != after[4]["embedding"]
