"""Public predictions preserve records while embeddings retain explicit coordinates."""

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import pytest
import torch

import relflow as rf
from relflow.architecture.output import plan


def configured():
    return rf.Model.xs(
        d_model=8,
        embed=True,
        accounts=rf.Branch(
            length=4,
            embed=True,
            amount=rf.Number,
            approved=rf.Boolean(mask=True, embed=True),
            transactions=rf.Branch(
                length=3,
                embed=True,
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
    rows = output.to_pylist()

    assert output.column_names == ["inputs", "predictions", "representations"]
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
    account_type = output.schema.field("predictions").type.field("accounts").type
    assert pa.types.is_list(account_type)
    assert account_type.value_field.name == "element"
    assert pa.types.is_list(account_type.value_type.field("transactions").type)
    assert output.schema.metadata == {b"relflow.model_version": model.version.encode()}

    chunked = model.predict(source, batch_size=1, retain=("id",))
    assert chunked.schema.equals(output.schema, check_metadata=True)
    assert chunked["predictions"].to_pylist()[:2] == [row["predictions"] for row in rows[:2]]
    assert chunked["predictions"].to_pylist()[2]["accounts"][0] is None


def test_representations_have_schema_paths_local_ordinals_and_unique_output_indices():
    model = configured()
    source = observations()
    rows = model.predict(source).to_pylist()
    chunked = model.predict(source, batch_size=1).to_pylist()
    for first, second in zip(rows, chunked, strict=True):
        assert [(value["path"], value["parents"], value["output_index"]) for value in first["representations"]] == [
            (value["path"], value["parents"], value["output_index"]) for value in second["representations"]
        ]
    expected = [
        ((), (), 0),
        (("accounts",), (), 0),
        (("accounts", "approved"), (1,), 0),
        (("accounts", "approved"), (2,), 0),
        (("accounts", "transactions"), (1,), 0),
        (("accounts", "transactions"), (2,), 0),
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
    assert output.schema.equals(model.predict(observations()).schema, check_metadata=True)
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, output.schema) as writer:
        writer.write_table(output)
    assert ipc.open_stream(sink.getvalue()).read_all().equals(output, check_metadata=True)
    path = tmp_path / "predictions.parquet"
    pq.write_table(output, path)
    assert pq.read_table(path).equals(output, check_metadata=True)


def test_embeddings_only_omit_prediction_and_input_sentinel_columns():
    model = rf.Model.xs(d_model=8, embed=True, amount=rf.Number)
    source = pa.table({"amount": [1.0]})
    output = model.predict(source)
    assert output.column_names == ["representations"]
    assert output.to_pylist()[0]["representations"][0]["path"] == []
    empty = model.predict(source.slice(0, 0))
    assert empty.schema.equals(output.schema, check_metadata=True)
    assert empty.num_rows == 0


def test_model_without_selected_output_rejects_prediction_plan():
    model = rf.Model.xs(d_model=8, amount=rf.Number)
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
        model = rf.Model.xs(d_model=8, events=rf.Branch(length=3, amount=rf.Number, note=Request(mask=True)))
        source = pa.Table.from_pylist([{"events": [None, {"amount": 1.0}]}])
        output = model.predict(source)
        rows = output.to_pylist()
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
        model = rf.Model.xs(d_model=8, note=Request(mask=True))
        with pytest.raises(ValueError, match="/note.*list children named 'element'.*'item'"):
            plan(model)
    finally:
        TENSORFIELDS.pop(extension.name, None)
