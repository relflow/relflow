"""Automatic collection geometry stays stable at the public Arrow boundary."""

from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

import relflow as rf
from tests.architecture.test_extension_contract import build_extension

EVENTS = pa.list_(pa.struct([("amount", pa.float64()), ("label", pa.bool_())]))


def observations(events):
    return pa.Table.from_pylist([{"events": value} for value in events], schema=pa.schema([("events", EVENTS)]))


def test_variable_targets_keep_types_offsets_and_stream_across_batch_widths(tmp_path):
    model = rf.Model.xs(d_model=8, events=rf.Branch(amount=rf.Number, label=rf.Boolean(mask=True, embed=True))).eval()
    values = [[], None, [None, {"amount": 2.0, "label": True}], [{"amount": float(i)} for i in range(5)]]
    batches = [model.predict(observations([value])) for value in values]
    combined = pa.concat_tables(batches)
    batched = model.predict(observations(values), batch_size=2)

    assert all(batch.schema == combined.schema for batch in batches)
    assert batched.schema == combined.schema
    events = combined["predictions"].combine_chunks().field("events")
    assert pa.types.is_list(events.type)
    assert events.offsets.to_pylist() == [0, 0, 0, 2, 7]
    assert events.is_valid().to_pylist() == [True, False, True, True]
    assert events[2].as_py()[0] is None
    assert len(events[2].as_py()) == 2
    assert "embedding" not in events.type.value_type.field("label").type.names
    representations = combined["representations"].to_pylist()
    assert [item["parents"] for item in representations[2]] == [[1]]
    assert all(item["path"] == ["events", "label"] for row in representations for item in row)

    writer = rf.Writer(tmp_path)
    for index, batch in enumerate(batches):
        writer.write_on_batch_end(SimpleNamespace(global_rank=0), model, batch, None, None, index, 0)
    writer.close()
    assert pq.read_table(tmp_path / "rank-0.parquet", schema=combined.schema).equals(combined)
    empty = model.predict(observations([]))
    assert empty.schema == combined.schema
    assert empty.num_rows == 0


@pytest.mark.parametrize("length", [1, 4])
def test_explicit_limits_keep_observed_output_lengths(length):
    model = rf.Model.xs(
        d_model=8, events=rf.Branch(length=length, amount=rf.Number, label=rf.Boolean(mask=True))
    ).eval()
    source = observations([[{"amount": 2.0}]])

    assert model.encode(source)["/events/amount"].state.shape == (1, 1, 1)
    output = model.predict(source)["predictions"].combine_chunks().field("events")
    assert pa.types.is_list(output.type)
    assert output.type.value_field.name == "element"
    assert len(output[0].as_py()) == 1
    assert list(output[0].as_py()[0]) == ["label"]


def test_nested_variable_axes_and_pass_through_embeddings_preserve_structural_slots():
    model = rf.Model.xs(
        d_model=8,
        events=rf.Branch(
            embed=True,
            reduction=None,
            amount=rf.Number,
            items=rf.Branch(reduction=None, embed=True, amount=rf.Number, label=rf.Boolean(mask=True)),
        ),
    ).eval()
    source = pa.Table.from_pylist(
        [
            {"events": [{"amount": 1.0, "items": [None, {"amount": 2.0, "label": True}]}]},
            {"events": [{"amount": 3.0, "items": []}, {"amount": 4.0, "items": [{"amount": 5.0}]}]},
        ]
    )
    output = model.predict(source)
    assert output.schema == model.predict(source, batch_size=1).schema
    rows = output.to_pylist()
    assert len(rows[0]["predictions"]["events"]) == 1
    assert len(rows[0]["predictions"]["events"][0]["items"]) == 2
    assert rows[0]["predictions"]["events"][0]["items"][0] is None
    first = rows[0]["representations"]
    outer = [item for item in first if item["path"] == ["events"]]
    inner = [item for item in first if item["path"] == ["events", "items"]]
    # The null item owns no tokens at either level; its slots stay reserved in the positions.
    assert len(outer) == 3
    assert len(inner) == 2
    assert all(item["parents"] == [] for item in outer)
    assert all(item["parents"] == [0] for item in inner)
    assert [item["output_index"] for item in outer] == [0, 2, 4]
    assert [item["output_index"] for item in inner] == [1, 3]
    assert all(any(value != 0.0 for value in item["embedding"]) for item in outer[:1] + inner[:1])
    second_inner = [item for item in rows[1]["representations"] if item["path"] == ["events", "items"]]
    assert len(second_inner) == 2
    assert all(item["parents"] == [1] for item in second_inner)


def test_checkpoint_keeps_symbolic_geometry_and_predictions(tmp_path):
    torch.manual_seed(103)
    model = rf.Model.xs(d_model=8, events=rf.Branch(amount=rf.Number, label=rf.Boolean(mask=True))).eval()
    source = observations([[{"amount": float(i)} for i in range(4)]])
    expected = model.predict(source).to_pylist()[0]["predictions"]["events"]
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    restored = rf.Model.load(checkpoint).eval()

    branch = restored.schema.branches[rf.Address("events")]
    assert branch.length is None
    assert branch.overflow is None
    assert restored.schema.model_dump() == model.schema.model_dump()
    actual = restored.predict(source).to_pylist()[0]["predictions"]["events"]
    torch.testing.assert_close(
        torch.tensor([value["label"]["content"]["probability"] for value in actual]),
        torch.tensor([value["label"]["content"]["probability"] for value in expected]),
    )


def test_pass_through_outputs_keep_intrinsic_slots_separate_from_item_coordinates():
    model = rf.Model.xs(d_model=8, events=rf.Branch(reduction=None, embed=True, amount=rf.Number)).eval()
    result = model.predict(observations([[], None, [None]]))
    assert result.column_names == ["representations"]
    rows = result["representations"].to_pylist()
    assert rows == [[], [], []]


def test_unbounded_warning_names_active_paths_once_per_generation(monkeypatch):
    from relflow.logging import logger

    warnings = []
    monkeypatch.setattr(type(logger), "warning", lambda self, message, *args, **kwargs: warnings.append(message))
    model = rf.Model.xs(
        d_model=8,
        events=rf.Branch(items=rf.Branch(amount=rf.Number), answer=rf.Number(mask=True)),
        inactive=rf.Branch(amount=rf.Number(active=False)),
    )
    assert len(warnings) == 1
    assert "'/events'" in warnings[0]
    assert "'/events/items'" in warnings[0]
    assert "inactive" not in warnings[0]
    assert "denial of service" in warnings[0]
    model.warn_unbounded()
    assert len(warnings) == 1
    model.update(rf.where("address") == rf.Address("events"), length=3)
    assert len(warnings) == 2
    assert "'/events/items'" in warnings[1]
    assert "'/events'," not in warnings[1]
    model.warn_unbounded()
    assert len(warnings) == 2


def test_third_party_tensorfields_follow_automatic_axes_without_shared_special_cases():
    extension, Request = build_extension()
    try:
        model = rf.Model.xs(d_model=8, events=rf.Branch(value=Request, label=Request(mask=True, embed=True))).eval()
        source = pa.Table.from_pylist(
            [{"events": []}, {"events": [{"value": "one", "label": "yes"}]}, {"events": [{"value": "two"}] * 5}],
            schema=pa.schema([("events", pa.list_(pa.struct([("value", pa.string()), ("label", pa.string())])))]),
        )
        result = model.predict(source, batch_size=1)
        assert result.column_names == ["representations"]
        representations = result["representations"].to_pylist()
        assert [len(row) for row in representations] == [0, 1, 5]
        assert all(len(value["embedding"]) == 8 for row in representations for value in row)
        assert all(value["path"] == ["events", "label"] for row in representations for value in row)
        assert [value["parents"] for value in representations[2]] == [[index] for index in range(5)]
    finally:
        rf.TENSORFIELDS.pop(extension.name, None)


def test_pass_through_representations_skip_null_items_but_keep_their_slots():
    model = rf.Model.xs(
        d_model=8, events=rf.Branch(reduction=None, embed=True, amount=rf.Number(embed=True), tag=rf.Hash)
    ).eval()
    source = pa.Table.from_pylist([{"events": [{"amount": 1.0, "tag": "a"}, None, {"amount": 2.0, "tag": "b"}]}])
    rows = model.predict(source)["representations"].to_pylist()[0]
    tokens = [item for item in rows if item["path"] == ["events"]]
    leaves = [item for item in rows if item["path"] == ["events", "amount"]]

    # Field-major slots: amount at 0..2 and tag at 3..5; the null item's slots 1 and 4 own no token.
    assert [item["output_index"] for item in tokens] == [0, 2, 3, 5]
    assert all(any(value != 0.0 for value in item["embedding"]) for item in tokens)
    assert [item["parents"] for item in leaves] == [[0], [2]]
