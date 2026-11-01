"""Random branch windows retain whole records before masks and tensorization."""

import random

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.data.iterables import geometry
from relflow.data.ragged import coalesce
from relflow.structs.experiment import Schema


def test_random_overflow_serializes_and_requires_a_finite_cap():
    assert rf.Overflow.random == "random"
    branch = rf.Branch(length=3, overflow="random", value=rf.Number)
    assert rf.Branch.model_validate_json(branch.model_dump_json()).overflow == rf.Overflow.random
    configured = Schema.from_tree(d_model=8, n_layers=1, n_heads=2, events=branch)
    restored = Schema.model_validate_json(configured.model_dump_json())
    assert restored.branches["/events"].overflow == rf.Overflow.random
    with pytest.raises(ValueError, match="unbounded branch.*finite length"):
        rf.Branch(overflow="random", value=rf.Number)


@pytest.mark.parametrize("strata", list(rf.Strata))
def test_random_overflow_keeps_contiguous_full_windows_and_uses_tail_outside_training(strata):
    configured = Schema.from_tree(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(length=3, overflow="random", value=rf.Number)
    )
    source = pa.table({"events": [[{"value": index} for index in range(8)] for _ in range(128)]})
    resolved = coalesce(source, configured, strata, seed=7, epoch=2, ordinal=3)
    values = np.asarray(resolved.fields["/events/value"].pristine.values).reshape(128, 3)

    assert np.all(np.diff(values, axis=1) == 1)
    assert resolved.layouts["/events"].lengths.tolist() == [[3]] * 128
    if strata == rf.Strata.train:
        assert set(values[:, 0]) == set(range(6))
    else:
        assert values.tolist() == [[5, 6, 7]] * 128
        other = coalesce(source, configured, strata, seed=91, epoch=15, ordinal=8)
        assert other.fields["/events/value"].pristine.values.equals(resolved.fields["/events/value"].pristine.values)


def test_random_windows_replay_and_vary_by_epoch_seed_batch_and_branch():
    fields = {name: rf.Branch(length=3, overflow="random", value=rf.Number) for name in ("left", "right")}
    configured = Schema.from_tree(d_model=8, n_layers=1, n_heads=2, fields=fields)
    rows = [[{"value": index} for index in range(12)] for _ in range(64)]
    source = pa.table({"left": rows, "right": rows})
    original = coalesce(source, configured, rf.Strata.train, seed=7, epoch=2, ordinal=3)
    before = configured.model_dump_json()
    replay = coalesce(source, configured, rf.Strata.train, seed=7, epoch=2, ordinal=3)
    expected = original.fields["/left/value"].pristine.values

    assert replay.fields["/left/value"].pristine.values.equals(expected)
    assert not original.fields["/right/value"].pristine.values.equals(expected)
    for options in (
        {"seed": 8, "epoch": 2, "ordinal": 3},
        {"seed": 7, "epoch": 3, "ordinal": 3},
        {"seed": 7, "epoch": 2, "ordinal": 4},
    ):
        changed = coalesce(source, configured, rf.Strata.train, **options)
        assert not changed.fields["/left/value"].pristine.values.equals(expected)
    assert configured.model_dump_json() == before


def test_random_windows_do_not_use_or_advance_process_random_state():
    configured = Schema.from_tree(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(length=3, overflow="random", value=rf.Number)
    )
    source = pa.table({"events": [[{"value": index} for index in range(12)] for _ in range(32)]})
    numpy_state, python_state, torch_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    first = coalesce(source, configured, rf.Strata.train, seed=5)

    assert all(np.array_equal(a, b) for a, b in zip(numpy_state, np.random.get_state(), strict=True))
    assert random.getstate() == python_state
    assert torch.equal(torch.get_rng_state(), torch_state)
    try:
        np.random.seed(934)
        random.seed(934)
        torch.manual_seed(934)
        second = coalesce(source, configured, rf.Strata.train, seed=5)
        assert first.fields["/events/value"].pristine.values.equals(second.fields["/events/value"].pristine.values)
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        torch.set_rng_state(torch_state)


def test_random_windows_preserve_null_empty_short_and_null_item_geometry():
    configured = Schema.from_tree(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(length=3, overflow="random", value=rf.Number)
    )
    event = pa.struct([("value", pa.int64())])
    source = pa.table(
        {"events": pa.array([None, [], [None], [{"value": 1}, None, {"value": None}]], type=pa.list_(event))}
    )
    for strata in rf.Strata:
        resolved = coalesce(source, configured, strata, seed=7)
        layout = resolved.layouts["/events"]
        axes = geometry(resolved.layouts, rows=4)["/events"]
        assert layout.lengths.tolist() == [[0], [0], [1], [3]]
        assert layout.valid.tolist() == [[False], [True], [True], [True]]
        assert axes.slots.tolist() == [
            [[False, False, False]],
            [[False, False, False]],
            [[False, False, False]],
            [[True, False, True]],
        ]
        assert resolved.fields["/events/value"].pristine.values.to_pylist() == [1]


def test_random_windows_align_nested_records_queries_masks_and_geometry():
    configured = Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            query="payload.history",
            length=3,
            overflow="random",
            value=rf.Number,
            target=rf.Number(mask=rf.Mask(query="selected", reconstruct=True)),
            children=rf.Branch(length=2, overflow="random", parent=rf.Number, value=rf.Number),
        ),
    )
    source = pa.Table.from_pylist(
        [
            {
                "payload": {
                    "history": [
                        None
                        if index == 3
                        else {
                            "value": row * 100 + index,
                            "target": -(row * 100 + index),
                            "selected": index % 2 == 0,
                            "children": [{"parent": row * 100 + index, "value": child} for child in range(5)],
                        }
                        for index in range(9)
                    ]
                }
            }
            for row in range(16)
        ]
    )
    resolved = coalesce(source, configured, rf.Strata.train, seed=2, epoch=3)
    outer, inner = resolved.layouts["/events"], resolved.layouts["/events/children"]
    records = dict(zip(outer.placement, outer.records.to_pylist(), strict=True))
    target = resolved.fields["/events/target"]
    for position, record in records.items():
        assert target.trainable[position] == record["selected"]
    assert target.pristine.values.to_pylist() == [-record["value"] for record in records.values()]
    assert inner.shape == (16, 1, 3, 2)
    for position, record in zip(inner.placement, inner.records.to_pylist(), strict=True):
        assert record["parent"] == records[position // 2]["value"]
    nested = resolved.fields["/events/children/value"].pristine.values.to_numpy().reshape(-1, 2)
    assert np.all(np.diff(nested, axis=1) == 1)
    axes = geometry(resolved.layouts, rows=16)
    assert axes["/events"].slots.reshape(-1).sum().item() == len(records)
    assert axes["/events/children"].slots.reshape(-1).sum().item() == len(inner.records)


def test_random_windows_are_invariant_to_dictionary_chunks_and_sliced_arrow_storage():
    configured = Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(length=3, overflow="random", value=rf.Number, label=rf.Category),
    )
    size = 7
    offsets = pa.array(np.arange(0, 9 * size, size), type=pa.int32())
    records = pa.StructArray.from_arrays(
        [pa.array(np.arange(8 * size)), pa.array(["a", "b"] * (4 * size)).dictionary_encode()],
        names=["value", "label"],
    )
    events = pa.ListArray.from_arrays(offsets, records)
    source = pa.table({"events": events.slice(1, 6)})
    copied = pa.table({"events": pa.array(source["events"].to_pylist(), type=source["events"].type)})
    chunked = pa.concat_tables([source.slice(0, 2), source.slice(2, 1), source.slice(3)])
    expected = coalesce(source, configured, rf.Strata.train, seed=7, epoch=3, ordinal=4)
    for table in (copied, chunked):
        actual = coalesce(table, configured, rf.Strata.train, seed=7, epoch=3, ordinal=4)
        for address in expected.fields:
            assert (
                actual.fields[address].pristine.values.to_pylist()
                == expected.fields[address].pristine.values.to_pylist()
            )
            assert (
                actual.fields[address].pristine.placement.to_pylist()
                == expected.fields[address].pristine.placement.to_pylist()
            )
        assert actual.layouts["/events"].lengths.tolist() == expected.layouts["/events"].lengths.tolist()


def test_random_overflow_prediction_keeps_tail_null_slots_and_local_parent_ordinals():
    model = rf.Model.xs(
        d_model=8,
        events=rf.Branch(length=3, overflow="random", amount=rf.Number, label=rf.Boolean(mask=True, embed=True)),
    ).eval()
    source = pa.Table.from_pylist(
        [
            {"events": [{"amount": 1.0}, {"amount": 2.0}, None, {"amount": 4.0}]},
            {"events": []},
            {"events": None},
        ]
    )
    output = model.predict(source)
    rows = output.to_pylist()
    assert len(rows[0]["predictions"]["events"]) == 3
    assert rows[0]["predictions"]["events"][1] is None
    assert [item["parents"] for item in rows[0]["representations"]] == [[0], [2]]
    assert rows[1]["predictions"]["events"] == []
    assert rows[2]["predictions"]["events"] is None
    assert output.schema == model.predict(source, batch_size=1).schema
