"""Collection geometry follows Arrow slots before leaf values or masking."""

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.data.arrow import Geometry
from relflow.data.iterables import encode, mock
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, Tokens
from relflow.structs.experiment import Schema


def schema(**fields):
    return Schema.from_tree(d_model=8, n_layers=1, n_heads=2, **fields)


def events(rows):
    item = pa.struct([pa.field("left", pa.float64()), pa.field("right", pa.float64())])
    return pa.table({"events": pa.array(rows, type=pa.list_(item))})


def test_null_objects_count_as_slots_before_leaf_values_and_masks():
    configured = schema(events=rf.Branch(left=rf.Number, right=rf.Number(mask=True)))
    source = events([[{"left": None, "right": None}, None, {"left": None, "right": 4.0}], [], None])

    resolved = coalesce(source, configured, Strata.train)
    layout = resolved.layouts["/events"]

    assert layout.shape == (3, 1, 3)
    assert layout.lengths.tolist() == [[3], [0], [0]]
    assert layout.present.tolist() == [[True], [True], [True]]
    assert layout.valid.tolist() == [[True], [True], [False]]
    assert layout.placement.tolist() == [0, 2]
    assert all(field.pristine.shape == layout.shape for field in resolved.fields.values())
    assert resolved.fields["/events/left"].pristine.dense[0].tolist() == [[Tokens.null, Tokens.padded, Tokens.null]]
    assert not np.asarray(resolved.fields["/events/right"].present).any()


@pytest.mark.parametrize("length", [None, 8])
def test_geometry_grows_and_shrinks_without_rewriting_symbolic_schema(length):
    configured = schema(events=rf.Branch(length=length, left=rf.Number))
    before = configured.model_dump()

    for rows, width in [([[{"left": 1.0}]], 1), ([[None, None, None]], 3), ([[], None], 0)]:
        resolved = coalesce(events(rows), configured, Strata.predict)
        assert resolved.layouts["/events"].shape == (len(rows), 1, width)
        assert resolved.fields["/events/left"].pristine.shape == (len(rows), 1, width)
        assert configured.model_dump() == before


@pytest.mark.parametrize("policy,expected", [("head", [1.0]), ("tail", [2.0, 3.0])])
def test_ancestor_retention_precedes_descendant_extent_resolution(policy, expected):
    configured = schema(events=rf.Branch(length=1, overflow=policy, items=rf.Branch(value=rf.Number)))
    item = pa.struct([pa.field("value", pa.float64())])
    event = pa.struct([pa.field("items", pa.list_(item))])
    source = pa.table(
        {
            "events": pa.array(
                [[{"items": [{"value": 1.0}]}, {"items": [{"value": 2.0}, {"value": 3.0}]}]], type=pa.list_(event)
            )
        }
    )

    resolved = coalesce(source, configured, Strata.predict)

    assert resolved.layouts["/events"].shape == (1, 1, 1)
    assert resolved.layouts["/events/items"].shape == (1, 1, 1, len(expected))
    assert resolved.fields["/events/items/value"].pristine.values.to_pylist() == expected


def test_typed_empty_nested_inputs_have_zero_extents_and_keep_the_row_axis():
    configured = schema(events=rf.Branch(items=rf.Branch(length=4, value=rf.Number)))
    item = pa.struct([pa.field("value", pa.float64())])
    event = pa.struct([pa.field("items", pa.list_(item))])
    source = pa.table({"events": pa.array([], type=pa.list_(event))})

    resolved = coalesce(source, configured, Strata.predict)

    assert resolved.layouts["/"].shape == (0, 1)
    assert resolved.layouts["/events"].shape == (0, 1, 0)
    assert resolved.layouts["/events/items"].shape == (0, 1, 0, 0)
    assert resolved.fields["/events/items/value"].pristine.dense.shape == (0, 1, 0, 0)


def test_tensor_geometry_survives_encoded_cloning_device_transfer_and_row_selection():
    configured = schema(events=rf.Branch(left=rf.Number), ignored=rf.Branch(value=rf.Number(active=False)))
    source = events([[{"left": 1.0}, None], []])

    encoded = encode(source, configured, Strata.predict, {})
    inputs = encoded.tensors.clone().to("cpu")
    axes = inputs[rf.Address()]

    assert set(axes.keys()) == {"/", "/events"}
    assert isinstance(axes["/events"], Geometry)
    assert axes["/events"].lengths.dtype == torch.int64
    assert axes["/events"].lengths.tolist() == [[2], [0]]
    assert axes["/events"].slots.tolist() == [[[True, False]], [[False, False]]]
    assert inputs[1:]["/"]["/events"].lengths.tolist() == [[0]]
    assert inputs.batch_size == torch.Size([2])


def test_mock_uses_empty_collection_axes_and_excludes_inactive_subtrees():
    configured = schema(
        events=rf.Branch(items=rf.Branch(length=4, value=rf.Number)), ignored=rf.Branch(value=rf.Number(active=False))
    )

    inputs = mock(configured, batch_size=3)

    assert inputs["/events/items/value"].state.shape == (3, 1, 0, 0)
    assert set(inputs["/"].keys()) == {"/", "/events", "/events/items"}
    assert inputs["/"]["/"].lengths.tolist() == [1, 1, 1]
    assert inputs["/"]["/events/items"].lengths.shape == (3, 1, 0)


@pytest.mark.parametrize("missing", [False, True])
def test_untyped_null_or_absent_collections_do_not_require_descendant_atoms(missing):
    configured = schema(
        events=rf.Branch(
            items=rf.Branch(value=rf.Number),
            mask=rf.Mask(query="unavailable_selector", skip=True, reconstruct=True),
        )
    )
    source = pa.table({"unused": [1, 2]}) if missing else pa.table({"events": [None, None]})

    resolved = coalesce(source, configured, Strata.predict)

    assert resolved.layouts["/events"].shape == (2, 1, 0)
    assert resolved.layouts["/events"].present.tolist() == [[not missing], [not missing]]
    assert not resolved.layouts["/events"].valid.any()
    assert resolved.layouts["/events/items"].shape == (2, 1, 0, 0)
    assert resolved.fields["/events/items/value"].vacant


@pytest.mark.parametrize("source", [pa.table({"events": [3]}), pa.table({"events": [[3]]})])
def test_nonnull_atoms_are_not_silently_treated_as_absent_collections(source):
    configured = schema(events=rf.Branch(value=rf.Number))

    with pytest.raises(ValueError, match="expected a (list|struct)"):
        coalesce(source, configured, Strata.predict)


@pytest.mark.parametrize("missing", [False, True])
def test_vacant_training_still_observes_and_learns_every_registered_field(monkeypatch, missing):
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(number=rf.Number, category=rf.Category, quantile=rf.Quantile),
    )
    calls = []
    for name in ("number", "category", "quantile"):
        extension = rf.TENSORFIELDS[name]
        learner = extension.learn

        def learn(*args, original=learner, **kwargs):
            calls.append(kwargs["address"])
            return original(*args, **kwargs)

        monkeypatch.setitem(extension.components, rf.Component.learn, learn)

    source = pa.table({"unused": [1, 2]}) if missing else pa.table({"events": [None, None]})
    before = {name: value.clone() for name, value in model.state_dict().items() if torch.is_tensor(value)}

    inputs = model.encode(source, strata=Strata.train)

    assert set(calls) == {"/events/number", "/events/category", "/events/quantile"}
    assert all(inputs[address].state.shape == (2, 1, 0) for address in calls)
    assert rf.Number.normalization(model, "/events/number")["count"] == 0
    assert rf.Category.vocabulary(model, "/events/category") == ()
    for name, value in before.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
