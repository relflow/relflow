"""Input-only extensions omit answers while preserving geometry and resources."""

from copy import deepcopy
from datetime import datetime

import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture.binding import bind
from relflow.data.iterables import encode
from relflow.structs.enums import Strata, TensorKey

CASES = [
    (rf.DateParts, [datetime(2026, 1, 1), None, datetime(2026, 2, 1)], {"dateparts": ["day_of_week"]}),
    (rf.Set, [["a", "b", "a"], [], None], {"p_unavailable": 0.0}),
    (rf.Group, ["a", None, "b"], {"n_groups": 3, "p_unavailable": 0.0}),
]


def nested(values):
    return pa.Table.from_pylist(
        [
            {"events": [{"value": values[0]}, {"value": values[1]}, None]},
            {"events": []},
            {"events": None},
            {"events": [{"value": values[2]}]},
        ]
    )


@pytest.mark.parametrize(
    ("field_type", "values", "options", "owner"),
    [
        pytest.param(field_type, values, options, owner, id=f"{field_type.model_fields['type'].default}-{owner}")
        for field_type, values, options in CASES
        for owner in ("root", "branch", "leaf")
        # Inherited role selection is shared; every extension still covers every
        # split and objective, and Set covers each inheritance boundary too.
        if owner == "leaf" or field_type is rf.Set
    ],
)
@pytest.mark.parametrize("objective", [False, True])
@pytest.mark.parametrize("strata", list(Strata))
def test_nested_target_roles_follow_reconstruction_owner(field_type, values, options, owner, objective, strata):
    policy = rf.Mask(rate=0.5, reconstruct=objective)
    model = rf.Model.xs(
        d_model=8,
        attention=None,
        mask=policy if owner == "root" else False,
        events=rf.Branch(
            length=3,
            attention=None,
            mask=policy if owner == "branch" else False,
            value=field_type(mask=policy if owner == "leaf" else False, **options),
        ),
    )
    encoded = model.encode(nested(values), strata=strata)
    field = encoded["/events/value"]
    assert field.targets.is_empty() == (not objective)
    assert field.targets.batch_size == field.state.shape == (4, 1, 3)
    assert not field.present[1:3].any()
    if not objective:
        assert not field.trainable.any()
    else:
        assert set(field.targets.keys()) == {TensorKey.state, TensorKey.content}
        assert field.targets[TensorKey.state].shape == field.state.shape
    copied = encoded.clone().to("cpu")
    assert copied["/events/value"].targets.is_empty() == (not objective)
    torch.testing.assert_close(copied["/events/value"].state, field.state)
    output = model.nodes["/events/value"].embedder.embed(field).payload
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("field_type", [rf.Set, rf.Group])
@pytest.mark.parametrize("objective", [False, True])
def test_target_omission_preserves_prefetched_vocabulary_bindings(field_type, objective):
    options = {"n_groups": 3} if field_type is rf.Group else {}
    model = rf.Model.xs(
        d_model=8,
        attention=None,
        embed=True,
        value=field_type(
            p_unavailable=0.0,
            mask=rf.Mask(query="selected", reconstruct=True) if objective else False,
            **options,
        ),
    )
    sources = [["b", "a", "b"], ["c", "b", "d"]]
    pending = [
        encode(
            pa.table(
                {
                    "value": [[label] for label in labels] if field_type is rf.Set else labels,
                    "selected": [False, True, False],
                }
            ),
            model.schema,
            Strata.train,
            model.interprocess_encoding_context,
        )
        for labels in sources
    ]
    originals = deepcopy(pending)
    for index in (1, 0, 1):
        bound = bind(model, pending[index], Strata.train)
        assert not bound.bindings
        field = bound.tensors["/value"]
        assert field.targets.is_empty() == (not objective)
        vocabulary = field_type.vocabulary(model, "/value")
        expected = torch.tensor([vocabulary.index(label) for label in sources[index]])
        content = field.content["indices"].reshape(3) if field_type is rf.Set else field.content.reshape(3)
        valued = field.state.reshape(3).eq(rf.Tokens.valued)
        torch.testing.assert_close(content[valued], expected[valued])
        if objective:
            answers = field.targets[TensorKey.content]
            answers = answers["indices"].reshape(3) if field_type is rf.Set else answers.reshape(3)
            supervised = field.targets[TensorKey.state].reshape(3).eq(rf.Tokens.valued)
            torch.testing.assert_close(answers[supervised], expected[supervised])
        observed = bound.observations["/value"][TensorKey.content]
        expected_counts = torch.zeros_like(observed)
        expected_counts.scatter_add_(0, expected, torch.ones_like(expected))
        torch.testing.assert_close(observed, expected_counts)
        assert (pending[index].tensors == originals[index].tensors).all()
        assert pending[index].bindings["/value"].labels.equals(originals[index].bindings["/value"].labels)
        for address, observation in pending[index].observations.items():
            assert (observation == originals[index].observations[address]).all()


@pytest.mark.parametrize(("field_type", "values", "options"), CASES)
def test_input_only_targets_remain_empty_after_checkpoint_reload(tmp_path, field_type, values, options):
    model = rf.Model.xs(
        d_model=8,
        attention=None,
        embed=True,
        events=rf.Branch(length=3, attention=None, value=field_type(**options)),
    )
    source = nested(values)
    model.encode(source, strata=Strata.train)
    before = model.predict(source)
    checkpoint = tmp_path / "model.pt"
    model.save(checkpoint)
    restored = rf.Model.load(checkpoint)
    assert restored.encode(source)["/events/value"].targets.is_empty()
    assert restored.predict(source).equals(before)
