import pydantic
import pytest

import relflow as rf


def test_reduction_configs_are_frozen_strict_tagged_values() -> None:
    assert rf.Mean().model_dump() == {"type": "mean"}
    assert rf.Attention(n_outputs=2, n_heads=4, n_layers=3, dropout=0.1).model_dump() == {
        "type": "attention",
        "n_outputs": 2,
        "n_heads": 4,
        "n_layers": 3,
        "dropout": 0.1,
        "position": True,
    }
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        rf.Attention().n_layers = 2
    with pytest.raises(pydantic.ValidationError):
        rf.Attention(n_heads=True)
    with pytest.raises(pydantic.ValidationError):
        rf.Attention(n_layers=1.5)
    with pytest.raises(pydantic.ValidationError):
        rf.Attention(n_outputs=True)
    with pytest.raises(pydantic.ValidationError):
        rf.Attention(dropout=1.0)
    with pytest.raises(pydantic.ValidationError):
        rf.Attention(position=1)


def test_branch_reduction_defaults_and_round_trip() -> None:
    branch = rf.Branch(value=rf.Number)

    assert branch.reduction == rf.Attention()

    configured = rf.Branch(reduction=rf.Attention(n_outputs=3, n_heads=2, n_layers=2, dropout=0.2), value=rf.Number)
    payload = configured.model_dump(mode="json", round_trip=True)

    assert rf.Branch.model_validate(payload).model_dump(mode="json") == configured.model_dump(mode="json")

    with pytest.raises(pydantic.ValidationError, match="convolution"):
        rf.Branch.model_validate(
            {
                "name": "items",
                "type": "branch",
                "reduction": {"type": "convolution", "n_outputs": 2},
                "fields": [{"name": "value", "type": "number"}],
            }
        )


def test_schema_round_trip_preserves_reductions() -> None:
    schema = rf.Schema.from_tree(
        groups=rf.Branch(
            length=2,
            reduction=rf.Attention(n_outputs=2, n_heads=2, n_layers=2, dropout=0.1),
            items=rf.Branch(length=3, reduction=None, value=rf.Number),
        ),
        averages=rf.Branch(length=4, reduction=rf.Mean(), value=rf.Number),
        target=rf.Number(mask=True),
        d_model=16,
        n_layers=1,
        n_heads=4,
        reduction=rf.Attention(n_outputs=3, n_heads=2, n_layers=2),
    )

    restored = rf.Schema.model_validate_json(schema.model_dump_json(round_trip=True))

    assert restored.model_dump(mode="json") == schema.model_dump(mode="json")


def test_output_count_belongs_only_to_sized_reduction_configs() -> None:
    with pytest.raises(pydantic.ValidationError):
        rf.Mean(n_outputs=2)

    assert rf.Branch(reduction=None, value=rf.Number).reduction is None


def test_schema_computes_effective_structural_branch_outputs() -> None:
    schema = rf.Schema.from_tree(
        root_value=rf.Number(),
        inactive=rf.Number(active=False),
        items=rf.Branch(
            length=3,
            reduction=None,
            value=rf.Number,
            ignored=rf.Number(active=False),
            details=rf.Branch(length=2, reduction=rf.Attention(n_outputs=4), value=rf.Number),
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
        reduction=rf.Attention(n_outputs=2),
    )

    assert schema.branch_outputs == {
        "record/items/details": 4,
        "record/items": 15,
        "record": 2,
    }


def test_schema_branch_outputs_excludes_inactive_subtrees() -> None:
    schema = rf.Schema.from_tree(
        inactive=rf.Branch(length=3, reduction=None, value=rf.Number(active=False)),
        target=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
        reduction=None,
    )

    assert schema.branch_outputs == {"record/inactive": 0, "record": 1}


def test_attention_reduction_head_geometry_validates_after_binding() -> None:
    with pytest.raises(ValueError, match="Attention reduction requires n_heads to divide d_model"):
        rf.Schema.from_tree(value=rf.Number(), d_model=16, n_layers=1, n_heads=4, reduction=rf.Attention(n_heads=6))

    schema = rf.Schema.from_tree(
        value=rf.Number(), d_model=12, n_layers=1, n_heads=4, reduction=rf.Attention(n_heads=3)
    )
    assert schema.fields.reduction == rf.Attention(n_heads=3)


def test_branch_output_width_cache_refreshes_after_schema_update() -> None:
    schema = rf.Schema.from_tree(
        items=rf.Branch(length=3, reduction=None, value=rf.Number),
        target=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    cached = schema.branch_outputs

    schema.update(rf.where("name") == "items", reduction=rf.Attention(n_outputs=2))

    assert schema.branch_outputs is not cached
    assert schema.branch_outputs["record/items"] == 2
