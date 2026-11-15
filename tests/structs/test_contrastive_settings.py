import pydantic
import pytest
import yaml

import relflow as rf


def test_contrastive_defaults():
    settings = rf.Contrastive()
    assert settings.views == rf.Crops(width=128, overlap=0.5, mask=0.5)
    assert settings.loss == rf.Hierarchical()
    assert settings.weight == 1.0


@pytest.mark.parametrize("options", [{"width": 0}, {"overlap": 0.0}, {"overlap": 1.0}])
def test_crops_reject_invalid_settings(options):
    with pytest.raises(pydantic.ValidationError):
        rf.Crops(**options)


@pytest.mark.parametrize("mask", [-0.1, 1.0, 2.0])
def test_crops_reject_a_mask_outside_zero_to_one(mask):
    with pytest.raises(pydantic.ValidationError, match=r"Crops mask must be in \[0, 1\)"):
        rf.Crops(mask=mask)


def test_crops_accept_zero_mask():
    assert rf.Crops(mask=0.0).mask == 0.0


def test_contrastive_rejects_non_positive_weight():
    with pytest.raises(pydantic.ValidationError):
        rf.Contrastive(weight=0.0)


def test_plain_branch_dump_has_no_contrastive_key():
    branch = rf.Branch(length=8, x=rf.Number)
    assert branch.contrastive is None
    for mode in ("python", "json"):
        assert "contrastive" not in branch.model_dump(mode=mode)
        assert "contrastive" not in branch.model_dump(mode=mode, round_trip=True)


def test_branch_contrastive_round_trips_through_yaml():
    settings = rf.Contrastive(views=rf.Crops(width=4, overlap=0.25, mask=0.3), weight=2.0)
    branch = rf.Branch(length=8, x=rf.Number, contrastive=settings)
    text = yaml.safe_dump(branch.model_dump(mode="json", round_trip=True))
    restored = rf.Branch.model_validate(yaml.safe_load(text))
    assert restored.contrastive == settings
    assert yaml.safe_dump(restored.model_dump(mode="json", round_trip=True)) == text


def test_yaml_without_contrastive_loads_a_plain_branch_and_dumps_back_unchanged():
    text = yaml.safe_dump(rf.Branch(length=8, x=rf.Number).model_dump(mode="json", round_trip=True))
    restored = rf.Branch.model_validate(yaml.safe_load(text))
    assert restored.contrastive is None
    assert yaml.safe_dump(restored.model_dump(mode="json", round_trip=True)) == text


def test_branch_contrastive_round_trips_through_the_saved_schema():
    settings = rf.Contrastive(views=rf.Crops(width=4, overlap=0.25), weight=2.0)
    schema = rf.Schema.from_tree(
        d_model=8, n_layers=1, n_heads=2, ett=rf.Branch(length=8, x=rf.Number, contrastive=settings)
    )
    loaded = rf.Schema.model_validate_json(schema.model_dump_json())
    assert loaded.branches["/ett"].contrastive == schema.branches["/ett"].contrastive
    assert loaded.model_dump() == schema.model_dump()


def test_contrastive_rejects_a_child_without_a_fixed_width_per_row():
    with pytest.raises(ValueError, match=r"branch '/ett'.*'/ett/inner'.*reduction"):
        rf.Model.xs(
            d_model=8,
            ett=rf.Branch(
                length=8,
                contrastive=rf.Contrastive(),
                x=rf.Number,
                inner=rf.Branch(length=2, reduction=None, y=rf.Number),
            ),
        )
