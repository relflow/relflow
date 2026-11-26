"""Twigs bind like leaves, stay out of the token graph, and validate their branch."""

from textwrap import dedent

import pytest

import relflow as rf
from relflow.structs.experiment import Schema


def events(**options) -> rf.Branch:
    return rf.Branch(length=4, at=rf.Index, amount=rf.Number, **options)


def test_bare_and_configured_index_bind_like_leaves_without_runtime_modules():
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=2,
        events=events(),
        visits=rf.Branch(when=rf.Index(query="started", granularity="hours"), amount=rf.Number),
        label=rf.Category(mask=True),
    )

    twigs = model.schema.twigs
    assert set(twigs) == {"/events/at", "/visits/when"}
    assert isinstance(twigs["/events/at"], rf.Index) and isinstance(twigs["/events/at"], rf.Twig)
    assert twigs["/visits/when"].queries() == ("started",)
    assert twigs["/visits/when"].granularity == "hours"
    assert "/events/at" not in model.schema.requests
    assert "/events/at" not in model.nodes
    assert model.schema.branch_widths["/events"] == 4


def test_index_owns_only_its_source_and_granularity():
    for option in ({"n_heads": 2}, {"embed": True}, {"mask": True}, {"dropout": 0.1}, {"granularity": "months"}):
        with pytest.raises(ValueError):
            rf.Index(**option)
    with pytest.raises(TypeError, match="twig names come from the parent"):
        rf.Index(name="at")


def test_index_rejects_the_root_a_second_index_and_ancestor_sources():
    with pytest.raises(ValueError, match="model root holds one record per observation"):
        rf.Model(d_model=16, n_layers=1, n_heads=2, at=rf.Index, amount=rf.Number)
    with pytest.raises(ValueError, match=r"declares Index twigs \['at', 'also'\]; a branch takes at most one"):
        rf.Model(d_model=16, n_layers=1, n_heads=2, events=rf.Branch(at=rf.Index, also=rf.Index, amount=rf.Number))
    with pytest.raises(ValueError, match="reads an ancestor branch"):
        rf.Model(d_model=16, n_layers=1, n_heads=2, events=rf.Branch(at=rf.Index(query="../at"), amount=rf.Number))


def test_index_requires_rotary_attention_in_its_branch():
    with pytest.raises(ValueError, match="has no rotary attention to position"):
        rf.Model(d_model=16, n_layers=1, n_heads=2, events=events(attention=None, reduction=rf.Mean()))
    with pytest.raises(ValueError, match="has no rotary attention to position"):
        rf.Model(
            d_model=16,
            n_layers=1,
            n_heads=2,
            events=events(attention=None, reduction=rf.Attention(position=False)),
        )
    pooled = rf.Model(d_model=16, n_layers=1, n_heads=2, events=events(attention=None))
    assert pooled.schema.twigs["/events/at"].address == "/events/at"


def test_index_round_trips_through_json_yaml_presets_and_class_declarations(tmp_path):
    configured = Schema.from_tree(
        d_model=16, n_layers=1, n_heads=2, events=rf.Branch(at=rf.Index(granularity="days"), amount=rf.Number)
    )
    restored = Schema.model_validate_json(configured.model_dump_json())
    assert restored.model_dump() == configured.model_dump()
    assert isinstance(restored.twigs["/events/at"], rf.Index)

    path = tmp_path / "model.yaml"
    path.write_text(
        dedent(
            """
            d_model: 16
            n_layers: 1
            n_heads: 2
            fields:
              events:
                type: branch
                fields:
                  at:
                    type: index
                    granularity: hours
                  amount:
                    type: number
            """
        ),
        encoding="utf-8",
    )
    loaded = rf.Model.from_yaml(path)
    assert loaded.schema.twigs["/events/at"].granularity == "hours"

    preset = rf.Model(preset="small", d_model=16, events=rf.Branch(at=rf.Index, amount=rf.Number))
    assert preset.schema.twigs["/events/at"].model_dump() == rf.Index().model_dump() | {"name": "at"}

    class Events(rf.Branch):
        at = rf.Index
        amount = rf.Number

    declared = rf.Model(d_model=16, n_layers=1, n_heads=2, events=Events(length=4))
    assert (
        declared.schema.model_dump() == rf.Model(d_model=16, n_layers=1, n_heads=2, events=events()).schema.model_dump()
    )


def test_schema_mutations_add_update_and_delete_an_index():
    model = rf.Model(d_model=16, n_layers=1, n_heads=2, events=rf.Branch(amount=rf.Number))

    model.extend(rf.where("address") == "/events", at=rf.Index)
    assert set(model.schema.twigs) == {"/events/at"}
    with pytest.raises(ValueError, match="a branch takes at most one"):
        model.extend(rf.where("address") == "/events", also=rf.Index)

    model.update(rf.where("type") == "index", granularity="hours")
    assert model.schema.twigs["/events/at"].granularity == "hours"
    with pytest.raises(ValueError):
        model.update(rf.where("type") == "index", granularity="months")
    assert model.schema.twigs["/events/at"].granularity == "hours"

    model.delete(rf.where("type") == "index")
    assert not model.schema.twigs
