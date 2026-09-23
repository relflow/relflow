"""Authored YAML follows the same schema, extension, and preset contracts as Python."""

from pathlib import Path
from textwrap import dedent

import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.tensorfields.base import TENSORFIELDS
from tests.architecture.test_extension_contract import build_extension


def template(directory: Path, content: str) -> Path:
    path = directory / "model.yaml"
    path.write_text(dedent(content), encoding="utf-8")
    return path


@pytest.mark.parametrize("preset", ["xs", "sm", "md", "lg", "xl"])
def test_yaml_and_python_resolve_the_same_preset_schema(tmp_path, preset):
    path = template(
        tmp_path,
        f"""
        preset: {preset}
        d_model: 16
        embed: true
        reduction:
          type: attention
          n_outputs: 1
        fields:
          origin:
            type: category
          booking:
            type: branch
            length: 2
            overflow: tail
            reduction:
              type: mean
            fields:
              length:
                type: number
                objective: mse
                dropout: null
              cabin:
                type: enum
                values: [economy, business]
                mask: true
                pooling: mean
        """,
    )

    loaded = rf.Model.from_yaml(path)
    expected = getattr(rf.Model, preset)(
        d_model=16,
        embed=True,
        reduction=rf.Attention(n_outputs=1),
        origin=rf.Category,
        booking=rf.Branch(
            length=2,
            overflow="tail",
            reduction=rf.Mean(),
            fields={
                "length": rf.Number(objective="mse", dropout=None),
                "cabin": rf.Enum(values=("economy", "business"), mask=True, pooling="mean"),
            },
        ),
    )

    assert loaded.schema.model_dump() == expected.schema.model_dump()
    assert loaded.preset == expected.preset
    assert rf.Enum.vocabulary(loaded, rf.Address("booking", "cabin")) == ("economy", "business")


def test_yaml_without_preset_keeps_ordinary_constructor_defaults(tmp_path):
    path = template(
        tmp_path,
        """
        d_model: 16
        n_layers: 1
        n_heads: 2
        fields:
          amount:
            type: number
          events:
            type: branch
            fields:
              kind:
                type: category
        """,
    )

    loaded = rf.Model.from_yaml(str(path))
    expected = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=2,
        amount=rf.Number,
        events=rf.Branch(kind=rf.Category),
    )

    assert loaded.preset is None
    assert loaded.schema.model_dump() == expected.schema.model_dump()


def test_python_overrides_replace_root_options_and_forward_runtime_configuration(tmp_path):
    path = template(
        tmp_path,
        """
        preset: md
        d_model: 32
        n_layers: 2
        batch_size: 3
        reduction:
          type: attention
          n_outputs: 3
          n_heads: 4
        fields:
          events:
            type: branch
            fields:
              amount:
                type: number
        """,
    )
    optimizer = rf.adamw(learning_rate=1e-3)
    scheduler = {"interval": "epoch"}

    loaded = rf.Model.from_yaml(
        path,
        d_model=16,
        n_layers=1,
        batch_size=7,
        attention=None,
        reduction=rf.Attention(n_outputs=1),
        optimizer=optimizer,
        scheduler=scheduler,
    )

    assert loaded.schema.d_model == 16
    assert loaded.schema.fields.n_layers == 1
    assert loaded.schema.fields.attention is None
    assert loaded.schema.fields.reduction == rf.Attention(n_outputs=1)
    assert loaded.schema.branches["/events"].n_layers == rf.presets.MD.branch.n_layers
    assert loaded.schema.branches["/events"].reduction == rf.presets.MD.branch.reduction
    assert loaded.batch_size == 7
    assert loaded.optimizer is optimizer
    assert loaded.scheduler is scheduler


def test_yaml_preserves_explicit_nulls_and_normalizes_mask_mappings(tmp_path):
    path = template(
        tmp_path,
        """
        preset: sm
        d_model: 16
        batch_size: 3
        attention: null
        reduction: null
        dropout: null
        mask:
          rate: 0.2
        fields:
          amount:
            type: number
            dropout: null
            decoder_position: null
            mask:
              - rate: 0.1
              - skip: true
                reconstruct: true
          events:
            type: branch
            attention: null
            reduction: null
            dropout: null
            mask:
              reconstruct: true
              rate: 0.3
            fields:
              kind:
                type: category
        """,
    )

    loaded = rf.Model.from_yaml(path)
    expected = rf.Model.sm(
        d_model=16,
        batch_size=3,
        attention=None,
        reduction=None,
        dropout=None,
        mask=rf.Mask(rate=0.2),
        amount=rf.Number(
            dropout=None,
            decoder_position=None,
            mask=[rf.Mask(rate=0.1), rf.Mask(skip=True, reconstruct=True)],
        ),
        events=rf.Branch(
            attention=None,
            reduction=None,
            dropout=None,
            mask=rf.Mask(reconstruct=True, rate=0.3),
            kind=rf.Category,
        ),
    )

    assert loaded.batch_size == 3
    assert loaded.schema.model_dump() == expected.schema.model_dump()


def test_yaml_subclass_preserves_preset_through_checkpoint_and_mutation(tmp_path):
    class Specialized(rf.Model):
        pass

    path = template(
        tmp_path,
        """
        preset: xs
        d_model: 16
        fields:
          amount:
            type: number
        """,
    )
    original = Specialized.from_yaml(path)
    loaded = Specialized.load(original.save(tmp_path / "model.ckpt"))

    assert type(original) is Specialized
    assert type(loaded) is Specialized
    assert loaded.preset == rf.presets.XS
    assert loaded.schema.model_dump() == original.schema.model_dump()
    for key, weight in original.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], weight, rtol=0, atol=0)

    loaded.extend(rf.where("address") == rf.Address(), events=rf.Branch(value=rf.Number))
    assert loaded.schema.branches["/events"].n_heads == rf.presets.XS.branch.n_heads
    assert loaded.schema.requests["/events/value"].n_heads == rf.presets.XS.leaf.n_heads


def test_yaml_resolves_late_registered_extensions_and_their_validators(tmp_path):
    rf.Model.from_yaml(template(tmp_path, "preset: xs\nd_model: 16\nfields: {amount: {type: number}}"))
    extension, Request = build_extension()
    try:
        path = template(
            tmp_path,
            f"""
            preset: xs
            d_model: 16
            fields:
              events:
                type: branch
                fields:
                  payload:
                    type: {extension.name}
                    family: bytes
                    mask: true
            """,
        )
        model = rf.Model.from_yaml(path)
        request = model.schema.requests["/events/payload"]

        assert isinstance(request, Request)
        assert request.family == "bytes"
        assert request.n_heads == rf.presets.XS.leaf.n_heads
        assert request.mask == rf.Number(mask=True).mask

        path.write_text(path.read_text().replace("family: bytes", "family: unknown"))
        with pytest.raises(ValueError) as caught:
            rf.Model.from_yaml(path)
        assert str(path) in str(caught.value)
        assert "fields.events.fields.payload" in str(caught.value)
        assert "family" in str(caught.value)
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_shared_yaml_anchors_bind_independent_names_and_mutable_nodes(tmp_path):
    path = template(
        tmp_path,
        """
        preset: xs
        d_model: 16
        fields:
          outbound: &leg
            type: branch
            length: 2
            fields:
              fare:
                type: number
                dropout: null
          inbound: *leg
        """,
    )
    model = rf.Model.from_yaml(path)

    assert set(model.schema.requests) == {rf.Address("outbound", "fare"), rf.Address("inbound", "fare")}
    for leg in ("outbound", "inbound"):
        branch = model.schema.branches[rf.Address(leg)]
        request = model.schema.requests[rf.Address(leg, "fare")]
        assert branch.name == leg
        assert branch.length == 2
        assert branch.n_heads == rf.presets.XS.branch.n_heads
        assert request.n_heads == rf.presets.XS.leaf.n_heads
        assert request.dropout is None

    model.update(rf.where("address") == rf.Address("outbound", "fare"), weight=0.25)

    assert model.schema.requests["/outbound/fare"].weight == 0.25
    assert model.schema.requests["/inbound/fare"].weight == 1.0


def test_recursive_yaml_branch_alias_reports_the_source(tmp_path):
    path = template(
        tmp_path,
        """
        preset: xs
        fields:
          events: &events
            type: branch
            fields:
              child: *events
        """,
    )

    with pytest.raises(ValueError) as caught:
        rf.Model.from_yaml(path)

    assert str(path) in str(caught.value)
    assert "recursive" in str(caught.value)
    assert "events" in str(caught.value)


def test_yaml_model_predicts_nested_targets_and_exports_root_embedding(tmp_path):
    path = template(
        tmp_path,
        """
        preset: xs
        d_model: 16
        embed: true
        fields:
          events:
            type: branch
            length: 2
            fields:
              amount:
                type: number
              target:
                type: number
                mask: true
                pooling: mean
        """,
    )
    model = rf.Model.from_yaml(path)
    data = pa.Table.from_pylist([{"events": [{"amount": 1.0}]}, {"events": [{"amount": 2.0}, {"amount": 3.0}]}])

    predictions = model.predict(data)["predictions"].combine_chunks()

    assert predictions.field("/").type.field("embedding").type == pa.list_(pa.float32(), 16)
    assert predictions.field("/events/target").values.field("inferred").to_pylist() == [True, False, True, True]


@pytest.mark.parametrize(
    "content, fragment",
    [
        ("", "mapping"),
        ("[]", "mapping"),
        ("preset: enormous\nfields: {}", "preset"),
        ("preset: 3\nfields: {}", "preset"),
        ("preset: xs\nn_head: 2\nfields: {}", "n_head"),
        ("preset: xs\nfields: []", "fields"),
        ("preset: xs\nfields: {amount: {}}", "type"),
        ("preset: xs\nfields: {amount: {type: missing}}", "missing"),
        ("preset: xs\nfields: {amount: {type: number, name: other}}", "name"),
        ("preset: xs\nfields: {amount: {type: number, surprise: 1}}", "surprise"),
        ("preset: xs\nfields: {kind: {type: enum, values: [a, a]}}", "unique"),
        ("preset: xs\noptimizer: adamw\nfields: {}", "optimizer"),
        ("preset: xs\nscheduler: {}\nfields: {}", "scheduler"),
        ("fields: {amount: {type: number}}", "d_model"),
        ("preset: [xs", "line 1"),
    ],
)
def test_invalid_templates_report_the_source_and_problem(tmp_path, content, fragment):
    path = template(tmp_path, content)

    with pytest.raises(ValueError) as caught:
        rf.Model.from_yaml(path)

    assert str(path) in str(caught.value)
    assert fragment in str(caught.value)


@pytest.mark.parametrize(
    "content, key",
    [
        ("preset: xs\npreset: sm\nfields: {}", "preset"),
        ("preset: xs\nfields:\n  amount: {type: number}\n  amount: {type: category}", "amount"),
        ("preset: xs\nfields: {amount: {type: number, mask: true, mask: false}}", "mask"),
        (
            """
            preset: xs
            fields:
              amount: &number
                type: number
                objective: mse
              price:
                <<: *number
                objective: huber
            """,
            "objective",
        ),
    ],
)
def test_duplicate_yaml_keys_are_rejected_before_overwriting(tmp_path, content, key):
    path = template(tmp_path, content)

    with pytest.raises(ValueError) as caught:
        rf.Model.from_yaml(path)

    assert str(path) in str(caught.value)
    assert "duplicate" in str(caught.value).lower()
    assert key in str(caught.value)


def test_yaml_python_tags_cannot_construct_objects(tmp_path):
    path = template(tmp_path, "!!python/object/apply:builtins.set [[a, b]]")

    with pytest.raises(ValueError) as caught:
        rf.Model.from_yaml(path)

    assert str(path) in str(caught.value)
    assert "python/object/apply" in str(caught.value)


@pytest.mark.parametrize("option", ["d_model", "n_layers", "batch_size", "attention"])
def test_invalid_root_option_values_without_preset_report_the_source(tmp_path, option):
    options = {"d_model": 16, "n_layers": 1, "n_heads": 2, option: "wrong"}
    path = template(
        tmp_path,
        "\n".join(f"{name}: {value}" for name, value in options.items()) + "\nfields: {amount: {type: number}}",
    )

    with pytest.raises(ValueError) as caught:
        rf.Model.from_yaml(path)

    assert str(path) in str(caught.value)
    assert option in str(caught.value)


@pytest.mark.parametrize("overrides", [{"preset": "sm"}, {"fields": {}}, {"n_head": 2}])
def test_python_overrides_cannot_replace_template_fields_or_preset(tmp_path, overrides):
    path = template(tmp_path, "preset: xs\nfields: {amount: {type: number}}")

    with pytest.raises(TypeError, match=next(iter(overrides))):
        rf.Model.from_yaml(path, **overrides)
