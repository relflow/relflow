"""Node-local feed-forward widths survive construction, persistence and growth."""

from pathlib import Path

import pyarrow as pa
import pytest
import torch
from beartype.roar import BeartypeCallHintParamViolation

import relflow as rf
from relflow.architecture.encoder import RotaryTransformerEncoderLayer
from relflow.architecture.pool import CrossAttentionBlock, LearnedQueryCrossAttention


def configured(*, preset: str | None = None, ffnx: int = 8) -> rf.Model:
    constructor = rf.Model if preset is None else getattr(rf.Model, preset)
    return constructor(
        d_model=8,
        n_layers=1,
        n_heads=2,
        ffnx=ffnx,
        batch_size=2,
        dropout=0.0,
        embed=True,
        reduction=rf.Attention(n_outputs=1, n_layers=1, n_heads=2, dropout=0.0),
        amount=rf.Number,
        offset=rf.Number,
        events=rf.Branch(
            length=2,
            n_layers=1,
            n_heads=2,
            ffnx=3,
            dropout=0.0,
            reduction=rf.Attention(n_outputs=1, n_layers=1, n_heads=2, dropout=0.0),
            value=rf.Number,
            quantity=rf.Number,
            target=rf.Number(mask=True, ffnx=2, objective="mse", pooling="query", n_heads=2, n_linear=1, dropout=0.0),
        ),
        target=rf.Number(mask=True, ffnx=5, objective="mse", pooling="query", n_heads=2, n_linear=1, dropout=0.0),
    )


def observations() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"amount": 1.0, "offset": 0.5, "events": [{"value": 2.0, "quantity": 1.0, "target": 0.2}], "target": 0.3},
            {
                "amount": 3.0,
                "offset": 1.0,
                "events": [
                    {"value": 1.0, "quantity": 2.0, "target": 0.4},
                    {"value": 4.0, "quantity": 1.0, "target": 0.7},
                ],
                "target": 0.8,
            },
        ]
    )


def assert_blocks(model: rf.Model) -> None:
    """Check encoder, coordinate, branch-pooling and decoder-pooling blocks."""
    layers = [
        module for module in model.modules() if isinstance(module, (RotaryTransformerEncoderLayer, CrossAttentionBlock))
    ]
    pools = [module for module in model.modules() if isinstance(module, LearnedQueryCrossAttention)]
    assert len(layers) >= 8
    assert len(pools) >= 4
    scopes = [(model.nodes[address].encoder, branch.ffnx) for address, branch in model.schema.branches.items()]
    for address in model.schema.objectives:
        request = model.schema.requests[address]
        decoder = model.nodes[address].decoder
        scopes.append((decoder, request.ffnx))
        projection = decoder.context_projection
        assert projection is not None
        assert isinstance(projection[0], torch.nn.RMSNorm)
        assert projection[1].out_features == model.schema.d_model * request.ffnx
        assert projection[2].approximate == "tanh"
        assert projection[3].in_features == model.schema.d_model * request.ffnx
    for scope, ffnx in scopes:
        for layer in scope.modules():
            if not isinstance(layer, (RotaryTransformerEncoderLayer, CrossAttentionBlock)):
                continue
            assert isinstance(layer.attention_norm, torch.nn.RMSNorm)
            assert isinstance(layer.ffn_norm, torch.nn.RMSNorm)
            assert layer.attention_norm.eps == layer.ffn_norm.eps == 1e-5
            assert isinstance(layer.ffn[0], torch.nn.Linear)
            assert layer.ffn[0].out_features == ffnx * model.schema.d_model
            assert isinstance(layer.ffn[1], torch.nn.GELU)
            assert layer.ffn[1].approximate == "tanh"
            assert layer.ffn[3].in_features == ffnx * model.schema.d_model
            assert layer.ffn[3].out_features == model.schema.d_model
    assert all(isinstance(pool.norm, torch.nn.RMSNorm) and pool.norm.eps == 1e-5 for pool in pools)


def template(path: Path, *, preset: str | None = None, ffnx: int = 8) -> Path:
    prefix = "" if preset is None else f"preset: {preset}\n"
    path.write_text(
        prefix
        + f"ffnx: {ffnx!r}\n"
        + """d_model: 8
n_layers: 1
n_heads: 2
batch_size: 2
dropout: 0.0
embed: true
reduction: {type: attention, n_outputs: 1, n_layers: 1, n_heads: 2, dropout: 0.0}
fields:
  amount: {type: number}
  offset: {type: number}
  events:
    type: branch
    length: 2
    n_layers: 1
    n_heads: 2
    ffnx: 3
    dropout: 0.0
    reduction: {type: attention, n_outputs: 1, n_layers: 1, n_heads: 2, dropout: 0.0}
    fields:
      value: {type: number}
      quantity: {type: number}
      target: {type: number, mask: true, ffnx: 2, objective: mse, pooling: query, n_heads: 2, n_linear: 1, dropout: 0.0}
  target: {type: number, mask: true, ffnx: 5, objective: mse, pooling: query, n_heads: 2, n_linear: 1, dropout: 0.0}
""",
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("preset", [None, "xs", "md"])
def test_python_yaml_and_presets_build_the_same_node_local_architecture(tmp_path, preset):
    python = configured(preset=preset, ffnx=6)
    yaml = rf.Model.from_yaml(template(tmp_path / "model.yaml", preset=preset, ffnx=6))
    assert python.schema.model_dump() == yaml.schema.model_dump()
    assert python.preset == yaml.preset
    for model in (python, yaml):
        assert "ffnx" not in model.schema.model_dump()
        assert model.schema.fields.ffnx == 6
        assert model.schema.branches[rf.Address("events")].ffnx == 3
        assert model.schema.requests[rf.Address("target")].ffnx == 5
        assert model.schema.requests[rf.Address("events", "target")].ffnx == 2
        assert model.schema.requests[rf.Address("amount")].ffnx == 8
        assert_blocks(model)
        result = model.predict(observations())
        predictions = result["predictions"].combine_chunks()
        assert predictions.field("events").values.type.get_field_index("target") >= 0
        assert predictions.type.get_field_index("target") >= 0
        embedding = result["representations"].combine_chunks().values.field("embedding")
        assert torch.isfinite(torch.as_tensor(embedding.values.to_numpy().copy())).all()


def test_schema_json_roundtrip_rebuilds_fixed_modules_without_configuration_metadata():
    original = configured(ffnx=6)
    restored = rf.Model(schema=rf.Schema.model_validate_json(original.schema.model_dump_json()), batch_size=2)
    assert restored.schema.model_dump() == original.schema.model_dump()
    assert "ffnx" not in rf.Schema.model_fields
    assert_blocks(restored)


def test_schema_constructor_rejects_explicit_root_width_overrides():
    schema = configured(ffnx=6).schema
    assert rf.Model(schema=schema).schema.fields.ffnx == 6
    for value in (8, 6, 4):
        with pytest.raises(TypeError, match="schema.*ffnx"):
            rf.Model(schema=schema, ffnx=value)


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "4", None])
def test_ffnx_requires_a_positive_integer_at_every_constructor_boundary(tmp_path, value):
    with pytest.raises((ValueError, TypeError, BeartypeCallHintParamViolation), match="ffnx"):
        configured(ffnx=value)
    with pytest.raises(ValueError, match="ffnx"):
        rf.Branch(ffnx=value, value=rf.Number)
    with pytest.raises(ValueError, match="ffnx"):
        rf.Number(ffnx=value)
    path = template(tmp_path / "invalid.yaml", ffnx=value)
    with pytest.raises(ValueError, match="ffnx"):
        rf.Model.from_yaml(path)


def test_node_defaults_are_independent_and_option_collisions_use_fields():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        ffnx=6,
        fields={"ffnx": rf.Number},
        events=rf.Branch(length=2, fields={"ffnx": rf.Number}),
        target=rf.Number(mask=True),
    )
    assert model.schema.fields.ffnx == 6
    assert model.schema.branches[rf.Address("events")].ffnx == 8
    assert model.schema.requests[rf.Address("target")].ffnx == 8
    assert model.schema.requests[rf.Address("ffnx")].ffnx == 8
    assert model.schema.requests[rf.Address("events", "ffnx")].ffnx == 8


def test_custom_preset_defaults_and_explicit_node_widths_remain_local():
    values = rf.presets.XS.model_dump()
    values["root"]["ffnx"] = 4
    values["branch"]["ffnx"] = 3
    values["leaf"]["ffnx"] = 2
    profile = rf.presets.Preset.model_validate(values)
    constructor = rf.presets.factory(profile).__get__(None, rf.Model)
    definition = rf.Branch(
        value=rf.Number,
        target=rf.Number(mask=True, ffnx=7),
        fixed=rf.Branch(ffnx=8, value=rf.Number(ffnx=8)),
    )
    model = constructor(d_model=8, events=definition, target=rf.Number(mask=True))
    assert definition.ffnx == 8 and "ffnx" not in definition.model_fields_set
    assert model.schema.fields.ffnx == 4
    assert model.schema.branches[rf.Address("events")].ffnx == 3
    assert model.schema.requests[rf.Address("events", "value")].ffnx == 2
    assert model.schema.requests[rf.Address("events", "target")].ffnx == 7
    assert model.schema.branches[rf.Address("events", "fixed")].ffnx == 8
    assert model.schema.requests[rf.Address("events", "fixed", "value")].ffnx == 8
    assert model.schema.requests[rf.Address("target")].ffnx == 2


@pytest.mark.parametrize("address", [rf.Address(), rf.Address("events"), rf.Address("events", "target")])
def test_invalid_width_mutation_preserves_schema_graph_and_weights(address):
    model = configured()
    schema = model.schema.model_dump()
    nodes = model.nodes
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    with pytest.raises(ValueError, match="ffnx"):
        model.update(rf.where("address") == address, ffnx=0)
    assert model.nodes is nodes
    assert model.schema.model_dump() == schema
    for name, value in weights.items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


def test_checkpoint_restores_exact_weights_predictions_and_preset(tmp_path):
    original = configured(preset="xs").eval()
    before = original.predict(observations())
    path = original.save(tmp_path / "model.ckpt")
    restored = rf.Model.load(path).eval()
    assert restored.schema.model_dump() == original.schema.model_dump()
    assert restored.preset == original.preset
    assert_blocks(restored)
    assert restored.state_dict().keys() == original.state_dict().keys()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    assert restored.predict(observations()).equals(before)


def test_extension_after_restore_uses_fixed_blocks_and_preserves_retained_weights(tmp_path):
    original = configured(preset="xs")
    restored = rf.Model.load(original.save(tmp_path / "before-growth.ckpt"))
    retained = {name: value.clone() for name, value in restored.state_dict().items()}
    restored.extend(
        rf.where("address") == rf.Address(),
        purchases=rf.Branch(
            length=2,
            ffnx=4,
            reduction=rf.Attention(n_outputs=1),
            value=rf.Number,
            quantity=rf.Number,
            target=rf.Number(mask=True, ffnx=6, objective="mse"),
        ),
    )
    restored.update(rf.where("address") == rf.Address("purchases"), n_layers=2, ffnx=7)
    restored.update(rf.where("address") == rf.Address("purchases", "target"), ffnx=3)
    assert_blocks(restored)
    branch = restored.nodes[rf.Address("purchases")].encoder
    assert len(branch.encoder) == 2
    assert branch.encoder[0].ffn[0].out_features == 56
    assert isinstance(restored.nodes[rf.Address("purchases", "target")].decoder.pool.norm, torch.nn.RMSNorm)
    for name, value in retained.items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    loaded = rf.Model.load(restored.save(tmp_path / "after-growth.ckpt"))
    assert loaded.schema.model_dump() == restored.schema.model_dump()
    assert_blocks(loaded)


@pytest.mark.integration
def test_fixed_nested_architecture_completes_a_cpu_optimizer_step(tmp_path):
    model = configured()
    data = observations()
    datamodule = rf.ArrowDataModule(
        train=data,
        validate=data,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
        shuffle=False,
    )
    parameter = model.nodes[rf.Address()].encoder.encoder[0].ffn[3].weight
    before = parameter.detach().clone()
    model.fit(
        datamodule,
        learning_rate=1e-3,
        fused=False,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        early_stopping=False,
        rollback=False,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )
    assert model.trainer.global_step == 1
    assert torch.isfinite(parameter).all()
    assert not torch.equal(parameter.detach(), before)
    assert torch.isfinite(model.trainer.callback_metrics["loss/train_epoch"])
    assert torch.isfinite(model.trainer.callback_metrics["loss/validate"])
