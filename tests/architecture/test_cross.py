import polars as pl
import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import Cross, RotaryTransformerEncoderLayer
from relflow.structs.packages import Parcel


@pytest.mark.parametrize("at_root", [False, True])
def test_cross_is_local_in_python_yaml_and_checkpoints(tmp_path, at_root):
    model = rf.Model(
        d_model=8,
        n_heads=2,
        n_layers=1,
        cross=at_root,
        fields={
            "cross": rf.Number,
            "items": rf.Branch(cross=not at_root, length=2, fields={"cross": rf.Number, "other": rf.Number}),
            "target": rf.Number(predict=True),
        },
    ).eval()
    path = tmp_path / "model.yaml"
    path.write_text(
        "d_model: 8\nn_heads: 2\nn_layers: 1\n"
        f"cross: {str(at_root).lower()}\n"
        "fields:\n"
        "  cross: {type: number}\n"
        "  items:\n"
        "    type: branch\n"
        "    length: 2\n"
        f"    cross: {str(not at_root).lower()}\n"
        "    fields: {cross: {type: number}, other: {type: number}}\n"
        "  target: {type: number, predict: true}\n"
    )
    assert rf.Model.from_yaml(path).schema.model_dump() == model.schema.model_dump()
    for address, enabled in (("/", at_root), ("/items", not at_root)):
        encoder = model.nodes[address].encoder
        assert isinstance(encoder.coordinate_encoder, Cross if enabled else RotaryTransformerEncoderLayer)
        assert isinstance(encoder.encoder[0], RotaryTransformerEncoderLayer)

    inputs = pl.DataFrame([{"cross": 1, "items": [{"cross": 2, "other": 3}, None]}, {"cross": None, "items": []}])
    before = model.predict(inputs)
    restored = rf.Model.load(model.save(tmp_path / "model.ckpt")).eval()
    assert restored.schema.model_dump() == model.schema.model_dump()
    assert before.equals(restored.predict(inputs))


def test_old_checkpoint_without_cross_retains_coordinate_attention(tmp_path):
    model = rf.Model(d_model=8, n_heads=2, n_layers=1, first=rf.Number, second=rf.Number)
    path = model.save(tmp_path / "legacy.ckpt")
    state = torch.load(path, weights_only=False)
    state["schema"]["fields"].pop("cross", None)
    torch.save(state, path)
    restored = rf.Model.load(path)
    assert restored.schema.fields.cross is False
    assert isinstance(restored.nodes["/"].encoder.coordinate_encoder, RotaryTransformerEncoderLayer)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_one_input_and_a_hidden_target_do_not_allocate_a_cross():
    model = rf.Model(d_model=8, n_heads=2, n_layers=1, cross=True, value=rf.Number, label=rf.Number(predict=True))
    assert model.nodes["/"].encoder.coordinate_encoder is None


def test_extending_and_deleting_fields_retains_each_surviving_cross_projection():
    model = rf.Model(d_model=8, n_heads=2, n_layers=1, cross=True, first=rf.Number, second=rf.Number)
    before = model.nodes["/"].encoder.coordinate_encoder
    model.extend(rf.where("address") == "/", third=rf.Number)
    extended = model.nodes["/"].encoder.coordinate_encoder
    torch.testing.assert_close(extended.down.weight[:, :16], before.down.weight)
    torch.testing.assert_close(extended.up.weight[:16], before.up.weight)
    torch.testing.assert_close(extended.up.bias[:16], before.up.bias)

    model.delete(rf.where("address") == "/first")
    remaining = model.nodes["/"].encoder.coordinate_encoder
    torch.testing.assert_close(remaining.down.weight, extended.down.weight[:, 8:])
    torch.testing.assert_close(remaining.up.weight, extended.up.weight[8:])
    torch.testing.assert_close(remaining.up.bias, extended.up.bias[8:])


def test_cross_mixes_fixed_field_and_child_slots_per_record_and_masks_gradients():
    model = rf.Model(
        d_model=4,
        n_heads=2,
        n_layers=1,
        items=rf.Branch(
            cross=True,
            attention=None,
            n_heads=2,
            length=3,
            value=rf.Number,
            detail=rf.Branch(length=2, n_heads=2, reduction=rf.Attention(n_outputs=2), value=rf.Number),
        ),
    )
    encoder = model.nodes["/items"].encoder
    cross = encoder.coordinate_encoder
    assert isinstance(cross, Cross)
    assert len(encoder.encoder) == 0
    assert cross.down.weight.shape == (8, 12)
    assert cross.down.bias is None
    assert cross.up.weight.shape == (12, 8)
    leaf = torch.randn(1, 3, 4, requires_grad=True)
    child = torch.randn(1, 6, 4, requires_grad=True)
    leaf_live = torch.tensor([[False, True, False]])
    child_live = torch.tensor([[True, True, False, True, False, False]])
    parcels = [
        Parcel(origin="/items/value", destination="/items", payload=leaf, present=leaf_live, batch_size=1),
        Parcel(origin="/items/detail", destination="/items", payload=child, present=child_live, batch_size=1),
    ]
    mixed_leaf, mixed_child = encoder.contextualize(parcels)

    visible = torch.cat([leaf_live.reshape(3, 1), child_live.reshape(3, 2)], dim=1)
    slots = torch.cat([leaf.reshape(3, 1, 4), child.reshape(3, 2, 4)], dim=1)
    x = slots.masked_fill(~visible.unsqueeze(-1), 0).reshape(3, 12)
    expected = (x + x * ((x @ cross.down.weight.T) @ cross.up.weight.T + cross.up.bias)).reshape(3, 3, 4)
    torch.testing.assert_close(mixed_leaf.payload, expected[:, 0].reshape_as(leaf))
    torch.testing.assert_close(mixed_child.payload, expected[:, 1:].reshape_as(child))
    assert torch.equal(mixed_leaf.present, leaf_live) and torch.equal(mixed_child.present, child_live)
    (mixed_leaf.payload.sum() + mixed_child.payload.sum()).backward()
    assert leaf.grad[~leaf_live].count_nonzero() == child.grad[~child_live].count_nonzero() == 0
    assert all(p.grad.isfinite().all() and p.grad.count_nonzero() for p in cross.parameters())
