"""Empty model paths retain backward participation whenever gradients are enabled."""

from contextlib import nullcontext

import pytest
import torch

import relflow as rf
from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.pool import LearnedQueryCrossAttention
from relflow.structs.enums import TensorKey, Tokens
from relflow.structs.packages import Parcel
from relflow.tensorfields.base import TensorFieldBase


class Field(TensorFieldBase):
    @classmethod
    def new(cls, **kwargs):
        raise NotImplementedError


def empty_path(kind, batch_size):
    model = rf.Model(
        value=rf.Number,
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    memory = torch.full((batch_size, 3, 8), torch.nan)
    present = torch.zeros(batch_size, 3, dtype=torch.bool)
    match kind:
        case "embedder":
            module = model.nodes["/value"].embedder
            field = Field()
            field.state = torch.full((batch_size, 3), Tokens.padded.value)
            field.present = present
            field.content = memory[..., 0]

            def invoke():
                return module.embed(field).payload

            expected_shape = (batch_size, 3, 8)
            parameters = tuple(module.parameters())
        case "decoder":
            module = model.nodes["/answer"].decoder

            def invoke():
                return module([], batch_size=batch_size, device=torch.device("cpu"), embed=True).payload[
                    TensorKey.embedding
                ]

            expected_shape = (batch_size, 1, 8)
            # The empty-heritage anchor covers the pool; unused sibling-context
            # parameters have never participated in this backward path.
            parameters = tuple(module.pool.parameters())
        case "branch":
            module = BranchEncoder(model.schema, "/")
            parcel = Parcel(
                payload=memory,
                present=present,
                origin="/value",
                destination="/",
                batch_size=batch_size,
            )

            def invoke():
                return module([parcel]).payload

            expected_shape = (batch_size, 8)
            parameters = tuple(module.parameters())
        case "pool":
            module = LearnedQueryCrossAttention(2, 8, 2, 0.2, mass_capacity=3)

            def invoke():
                return module(memory, present=present)

            expected_shape = (batch_size, 2, 8)
            parameters = tuple(module.parameters())
        case "attention":
            module = RotaryMultiheadAttention(8, 2, 0.2, n_kv_heads=1)
            query = torch.randn(batch_size, 2, 8)
            empty = memory[:, :0]
            padding = present[:, :0]

            def invoke():
                return module(query, empty, empty, key_padding_mask=padding)

            expected_shape = (batch_size, 2, 8)
            parameters = tuple(module.parameters())
        case _:
            raise ValueError(kind)
    return module, invoke, parameters, expected_shape


@pytest.mark.parametrize("kind", ["embedder", "decoder", "branch", "pool", "attention"])
@pytest.mark.parametrize("batch_size", [0, 2])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("mode", ["grad", "no_grad", "inference"])
def test_empty_paths_preserve_autograd_and_zero_outputs(kind, batch_size, training, mode):
    module, invoke, parameters, expected_shape = empty_path(kind, batch_size)
    module.train(training)
    state = {name: value.detach().clone() for name, value in module.named_buffers()}
    identities = tuple(id(parameter) for parameter in module.parameters())
    rng = torch.get_rng_state().clone()
    context = {"grad": nullcontext, "no_grad": torch.no_grad, "inference": torch.inference_mode}[mode]

    with context():
        output = invoke()

    assert output.shape == expected_shape
    assert output.dtype == next(module.parameters()).dtype
    assert output.device.type == "cpu"
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0
    assert output.requires_grad is (mode == "grad")
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(child.training is training for child in module.modules())
    assert identities == tuple(id(parameter) for parameter in module.parameters())
    for name, value in module.named_buffers():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    if mode == "grad":
        output.sum().backward()
        for parameter in parameters:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert torch.count_nonzero(parameter.grad) == 0
    else:
        assert all(parameter.grad is None for parameter in module.parameters())
