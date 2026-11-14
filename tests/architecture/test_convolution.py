import pytest
import torch

import relflow as rf
from relflow.architecture.convolution import Block, ConvMixer, padding
from relflow.structs.enums import Direction


def reach(mixer: ConvMixer, *, length: int, at: int) -> tuple[int, int]:
    """First and last row whose input changes the output at one row."""
    torch.manual_seed(0)
    tokens = torch.randn(1, length, 8, requires_grad=True)
    mixer(tokens, present=None, positions=None, steps=None, packing=None)[:, at].sum().backward()
    touched = (tokens.grad.abs().sum(dim=-1)[0] > 0).nonzero().flatten()
    return int(touched.min()), int(touched.max())


@pytest.mark.parametrize("n_blocks", [1, 2, 6])
def test_bidirectional_receptive_field_is_four_times_depth_dilation(n_blocks):
    side = 2 * (2**n_blocks - 1)
    centre = side + 3
    mixer = ConvMixer(width=8, n_blocks=n_blocks, direction=Direction.bidirectional)
    assert reach(mixer, length=2 * centre + 1, at=centre) == (centre - side, centre + side)


@pytest.mark.parametrize("n_blocks", [1, 3])
def test_causal_conv_sees_only_past_rows(n_blocks):
    back = 4 * (2**n_blocks - 1)
    at = back + 3
    mixer = ConvMixer(width=8, n_blocks=n_blocks, direction=Direction.causal)
    assert reach(mixer, length=at + 10, at=at) == (at - back, at)
    torch.manual_seed(1)
    tokens = torch.randn(2, 40, 8)
    changed = tokens.clone()
    changed[:, at + 1 :] = torch.randn(2, 40 - at - 1, 8)
    kwargs = {"present": None, "positions": None, "steps": None, "packing": None}
    assert torch.equal(mixer(tokens, **kwargs)[:, : at + 1], mixer(changed, **kwargs)[:, : at + 1])


def test_missing_rows_are_zero_and_do_not_leak():
    torch.manual_seed(2)
    mixer = ConvMixer(width=8, n_blocks=2, direction=Direction.bidirectional)
    present = torch.ones(1, 9, dtype=torch.bool)
    present[0, 3] = False
    tokens = torch.randn(1, 9, 8)
    changed = tokens.clone()
    changed[0, 3] = torch.randn(8)
    kwargs = {"present": present, "positions": None, "steps": None, "packing": None}
    output = mixer(tokens, **kwargs)
    assert not output[0, 3].any()
    assert torch.equal(output, mixer(changed, **kwargs))


def test_every_direction_has_a_padding():
    assert set(padding) == set(Direction)


class Reference(torch.nn.Module):
    """The block as the series embedder built it before the move."""

    def __init__(self, *, width: int, dilation: int) -> None:
        super().__init__()
        self.norm = torch.nn.RMSNorm(width)
        self.first = torch.nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.second = torch.nn.Conv1d(width, width, kernel_size=3, padding=dilation, dilation=dilation)
        self.activation = torch.nn.GELU(approximate="tanh")

    def forward(self, x, *, observed):
        gone = ~observed.unsqueeze(-1)
        hidden = self.activation(self.norm(x)).transpose(1, 2)
        hidden = self.activation(self.first(hidden).transpose(1, 2)).masked_fill(gone, 0.0)
        hidden = self.second(hidden.transpose(1, 2)).transpose(1, 2)
        return (x + hidden).masked_fill(gone, 0.0)


def test_bidirectional_block_keeps_state_dict_keys_and_outputs():
    torch.manual_seed(3)
    block = Block(width=8, dilation=2, direction=Direction.bidirectional)
    reference = Reference(width=8, dilation=2)
    assert block.state_dict().keys() == reference.state_dict().keys()
    reference.load_state_dict(block.state_dict())
    x = torch.randn(2, 11, 8)
    observed = torch.rand(2, 11) > 0.3
    assert torch.equal(block(x, observed=observed), reference(x, observed=observed))


def test_series_embedder_state_keys_are_unchanged():
    from relflow.tensorfields.extensions.series import Embedder

    schema = rf.Schema.from_tree(d_model=8, n_layers=1, n_heads=2, s=rf.Series(n_dim=1, n_blocks=2))
    keys = Embedder(schema=schema, address="/s").state_dict().keys()
    assert {"blocks.1.norm.weight", "blocks.1.first.weight", "blocks.1.second.bias"} <= keys
