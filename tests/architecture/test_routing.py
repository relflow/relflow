from copy import deepcopy

import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel
from relflow.tensorfields.base import EmbedderBase, TensorFieldBase, TensorInput


class Field(TensorFieldBase):
    @classmethod
    def new(cls, input, target, present, trainable, inferred, address, schema, strata, context):
        raise NotImplementedError


class Probe(EmbedderBase):
    def __init__(self, schema):
        super().__init__(schema, "/value")
        self.linear = torch.nn.Linear(7, schema.d_model)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, inputs):
        self.inputs = inputs
        assert inputs.content["nested", "matrix"].is_contiguous()
        values = torch.cat(
            (inputs.content["scalar"].unsqueeze(-1), inputs.content["nested", "matrix"].view(inputs.state.numel(), -1)),
            dim=1,
        )
        present = inputs.state.ne(Tokens.other.value)
        self.parcel = Parcel(
            payload=self.dropout(self.linear(values)).masked_fill(~present.unsqueeze(-1), torch.nan),
            present=present,
            origin=self.address,
            destination=self.destination,
            batch_size=inputs.state.numel(),
        )
        return self.parcel


def test_plain_input_node_allocates_only_an_embedder():
    model = rf.Model(value=rf.Number, d_model=8, n_layers=1, n_heads=2)

    node = model.nodes["/value"]

    assert hasattr(node, "embedder")
    assert not hasattr(node, "decoder")


@pytest.mark.parametrize("indices", [None, [1, 4], [5, 1, 1], []])
def test_tensorfield_take_gathers_every_content_prefix(indices):
    field = Field()
    field.state = torch.arange(6).reshape(2, 3)
    field.content = TensorDict(
        {
            "scalar": torch.arange(6).reshape(2, 3),
            "vector": torch.arange(24).reshape(2, 3, 4),
        },
        batch_size=[2, 3],
    )

    selected = torch.arange(6) if indices is None else torch.tensor(indices, dtype=torch.int64)
    compact = field.take(None if indices is None else selected)

    assert isinstance(compact, TensorInput)
    assert compact.batch_size == torch.Size([len(selected)])
    assert not hasattr(compact, "targets")
    assert not hasattr(compact, "present")
    assert not hasattr(compact, "trainable")
    assert not hasattr(compact, "inferred")
    assert torch.equal(compact.state, selected)
    assert torch.equal(compact.content["scalar"], selected)
    assert torch.equal(
        compact.content["vector"],
        field.content["vector"].reshape(6, 4).index_select(0, selected),
    )


@pytest.mark.parametrize("indices", [None, [1, 4]])
def test_tensorfield_take_rejects_an_invalid_content_prefix(indices):
    field = Field()
    field.state = torch.arange(6).reshape(2, 3)
    field.content = torch.arange(6).reshape(3, 2)

    with pytest.raises(ValueError, match="content must start with state shape"):
        field.take(None if indices is None else torch.tensor(indices))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_tensorfield_take_preserves_empty_geometry_and_index_validation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    field = Field()
    field.state = torch.arange(6, device=device).reshape(2, 3)
    field.content = torch.arange(12.0, device=device).reshape(2, 3, 2)
    for indices in ([-1], [6], [10]):
        with pytest.raises(IndexError, match="indices must be between 0 and 5"):
            field.take(torch.tensor(indices, dtype=torch.int64, device=device))
    for indices in (torch.tensor([1.0], device=device), torch.tensor([[1]], device=device)):
        with pytest.raises(TypeError, match="one-dimensional int64"):
            field.take(indices)
    if device == "cuda":
        with pytest.raises(ValueError, match="indices must use state device"):
            field.take(torch.tensor([1]))

    field.state = field.state[:0]
    field.content = TensorDict(
        {"nested": TensorDict({"vector": field.content[:0]}, batch_size=[0, 3], device=device)},
        batch_size=[0, 3],
        device=device,
    )
    compact = field.take(None)
    assert compact.state.shape == (0,)
    assert compact.content["nested", "vector"].shape == (0, 2)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_all_present_extension_matches_padded_geometry_and_preserves_copy_isolation(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    torch.manual_seed(13)
    schema = rf.Model(value=rf.Number, d_model=8, n_heads=2, n_layers=1).schema
    full = Probe(schema).to(device).train()
    padded = deepcopy(full)
    fields = []
    for length in (3, 4):
        field = Field()
        field.state = torch.full((2, length), Tokens.padded.value, device=device)
        field.state[:, :3] = torch.tensor([[0, 4, 1], [0, 0, 0]], device=device)
        field.present = field.state.ne(Tokens.padded.value)
        scalar = torch.full((2, length), torch.nan, device=device)
        scalar[:, :3] = torch.arange(6.0, device=device).reshape(2, 3)
        matrix = torch.full((2, length, 2, 3), torch.nan, device=device).transpose(-1, -2)
        matrix[:, :3] = torch.arange(36.0, device=device).reshape(2, 3, 3, 2)
        assert not matrix.is_contiguous()
        field.content = TensorDict(
            {
                "scalar": scalar.requires_grad_(),
                "nested": TensorDict({"matrix": matrix.requires_grad_()}, batch_size=[2, length], device=device),
            },
            batch_size=[2, length],
            device=device,
        )
        fields.append(field)

    torch.manual_seed(19)
    actual = full.embed(fields[0])
    torch.manual_seed(19)
    expected = padded.embed(fields[1])
    torch.testing.assert_close(actual.payload, expected.payload[:, :3])
    torch.testing.assert_close(actual.present, expected.present[:, :3])
    assert torch.isfinite(actual.payload).all()
    assert torch.count_nonzero(expected.payload[:, 3]) == 0
    weights = torch.linspace(0.1, 1, actual.payload.numel(), device=device).reshape_as(actual.payload)
    (actual.payload * weights).sum().backward()
    (expected.payload[:, :3] * weights).sum().backward()
    for left, right in zip(full.parameters(), padded.parameters(), strict=True):
        assert left.grad is not None and right.grad is not None
        torch.testing.assert_close(left.grad, right.grad)
        assert torch.isfinite(left.grad).all()
    for key in ("scalar", ("nested", "matrix")):
        torch.testing.assert_close(fields[0].content[key].grad, fields[1].content[key].grad[:, :3])

    state = fields[0].state.clone()
    content = fields[0].content.clone()
    presence = actual.present.clone()
    with torch.no_grad():
        full.inputs.state.fill_(9)
        full.inputs.content["scalar"].fill_(999)
        full.inputs.content["nested", "matrix"].fill_(999)
        full.parcel.present.logical_not_()
    torch.testing.assert_close(fields[0].state, state)
    torch.testing.assert_close(fields[0].content["scalar"], content["scalar"])
    torch.testing.assert_close(fields[0].content["nested", "matrix"], content["nested", "matrix"])
    torch.testing.assert_close(actual.present, presence)


def test_embedder_compacts_present_coordinates_and_restores_fixed_geometry():
    model = rf.Model(
        value=rf.Number(mask=rf.Mask(query="skip", skip=True, dropout=False)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table({"value": [1.0, 2.0, 3.0], "skip": [False, True, False]}),
        strata=Strata.train,
    )
    field = inputs["/value"]
    embedder = model.nodes["/value"].embedder
    seen = []
    handle = embedder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))

    try:
        parcel = embedder.embed(field)
    finally:
        handle.remove()

    assert len(seen) == 1
    assert isinstance(seen[0], TensorInput)
    assert seen[0].state.shape == (2,)
    assert parcel.payload.shape == (3, 1, 8)
    assert torch.equal(parcel.present, field.present)
    assert torch.equal(parcel.payload[1], torch.zeros_like(parcel.payload[1]))


def test_embedder_is_not_called_for_an_all_skipped_field():
    model = rf.Model(
        value=rf.Number(mask=rf.Mask(skip=True, dropout=False)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    field = model.encode(pa.table({"value": [1.0, 2.0]}), strata=Strata.train)["/value"]
    embedder = model.nodes["/value"].embedder
    calls = 0

    def count(module, args):
        nonlocal calls
        calls += 1

    handle = embedder.register_forward_pre_hook(count)
    try:
        parcel = embedder.embed(field)
    finally:
        handle.remove()

    assert calls == 0
    assert parcel.payload.shape == (2, 1, 8)
    assert not parcel.present.any()
    assert torch.equal(parcel.payload, torch.zeros_like(parcel.payload))


def test_runtime_keeps_mixed_skip_rows_out_of_branch_context():
    model = rf.Model(
        value=rf.Number(mask=rf.Mask(query="skip", skip=True, dropout=False)),
        d_model=8,
        n_layers=1,
        n_heads=2,
        embed=True,
    )
    inputs = model.encode(
        pa.table({"value": [1.0, 2.0], "skip": [False, True]}),
        strata=Strata.train,
    )

    predictions = model(inputs, strata=Strata.train)
    root = next(prediction for prediction in predictions if prediction.address == "/")
    embedding = root.payload[TensorKey.embedding]

    assert torch.isfinite(embedding).all()
    assert not torch.equal(embedding[0], torch.zeros_like(embedding[0]))
    assert torch.equal(embedding[1], torch.zeros_like(embedding[1]))


def test_peer_objective_selection_runs_the_same_decoder_and_anchors_local_backward(monkeypatch):
    model = rf.Model(
        context=rf.Number,
        first=rf.Boolean(mask=rf.Mask(query="first_selected", dropout=False, reconstruct=True)),
        second=rf.Boolean(mask=rf.Mask(query="second_selected", dropout=False, reconstruct=True)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table(
            {
                "context": [1.0, 2.0],
                "first": [True, False],
                "second": [False, True],
                "first_selected": [False, False],
                "second_selected": [False, False],
            }
        ),
        strata=Strata.train,
    )
    calls = []

    def select_peer(local):
        calls.append(local.clone())
        return torch.tensor([0, 1], dtype=local.dtype, device=local.device)

    monkeypatch.setattr("relflow.architecture.runtime.all_reduce_max", select_peer)

    predictions = model(inputs, strata=Strata.train)
    output = model.training_step(inputs, batch_idx=0)

    assert len(calls) == 2
    assert all(torch.equal(call, torch.zeros(2, dtype=torch.uint8)) for call in calls)
    assert [prediction.address for prediction in predictions] == ["/second"]
    assert output["loss"].requires_grad
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.nodes["/second"].decoder.parameters())
    assert all(parameter.grad is None for parameter in model.nodes["/first"].decoder.parameters())


@pytest.mark.parametrize(
    ("strata", "step"),
    [
        (Strata.validate, "validation_step"),
        (Strata.test, "test_step"),
    ],
)
def test_evaluation_loss_does_not_require_a_gradient_anchor(strata, step):
    model = rf.Model(
        context=rf.Number,
        label=rf.Boolean(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table(
            {
                "context": [1.0, 2.0],
                "label": [True, False],
            }
        ),
        strata=strata,
    )

    with torch.inference_mode():
        output = getattr(model, step)(inputs, batch_idx=0)

    assert torch.isfinite(output["loss"])
    assert output["loss"].item() > 0.0
    assert not output["loss"].requires_grad


def test_evaluation_returns_zero_when_only_a_peer_rank_has_targets(monkeypatch):
    model = rf.Model(
        context=rf.Number,
        label=rf.Boolean(mask=rf.Mask(query="selected", dropout=False, reconstruct=True)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table(
            {
                "context": [1.0, 2.0],
                "label": [True, False],
                "selected": [False, False],
            }
        ),
        strata=Strata.validate,
    )
    monkeypatch.setattr(
        "relflow.architecture.runtime.all_reduce_max",
        lambda local: torch.ones_like(local),
    )

    with torch.inference_mode():
        output = model.validation_step(inputs, batch_idx=0)

    assert output["loss"].item() == 0.0
    assert not output["loss"].requires_grad


def test_globally_empty_objective_skips_the_training_update(monkeypatch):
    model = rf.Model(
        context=rf.Number,
        label=rf.Boolean(mask=rf.Mask(query="selected", dropout=False, reconstruct=True)),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    inputs = model.encode(
        pa.table(
            {
                "context": [1.0, 2.0],
                "label": [True, False],
                "selected": [False, False],
            }
        ),
        strata=Strata.train,
    )
    monkeypatch.setattr("relflow.architecture.runtime.all_reduce_max", lambda local: local)

    output = model.training_step(inputs, batch_idx=0)

    assert output is None
