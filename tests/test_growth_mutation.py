"""Schema edits preserve extension-owned learned vocabulary storage."""

from copy import deepcopy
from typing import Literal
from uuid import uuid4

import pyarrow as pa
import pydantic
import pytest
import torch

import relflow as rf
from relflow.helpers.state import Rebuild, compatible
from relflow.tensorfields.base import TENSORFIELDS, Extension
from relflow.tensorfields.extensions import number

ADDRESS = rf.Address("/value")
KINDS = ("category", "set", "cluster")
LABELS = ("a", "b", "c", "d", "e", "a")


def build(kind):
    options = {"p_unavailable": 0, "mask": rf.Mask(reconstruct=True)}
    if kind == "cluster":
        field = rf.Cluster(bounds=3, revive_temperature=0, **options)
    else:
        field = {"category": rf.Category, "set": rf.Set}[kind](**options)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention=None,
        reduction=rf.Mean(),
        value=field,
    )
    values = [[label] for label in LABELS] if kind == "set" else list(LABELS)
    model.encode(pa.table({"value": values}), strata="train")
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(7.25)
    model.eval()
    return model


def check_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
        else:
            assert value == expected[name]


def check_allocation(model, kind, size):
    node = model.nodes[ADDRESS]
    rows = size + (kind == "cluster")
    assert node.embedder.vocab.size == size
    counter = node.embedder.counters["content"]
    assert counter.size == rows
    assert counter.counts.shape == counter._pending_counts.shape == (rows,)
    embedding = node.embedder.embeddings["cluster" if kind == "cluster" else "content"]
    assert embedding.num_embeddings == rows
    assert embedding.weight.shape == (rows, 3 if kind == "cluster" else 8)
    if kind != "cluster":
        linear = node.decoder.linears["content"]
        assert linear.out_features == size
        assert linear.weight.shape == (size, 8)
        assert linear.bias.shape == (size,)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("checkpoint", [False, True], ids=["live", "loaded"])
def test_neutral_update_preserves_grown_state(kind, checkpoint, tmp_path):
    model = build(kind)
    if checkpoint:
        path = tmp_path / "grown.ckpt"
        model.save(path)
        model = rf.Model.load(path)
        model.eval()
    check_allocation(model, kind, 8)
    before = deepcopy(model.state_dict())
    old = model.nodes[ADDRESS]
    pending = old.embedder.counters["content"]._pending_counts.clone()

    model.update(rf.where("address") == ADDRESS, description="Neutral annotation")

    assert model.nodes[ADDRESS] is not old
    assert not model.training
    check_allocation(model, kind, 8)
    check_state(model.state_dict(), before)
    torch.testing.assert_close(model.nodes[ADDRESS].embedder.counters["content"]._pending_counts, pending)
    assert model.nodes[ADDRESS].embedder.vocab.snapshot() == list(dict.fromkeys(LABELS))

    # Repeated rebuilds must retain learned storage without a schema allocation.
    with model.override(rf.where("address") == ADDRESS, description="Temporary annotation"):
        check_state(model.state_dict(), before)
    check_state(model.state_dict(), before)
    assert "capacity" not in model.schema.requests[ADDRESS].model_dump()
    assert "size" not in model.schema.requests[ADDRESS].model_dump()


@pytest.mark.parametrize("kind", KINDS)
def test_extending_schema_preserves_existing_grown_resources(kind):
    model = build(kind)
    before = deepcopy(model.nodes[ADDRESS].embedder.state_dict())
    outputs = deepcopy(model.nodes[ADDRESS].decoder.linears.state_dict())

    model.extend(extra=rf.Number)

    check_allocation(model, kind, 8)
    check_state(model.nodes[ADDRESS].embedder.state_dict(), before)
    check_state(model.nodes[ADDRESS].decoder.linears.state_dict(), outputs)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("option", ["size", "capacity"])
def test_allocation_options_are_rejected_without_changing_grown_state(kind, option):
    model = build(kind)
    before = deepcopy(model.state_dict())
    schema = model.schema.model_dump()
    for validate in (False, True):
        with pytest.raises((AttributeError, pydantic.ValidationError), match=option):
            model.update(rf.where("address") == ADDRESS, **{option: 4}, validate=validate)
        check_state(model.state_dict(), before)
        assert model.schema.model_dump() == schema


def test_cluster_bounds_change_retains_grown_capacity_but_reinitializes_changed_columns():
    model = build("cluster")
    old = model.nodes[ADDRESS]
    counts = old.embedder.counters["content"].counts.clone()

    model.update(rf.where("address") == ADDRESS, n_clusters=(5, 5))

    node = model.nodes[ADDRESS]
    assert node.embedder.capacity == 8
    assert node.embedder.size == 5
    assert node.embedder.embeddings["cluster"].weight.shape == (9, 5)
    assert node.embedder.embeddings["content"].weight.shape == (8, 5)
    assert node.decoder.linears["cluster"].weight.shape == (5, 8)
    assert node.embedder.committed.shape == node.embedder.usage_ema.shape == (5,)
    assert not torch.all(node.embedder.embeddings["cluster"].weight == 7.25)
    torch.testing.assert_close(node.embedder.counters["content"].counts, counts)
    assert node.embedder.vocab.snapshot() == list(dict.fromkeys(LABELS))


def test_third_party_resource_owns_subtree_rebuild_without_architecture_changes():
    name = f"growth_mutation_{uuid4().hex}"
    extension = Extension(name=name, types=(float,))

    @extension.register
    class Request(number.Request):
        type: Literal[name] = name
        allocation: int = 2

    @extension.register
    class TensorField(number.TensorField):
        pass

    class Resource(torch.nn.Module):
        def __init__(self, size):
            super().__init__()
            self.initial_size = size
            self.table = torch.nn.Embedding(size, 3)

        def rebuild_state(self, previous):
            if self.initial_size == previous.initial_size:
                self.table = torch.nn.Embedding(previous.table.num_embeddings, 3)
            return compatible(self.state_dict(), previous.state_dict())

    @extension.register
    class Embedder(number.Embedder):
        def __init__(self, schema: rf.Schema, address: rf.Address):
            super().__init__(schema=schema, address=address)
            self.resource = Resource(schema.requests[address].allocation)

    try:
        model = rf.Model(d_model=8, n_layers=1, n_heads=2, value=Request())
        old = model.nodes[ADDRESS].embedder.resource
        assert isinstance(old, Rebuild)
        old.table = torch.nn.Embedding(7, 3)
        with torch.no_grad():
            old.table.weight.fill_(7.25)

        model.update(rf.where("address") == ADDRESS, description="Third-party neutral edit")

        resource = model.nodes[ADDRESS].embedder.resource
        assert resource is not old
        assert resource.table.num_embeddings == 7
        torch.testing.assert_close(resource.table.weight, old.table.weight, atol=0, rtol=0)

        model.update(rf.where("address") == ADDRESS, allocation=4)

        resource = model.nodes[ADDRESS].embedder.resource
        assert resource.table.num_embeddings == 4
        assert not torch.all(resource.table.weight == 7.25)
    finally:
        TENSORFIELDS.pop(name, None)
