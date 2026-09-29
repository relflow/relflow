"""Vocabulary storage is absent from the schema and does not define classes."""

from copy import deepcopy

import pyarrow as pa
import pydantic
import pytest
import torch

import relflow as rf
from relflow.helpers.resize import Resize
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions import category, cluster
from relflow.tensorfields.extensions import set as sets

ADDRESS = "/label"
FIELDS = (rf.Category, rf.Set, rf.Cluster)


def build(kind):
    field = {
        "category": rf.Category(mask=True, topk=[2, 10000]),
        "set": rf.Set(mask=True),
        "cluster": rf.Cluster(bounds=3, mask=True, revive_temperature=0),
    }[kind]
    return rf.Model(d_model=8, n_layers=1, n_heads=2, dropout=0.0, x=rf.Number, label=field)


def source(kind, labels):
    values = [[label] if label is not None else None for label in labels] if kind == "set" else labels
    return pa.table({"x": [0.0] * len(labels), "label": values})


def reserve(model, width):
    """Force spare runtime storage as an experimental control, not a field option."""
    node = model.nodes[ADDRESS]
    resize = Resize()
    if model.schema.requests[ADDRESS].type == "cluster":
        node.embedder.resize(width, resize)
    else:
        resize.embedding(node.embedder.embeddings["content"], width)
        resize.linear(node.decoder.linears["content"], width)
        node.embedder.counters["content"].resize(width, resize)
        resize.attribute(node.embedder.vocab, "size", width)
    resize.commit()


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("option", ["size", "capacity"])
def test_allocation_options_are_rejected_by_constructor_and_schema(field, option):
    required = {"bounds": 3} if field is rf.Cluster else {}
    with pytest.raises(pydantic.ValidationError, match=option):
        field(**required, **{option: 8})
    with pytest.raises(pydantic.ValidationError, match=option):
        field.model_validate({"name": "label", **required, option: 8})
    assert option not in field.model_json_schema()["properties"]
    assert option not in field(**required).model_dump()


@pytest.mark.parametrize("kind", ["category", "set", "cluster"])
@pytest.mark.parametrize("width", [8, 128])
def test_spare_storage_preserves_predictions_schema_and_checkpoint(kind, width, tmp_path):
    model = build(kind)
    model.encode(source(kind, ["a", "b", "c"]), strata="train")
    schema = deepcopy(model.schema.model_dump())
    labels = tuple(model.nodes[ADDRESS].embedder.vocab.master)
    before = model.predict(source(kind, ["unseen", "a", None]))["predictions"].to_pylist()

    reserve(model, width)

    assert model.schema.model_dump() == schema
    assert tuple(model.nodes[ADDRESS].embedder.vocab.master) == labels
    assert model.predict(source(kind, ["unseen", "a", None]))["predictions"].to_pylist() == before
    loaded = rf.Model.load(model.save(tmp_path / "spare.ckpt"))
    assert loaded.schema.model_dump() == schema
    assert loaded.nodes[ADDRESS].embedder.vocab.size == width
    assert loaded.predict(source(kind, ["unseen", "a", None]))["predictions"].to_pylist() == before


@pytest.mark.parametrize("kind", ["category", "set", "cluster"])
@pytest.mark.parametrize("labels", [["a", "b", "unseen", None], ["unseen", None]])
def test_spare_storage_preserves_loss_and_gradients(kind, labels):
    small, large = build(kind), build(kind)
    small.encode(source(kind, ["a", "b", "c"]), strata="train")
    large.load_state_dict(small.state_dict())
    reserve(large, 128)
    probe = torch.randn(len(labels), 8)
    results, gradients = [], []
    objective = {"category": category.loss, "set": sets.loss, "cluster": cluster.loss}[kind]
    for model in (small, large):
        model.eval()
        model.track = lambda names, value: value
        batch = model.encode(source(kind, labels), strata="test")[ADDRESS]
        context = probe.clone().requires_grad_()
        payload = model.nodes[ADDRESS].decoder.decode(context)
        value = objective(model, Prediction(address=ADDRESS, payload=payload), batch, rf.Strata.test)
        value.backward()
        assert torch.isfinite(value)
        results.append(value)
        gradients.append(context.grad)
    torch.testing.assert_close(results[0], results[1])
    torch.testing.assert_close(gradients[0], gradients[1])
    for name, parameter in small.named_parameters():
        other = dict(large.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
            continue
        if parameter.shape == other.shape:
            torch.testing.assert_close(parameter.grad, other.grad)
        elif kind == "cluster":
            torch.testing.assert_close(parameter.grad[:-1], other.grad[: parameter.shape[0] - 1])
            torch.testing.assert_close(parameter.grad[-1], other.grad[-1])
            assert not other.grad[parameter.shape[0] - 1 : -1].any()
        else:
            torch.testing.assert_close(parameter.grad, other.grad[: parameter.shape[0]])
            assert not other.grad[parameter.shape[0] :].any()


@pytest.mark.parametrize("known", [[], ["only"], ["a", "b", "c"]])
def test_topk_is_capped_by_discovered_labels_including_empty_vocabulary(known):
    model = build("category")
    if known:
        model.encode(source("category", known), strata="train")
    predicted = model.predict(source("category", ["unseen"]))["predictions"].to_pylist()[0][ADDRESS]["content"]
    assert len(predicted["topk"]) == len(known)
    assert {item["value"] for item in predicted["topk"]} == set(known)
    if not known:
        assert predicted["value"] is None
    batch = model.encode(source("category", [known[0] if known else "unseen"]), strata="test")[ADDRESS]
    model.track = lambda names, value: value
    payload = model.nodes[ADDRESS].decoder.decode(torch.randn(1, 8))
    value = category.loss(model, Prediction(address=ADDRESS, payload=payload), batch, rf.Strata.test)
    value.backward()
    assert torch.isfinite(value)
    metric = model.nodes[ADDRESS].decoder.metrics["test_metrics"].compute()
    assert metric["accuracy.top10000"] == 1 if known else torch.isnan(metric["accuracy.top10000"])
