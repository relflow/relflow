"""Vocabulary storage is absent from the schema and does not define classes."""

from copy import deepcopy

import pyarrow as pa
import pydantic
import pytest
import torch

import relflow as rf
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions import category, group
from relflow.tensorfields.extensions import set as sets
from relflow.tensorfields.shared.resize import Resize

ADDRESS = "/label"
FIELDS = (rf.Category, rf.Set, rf.Group)


def build(kind):
    if kind == "group":
        field = rf.Group(n_groups=3, mask=True)
    else:
        field = {
            "category": rf.Category(mask=True, topk=[2, 10000]),
            "set": rf.Set(mask=True),
        }[kind]
    return rf.Model.xs(d_model=8, x=rf.Number, label=field)


def source(kind, labels):
    values = [[label] if label is not None else None for label in labels] if kind == "set" else labels
    return pa.table({"x": [0.0] * len(labels), "label": values})


def reserve(model, width):
    """Force spare runtime storage as an experimental control, not a field option."""
    node = model.nodes[ADDRESS]
    resize = Resize()
    if model.schema.requests[ADDRESS].type == "group":
        node.embedder.resize(width, resize)
    else:
        resize.embedding(node.embedder.embeddings["content"], width)
        resize.linear(node.decoder.linears["content"], width)
        node.embedder.counters["content"].resize(width, resize)
        resize.attribute(node.embedder.vocab, "size", width)
    resize.commit()


def approximate(value):
    """Allow float32 GEMM rounding while keeping prediction structure and labels exact."""
    if isinstance(value, float):
        return pytest.approx(value, rel=1e-6, abs=1e-7)
    if isinstance(value, dict):
        return {name: approximate(child) for name, child in value.items()}
    if isinstance(value, list):
        return [approximate(child) for child in value]
    return value


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("option", ["size", "capacity"])
def test_allocation_options_are_rejected_by_constructor_and_schema(field, option):
    required = {"n_groups": 3} if field is rf.Group else {}
    with pytest.raises(pydantic.ValidationError, match=option):
        field(**required, **{option: 8})
    with pytest.raises(pydantic.ValidationError, match=option):
        field.model_validate({**required, option: 8})
    assert option not in field.model_json_schema()["properties"]
    assert option not in field(**required).model_dump()


@pytest.mark.parametrize("kind", ["category", "set", "group"])
@pytest.mark.parametrize("width", [8, 128])
def test_spare_storage_preserves_predictions_schema_and_checkpoint(kind, width, tmp_path):
    model = build(kind)
    model.encode(source(kind, ["a", "b", "c"]), strata="train")
    schema = deepcopy(model.schema.model_dump())
    labels = tuple(model.nodes[ADDRESS].embedder.vocab.master)
    probe = source(kind, ["unseen", "a", None])
    before = model.predict(probe)
    expected = approximate(before["predictions"].to_pylist())

    reserve(model, width)

    assert model.schema.model_dump() == schema
    assert tuple(model.nodes[ADDRESS].embedder.vocab.master) == labels
    after = model.predict(probe)
    assert after.schema.equals(before.schema, check_metadata=True)
    assert after["predictions"].to_pylist() == expected
    loaded = rf.Model.load(model.save(tmp_path / "spare.ckpt"))
    assert loaded.schema.model_dump() == schema
    assert loaded.nodes[ADDRESS].embedder.vocab.size == width
    restored = loaded.predict(probe)
    assert restored.schema.equals(before.schema, check_metadata=True)
    assert restored["predictions"].to_pylist() == expected


@pytest.mark.parametrize("kind", ["category", "set", "group"])
@pytest.mark.parametrize("labels", [["a", "b", "unseen", None], ["unseen", None]])
def test_spare_storage_preserves_loss_and_gradients(kind, labels):
    small, large = build(kind), build(kind)
    small.encode(source(kind, ["a", "b", "c"]), strata="train")
    large.load_state_dict(small.state_dict())
    reserve(large, 128)
    probe = torch.randn(len(labels), 8)
    results, gradients = [], []
    objective = {"category": category.loss, "set": sets.loss, "group": group.loss}[kind]
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
        elif kind == "group":
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
    predicted = model.predict(source("category", ["unseen"]))["predictions"].to_pylist()[0]["label"]["content"]
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
