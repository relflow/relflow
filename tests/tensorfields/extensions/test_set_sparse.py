"""Set inputs scale with observed members while preserving their dense objective."""

from copy import deepcopy

import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.binding import bind
from relflow.data.iterables import encode
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.set import loss

ADDRESS = rf.Address("/tags")


def model(*, mask=False, p_unavailable=0.0):
    return rf.Model.xs(
        d_model=8,
        attention=None,
        tags=rf.Set(mask=mask, p_unavailable=p_unavailable),
    )


def table(values):
    return pa.table({"tags": pa.array(values, type=pa.list_(pa.string()))})


def dense_embedding(module, field, values, parameters):
    """Independent reference for the original binary-membership mean."""
    weights, state_weights, unavailable = parameters
    vocabulary = rf.Set.vocabulary(module, ADDRESS)
    membership = weights.new_zeros(len(values), weights.shape[0])
    unknown = torch.zeros(len(values), dtype=torch.bool, device=weights.device)
    for row, members in enumerate(values):
        for label in set(members or ()) - {None}:
            if label in vocabulary:
                membership[row, vocabulary.index(label)] = 1
            else:
                unknown[row] = True
    state = field.state.reshape(-1)
    content = membership @ weights / membership.sum(-1, keepdim=True).clamp_min(1)
    content = content + unknown.unsqueeze(-1) * unavailable
    return state_weights[state] + content * state.eq(rf.Tokens.valued).unsqueeze(-1)


@pytest.mark.parametrize("objective", [False, True])
def test_small_sets_do_not_allocate_per_vocabulary_member(objective):
    sizes = []
    for cardinality in (4, 4096):
        module = model(mask=rf.Mask(dropout=False, reconstruct=True) if objective else False)
        module.encode(table([[f"label-{index}"] for index in range(cardinality)]), strata="train")
        field = module.encode(table([["label-0"], ["label-2", "label-3"], [], None]), strata="train")[ADDRESS]
        contents = [field.content]
        if objective:
            contents.append(field.targets[rf.TensorKey.content])
        for content in contents:
            assert set(content.keys()) == {"indices", "unavailable"}
            assert content["indices"].dtype == torch.int64
            assert content["indices"].shape == (*field.state.shape, 2)
            assert content["unavailable"].shape == field.state.shape
        sizes.append(sum(value.numel() * value.element_size() for value in field.content.values()))
    assert sizes[0] == sizes[1]
    assert sizes[1] < 4 * 4096 * torch.tensor(0.0).element_size() / 100


def test_compact_pooling_matches_dense_values_and_parameter_gradients():
    module = model()
    module.encode(table([["a", "b", "c"]]), strata="train")
    values = [["a", "b", "a"], ["b", "a"], ["a", "new", "other", "new"], [], None, [None], ["new"]]
    field = module.encode(table(values))[ADDRESS]
    embedder = module.nodes[ADDRESS].embedder
    with torch.no_grad():
        embedder.unavailable.copy_(torch.arange(8) / 10)
    parameters = [embedder.embeddings["content"].weight, embedder.embeddings["state"].weight, embedder.unavailable]
    reference_parameters = [parameter.detach().clone().requires_grad_() for parameter in parameters]
    actual = embedder(field.take(None)).payload
    expected = dense_embedding(module, field, values, reference_parameters)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual[0], actual[1])
    assert field.content["unavailable"].reshape(-1).tolist() == [0, 0, 2, 0, 0, 0, 1]
    factors = torch.linspace(-2, 3, actual.numel()).reshape_as(actual)
    gradients = torch.autograd.grad((actual * factors).sum(), parameters)
    reference_gradients = torch.autograd.grad((expected * factors).sum(), reference_parameters)
    for gradient, reference in zip(gradients, reference_gradients, strict=True):
        torch.testing.assert_close(gradient, reference)


def test_empty_member_axis_is_valid_and_preserves_null_state():
    module = model()
    field = module.encode(table([[], None, [None]]))[ADDRESS]
    assert field.content["indices"].shape == (3, 1, 0)
    assert field.content["unavailable"].count_nonzero() == 0
    embedder = module.nodes[ADDRESS].embedder
    actual = embedder(field.take(None)).payload
    expected = embedder.embeddings["state"](field.state.reshape(-1))
    torch.testing.assert_close(actual, expected)
    assert field.state.reshape(-1).tolist() == [rf.Tokens.valued, rf.Tokens.null, rf.Tokens.valued]
    actual.sum().backward()
    assert embedder.embeddings["content"].weight.grad is not None
    assert embedder.embeddings["content"].weight.grad.count_nonzero() == 0


def test_unavailable_augmentation_preserves_compact_pristine_targets():
    source = table([["a", "b", "a"], [], None])
    field = model(p_unavailable=1.0).encode(source, strata="train")[ADDRESS]
    supervised = model(mask=True, p_unavailable=1.0).encode(source, strata="train")[ADDRESS]
    answers = supervised.targets[rf.TensorKey.content]
    assert field.content["indices"].eq(-1).all()
    assert field.content["unavailable"].reshape(-1).tolist() == [2, 0, 0]
    assert answers["indices"].ge(0).sum().item() == 2
    assert answers["unavailable"].count_nonzero() == 0
    assert supervised.content["indices"].shape == answers["indices"].shape == (3, 1, 2)
    assert supervised.content["indices"].eq(-1).all()


@pytest.mark.parametrize("objective", [False, True])
def test_nested_member_axis_survives_clone_device_transfer_and_coordinate_selection(objective):
    module = rf.Model.xs(
        d_model=8,
        attention=None,
        events=rf.Branch(
            length=3,
            tags=rf.Set(
                p_unavailable=0.0,
                mask=rf.Mask(dropout=False, reconstruct=True) if objective else False,
            ),
        ),
    )
    source = pa.Table.from_pylist(
        [
            {"events": [{"tags": ["a", "b"]}, {"tags": []}]},
            {"events": [{"tags": ["b"]}, None, {"tags": None}]},
        ]
    )
    field = module.encode(source, strata="train")["/events/tags"]
    copied = field.clone().to("cpu")
    assert copied.batch_size == field.batch_size == torch.Size([2])
    assert copied.content["indices"].shape == (2, 1, 3, 2)
    if objective:
        assert copied.targets[rf.TensorKey.content]["indices"].shape == (2, 1, 3, 2)
    torch.testing.assert_close(copied.content["indices"], field.content["indices"])
    assert copied.content["indices"].data_ptr() != field.content["indices"].data_ptr()
    coordinates = torch.tensor([5, 0, 3, 0])
    selected = copied.take(coordinates)
    assert selected.batch_size == torch.Size([4])
    assert selected.content["indices"].shape == (4, 2)
    torch.testing.assert_close(selected.content["indices"], copied.content["indices"].reshape(6, 2)[coordinates])
    embedder = module.nodes["/events/tags"].embedder
    torch.testing.assert_close(embedder(selected).payload, embedder(copied.take(None)).payload[coordinates])
    selected.content["indices"].fill_(-1)
    torch.testing.assert_close(copied.content["indices"], field.content["indices"])


@pytest.mark.parametrize("strata", [rf.Strata.train, rf.Strata.validate])
def test_prefetched_member_ids_rebind_after_growth_without_mutating_workers(strata):
    module = model()
    module.encode(table([["a"]]), strata="train")
    sources = [table([["b", "a", "b"], [], None]), table([["c", "b"], ["a"]])]
    prefetched = [encode(source, module.schema, strata, module.interprocess_encoding_context) for source in sources]
    original = deepcopy(prefetched)
    module.encode(table([["d", "e", "f", "g"]]), strata="train")
    for index in (1, 0, 1):
        field = bind(module, prefetched[index], strata).tensors[ADDRESS]
        values = sources[index]["tags"].to_pylist()
        embedder = module.nodes[ADDRESS].embedder
        parameters = [embedder.embeddings["content"].weight, embedder.embeddings["state"].weight, embedder.unavailable]
        expected = dense_embedding(module, field, values, parameters)
        torch.testing.assert_close(embedder(field.take(None)).payload, expected)
        assert field.content["indices"].shape[-1] <= 2
        assert (prefetched[index].tensors == original[index].tensors).all()
        assert prefetched[index].bindings[ADDRESS].labels.equals(original[index].bindings[ADDRESS].labels)
        for address, observation in prefetched[index].observations.items():
            assert (observation == original[index].observations[address]).all()
    vocabulary = rf.Set.vocabulary(module, ADDRESS)
    assert ("b" in vocabulary) == (strata == rf.Strata.train)
    assert ("c" in vocabulary) == (strata == rf.Strata.train)


def test_compact_supervised_targets_match_dense_binary_loss_and_gradients():
    module = model(mask=True)
    module.encode(table([["a", "b", "c"]]), strata="train")
    module.track = lambda names, value: value
    values = [["b", "a", "a"], ["a", "novel"], ["novel"], [], None]
    field = module.encode(table(values), strata="test")[ADDRESS]
    capacity = module.nodes[ADDRESS].embedder.size
    state_logits = torch.randn(len(values), 1, len(rf.Tokens), requires_grad=True)
    content_logits = torch.randn(len(values), 1, capacity, requires_grad=True)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {rf.TensorKey.state: state_logits, rf.TensorKey.content: content_logits},
            batch_size=[len(values)],
        ),
    )
    actual = loss(module=module, prediction=prediction, batch=field, strata=rf.Strata.test)
    reference_state = state_logits.detach().clone().requires_grad_()
    reference_content = content_logits.detach().clone().requires_grad_()
    targets = torch.zeros(len(values), 3)
    targets[0, :2] = 1
    targets[1, 0] = 1
    state_targets = field.targets[rf.TensorKey.state].reshape(-1)
    trainable = field.trainable.reshape(-1)
    valued = trainable & state_targets.eq(rf.Tokens.valued)
    expected = torch.nn.functional.cross_entropy(
        reference_state.reshape(-1, len(rf.Tokens)), state_targets, reduction="none"
    )
    expected = expected[trainable].mean() + torch.nn.functional.binary_cross_entropy_with_logits(
        reference_content.reshape(len(values), capacity)[valued, :3], targets[valued]
    )
    torch.testing.assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(actual, (state_logits, content_logits))
    reference_gradients = torch.autograd.grad(expected, (reference_state, reference_content))
    for gradient, reference in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(gradient, reference)
    assert content_logits.grad is None
    assert actual_gradients[1][..., 3:].count_nonzero() == 0
