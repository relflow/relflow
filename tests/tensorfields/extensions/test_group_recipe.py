import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.helpers.resize import Resize
from relflow.structs.enums import TensorKey, Tokens
from relflow.tensorfields.extensions.group import Embedder, TensorField, vocabulary_logits


@pytest.fixture
def embedder():
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        identity=rf.Group(n_groups=3, mask=True, p_unavailable=0.0),
    )
    return Embedder(schema=schema, address="/identity")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_identity_likelihood_is_a_normalized_mixture_with_identity_priors(embedder, dtype):
    resize = Resize()
    embedder.resize(8, resize)
    resize.commit()
    embedder.vocab.extend(["frequent", "middle", "rare"])
    embedder.to(dtype=dtype)
    weight = embedder.embeddings[TensorKey.group.name].weight
    with torch.no_grad():
        weight.fill_(500)
        weight[:3].copy_(torch.tensor([[8, 2, 1], [2, 2, 2], [1, 1, 5]]).float().log())
        # Commitment does not remove an otherwise available assignment candidate.
        embedder.committed[-1] = False
    probabilities = torch.tensor([[0.2, 0.5, 0.3], [0.7, 0.1, 0.2]], dtype=dtype, requires_grad=True)

    logits = vocabulary_logits(embedder, probabilities)

    conditional = torch.tensor([[8 / 11, 2 / 5, 1 / 8], [2 / 11, 2 / 5, 2 / 8], [1 / 11, 1 / 5, 5 / 8]])
    expected = probabilities.detach().float() @ conditional.T
    tolerance = 0.005 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(logits.exp(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(logits.exp().sum(-1), torch.ones(2), atol=tolerance, rtol=tolerance)
    # Normalization across identities must preserve their nonuniform prior.
    assert not torch.allclose(expected[0], torch.full((3,), 1 / 3))

    (logits.exp() * torch.tensor([1.0, 2.0, 4.0])).sum().backward()
    assert weight.grad is not None and probabilities.grad is not None
    assert torch.isfinite(weight.grad).all() and torch.isfinite(probabilities.grad).all()
    assert bool(weight.grad[:3].ne(0).any())
    assert torch.count_nonzero(weight.grad[3:]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_identity_likelihood_excludes_retired_groups(embedder, dtype):
    resize = Resize()
    embedder.resize(4, resize)
    resize.commit()
    embedder.vocab.extend(["a", "b"])
    embedder.to(dtype=dtype)
    embedder.retired[1] = True
    weight = embedder.embeddings[TensorKey.group.name].weight
    with torch.no_grad():
        weight.zero_()
        weight[:2, 0] = torch.tensor([3.0, 1.0]).log().to(dtype)
        weight[:2, 1] = torch.tensor([100.0, -100.0], dtype=dtype)
        weight[:2, 2] = torch.tensor([1.0, 4.0]).log().to(dtype)
    probabilities = torch.tensor([[0.25, 0.0, 0.75]], dtype=dtype, requires_grad=True)

    logits = vocabulary_logits(embedder, probabilities)

    expected = torch.tensor([[0.25 * 3 / 4 + 0.75 / 5, 0.25 / 4 + 0.75 * 4 / 5]])
    tolerance = 0.005 if dtype == torch.bfloat16 else 1e-6
    torch.testing.assert_close(logits.exp(), expected, atol=tolerance, rtol=tolerance)
    logits[0, 0].backward()
    assert weight.grad is not None and probabilities.grad is not None
    assert torch.isfinite(weight.grad).all() and torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(weight.grad[:, 1]) == 0
    assert probabilities.grad[0, 1] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_empty_identity_likelihood_keeps_batch_shape_and_finite_gradients(embedder, dtype):
    embedder.to(dtype=dtype)
    probabilities = torch.tensor([[0.2, 0.5, 0.3], [1.0, 0.0, 0.0]], dtype=dtype, requires_grad=True)

    logits = vocabulary_logits(embedder, probabilities)

    assert logits.shape == (2, 0)
    logits.sum().backward()
    weight = embedder.embeddings[TensorKey.group.name].weight
    assert weight.grad is not None and probabilities.grad is not None
    assert torch.isfinite(weight.grad).all() and torch.isfinite(probabilities.grad).all()
    assert torch.count_nonzero(weight.grad) == 0
    assert torch.count_nonzero(probabilities.grad) == 0


def test_group_initialization_balances_identities_and_keeps_unknown_neutral(embedder):
    weight = embedder.embeddings[TensorKey.group.name].weight
    assert weight.shape == (2, 3)
    torch.testing.assert_close(weight[0], torch.tensor([0.1, 0.0, 0.0]))
    assert torch.count_nonzero(weight[embedder.capacity]) == 0

    resize = Resize()
    embedder.resize(8, resize)
    resize.commit()

    weight = embedder.embeddings[TensorKey.group.name].weight
    expected = torch.zeros(8, 3)
    expected[torch.arange(8), torch.arange(8) % 3] = 0.1
    torch.testing.assert_close(weight[:8], expected)
    assert torch.count_nonzero(weight[embedder.capacity]) == 0


def test_group_growth_preserves_learned_rows_and_unknown_while_avoiding_retired_groups(embedder):
    resize = Resize()
    embedder.resize(4, resize)
    resize.commit()
    weight = embedder.embeddings[TensorKey.group.name].weight
    with torch.no_grad():
        weight[:4].copy_(torch.arange(12).reshape(4, 3))
        weight[4].fill_(-7)
        embedder.retired[1] = True
    learned = weight[:4].detach().clone()
    unknown = weight[4].detach().clone()

    resize = Resize()
    embedder.resize(8, resize)
    resize.commit()

    weight = embedder.embeddings[TensorKey.group.name].weight
    torch.testing.assert_close(weight[:4], learned)
    torch.testing.assert_close(weight[8], unknown)
    expected = torch.zeros(4, 3)
    expected[torch.arange(4), torch.tensor([0, 2, 0, 2])] = 0.1
    torch.testing.assert_close(weight[4:8], expected)


def test_training_routes_each_identity_deterministically_with_straight_through_gradients(embedder):
    resize = Resize()
    embedder.resize(4, resize)
    resize.commit()
    with torch.no_grad():
        embedder.embeddings[TensorKey.state.name].weight.zero_()
        embedder.embeddings[TensorKey.content.name].weight.zero_()
        embedder.embeddings[TensorKey.content.name].weight[:3].copy_(torch.eye(3))
        embedder.embeddings[TensorKey.group.name].weight[:2].copy_(torch.tensor([[2.0, 0.0, -1.0], [-1.0, 0.5, 1.5]]))
    field = TensorField(
        state=torch.full((2, 2), Tokens.valued.value, dtype=torch.int64),
        content=torch.tensor([[0, 0], [1, 0]], dtype=torch.int64),
        present=torch.ones((2, 2), dtype=torch.bool),
        trainable=torch.zeros((2, 2), dtype=torch.bool),
        inferred=torch.zeros((2, 2), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=2,
    )
    embedder.train()
    rng = torch.get_rng_state().clone()
    training = embedder.embed(field).payload

    assert torch.equal(torch.get_rng_state(), rng)
    torch.testing.assert_close(training[0, 0], training[0, 1])
    torch.testing.assert_close(training[0, 0], training[1, 1])
    torch.testing.assert_close(
        training[..., :3], torch.tensor([[[1.0, 0, 0], [1.0, 0, 0]], [[0, 0, 1.0], [1.0, 0, 0]]])
    )
    embedder.eval()
    torch.testing.assert_close(training, embedder.embed(field).payload)

    training.square().sum().backward()
    gradient = embedder.embeddings[TensorKey.group.name].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert bool(gradient[:2].ne(0).any())
    assert torch.count_nonzero(gradient[2:]) == 0
