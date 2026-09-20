"""Validate the isolated norm-budget experiment before interpreting its proofs."""

from functools import partial
from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch

import relflow as rf
from experiments.vocabulary_norm import ROOT, Budget, Norm, category, load, project, radius, trajectory
from relflow.tensorfields.base import TensorInput


@pytest.fixture
def experiment():
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, entity=rf.Category(size=8, p_unavailable=0.0))
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    trainer = SimpleNamespace(world_size=1, optimizers=[optimizer])
    journal = []
    callback = Norm(Budget(0.0, 4), journal)
    callback.on_fit_start(trainer, model)
    embedder = model.nodes["record/entity"].embedder
    yield model, embedder, optimizer, trainer, callback, journal
    if callback.handles:
        callback.on_fit_end(trainer, model)


def inputs(values, states=None):
    content = torch.tensor(values, dtype=torch.int64)
    state = torch.zeros_like(content) if states is None else torch.tensor(states, dtype=torch.int64)
    return TensorInput(content=content, state=state, batch_size=len(values))


def test_budget_grows_monotonically_and_stops_at_one():
    counts = torch.tensor([0, 1, 2, 4, 8])
    torch.testing.assert_close(radius(counts, Budget(0.0, 4)), torch.tensor([0.0, 0.25, 0.5, 1.0, 1.0]))
    torch.testing.assert_close(radius(counts, Budget(0.01, 4)), torch.tensor([0.01, 0.2575, 0.505, 1.0, 1.0]))
    assert radius(counts, Budget(0.0)).eq(1).all()


def test_projection_caps_without_forcing_small_vectors_outward():
    weights = torch.tensor([[3.0, 4.0], [0.03, 0.04], [0.0, 0.0]])
    project(weights, torch.ones(3))
    torch.testing.assert_close(weights, torch.tensor([[0.6, 0.8], [0.03, 0.04], [0.0, 0.0]]))


def test_admission_and_forward_do_not_count_as_learning(experiment):
    model, embedder, _, _, _, _ = experiment
    weights = embedder.embeddings["content"].weight.detach().clone()
    model.encode(pa.Table.from_pylist([{"entity": "new"}]), strata="train")
    assert rf.Category.vocabulary(model, "record/entity") == ("new",)
    embedder(inputs([0]))
    assert embedder.budget_counts.eq(0).all()
    assert embedder.budget_pending.eq(0).all()
    torch.testing.assert_close(embedder.embeddings["content"].weight, weights)


def test_only_backpropagated_valued_known_occurrences_count(experiment):
    _, embedder, optimizer, _, _, _ = experiment
    batch = inputs([0, 0, 1, 8, 2, 3, 4], [0, 0, 0, 0, rf.Tokens.masked, rf.Tokens.null, 0])
    payload = embedder(batch).payload
    payload[:6].sum().backward()
    assert embedder.budget_counts.eq(0).all()
    assert embedder.budget_pending.tolist() == [2, 1, 0, 0, 0, 0, 0, 0]
    optimizer.step()
    assert embedder.budget_counts.tolist() == [2, 1, 0, 0, 0, 0, 0, 0]
    assert embedder.budget_pending.eq(0).all()
    torch.testing.assert_close(
        embedder.embeddings["content"].weight.norm(dim=-1), torch.tensor([0.5, 0.25, 0, 0, 0, 0, 0, 0])
    )


def test_gradient_accumulation_commits_counts_once(experiment):
    _, embedder, optimizer, _, callback, _ = experiment
    for _ in range(2):
        embedder(inputs([1, 1])).payload.sum().backward()
    assert embedder.budget_pending[1] == 4
    assert embedder.budget_counts[1] == 0
    optimizer.step()
    assert embedder.budget_counts[1] == 4
    assert callback.steps == 1
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    assert embedder.budget_counts[1] == 4


def test_evaluation_and_no_grad_do_not_count(experiment):
    model, embedder, _, _, _, _ = experiment
    model.eval()
    embedder(inputs([0])).payload.sum().backward()
    model.train()
    with torch.no_grad():
        embedder(inputs([0]))
    assert embedder.budget_pending.eq(0).all()


def test_counts_survive_successive_fits_and_hooks_are_removed(experiment):
    model, embedder, optimizer, trainer, callback, journal = experiment
    embedder(inputs([1])).payload.sum().backward()
    optimizer.step()
    callback.on_fit_end(trainer, model)
    assert not callback.handles
    assert not embedder._forward_hooks
    weights = embedder.embeddings["content"].weight.detach().clone()
    following = Norm(callback.budget, journal)
    following.on_fit_start(trainer, model)
    torch.testing.assert_close(embedder.embeddings["content"].weight, weights)
    optimizer.zero_grad(set_to_none=True)
    embedder(inputs([1, 2])).payload.sum().backward()
    optimizer.step()
    following.on_fit_end(trainer, model)
    assert embedder.budget_counts.tolist() == [0, 2, 1, 0, 0, 0, 0, 0]
    assert len(journal) == 2
    assert all(stage["fields"]["record/entity"]["maximum_bound_excess"] < 1e-6 for stage in journal)


def test_rolling_proof_counts_learning_in_every_training_stage():
    proof = load(ROOT / "proofs/vocabulary/rolling_input_admission.py")
    journal = []
    factory = partial(Norm, Budget(0.0, 128), journal, partial(trajectory, proof, 7801))
    category.callback(factory)
    try:
        proof.run(7801, 2, "cpu")
    finally:
        category.callback_factories.remove(factory)
    counts = [stage["fields"]["record/entity"]["gradient_occurrences"] for stage in journal]
    assert 0 < counts[0] < counts[1] < counts[2]
    assert all(stage["fields"]["record/entity"]["steps"] == 2 for stage in journal)
