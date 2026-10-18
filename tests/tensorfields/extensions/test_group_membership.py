"""Group's own likelihood supplies proposals without another tensorfield objective."""

import pytest
import torch

from relflow.tensorfields.membership import propose


def brute_force(assignments, context, identities, available):
    weights = assignments.double()
    errors = torch.zeros_like(weights)
    proposals = {}
    for identity in range(len(weights)):
        for candidate in available.nonzero(as_tuple=True)[0].tolist():
            proposal = torch.full_like(weights[identity], 1e-8).log()
            proposal[candidate] = 0
            proposal += weights[identity].logsumexp(0) - proposal.logsumexp(0)
            changed = weights.clone()
            changed[identity] = proposal
            conditional = changed.log_softmax(0)[identity]
            selected = identities.eq(identity)
            likelihood = context[selected].double() + conditional
            likelihood[:, ~available] = -torch.inf
            errors[identity, candidate] = -likelihood.logsumexp(-1).sum()
            proposals[identity, candidate] = proposal
    return errors, proposals


def example():
    weights = torch.tensor([[9.0, 1.0, 2.0], [2.0, 5.0, 1.0], [1.0, 3.0, 6.0], [4.0, 1.0, 7.0]]).log()
    probabilities = torch.tensor([[0.90, 0.05, 0.05], [0.70, 0.20, 0.10], [0.05, 0.90, 0.05], [0.10, 0.10, 0.80]])
    identities = torch.tensor([0, 0, 1, 2])
    return weights, probabilities.log(), identities, torch.ones(3, dtype=torch.bool)


def test_own_likelihood_matches_literal_one_row_candidate_search():
    weights, context, identities, available = example()
    errors, proposals = brute_force(weights, context, identities, available)

    replacement, rows = propose(weights, context, identities, available)

    assert rows.tolist() == [0, 1, 2]
    for identity in rows.tolist():
        candidate = errors[identity].argmin().item()
        torch.testing.assert_close(replacement[identity].double(), proposals[identity, candidate])
    torch.testing.assert_close(replacement.logsumexp(-1), weights.logsumexp(-1))
    assert torch.equal(replacement[3], weights[3])


def test_other_identities_compete_for_normalized_likelihood():
    weights = torch.tensor([[1.0, 1.0], [1000.0, 0.01]]).log()
    context = torch.tensor([[0.9, 0.1]]).log()

    replacement, rows = propose(weights, context, torch.tensor([0]), torch.ones(2, dtype=torch.bool))

    assert rows.tolist() == [0]
    assert replacement[0].argmax() == 1
    assert context[0].argmax() == 0
    assert torch.equal(replacement[1], weights[1])


def test_proposals_preserve_inputs_gradients_and_random_state():
    weights, context, identities, available = example()
    weights.requires_grad_()
    context.requires_grad_()
    weights.grad = torch.ones_like(weights)
    before = [value.clone() for value in (weights, context, identities, available)]
    random = torch.random.get_rng_state().clone()

    replacement, _ = propose(weights, context, identities, available)

    assert not replacement.requires_grad
    assert torch.equal(weights.grad, torch.ones_like(weights))
    assert context.grad is None
    assert torch.equal(random, torch.random.get_rng_state())
    for original, current in zip(before, (weights, context, identities, available), strict=True):
        assert torch.equal(original, current)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_proposals_restore_assignment_dtype_without_losing_row_mass(dtype):
    weights, context, identities, available = example()
    weights = weights.to(dtype)
    replacement, rows = propose(weights, context, identities, available)
    assert replacement.dtype == dtype
    assert rows.numel() > 0
    torch.testing.assert_close(replacement.logsumexp(-1), weights.logsumexp(-1))


def test_true_ties_retain_the_entire_original_row():
    weights = torch.tensor([[4.0, -2.0, 1.0]])
    context = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]]).log()

    replacement, rows = propose(weights, context, torch.tensor([0, 0]), torch.ones(3, dtype=torch.bool))

    assert rows.numel() == 0
    assert torch.equal(replacement, weights)


def test_retired_columns_are_excluded_but_their_row_mass_is_preserved():
    weights, context, identities, _ = example()
    weights[:, 1] = torch.tensor([50.0, 10.0, 1.0, -4.0])
    context[:, 1] = 1000.0
    available = torch.tensor([True, False, True])
    errors, proposals = brute_force(weights, context, identities, available)
    errors[:, ~available] = torch.inf

    replacement, rows = propose(weights, context, identities, available)

    assert rows.numel() > 0
    assert not replacement[rows].argmax(-1).eq(1).any()
    torch.testing.assert_close(replacement.logsumexp(-1), weights.logsumexp(-1))
    for identity in rows.tolist():
        torch.testing.assert_close(
            replacement[identity].double(), proposals[identity, errors[identity].argmin().item()]
        )


@pytest.mark.parametrize("kind", ["empty_vocabulary", "empty_evidence", "no_available"])
def test_absent_evidence_or_candidates_preserve_assignments(kind):
    weights, context, identities, available = example()
    if kind == "empty_vocabulary":
        weights, context, identities = weights[:0], context[:0], identities[:0]
    elif kind == "empty_evidence":
        context, identities = context[:0], identities[:0]
    else:
        available.zero_()
    replacement, rows = propose(weights, context, identities, available)
    assert rows.numel() == 0
    assert torch.equal(replacement, weights)


@pytest.mark.parametrize("unknown", [-1, 4])
def test_unknown_identity_ids_are_rejected(unknown):
    weights, context, identities, available = example()
    identities[0] = unknown
    with pytest.raises(IndexError, match="known rows"):
        propose(weights, context, identities, available)


@pytest.mark.parametrize("empty_rank", [False, True])
def test_global_counts_and_errors_include_remote_evidence(monkeypatch, empty_rank):
    weights, context, identities, available = example()
    expected, expected_rows = propose(weights, context, identities, available)
    split = 0 if empty_rank else 1
    remote_errors, _ = brute_force(weights, context[split:], identities[split:], available)
    remote_counts = torch.bincount(identities[split:], minlength=len(weights))
    calls = []

    def reduce(value):
        calls.append(value.clone())
        return value + (remote_counts if value.dtype == torch.int64 else remote_errors)

    monkeypatch.setattr("relflow.tensorfields.membership.all_reduce_sum", reduce)
    replacement, rows = propose(weights, context[:split], identities[:split], available)

    assert len(calls) == 2
    assert rows.tolist() == expected_rows.tolist()
    torch.testing.assert_close(replacement, expected)


@pytest.mark.parametrize("offset", [-10000.0, 10000.0])
def test_extreme_common_log_mass_remains_finite(offset):
    weights, context, identities, available = example()
    replacement, rows = propose(weights + offset, context, identities, available)
    assert rows.numel() > 0
    assert torch.isfinite(replacement).all()
    torch.testing.assert_close(replacement.logsumexp(-1), (weights + offset).logsumexp(-1))


def test_zero_probability_groups_are_valid_context_evidence():
    weights, _, identities, available = example()
    context = torch.full((len(identities), weights.shape[-1]), -torch.inf)
    context[:, 0] = 0.0
    replacement, _ = propose(weights, context, identities, available)
    assert torch.isfinite(replacement).all()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_invalid_available_context_is_rejected(bad):
    weights, context, identities, available = example()
    context[0] = bad
    with pytest.raises(ValueError, match="positive probability mass"):
        propose(weights, context, identities, available)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_assignment_mass_is_rejected(bad):
    weights, context, identities, available = example()
    weights[0, 0] = bad
    with pytest.raises(ValueError, match="assignment log weights must be finite"):
        propose(weights, context, identities, available)
