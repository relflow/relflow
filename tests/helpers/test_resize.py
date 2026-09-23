from copy import deepcopy

import pytest
import torch
from torch import nn

from relflow.helpers.resize import Resize


def test_metadata_publication_follows_successful_validation_and_commit():
    embedding = nn.Embedding(2, 3)
    original = embedding.weight
    optimizer = torch.optim.AdamW(embedding.parameters())
    optimizer.state[original]["unsupported_history"] = torch.ones_like(original)
    seen = []
    plan = Resize([optimizer])
    plan.embedding(embedding, 4)
    plan.publish(lambda: seen.append(embedding.num_embeddings))

    with pytest.raises(ValueError, match="unsupported"):
        plan.commit()
    assert embedding.weight is original
    assert embedding.num_embeddings == 2
    assert not seen

    optimizer.state[original].clear()
    plan.commit()
    plan.commit()
    assert seen == [4]
    with pytest.raises(RuntimeError, match="after commit"):
        plan.publish(lambda: seen.append(0))


@pytest.mark.parametrize(
    ("optimizer_type", "options"),
    [
        (torch.optim.Adam, {}),
        (torch.optim.Adam, {"amsgrad": True}),
        (torch.optim.AdamW, {}),
        (torch.optim.AdamW, {"amsgrad": True}),
        (torch.optim.SGD, {"momentum": 0.9}),
    ],
)
@pytest.mark.parametrize("tail", [False, True])
def test_embedding_preserves_rows_accumulated_gradients_and_optimizer_state(optimizer_type, options, tail):
    module = nn.Embedding(3, 4, dtype=torch.float64)
    untouched = nn.Parameter(torch.ones(2, dtype=torch.float64))
    optimizer = optimizer_type(
        [{"params": [module.weight], "lr": 0.01}, {"params": [untouched], "lr": 0.02}],
        **options,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    inputs = torch.arange(3)
    (module(inputs).square().sum() + untouched.sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module(inputs).square().sum().backward()

    old = module.weight
    weights = old.detach().clone()
    old_state = optimizer.state[old]
    state_values = deepcopy(old_state)
    untouched_state = optimizer.state[untouched]
    groups = optimizer.param_groups
    group_objects = list(groups)
    parameter_lists = [group["params"] for group in groups]
    scheduler_state = deepcopy(scheduler.state_dict())
    plan = Resize([optimizer])
    plan.embedding(module, 6, tail=tail)
    assert module.weight is old
    assert module.num_embeddings == 3
    assert optimizer.param_groups[0]["params"][0] is old
    assert optimizer.state[old] is old_state
    assert len(plan.parameters) == 1
    assert plan.parameters[0][0] is old
    new = plan.parameters[0][1]

    # Backward after staging must be included in the eventual replacement.
    (module(inputs) * 2).sum().backward()
    gradients = old.grad.clone()
    plan.commit()

    destinations = torch.tensor([0, 1, 5] if tail else [0, 1, 2])
    introduced = torch.tensor([2, 3, 4] if tail else [3, 4, 5])
    assert module.weight is new
    assert module.num_embeddings == 6
    assert new.dtype == old.dtype
    assert new.device == old.device
    assert torch.equal(new[destinations], weights)
    assert torch.equal(new.grad[destinations], gradients)
    assert torch.count_nonzero(new.grad[introduced]) == 0
    assert torch.equal(old, weights)
    assert torch.equal(old.grad, gradients)
    assert old not in optimizer.state
    assert optimizer.state[untouched] is untouched_state
    for name, value in optimizer.state[new].items():
        if name == "step":
            assert value is old_state[name]
            assert torch.equal(value, state_values[name])
        else:
            assert torch.equal(value[destinations], state_values[name])
            assert torch.count_nonzero(value[introduced]) == 0
            assert torch.equal(old_state[name], state_values[name])

    assert optimizer.param_groups is groups
    assert len(groups) == 2
    for index, group in enumerate(groups):
        assert group is group_objects[index]
        assert group["params"] is parameter_lists[index]
    assert groups[0]["params"][0] is new
    assert groups[1]["params"][0] is untouched
    assert scheduler.optimizer is optimizer
    assert scheduler.state_dict() == scheduler_state

    module(introduced).sum().backward()
    assert torch.equal(new.grad[destinations], gradients)
    assert torch.equal(new.grad[introduced], torch.ones_like(new.grad[introduced]))
    before = new.detach().clone()
    optimizer.step()
    scheduler.step()
    assert torch.all(new[introduced] != before[introduced])
    assert [group["lr"] for group in groups] == [0.005, 0.01]


@pytest.mark.parametrize("tail", [False, True])
@pytest.mark.parametrize("padding", [None, 0, 2])
def test_embedding_uses_pytorch_initialization_and_preserves_padding_and_module_options(tail, padding):
    module = nn.Embedding(3, 4, padding_idx=padding, max_norm=3.0, norm_type=1.0, scale_grad_by_freq=True)
    module.eval()
    with torch.no_grad():
        module.weight.copy_(torch.arange(12).reshape(3, 4))
    before = module.weight.detach().clone()
    mapped_padding = 5 if tail and padding == 2 else padding
    with torch.random.fork_rng():
        torch.manual_seed(78)
        expected = nn.Embedding(6, 4, padding_idx=mapped_padding)
        torch.manual_seed(78)
        plan = Resize()
        plan.embedding(module, 6, tail=tail)
    assert module.padding_idx == padding
    plan.commit()
    introduced = torch.tensor([2, 3, 4] if tail else [3, 4, 5])
    destinations = torch.tensor([0, 1, 5] if tail else [0, 1, 2])
    assert torch.equal(module.weight[destinations], before)
    assert torch.equal(module.weight[introduced], expected.weight[introduced])
    assert module.padding_idx == mapped_padding
    assert module.max_norm == 3.0
    assert module.norm_type == 1.0
    assert module.scale_grad_by_freq
    assert not module.training
    if mapped_padding is not None:
        module(torch.tensor([mapped_padding])).sum().backward()
        assert torch.count_nonzero(module.weight.grad) == 0


@pytest.mark.parametrize("bias", [False, True])
def test_linear_preserves_weight_bias_gradients_and_history_and_initializes_new_outputs(bias):
    module = nn.Linear(4, 2, bias=bias, dtype=torch.float64)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.01, amsgrad=True)
    inputs = torch.arange(8, dtype=torch.float64).reshape(2, 4)
    module(inputs).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module(inputs).square().sum().backward()
    before = {name: parameter for name, parameter in module.named_parameters()}
    states = {name: deepcopy(optimizer.state[parameter]) for name, parameter in before.items()}
    with torch.random.fork_rng():
        torch.manual_seed(83)
        expected = nn.Linear(4, 5, bias=bias, dtype=torch.float64)
        torch.manual_seed(83)
        plan = Resize([optimizer])
        plan.linear(module, 5)
    assert module.out_features == 2
    assert all(getattr(module, name) is parameter for name, parameter in before.items())
    plan.commit()
    assert module.out_features == 5
    assert module.in_features == 4
    assert len(plan.parameters) == (2 if bias else 1)
    for name, new in module.named_parameters():
        old = before[name]
        assert torch.equal(new[:2], old)
        assert torch.equal(new[2:], getattr(expected, name)[2:])
        assert torch.equal(new.grad[:2], old.grad)
        assert torch.count_nonzero(new.grad[2:]) == 0
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            assert torch.equal(optimizer.state[new][key][:2], states[name][key])
            assert torch.count_nonzero(optimizer.state[new][key][2:]) == 0
    if not bias:
        assert module.bias is None
    module(inputs)[:, 2:].sum().backward()
    new_outputs = module(inputs)[:, 2:].detach().clone()
    optimizer.step()
    assert not torch.equal(module(inputs)[:, 2:], new_outputs)


def test_parameter_supports_arbitrary_row_mapping_for_extension_owned_tensors():
    module = nn.Module()
    old = nn.Parameter(torch.arange(12, dtype=torch.float64).reshape(3, 2, 2))
    module.register_parameter("table", old)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
    old.square().sum().backward()
    optimizer.step()
    old_gradients = old.grad.clone()
    old_momentum = optimizer.state[old]["momentum_buffer"].clone()
    values = torch.full((5, 2, 2), -7.0, dtype=old.dtype, requires_grad=True)
    indices = torch.tensor([4, 0, 3])
    plan = Resize([optimizer])
    new = plan.parameter(module, "table", values, indices)
    indices.fill_(1)
    plan.commit()
    assert module.table is new
    assert new.is_leaf
    assert new.requires_grad
    assert torch.equal(new[[4, 0, 3]], old)
    assert torch.equal(new[[1, 2]], values[[1, 2]])
    assert torch.equal(values, torch.full_like(values, -7))
    assert torch.equal(new.grad[[4, 0, 3]], old_gradients)
    assert torch.equal(optimizer.state[new]["momentum_buffer"][[4, 0, 3]], old_momentum)
    assert torch.count_nonzero(new.grad[[1, 2]]) == 0


@pytest.mark.parametrize(("initial", "current"), [(None, None), (None, 2.0), (7.0, None), (7.0, 2.0)])
def test_commit_reads_current_gradient_values_and_presence_after_external_synchronization(initial, current):
    module = nn.Embedding(3, 2)
    old = module.weight
    old.grad = None if initial is None else torch.full_like(old, initial)
    plan = Resize()
    plan.embedding(module, 6, tail=True)
    new = plan.parameters[0][1]
    assert new.grad is None

    # A caller may flush/average gradients and resolve cross-rank presence here.
    old.grad = None if current is None else torch.full_like(old, current)
    plan.commit()

    assert module.weight is new
    if current is None:
        assert new.grad is None
    else:
        assert torch.equal(new.grad[[0, 1, 5]], torch.full_like(old, current))
        assert torch.count_nonzero(new.grad[2:5]) == 0
        assert new.grad is not old.grad


def test_buffer_and_metadata_publish_caller_prepared_values_and_keep_persistence():
    module = nn.Module()
    module.size = 3
    module.register_buffer("counts", torch.tensor([7, 8, 9]))
    module.register_buffer("pending", torch.tensor([2, 3, 4]), persistent=False)
    counts = module.counts
    pending = module.pending
    values = torch.tensor([7, 8, 1, 1, 9])
    pending_values = torch.tensor([2, 3, 0, 0, 4])
    plan = Resize()
    plan.buffer(module, "counts", values)
    plan.buffer(module, "pending", pending_values)
    plan.attribute(module, "size", 5)
    assert module.counts is counts
    assert module.pending is pending
    assert module.size == 3
    assert plan.parameters == []
    plan.commit()
    assert module.counts is values
    assert module.pending is pending_values
    assert module.size == 5
    assert set(module.state_dict()) == {"counts"}
    assert set(dict(module.named_buffers())) == {"counts", "pending"}


@pytest.mark.parametrize("tail", [False, True])
def test_sparse_embedding_gradients_and_sgd_momentum_preserve_repeated_indices(tail):
    module = nn.Embedding(3, 2, sparse=True)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
    inputs = torch.tensor([2, 0, 2, 1])
    module(inputs).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    module(inputs).sum().backward()
    old = module.weight
    momentum = optimizer.state[old]["momentum_buffer"].to_dense()
    plan = Resize([optimizer])
    plan.embedding(module, 5, tail=tail)
    module(inputs).sum().backward()
    gradient = old.grad.to_dense()
    plan.commit()
    destinations = torch.tensor([0, 1, 4] if tail else [0, 1, 2])
    introduced = torch.tensor([2, 3] if tail else [3, 4])
    assert module.weight.grad.is_sparse
    assert torch.equal(module.weight.grad.to_dense()[destinations], gradient)
    assert torch.count_nonzero(module.weight.grad.to_dense()[introduced]) == 0
    state = optimizer.state[module.weight]["momentum_buffer"]
    assert state.is_sparse
    assert torch.equal(state.to_dense()[destinations], momentum)
    before = module.weight.detach().clone()
    module(introduced).sum().backward()
    optimizer.step()
    assert torch.all(module.weight[introduced] != before[introduced])


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        ("exp_avg", torch.zeros(2, 3), "must have shape"),
        ("exp_avg_sq", 3, "must have shape"),
        ("max_exp_avg_sq", torch.zeros(2, 1), "must have shape"),
        ("step", torch.zeros(2), "must be scalar"),
        ("factored_rows", torch.zeros(2), "unsupported"),
        ("unknown", torch.zeros(2), "unsupported"),
    ],
)
def test_incompatible_state_rejects_entire_plan_without_partial_mutation(name, value, error):
    embedding = nn.Embedding(2, 3)
    linear = nn.Linear(3, 2)
    linear.register_buffer("counts", torch.ones(2))
    first = torch.optim.SGD(embedding.parameters(), lr=0.1, momentum=0.9)
    second = torch.optim.AdamW(linear.parameters(), lr=0.1, amsgrad=True)
    linear(embedding(torch.arange(2))).sum().backward()
    first.step()
    second.step()
    # The invalid state follows valid migrations in both the parameter and optimizer lists.
    second.state[linear.bias][name] = value
    old_weight = embedding.weight
    old_linear_weight = linear.weight
    old_bias = linear.bias
    old_counts = linear.counts
    old_states = [(optimizer, list(optimizer.state.items())) for optimizer in (first, second)]
    old_gradients = [parameter.grad.clone() for parameter in (old_weight, old_linear_weight, old_bias)]
    plan = Resize([first, second])
    plan.embedding(embedding, 5, tail=True)
    plan.linear(linear, 5)
    plan.buffer(linear, "counts", torch.ones(5))
    with pytest.raises(ValueError, match=error):
        plan.commit()
    assert embedding.weight is old_weight
    assert embedding.num_embeddings == 2
    assert linear.weight is old_linear_weight
    assert linear.bias is old_bias
    assert linear.out_features == 2
    assert linear.counts is old_counts
    for optimizer, states in old_states:
        assert len(optimizer.state) == len(states)
        for parameter, state in states:
            assert optimizer.state[parameter] is state
    assert first.param_groups[0]["params"][0] is old_weight
    assert second.param_groups[0]["params"][0] is old_linear_weight
    assert second.param_groups[0]["params"][1] is old_bias
    for (old, new), gradient in zip(plan.parameters, old_gradients, strict=True):
        assert torch.equal(old.grad, gradient)
        assert new.grad is None


def test_uninitialized_optimizer_state_stays_lazy_and_frozen_parameters_stay_frozen():
    module = nn.Linear(3, 2)
    module.bias.requires_grad_(False)
    optimizer = torch.optim.AdamW(module.parameters(), lr=0.1)
    plan = Resize([optimizer])
    plan.linear(module, 5)
    plan.commit()
    assert not optimizer.state
    assert module.weight.grad is None
    assert module.bias.grad is None
    assert module.weight.requires_grad
    assert not module.bias.requires_grad
    before = module.weight.detach().clone()
    module(torch.ones(1, 3)).sum().backward()
    optimizer.step()
    assert torch.all(module.weight[2:] != before[2:])
    assert module.bias not in optimizer.state


def test_shared_parameter_migrates_in_each_optimizer_without_duplicating_groups():
    module = nn.Embedding(3, 2)
    first = torch.optim.Adam(module.parameters(), lr=0.01)
    second = torch.optim.SGD(module.parameters(), lr=0.1, momentum=0.9)
    module(torch.arange(3)).square().sum().backward()
    first.step()
    second.step()
    old = module.weight
    first_moment = first.state[old]["exp_avg"].clone()
    second_moment = second.state[old]["momentum_buffer"].clone()
    plan = Resize([first, second, first])
    plan.embedding(module, 5, tail=True)
    plan.commit()
    for optimizer, key, moment in ((first, "exp_avg", first_moment), (second, "momentum_buffer", second_moment)):
        assert old not in optimizer.state
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["params"][0] is module.weight
        assert torch.equal(optimizer.state[module.weight][key][[0, 1, 4]], moment)
        assert torch.count_nonzero(optimizer.state[module.weight][key][2:4]) == 0


@pytest.mark.parametrize("step", [4, 4.0, torch.tensor(4.0), torch.tensor([4.0])])
def test_scalar_steps_remain_unchanged_even_when_shape_matches_a_single_row_parameter(step):
    module = nn.Linear(2, 1)
    optimizer = torch.optim.Adam(module.parameters(), lr=0.01)
    module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.state[module.bias]["step"] = step
    plan = Resize([optimizer])
    plan.linear(module, 4)
    plan.commit()
    assert optimizer.state[module.bias]["step"] is step
    assert optimizer.state[module.bias]["exp_avg"].shape == (4,)


def test_same_size_is_noop_and_does_not_consume_rng():
    embedding = nn.Embedding(3, 2)
    linear = nn.Linear(2, 3)
    old_embedding = embedding.weight
    old_linear = linear.weight
    rng = torch.random.get_rng_state().clone()
    plan = Resize()
    plan.embedding(embedding, 3, tail=True)
    plan.linear(linear, 3)
    plan.commit()
    assert plan.parameters == []
    assert embedding.weight is old_embedding
    assert linear.weight is old_linear
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("size", [2, -1, 3.5, True])
def test_invalid_growth_rejects_before_staging(size):
    plan = Resize()
    with pytest.raises(ValueError, match="row size"):
        plan.embedding(nn.Embedding(3, 2), size)
    with pytest.raises(ValueError, match="row size"):
        plan.linear(nn.Linear(2, 3), size)
    assert plan.parameters == []


@pytest.mark.parametrize(
    ("shape", "dtype", "indices", "error"),
    [
        ((5, 4), torch.float32, None, "row axis"),
        ((2, 2), torch.float32, None, "row axis"),
        ((5, 2), torch.float64, None, "dtype"),
        ((5, 2), torch.float32, [0, 1], "one destination"),
        ((5, 2), torch.float32, [0, 1, 1], "unique"),
        ((5, 2), torch.float32, [0, 1, 5], "must be in"),
        ((5, 2), torch.float32, [0, 1, -1], "must be in"),
        ((5, 2), torch.float32, [0.0, 1.0, 2.0], "integers"),
    ],
)
def test_invalid_parameter_mapping_rejects_before_staging(shape, dtype, indices, error):
    module = nn.Embedding(3, 2)
    old = module.weight
    plan = Resize()
    with pytest.raises(ValueError, match=error):
        plan.parameter(module, "weight", torch.zeros(shape, dtype=dtype), indices)
    assert module.weight is old
    assert plan.parameters == []


def test_replacing_owner_after_staging_rejects_before_publication():
    first = nn.Embedding(3, 2)
    second = nn.Embedding(3, 2)
    old = first.weight
    plan = Resize()
    plan.embedding(first, 5)
    plan.embedding(second, 5)
    second.weight = nn.Parameter(torch.zeros_like(second.weight))
    with pytest.raises(ValueError, match="changed after staging"):
        plan.commit()
    assert first.weight is old
    assert first.num_embeddings == second.num_embeddings == 3


def test_commit_is_idempotent_and_successful_plan_cannot_be_reused():
    module = nn.Embedding(3, 2)
    plan = Resize()
    plan.embedding(module, 5)
    plan.commit()
    weight = module.weight
    plan.commit()
    assert module.weight is weight
    assert plan.parameters[0][1] is weight
    with pytest.raises(RuntimeError, match="after commit"):
        plan.embedding(module, 7)
