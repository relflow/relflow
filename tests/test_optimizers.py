from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
from io import BytesIO

import pytest
import torch

from relflow import adamw


def _module() -> torch.nn.Module:
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("linear", torch.nn.Linear(3, 2)),
                ("norm", torch.nn.LayerNorm(2)),
            ]
        )
    )


def _parameter_ids(parameters: Iterable[torch.nn.Parameter]) -> set[int]:
    return {id(parameter) for parameter in parameters}


def test_adamw_groups_bias_1d_and_named_parameters_without_decay() -> None:
    module = _module()
    parameters = dict(module.named_parameters())

    optimizer = adamw(
        learning_rate=0.123,
        weight_decay=0.5,
        betas=(0.8, 0.9),
        eps=1e-7,
    )(module)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 0.123
    assert optimizer.defaults["betas"] == (0.8, 0.9)
    assert optimizer.defaults["eps"] == 1e-7
    assert optimizer.defaults["fused"] is True
    assert optimizer.defaults["weight_decay"] == 0.5
    assert [group["weight_decay"] for group in optimizer.param_groups] == [0.5, 0.0]

    decay_group, no_decay_group = optimizer.param_groups
    assert _parameter_ids(decay_group["params"]) == {id(parameters["linear.weight"])}
    assert _parameter_ids(no_decay_group["params"]) == {
        id(parameters["linear.bias"]),
        id(parameters["norm.weight"]),
        id(parameters["norm.bias"]),
    }


def test_adamw_can_decay_bias_and_1d_parameters() -> None:
    module = _module()
    parameters = dict(module.named_parameters())

    optimizer = adamw(
        learning_rate=1e-3,
        weight_decay=0.2,
        decay_bias=True,
        decay_1d=True,
        no_decay_name_fragments=(),
    )(module)

    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["weight_decay"] == 0.2
    assert _parameter_ids(optimizer.param_groups[0]["params"]) == {id(parameter) for parameter in parameters.values()}


def test_adamw_skips_frozen_parameters() -> None:
    module = _module()
    parameters = dict(module.named_parameters())
    parameters["linear.weight"].requires_grad_(False)

    optimizer = adamw(learning_rate=1e-3)(module)

    optimized_parameters = {
        parameter_id for group in optimizer.param_groups for parameter_id in _parameter_ids(group["params"])
    }
    assert id(parameters["linear.weight"]) not in optimized_parameters


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fused_adamw_matches_updates_and_checkpoint_continuation(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    torch.manual_seed(29)
    reference = _module().to(device)
    reference.register_parameter("unused", torch.nn.Parameter(torch.randn(2, device=device)))
    candidate = deepcopy(reference)
    baseline = adamw(learning_rate=1e-3, fused=None)(reference)
    optimized = adamw(learning_rate=1e-3)(candidate)
    assert baseline.defaults["fused"] is None
    assert optimized.defaults["fused"] is True

    for step in range(8):
        inputs = torch.randn(16, 3, device=device)
        targets = torch.randn(16, 2, device=device)
        for module, optimizer in ((reference, baseline), (candidate, optimized)):
            optimizer.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(module(inputs), targets).backward()
            if step % 3 == 0:
                module.linear.bias.grad = None
            optimizer.step()

        for expected, actual in zip(reference.parameters(), candidate.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
            expected_state = baseline.state.get(expected, {})
            actual_state = optimized.state.get(actual, {})
            assert actual_state.keys() == expected_state.keys()
            for name, value in actual_state.items():
                torch.testing.assert_close(value.cpu(), expected_state[name].cpu(), atol=1e-6, rtol=1e-5)
        assert candidate.unused.grad is None
        assert candidate.unused not in optimized.state

        if step == 3:
            checkpoint = BytesIO()
            torch.save({"model": candidate.state_dict(), "optimizer": optimized.state_dict()}, checkpoint)
            checkpoint.seek(0)
            restored = torch.load(checkpoint, weights_only=True)
            candidate = deepcopy(reference)
            candidate.load_state_dict(restored["model"])
            optimized = adamw(learning_rate=1e-3)(candidate)
            optimized.load_state_dict(restored["optimizer"])


def test_unfused_adamw_checkpoint_resumes_with_original_backend_and_grad_scaler() -> None:
    torch.manual_seed(41)
    reference = _module()
    baseline = adamw(learning_rate=1e-3, fused=None)(reference)
    inputs = torch.randn(8, 3)
    targets = torch.randn(8, 2)
    torch.nn.functional.mse_loss(reference(inputs), targets).backward()
    baseline.step()

    candidate = deepcopy(reference)
    restored = adamw(learning_rate=1e-3, fused=None)(candidate)
    restored.load_state_dict(deepcopy(baseline.state_dict()))
    assert restored.defaults["fused"] is None
    assert all(group["fused"] is None for group in restored.param_groups)

    for module, optimizer in ((reference, baseline), (candidate, restored)):
        optimizer.zero_grad(set_to_none=True)
        scaler = torch.amp.GradScaler("cpu")
        loss = torch.nn.functional.mse_loss(module(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    for expected, actual in zip(reference.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)
        for name, value in baseline.state[expected].items():
            torch.testing.assert_close(restored.state[actual][name], value, atol=0.0, rtol=0.0)
