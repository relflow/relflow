"""Group's own identity proposals commit or roll back coherently across ranks."""

import copy
import os
from datetime import timedelta

import pyarrow as pa
import pytest
import torch
import torch.distributed as distributed

import relflow as rf
from relflow.structs.enums import TensorKey
from relflow.tensorfields import grouping
from relflow.tensorfields.extensions.group import resolve_group_embedder
from tests.distributed import launch

ADDRESS = "/identity"
CASES = (
    "accepted",
    "rejected",
    "nonfinite_training",
    "exception_training",
    "nonfinite_candidate",
    "exception_candidate",
    "frozen",
    "empty_calibration",
)


def exercise(directory):
    """Keep both ranks alive through successful proposals and local head failures."""
    torch.set_num_threads(1)
    distributed.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=int(os.environ["RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
        timeout=timedelta(seconds=30),
    )
    rank = distributed.get_rank()
    results = {}
    for case in CASES:
        torch.manual_seed(9381)
        model = rf.Model.xs(
            d_model=8,
            batch_size=128,
            identity=rf.Group(n_groups=3, mask=True, p_unavailable=0),
        )
        model.encode(pa.table({"identity": ["a", "b", "unobserved", "absent"]}), strata="train")
        embedder = resolve_group_embedder(model, ADDRESS)
        weight = embedder.embeddings[TensorKey.group.name].weight
        with torch.no_grad():
            weight[:4].copy_(torch.tensor([[2.0, -1, -2], [-2, 3, -1], [-1, -2, 4], [3, -1, 0]]))
            weight[-1].copy_(torch.tensor([0.5, -0.5, 1.0]))
        optimizer = torch.optim.AdamW([weight], amsgrad=True)
        optimizer.state[weight] = {
            "step": torch.tensor(13.0),
            "exp_avg": torch.full_like(weight, 3),
            "exp_avg_sq": torch.full_like(weight, 5),
            "max_exp_avg_sq": torch.full_like(weight, 7),
        }
        weight.grad = torch.full_like(weight, 11)
        if case == "frozen" and rank == 1:
            weight.requires_grad_(False)

        # Every phase spans three real model forwards. A failure in its first
        # chunk must not strand the peer while it evaluates subsequent chunks.
        train = pa.table({"identity": ["a", "b"] * 150})
        validate = pa.table({"identity": ["a", "b"] * 150})
        if case == "empty_calibration":
            training = grouping.Calibration(512, 1024**2)
            validation = grouping.Calibration(512, 1024**2)
            if rank == 0:
                training.add(train)
                validation.add(validate)
            else:
                assert training.rows == validation.rows == 0
            gathered_train = training.gather(512, 1024**2)
            gathered_validate = validation.gather(512, 1024**2)
            assert gathered_train is not None and gathered_train.equals(train)
            assert gathered_validate is not None and gathered_validate.equals(validate)
            train, validate = gathered_train, gathered_validate

        calls = 0

        def head(module, inputs, output):
            nonlocal calls
            calls += 1
            torch.rand(3)
            failure_call = 4 if case.endswith("training") else 7
            failing = rank == 1 and calls == failure_call
            if failing and case.startswith("exception"):
                raise RuntimeError("rank-one Group head failure")
            if failing and case.startswith("nonfinite"):
                return torch.full_like(output, float("nan"))

            # Fix only Group's own decoder logits. The actual reconstruction
            # likelihood, one-row proposal search, and validation gate all run.
            predictions = torch.full_like(output, -8)
            flattened = predictions.reshape(-1, embedder.size)
            identities = torch.arange(len(flattened), device=output.device).remainder(2)
            training_phase = 4 <= calls <= 6
            groups = identities if case == "rejected" and not training_phase else 1 - identities
            flattened.scatter_(1, groups.unsqueeze(-1), 8)
            return predictions

        model.train()
        decoder = model.nodes[ADDRESS].decoder
        decoder.eval()
        frozen = decoder.linears[TensorKey.state.name].weight
        frozen.requires_grad_(False)
        trainability = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        modes = {layer: layer.training for layer in model.modules()}
        parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
        before = weight.detach().clone()
        history = copy.deepcopy(optimizer.state[weight])
        buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
        vocabulary = embedder.vocab.snapshot()
        rng = torch.get_rng_state().clone()
        handle = decoder.linears[TensorKey.group.name].register_forward_hook(head)
        try:
            event = grouping.refine(model, ADDRESS, train, validate, [optimizer])
        finally:
            handle.remove()

        assert event["accepted"] == (case in {"accepted", "empty_calibration"})
        assert event["objective"] == "group_identity_nll"
        assert all(layer.training == training for layer, training in modes.items())
        assert all(parameter.requires_grad == trainability[name] for name, parameter in model.named_parameters())
        assert torch.equal(torch.get_rng_state(), rng)
        assert embedder.vocab.snapshot() == vocabulary
        assert not frozen.requires_grad
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
        for name, value in model.named_parameters():
            if value is not weight:
                torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
        if event["accepted"]:
            assert calls == 9
            assert event["changed_ids"] == 2
            assert event["proposed"] < event["before"]
            assert weight[:2].argmax(-1).tolist() == [1, 0]
            torch.testing.assert_close(weight[2:], before[2:], rtol=0, atol=0)
            torch.testing.assert_close(weight[:2].logsumexp(-1), before[:2].logsumexp(-1))
            assert optimizer.state[weight]["step"].item() == 13
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                assert not optimizer.state[weight][name][:2].any()
                torch.testing.assert_close(optimizer.state[weight][name][2:], history[name][2:], rtol=0, atol=0)
            assert not weight.grad[:2].any() and weight.grad[2:].eq(11).all()
        else:
            torch.testing.assert_close(weight, before, rtol=0, atol=0)
            for name, value in history.items():
                torch.testing.assert_close(optimizer.state[weight][name], value, rtol=0, atol=0)
            assert weight.grad.eq(11).all()
            if case.startswith("nonfinite"):
                assert "finite available-group probability mass" in event["reason"]
            elif case.startswith("exception"):
                assert "rank-one Group head failure" in event["reason"]
            elif case == "frozen":
                assert "trainable Group objective" in event["reason"]
                assert calls == 0
            else:
                assert calls == 9
                assert event["proposed"] > event["before"]
        results[case] = {
            "event": event,
            "weight": weight.detach().clone(),
            "gradient": weight.grad.clone(),
            "history": copy.deepcopy(optimizer.state[weight]),
            "head_calls": calls,
        }
    torch.save(results, directory / f"rank-{rank}.pt")


@pytest.mark.skipif(not distributed.is_available() or not distributed.is_gloo_available(), reason="requires CPU Gloo")
def test_group_refinement_coordinates_own_objective_failures_frozen_ranks_and_empty_calibration(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    launch(exercise, tmp_path)
    left, right = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    assert tuple(left) == tuple(right) == CASES
    for case in CASES:
        assert left[case]["event"] == right[case]["event"]
        for name in ("weight", "gradient"):
            torch.testing.assert_close(left[case][name], right[case][name], rtol=0, atol=0)
        for name, value in left[case]["history"].items():
            torch.testing.assert_close(value, right[case]["history"][name], rtol=0, atol=0)
    for phase in ("training", "candidate"):
        for failure in ("nonfinite", "exception"):
            case = f"{failure}_{phase}"
            assert left[case]["head_calls"] > right[case]["head_calls"]
