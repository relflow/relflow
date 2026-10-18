"""Adaptive groups retain identity routing and valid optimizer history."""

import math

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.group import GroupMergeCallback, GroupReviveCallback, GroupUsageCallback, loss

ADDRESS = "/identity"


def model(bounds, *, revive_temperature=0):
    configured = rf.Model.xs(
        d_model=8,
        identity=rf.Group(
            bounds=bounds,
            ema_decay=0,
            revive_temperature=revive_temperature,
            p_unavailable=0,
            embed=True,
            mask=rf.Mask(query="selected", reconstruct=True),
        ),
    )
    configured.encode(pa.table({"identity": ["a", "b", "c"], "selected": [True] * 3}), strata="train")
    configured.track = lambda names, value: value
    return configured


@pytest.fixture
def trainer():
    return lit.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )


def prepare_merge(configured):
    """Make groups zero and one redundant, with identities owned by each."""
    for token, probabilities in (("a", [0.6, 0.3, 0.1]), ("b", [0.2, 0.7, 0.1]), ("c", [0.1, 0.1, 0.8])):
        rf.Group.assign(configured, ADDRESS, token, probabilities)
    node = configured.nodes[ADDRESS]
    with torch.no_grad():
        content = node.embedder.embeddings["content"].weight
        content.zero_()
        content[0, :2] = 1
        content[1, 2] = 1
        decoder = node.decoder.linears["group"].weight
        decoder.zero_()
        decoder[:2, 0] = -1
        decoder[2, 1] = -2
        node.embedder.committed.fill_(True)
        node.embedder.usage_ema.copy_(torch.tensor([0.6, 0.3, 0.1]))
    node.decoder.metrics["train_metrics"].update(torch.tensor([3, 3, 0]), torch.zeros(2))


def test_identical_uniform_predictions_do_not_create_eight_distinct_groups(trainer):
    configured = model((2, 8))
    source = pa.table({"identity": ["a", "b", "c"] * 4, "selected": [True] * 12})
    batch = configured.encode(source, strata="test")[ADDRESS]
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                rf.TensorKey.state: torch.zeros(12, 1, len(rf.Tokens)),
                rf.TensorKey.group: torch.zeros(12, 1, 8),
            },
            batch_size=[12],
        ),
    )

    assert torch.isfinite(loss(configured, prediction, batch, rf.Strata.train))
    GroupUsageCallback().on_train_batch_end(trainer, configured, None, batch, 0)

    # The same uncertain prediction for every identity contains no evidence for extra groups.
    assert len(rf.Group.status(configured, ADDRESS)["committed"]) == 2


def test_balance_cross_entropy_does_not_overweight_individual_observations():
    configured = model((2, 2))
    source = pa.table({"identity": ["a", "b"] * 4, "selected": [True] * 8})
    batch = configured.encode(source, strata="test")[ADDRESS]
    logits = torch.tensor([[100.0, 0.0]] * 7 + [[2.0, 0.0]], requires_grad=True)
    balanced = []

    def track(names, value):
        if names[-2:] == (rf.Metric.loss, rf.TensorKey.group):
            balanced.append(value)
        return value

    configured.track = track
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {rf.TensorKey.state: torch.zeros(8, 1, len(rf.Tokens)), rf.TensorKey.group: logits.reshape(8, 1, 2)},
            batch_size=[8],
        ),
    )
    assert torch.isfinite(loss(configured, prediction, batch, rf.Strata.train))
    assert len(balanced) == 1
    gradient = torch.autograd.grad(balanced[0], logits)[0]

    # A normalized CE target has gradient p - q with L1 <= 2; the batch mean divides by N.
    assert torch.isfinite(gradient).all()
    assert (gradient.abs().sum(dim=-1) <= 2 / len(logits) + 1e-6).all()


def test_single_group_preserves_nonuniform_learned_identity_priors():
    configured = rf.Model.xs(d_model=8, identity=rf.Group(bounds=1, mask=True))
    configured.encode(pa.table({"identity": ["a", "b"]}), strata="train")
    configured.track = lambda names, value: value
    with torch.no_grad():
        configured.nodes[ADDRESS].embedder.embeddings["group"].weight[:2].copy_(torch.tensor([[0.9], [0.1]]).log())
    batch = configured.encode(pa.table({"identity": ["a"]}), strata="test")[ADDRESS]
    state = torch.zeros(1, 1, len(rf.Tokens))
    state[..., rf.Tokens.valued.value] = 50
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict({rf.TensorKey.state: state, rf.TensorKey.group: torch.zeros(1, 1, 1)}, batch_size=[1]),
    )

    objective = loss(configured, prediction, batch, rf.Strata.test)
    scores = configured.nodes[ADDRESS].decoder.metrics.compute(rf.Strata.test)
    extension = rf.TENSORFIELDS["group"]
    datatype = extension.component(rf.Component.output)(configured, ADDRESS)
    content = extension.component(rf.Component.write)(configured, prediction, datatype).field("content")

    # Sharing one group must still allow unequal label frequencies. Normalizing each
    # one-column assignment row would erase this learned prior and force 50/50 output.
    assert {assignment["group"] for assignment in rf.Group.assignments(configured, ADDRESS).values()} == {0}
    assert content.field("value").to_pylist() == ["a"]
    probability = content.field("probability")[0].as_py()
    assert probability == pytest.approx(0.9)
    assert objective.item() == pytest.approx(-math.log(0.9))
    assert scores["nll.content"].item() == pytest.approx(-math.log(probability))


def test_bfloat16_writer_and_reconstruction_metrics_choose_the_same_identity():
    configured = model((2, 2))
    for token, group_id in (("a", 0), ("b", 1), ("c", 0)):
        rf.Group.assign(configured, ADDRESS, token, group_id)
    batch = configured.encode(pa.table({"identity": ["b"], "selected": [True]}), strata="test")[ADDRESS]
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                rf.TensorKey.state: torch.zeros(1, 1, len(rf.Tokens)),
                rf.TensorKey.group: torch.tensor([[[0.0, 0.001]]], dtype=torch.bfloat16),
            },
            batch_size=[1],
        ),
    )

    assert torch.isfinite(loss(configured, prediction, batch, rf.Strata.test))
    scores = configured.nodes[ADDRESS].decoder.metrics.compute(rf.Strata.test)
    extension = rf.TENSORFIELDS["group"]
    datatype = extension.component(rf.Component.output)(configured, ADDRESS)
    output = extension.component(rf.Component.write)(configured, prediction, datatype)

    # Rounding the posterior to bf16 before label decoding destroys this margin.
    assert scores["accuracy.content"].item() == 1
    assert output.field("group").field("value").to_pylist() == [1]
    assert output.field("content").field("value").to_pylist() == ["b"]


def test_revival_never_overwrites_occupied_uncommitted_groups():
    configured = model((1, 3), revive_temperature=1)
    node = configured.nodes[ADDRESS]
    for group_id, token in enumerate(("a", "b", "c")):
        rf.Group.assign(configured, ADDRESS, token, group_id)
    node.embedder.adherence_ema.fill_(1)
    assert rf.Group.status(configured, ADDRESS)["committed"] == (0,)
    assert {assignment["group"] for assignment in rf.Group.assignments(configured, ADDRESS).values()} == {0, 1, 2}

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        plan = GroupReviveCallback.plan(node.embedder, configured.schema.requests[ADDRESS], epoch=0)

    # Committed is the balancing subset. Both other groups still encode known identities.
    assert plan is None, f"revival would overwrite occupied group IDs: {plan['dead_ids']}"


def test_revival_can_reuse_retired_groups_which_no_longer_encode_identities():
    configured = model((1, 2), revive_temperature=1)
    node = configured.nodes[ADDRESS]
    for token in ("a", "b", "c"):
        rf.Group.assign(configured, ADDRESS, token, 1)
    node.embedder.retired[1] = True
    assert {assignment["group"] for assignment in rf.Group.assignments(configured, ADDRESS).values()} == {0}

    plan = GroupReviveCallback.plan(node.embedder, configured.schema.requests[ADDRESS], epoch=0)

    assert plan is not None
    assert plan["dead_ids"] == [1]


def test_merge_redirects_loser_identities_and_encoder_routing_to_the_survivor(trainer):
    configured = model((1, 3))
    prepare_merge(configured)
    node = configured.nodes[ADDRESS]
    assert rf.Group.assignments(configured, ADDRESS)["b"]["group"] == 1
    assert node.embedder.embeddings["group"].weight[:3].max() < 0

    GroupMergeCallback().on_train_epoch_end(trainer, configured)

    assert rf.Group.status(configured, ADDRESS)["committed"] == (0, 2)
    assignments = rf.Group.assignments(configured, ADDRESS)
    assert {token: assignment["group"] for token, assignment in assignments.items()} == {"a": 0, "b": 0, "c": 2}
    for token, expected in (("a", (0.9, 0.0, 0.1)), ("b", (0.9, 0.0, 0.1)), ("c", (0.2, 0.0, 0.8))):
        assert assignments[token]["probabilities"] == pytest.approx(expected)
    routed = []
    handle = node.embedder.embeddings["content"].register_forward_pre_hook(
        lambda module, inputs: routed.extend(inputs[0].argmax(dim=-1).reshape(-1).tolist())
    )
    try:
        configured.predict(pa.table({"identity": ["a", "b", "c"], "selected": [False] * 3}))
    finally:
        handle.remove()
    assert routed == [0, 0, 2]


def test_negative_decoder_logits_cannot_predict_a_retired_group(trainer):
    configured = model((1, 3))
    prepare_merge(configured)
    GroupMergeCallback().on_train_epoch_end(trainer, configured)
    payload = configured.nodes[ADDRESS].decoder.decode(torch.ones(3, 1, 8))
    assert (payload[rf.TensorKey.group][..., 0] < 0).all()
    extension = rf.TENSORFIELDS["group"]
    datatype = extension.component(rf.Component.output)(configured, ADDRESS)
    output = extension.component(rf.Component.write)(configured, Prediction(address=ADDRESS, payload=payload), datatype)

    assert output.field("group").field("value").to_pylist() == [0, 0, 0]
    assert all(torch.isfinite(torch.tensor(value["probability"])) for value in output.field("content").to_pylist())


def test_retired_group_reconstruction_has_finite_loss_and_gradients(trainer):
    configured = model((1, 3))
    prepare_merge(configured)
    GroupMergeCallback().on_train_epoch_end(trainer, configured)
    batch = configured.encode(pa.table({"identity": ["a", "b", "c"], "selected": [True] * 3}), strata="test")[ADDRESS]
    payload = configured.nodes[ADDRESS].decoder.decode(torch.ones(3, 1, 8))

    objective = loss(configured, Prediction(address=ADDRESS, payload=payload), batch, rf.Strata.train)
    assert torch.isfinite(objective)
    objective.backward()

    gradients = [parameter.grad for parameter in configured.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert torch.isfinite(configured.nodes[ADDRESS].embedder.statistics).all()


def test_retirement_status_and_assignments_survive_checkpoint(trainer, tmp_path):
    configured = model((1, 3))
    prepare_merge(configured)
    before = rf.Group.status(configured, ADDRESS)
    GroupMergeCallback().on_train_epoch_end(trainer, configured)
    status = rf.Group.status(configured, ADDRESS)
    assert before["retired"] == ()
    assert status["retired"] == (1,)
    assert status["committed"] == (0, 2)
    assignments = rf.Group.assignments(configured, ADDRESS)
    assert all(assignment["probabilities"][1] == 0 for assignment in assignments.values())
    for assignment in (1, [0.5, 0.5, 0.0]):
        with pytest.raises(ValueError, match="retired"):
            rf.Group.assign(configured, ADDRESS, "new", assignment)
    assert rf.Group.vocabulary(configured, ADDRESS) == ("a", "b", "c")

    restored = rf.Model.load(configured.save(tmp_path / "retired.ckpt"))

    assert rf.Group.status(restored, ADDRESS) == status
    assert rf.Group.assignments(restored, ADDRESS) == assignments
    with pytest.raises(ValueError, match="retired"):
        rf.Group.assign(restored, ADDRESS, "new", 1)


@pytest.mark.parametrize("operation", ["merge", "revive"])
@pytest.mark.parametrize("frozen", ["all", "assignments", "prototypes", "decoder"])
def test_adaptive_replacements_preserve_frozen_group_state(operation, frozen, trainer):
    configured = model((1, 3) if operation == "merge" else (1, 2), revive_temperature=1)
    node = configured.nodes[ADDRESS]
    affected = {
        "assignments": node.embedder.embeddings["group"].weight,
        "prototypes": node.embedder.embeddings["content"].weight,
        "decoder": node.decoder.linears["group"].weight,
    }
    optimizer = torch.optim.AdamW(node.parameters(), lr=0.01, amsgrad=True, fused=False)
    for parameter in node.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    trainer.optimizers = [optimizer]
    if operation == "merge":
        prepare_merge(configured)
    else:
        node.embedder.retired[1] = True
        node.decoder.metrics["train_metrics"].update(torch.tensor([3, 3, 0]), torch.zeros(2))
    for name, parameter in affected.items():
        if frozen in ("all", name):
            parameter.requires_grad_(False)
    parameters = {name: parameter.detach().clone() for name, parameter in node.named_parameters()}
    buffers = {name: buffer.detach().clone() for name, buffer in node.named_buffers()}
    history = {
        parameter: {key: value.clone() for key, value in state.items()} for parameter, state in optimizer.state.items()
    }

    callback = GroupMergeCallback() if operation == "merge" else GroupReviveCallback()
    callback.on_train_epoch_end(trainer, configured)
    # Direct application must enforce the same atomic invariant as the callback.
    if operation == "merge":
        callback.apply(node.embedder, node.decoder, {"loser": 1, "winner": 0}, optimizers=[optimizer])
    else:
        plan = {
            "dead_ids": [1],
            "donors": [0],
            "noise_assign": torch.zeros(1, affected["assignments"].shape[0]),
            "noise_content": torch.zeros(1, affected["prototypes"].shape[0]),
            "noise_group": torch.zeros(1, affected["decoder"].shape[1]),
        }
        callback.apply(node.embedder, node.decoder, plan, optimizers=[optimizer])

    for name, parameter in node.named_parameters():
        torch.testing.assert_close(parameter, parameters[name], rtol=0, atol=0)
    for name, buffer in node.named_buffers():
        torch.testing.assert_close(buffer, buffers[name], rtol=0, atol=0)
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            torch.testing.assert_close(value, history[parameter][key], rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["merge", "revive"])
def test_adaptive_replacements_do_not_reuse_dead_groups_adamw_momentum(operation, trainer):
    configured = model((1, 3) if operation == "merge" else (1, 2), revive_temperature=1)
    node = configured.nodes[ADDRESS]
    affected = (
        (node.embedder.embeddings["group"].weight, 1),
        (node.embedder.embeddings["content"].weight, 1),
        (node.decoder.linears["group"].weight, 0),
    )
    optimizer = torch.optim.AdamW(configured.parameters(), lr=0.01, weight_decay=0, amsgrad=True, fused=False)
    for parameter in configured.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    trainer.optimizers = [optimizer]
    configured._trainer = trainer
    if operation == "merge":
        prepare_merge(configured)
    else:
        node.decoder.metrics["train_metrics"].update(torch.tensor([3, 3, 0]), torch.zeros(2))
        node.embedder.retired[1] = True
    # Donor history is zero. Resetting a replacement or inheriting donor moments are both valid.
    for parameter, axis in affected:
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            optimizer.state[parameter][key].select(axis, 0).zero_()
            assert optimizer.state[parameter][key].select(axis, 1).count_nonzero()
    history = {
        parameter: {key: value.clone() for key, value in state.items()} for parameter, state in optimizer.state.items()
    }
    before = [parameter.detach().select(axis, 1).clone() for parameter, axis in affected]

    callback = GroupMergeCallback() if operation == "merge" else GroupReviveCallback()
    callback.on_train_epoch_end(trainer, configured)

    if operation == "revive":
        assert not bool(node.embedder.retired[1])
        assert bool(node.embedder.committed[1])
    affected_parameters = {parameter for parameter, _ in affected}
    for parameter, state in history.items():
        for key, value in state.items():
            if key == "step" or parameter not in affected_parameters:
                torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
    untouched = 2 if operation == "merge" else 0
    for parameter, axis in affected:
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            torch.testing.assert_close(
                optimizer.state[parameter][key].select(axis, untouched),
                history[parameter][key].select(axis, untouched),
                rtol=0,
                atol=0,
            )
    replacements = [
        (parameter, axis, parameter.detach().select(axis, 1).clone())
        for (parameter, axis), old in zip(affected, before, strict=True)
        if not torch.equal(parameter.detach().select(axis, 1), old)
    ]
    assert replacements, "the controlled callback did not replace any group parameters"
    for parameter in configured.parameters():
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    for parameter, axis, replacement in replacements:
        # With zero fresh gradients and zero donor history, old loser momentum must not move replacements.
        torch.testing.assert_close(parameter.detach().select(axis, 1), replacement, rtol=0, atol=0)
