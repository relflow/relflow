"""Unknown inputs are trainable fallbacks, not corrupted reconstruction answers."""

import math
from datetime import timedelta

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
import torch.distributed as distributed
from lightning.pytorch.strategies import DDPStrategy
from tensordict import TensorDict

import relflow as rf
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions import group
from relflow.tensorfields.extensions import set as sets
from relflow.tensorfields.shared.reconstruction import Metrics
from tests.distributed import launch


def model(kind, *, p_unavailable=0.0, mask=True):
    request = (
        rf.Set(p_unavailable=p_unavailable, mask=mask)
        if kind == "set"
        else rf.Group(n_groups=4, p_unavailable=p_unavailable, mask=mask)
    )
    result = rf.Model.xs(d_model=8, x=rf.Number, label=request)
    labels = [["a"], ["b"]] if kind == "set" else ["a", "b"]
    result.encode(pa.table({"x": [0.0, 0.0], "label": labels}), strata="train")
    result.track = lambda names, value: value
    return result


def evaluate(module, labels, logits, strata=rf.Strata.test, *, encode_strata=rf.Strata.test):
    batch = module.encode(pa.table({"x": [0.0] * len(labels), "label": labels}), strata=encode_strata)["/label"]
    kind = module.schema.requests["/label"].type
    key = rf.TensorKey.content if kind == "set" else rf.TensorKey.group
    prediction = Prediction(
        address="/label",
        payload=TensorDict(
            {
                rf.TensorKey.state: torch.zeros(len(labels), 1, len(rf.Tokens), requires_grad=True),
                key: logits.unsqueeze(1),
            },
            batch_size=[len(labels)],
        ),
    )
    return (sets.loss if kind == "set" else group.loss)(module, prediction, batch, strata)


@pytest.mark.parametrize("kind", ["set", "group"])
def test_input_augmentation_preserves_pristine_answers(kind):
    module = model(kind, p_unavailable=1.0, mask=False)
    labels = [["a", "b"], [], None] if kind == "set" else ["a", "b", None]
    field = module.encode(pa.table({"x": [0.0] * 3, "label": labels}), strata="train")["/label"]
    supervised = model(kind, p_unavailable=1.0)
    answers = supervised.encode(pa.table({"x": [0.0] * 3, "label": labels}), strata="train")["/label"].targets[
        rf.TensorKey.content
    ]
    if kind == "set":
        assert field.content["indices"].eq(-1).all()
        assert field.content["unavailable"].reshape(-1).tolist() == [2, 0, 0]
        assert answers["indices"][0, 0].tolist() == [0, 1]
        assert answers["indices"][1:].eq(-1).all()
        assert not answers["unavailable"].any()
    else:
        assert field.content.reshape(-1)[:2].tolist() == [-1, -1]
        assert answers.reshape(-1)[:2].tolist() == [0, 1]


def test_group_state_loss_ignores_simulated_input_unavailability():
    module = model("group", p_unavailable=1.0, mask=rf.Mask(reconstruct=True))
    angles = torch.zeros(4, 1, requires_grad=True)

    objective = evaluate(module, ["a", "b", "b", "a"], angles, rf.Strata.train, encode_strata=rf.Strata.train)

    expected = torch.nn.functional.cross_entropy(
        torch.zeros(4, len(rf.Tokens)), torch.full((4,), rf.Tokens.valued.value)
    )
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert angles.grad is None
    assert module.nodes["/label"].embedder.embeddings["group"].weight.grad is None


def test_set_unknown_counts_preserve_set_semantics_and_reach_encoder():
    module = model("set", mask=False)
    labels = [["a", "new", "new", "other", None], ["other", "new", "a"], [], ["new"], None]
    batch = module.encode(pa.table({"x": [0.0] * 5, "label": labels}), strata="test")["/label"]
    assert batch.content["unavailable"].reshape(-1).tolist() == [2, 2, 0, 1, 0]
    indices = batch.content["indices"][0, 0]
    assert indices[indices.ge(0)].tolist() == [0]
    assert torch.equal(batch.content["indices"][0], batch.content["indices"][1])
    embedder = module.nodes["/label"].embedder
    with torch.no_grad():
        embedder.unavailable.fill_(1)
    embedded = embedder(batch.take(torch.arange(5))).payload
    torch.testing.assert_close(embedded[0], embedded[1])
    assert not torch.equal(embedded[2], embedded[3])
    assert not torch.equal(embedded[3], embedded[4])
    assert rf.Set.vocabulary(module, "/label") == ("a", "b")


def test_set_partial_unknown_targets_keep_known_positive_and_negative_gradients():
    module = model("set")
    logits = torch.randn(4, 8, requires_grad=True)
    evaluate(module, [["a", "new"], ["new"], [], None], logits).backward()
    expected = logits.detach().clone().requires_grad_()
    torch.nn.functional.binary_cross_entropy_with_logits(
        expected[:3, :2], torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    ).backward()
    torch.testing.assert_close(logits.grad, expected.grad)
    scores = module.nodes["/label"].decoder.metrics["test_metrics"].compute()
    assert scores["targets.known"] == 1
    assert scores["targets.unavailable"] == 2
    assert scores["coverage.content"] == pytest.approx(1 / 3)
    assert scores["sets.valued"] == 3
    assert scores["sets.complete"] == 1


def test_set_scores_do_not_depend_on_batch_partition():
    labels = [["a"], ["a", "new"], ["new"], [], ["b"], None]
    logits = torch.randn(6, 8)
    joint, split = model("set"), model("set")
    split.load_state_dict(joint.state_dict())
    evaluate(joint, labels, logits)
    for start, end in ((0, 1), (1, 3), (3, 6)):
        evaluate(split, labels[start:end], logits[start:end])
    first = joint.nodes["/label"].decoder.metrics["test_metrics"].compute()
    second = split.nodes["/label"].decoder.metrics["test_metrics"].compute()
    assert first["targets.known"] > 0
    for name, value in first.items():
        torch.testing.assert_close(value, second[name], equal_nan=True)


def test_set_unused_capacity_does_not_inflate_bit_accuracy_or_dilute_loss():
    small, large = model("set"), model("set")
    evaluate(small, [["a"]], torch.tensor([[-2.0, -2.0]]))
    evaluate(large, [["a"]], torch.full((1, 1024), -2.0))
    left = small.nodes["/label"].decoder.metrics["test_metrics"].compute()
    right = large.nodes["/label"].decoder.metrics["test_metrics"].compute()
    for name, value in left.items():
        torch.testing.assert_close(value, right[name], equal_nan=True)
    assert left["accuracy.content"] == 0.5
    assert left["accuracy.set"] == 0


def test_group_unknown_targets_do_not_change_assignments():
    module = model("group")
    embedder = module.nodes["/label"].embedder
    assignments = embedder.embeddings["group"].weight.detach().clone()
    logits = torch.randn(6, 1, requires_grad=True)
    objective = evaluate(module, ["new"] * 6, logits, rf.Strata.train)
    objective.backward()
    assert torch.isfinite(objective)
    assert logits.grad is None
    assert embedder.embeddings["group"].weight.grad is None
    torch.testing.assert_close(embedder.embeddings["group"].weight, assignments)


def test_group_known_and_unknown_targets_use_only_state_loss():
    base, mixed = model("group"), model("group")
    mixed.load_state_dict(base.state_dict())
    mixed.nodes["/label"].embedder.counters["content"].counts[0] = 1000
    known_logits = torch.randn(4, 1, requires_grad=True)
    mixed_logits = torch.cat((known_logits.detach(), torch.full((7, 1), 30.0))).requires_grad_()
    known_loss = evaluate(base, ["a", "b", "a", "b"], known_logits, rf.Strata.train)
    mixed_loss = evaluate(mixed, ["a", "b", "a", "b"] + ["new"] * 7, mixed_logits, rf.Strata.train)
    torch.testing.assert_close(known_loss, mixed_loss)
    known_loss.backward()
    mixed_loss.backward()
    assert known_logits.grad is None
    assert mixed_logits.grad is None
    assert base.nodes["/label"].embedder.embeddings["group"].weight.grad is None
    assert mixed.nodes["/label"].embedder.embeddings["group"].weight.grad is None


@pytest.mark.parametrize("kind", ["set", "group"])
def test_empty_vocabulary_keeps_state_loss_finite(kind):
    request = rf.Set(mask=True) if kind == "set" else rf.Group(n_groups=4, mask=True)
    module = rf.Model.xs(d_model=8, x=rf.Number, label=request)
    module.track = lambda names, value: value
    logits = torch.randn(2, 8 if kind == "set" else 1, requires_grad=True)
    labels = [["new"], []] if kind == "set" else ["new", "new"]
    objective = evaluate(module, labels, logits)
    objective.backward()
    assert torch.isfinite(objective)
    if kind == "set":
        assert logits.grad is not None and not logits.grad.any()
        scores = module.nodes["/label"].decoder.metrics["test_metrics"].compute()
        assert torch.isnan(scores["loss.content"])
    else:
        assert logits.grad is None
        assert not hasattr(module.nodes["/label"].decoder, "metrics")


def test_group_does_not_register_reconstruction_metrics():
    module = rf.Model.xs(d_model=8, label=rf.Group(n_groups=4, mask=True))
    assert not hasattr(module.nodes["/label"].decoder, "metrics")
    assert not any(isinstance(callback, Metrics) for callback in module.configure_callbacks())


def test_registered_metrics_share_one_callback_and_reset_each_stage():
    module = rf.Model.xs(
        d_model=8,
        category=rf.Category(mask=True),
        tags=rf.Set(mask=True),
        group=rf.Group(n_groups=4, mask=True),
    )
    assert sum(isinstance(callback, Metrics) for callback in module.configure_callbacks()) == 1
    assert not hasattr(module.nodes["/group"].decoder, "metrics")
    for address in ("/category", "/tags"):
        metric = module.nodes[address].decoder.metrics["test_metrics"]
        metric.counts.share_memory_().fill_(2)
    Metrics().on_test_epoch_start(None, module)
    for address in ("/category", "/tags"):
        metric = module.nodes[address].decoder.metrics["test_metrics"]
        assert not metric.counts.any()
        assert not metric.counts.is_shared()


def mixed_metrics():
    selection = rf.Mask(query="selected", reconstruct=True)
    module = rf.Model.xs(
        d_model=8,
        batch_size=2,
        x=rf.Number,
        label=rf.Category(mask=selection),
        tags=rf.Set(mask=selection),
        group=rf.Group(n_groups=4, mask=selection),
    )
    module.encode(
        pa.table(
            {
                "x": [0.0, 1.0],
                "label": ["a", "b"],
                "tags": [["a"], ["b"]],
                "group": ["a", "b"],
                "selected": [True, True],
            }
        ),
        strata="train",
    )
    labels = ["a" if i % 4 == 0 else "unseen" for i in range(16)]
    source = pa.table(
        {
            "x": [float(i) for i in range(16)],
            "label": labels,
            "tags": [[label] for label in labels],
            "group": labels,
            "selected": [i % 2 == 0 for i in range(16)],
        }
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(start_method="popen", process_group_backend="gloo", timeout=timedelta(seconds=60)),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    result = trainer.validate(
        module, datamodule=rf.ArrowDataModule(model=module, validate=source, num_workers=0), verbose=False
    )[0]
    for name in ("label", "tags"):
        assert result[f".{name}/validate.targets.known"] == 4
        assert result[f".{name}/validate.targets.unavailable"] == 4
        assert result[f".{name}/validate.coverage.content"] == 0.5
    if trainer.global_rank == 0:
        assert math.isfinite(result[".group/validate.loss.state"])
    else:
        assert ".group/validate.loss.state" not in result
        assert ".group/validate.accuracy.state" not in result
    for name in ("targets.known", "targets.unavailable", "coverage.content", "accuracy.content", "loss.content"):
        assert f".group/validate.{name}" not in result


@pytest.mark.integration
@pytest.mark.skipif(not distributed.is_available() or not distributed.is_gloo_available(), reason="requires CPU Gloo")
def test_mixed_vocabulary_metrics_include_a_distributed_objective_empty_rank():
    launch(mixed_metrics)


def test_set_checkpoint_preserves_learned_unavailable_embedding(tmp_path):
    module = model("set", mask=False)
    with torch.no_grad():
        module.nodes["/label"].embedder.unavailable.fill_(0.75)
    path = tmp_path / "set.rf"
    module.save(path)
    restored = rf.Model.load(path)
    torch.testing.assert_close(
        restored.nodes["/label"].embedder.unavailable, module.nodes["/label"].embedder.unavailable
    )
    assert rf.Set.vocabulary(restored, "/label") == ("a", "b")


@pytest.mark.parametrize("known,unknown", [(False, True), (1, 3), (1.5, 3.5), (b"a", b"new"), ("a", "new")])
def test_set_unknown_members_are_distinct_within_each_arrow_atom_family(known, unknown):
    module = rf.Model.xs(d_model=8, tags=rf.Set(p_unavailable=0.0))
    module.encode(pa.table({"tags": [[known]]}), strata="train")
    field = module.encode(pa.table({"tags": [[known, unknown, unknown, None], [unknown], []]}), strata="test")["/tags"]
    assert field.content["unavailable"].reshape(-1).tolist() == [1, 1, 0]
    indices = field.content["indices"].reshape(3, -1)
    assert indices[0][indices[0].ge(0)].tolist() == [0]
    assert indices[1:].eq(-1).all()


@pytest.mark.integration
@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_set_and_group_train_with_bfloat16_mixed_precision(accelerator):
    module = rf.Model.xs(
        d_model=8,
        batch_size=4,
        x=rf.Number,
        tags=rf.Set(p_unavailable=0.5, mask=rf.Mask(rate=0.5, reconstruct=True)),
        group=rf.Group(n_groups=4, p_unavailable=0.5, mask=True),
    )
    source = pa.table(
        {
            "x": [0.0, 1.0, 2.0, 3.0] * 4,
            "tags": [["a"], ["b", "new"], [], ["new"]] * 4,
            "group": ["a", "b", "new", "new"] * 4,
        }
    )
    module.fit(
        rf.ArrowDataModule(model=module, train=source, validate=source, num_workers=0),
        learning_rate=0.001,
        early_stopping=False,
        rollback=False,
        accelerator=accelerator,
        devices=1,
        precision="bf16-mixed",
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    assert module.trainer.global_step == 1
    scores = module.nodes["/tags"].decoder.metrics["validate_metrics"].compute()
    assert torch.isfinite(scores["loss.content"])
    assert torch.isfinite(module.trainer.callback_metrics[".group/validate.loss.state"])
    assert ".group/validate.loss.content" not in module.trainer.callback_metrics
