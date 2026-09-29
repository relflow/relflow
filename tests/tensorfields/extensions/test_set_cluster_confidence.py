"""Unknown inputs are trainable fallbacks, not corrupted reconstruction answers."""

from types import SimpleNamespace

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions import cluster
from relflow.tensorfields.extensions import set as sets
from relflow.tensorfields.shared.reconstruction import Metrics


def model(kind, *, p_unavailable=0.0, mask=True):
    request = (
        rf.Set(p_unavailable=p_unavailable, mask=mask)
        if kind == "set"
        else rf.Cluster(n_clusters=(2, 4), p_unavailable=p_unavailable, mask=mask)
    )
    result = rf.Model(d_model=8, n_heads=2, n_layers=1, x=rf.Number, label=request)
    labels = [["a"], ["b"]] if kind == "set" else ["a", "b"]
    result.encode(pa.table({"x": [0.0, 0.0], "label": labels}), strata="train")
    result.track = lambda names, value: value
    return result


def evaluate(module, labels, logits, strata=rf.Strata.test):
    # Encode without admission, including when testing the training objective.
    batch = module.encode(pa.table({"x": [0.0] * len(labels), "label": labels}), strata="test")["/label"]
    kind = module.schema.requests["/label"].type
    key = rf.TensorKey.content if kind == "set" else rf.TensorKey.cluster
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
    return (sets.loss if kind == "set" else cluster.loss)(module, prediction, batch, strata)


@pytest.mark.parametrize("kind", ["set", "cluster"])
def test_input_augmentation_preserves_pristine_answers(kind):
    module = model(kind, p_unavailable=1.0, mask=False)
    labels = [["a", "b"], [], None] if kind == "set" else ["a", "b", None]
    field = module.encode(pa.table({"x": [0.0] * 3, "label": labels}), strata="train")["/label"]
    supervised = model(kind, p_unavailable=1.0)
    answers = supervised.encode(pa.table({"x": [0.0] * 3, "label": labels}), strata="train")["/label"].targets[
        rf.TensorKey.content
    ]
    if kind == "set":
        assert not field.content["membership"].any()
        assert field.content["unavailable"].reshape(-1).tolist() == [2, 0, 0]
        assert answers["membership"].sum() == 2
        assert not answers["unavailable"].any()
    else:
        assert field.content.reshape(-1)[:2].tolist() == [-1, -1]
        assert answers.reshape(-1)[:2].tolist() == [0, 1]


def test_set_unknown_counts_preserve_set_semantics_and_reach_encoder():
    module = model("set", mask=False)
    labels = [["a", "new", "new", "other", None], ["other", "new", "a"], [], ["new"], None]
    batch = module.encode(pa.table({"x": [0.0] * 5, "label": labels}), strata="test")["/label"]
    assert batch.content["unavailable"].reshape(-1).tolist() == [2, 2, 0, 1, 0]
    assert torch.equal(batch.content["membership"][0], batch.content["membership"][1])
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


@pytest.mark.parametrize("kind", ["set", "cluster"])
def test_scores_do_not_depend_on_batch_partition(kind):
    labels = [["a"], ["a", "new"], ["new"], [], ["b"], None] if kind == "set" else ["a", "new", "new", "b", "a", None]
    width = 8 if kind == "set" else 4
    logits = torch.randn(6, width)
    joint, split = model(kind), model(kind)
    split.load_state_dict(joint.state_dict())
    evaluate(joint, labels, logits)
    for start, end in ((0, 1), (1, 3), (3, 6)):
        evaluate(split, labels[start:end], logits[start:end])
    first = joint.nodes["/label"].decoder.metrics["test_metrics"].compute()
    second = split.nodes["/label"].decoder.metrics["test_metrics"].compute()
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


def test_cluster_unknown_targets_do_not_change_assignment_or_adaptive_state():
    module = model("cluster")
    embedder = module.nodes["/label"].embedder
    usage, committed, adherence = (
        value.clone() for value in (embedder.usage_ema, embedder.committed, embedder.adherence_ema)
    )
    logits = torch.randn(6, 4, requires_grad=True)
    objective = evaluate(module, ["new"] * 6, logits, rf.Strata.train)
    objective.backward()
    assert torch.isfinite(objective)
    assert not logits.grad.any()
    assert not embedder.embeddings["cluster"].weight.grad.any()
    torch.testing.assert_close(embedder.usage_ema, usage)
    torch.testing.assert_close(embedder.committed, committed)
    torch.testing.assert_close(embedder.adherence_ema, adherence)
    scores = module.nodes["/label"].decoder.metrics["train_metrics"].compute()
    assert scores["targets.unavailable"] == 6
    assert scores["coverage.content"] == 0
    assert torch.isnan(scores["loss.content"])


def test_cluster_balancing_uses_known_rows_and_proper_unweighted_label_objective():
    base, mixed = model("cluster"), model("cluster")
    mixed.load_state_dict(base.state_dict())
    mixed.nodes["/label"].embedder.counters["content"].counts[0] = 1000
    known_logits = torch.randn(4, 4, requires_grad=True)
    mixed_logits = torch.cat((known_logits.detach(), torch.full((7, 4), 30.0))).requires_grad_()
    evaluate(base, ["a", "b", "a", "b"], known_logits, rf.Strata.train).backward()
    evaluate(mixed, ["a", "b", "a", "b"] + ["new"] * 7, mixed_logits, rf.Strata.train).backward()
    torch.testing.assert_close(known_logits.grad, mixed_logits.grad[:4])
    assert not mixed_logits.grad[4:].any()
    left, right = (module.nodes["/label"].embedder for module in (base, mixed))
    torch.testing.assert_close(left.usage_ema, right.usage_ema)
    torch.testing.assert_close(left.committed, right.committed)
    torch.testing.assert_close(left.adherence_ema, right.adherence_ema)
    assert not right.embeddings["cluster"].weight.grad[2:].any()


def test_cluster_balancing_can_escape_saturated_initial_assignments():
    module = model("cluster")
    logits = torch.tensor([[1000.0, 0.0, 0.0, 0.0]] * 4, requires_grad=True)
    objective = evaluate(module, ["a", "b", "a", "b"], logits, rf.Strata.train)
    objective.backward()
    assert torch.isfinite(objective)
    # Probability-space Sinkhorn underflows the unused columns to zero and
    # cannot revive them; a finite-logit assignment still needs balancing.
    assert logits.grad[:, 0].mean() > 0.1
    assert logits.grad[:, 1].mean() < -0.1


@pytest.mark.parametrize("kind", ["set", "cluster"])
def test_empty_vocabulary_has_differentiable_zero_content_loss(kind):
    request = rf.Set(mask=True) if kind == "set" else rf.Cluster(n_clusters=4, mask=True)
    module = rf.Model(d_model=8, n_heads=2, n_layers=1, x=rf.Number, label=request)
    module.track = lambda names, value: value
    logits = torch.randn(2, 8 if kind == "set" else 4, requires_grad=True)
    labels = [["new"], []] if kind == "set" else ["new", "new"]
    objective = evaluate(module, labels, logits)
    objective.backward()
    assert torch.isfinite(objective)
    assert not logits.grad.any()
    scores = module.nodes["/label"].decoder.metrics["test_metrics"].compute()
    assert torch.isnan(scores["loss.content"])


def test_registered_metrics_share_one_callback_and_reset_each_stage():
    module = rf.Model(
        d_model=8,
        n_heads=2,
        n_layers=1,
        category=rf.Category(mask=True),
        tags=rf.Set(mask=True),
        group=rf.Cluster(n_clusters=4, mask=True),
    )
    assert sum(isinstance(callback, Metrics) for callback in module.configure_callbacks()) == 1
    for node in module.nodes.values():
        if getattr(node, "decoder", None) is not None:
            metric = node.decoder.metrics["test_metrics"]
            metric.counts.share_memory_().fill_(2)
    Metrics().on_test_epoch_start(None, module)
    for node in module.nodes.values():
        if getattr(node, "decoder", None) is not None:
            metric = node.decoder.metrics["test_metrics"]
            assert not metric.counts.any()
            assert not metric.counts.is_shared()


def test_unknown_only_epoch_does_not_revive_or_merge_from_stale_cluster_statistics():
    module = model("cluster")
    embedder = module.nodes["/label"].embedder
    decoder = module.nodes["/label"].decoder
    with torch.no_grad():
        embedder.adherence_ema.fill_(1)
        embedder.committed[:3] = True
        embedder.embeddings["content"].weight[:, 1].copy_(embedder.embeddings["content"].weight[:, 0])
        decoder.linears["cluster"].weight[1].copy_(decoder.linears["cluster"].weight[0])
    before = {key: value.clone() for key, value in embedder.state_dict().items() if isinstance(value, torch.Tensor)}
    evaluate(module, ["new"] * 6, torch.randn(6, 4), rf.Strata.train)
    trainer = SimpleNamespace(current_epoch=0, is_global_zero=True)
    cluster.ClusterMergeCallback().on_train_epoch_end(trainer, module)
    cluster.ClusterReviveCallback().on_train_epoch_end(trainer, module)
    for key, value in before.items():
        torch.testing.assert_close(value, embedder.state_dict()[key])


def test_mixed_vocabulary_metrics_include_a_distributed_objective_empty_rank():
    selection = rf.Mask(query="selected", reconstruct=True)
    module = rf.Model(
        d_model=8,
        n_heads=2,
        n_layers=1,
        batch_size=2,
        x=rf.Number,
        label=rf.Category(mask=selection),
        tags=rf.Set(mask=selection),
        group=rf.Cluster(bounds=(2, 4), mask=selection),
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
        strategy="ddp_spawn",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    result = trainer.validate(
        module, datamodule=rf.ArrowDataModule(model=module, validate=source, num_workers=0), verbose=False
    )[0]
    for name in ("label", "tags", "group"):
        assert result[f".{name}/validate.targets.known"] == 4
        assert result[f".{name}/validate.targets.unavailable"] == 4
        assert result[f".{name}/validate.coverage.content"] == 0.5


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
    module = rf.Model(d_model=8, n_heads=2, n_layers=1, tags=rf.Set(p_unavailable=0.0))
    module.encode(pa.table({"tags": [[known]]}), strata="train")
    field = module.encode(pa.table({"tags": [[known, unknown, unknown, None], [unknown], []]}), strata="test")["/tags"]
    assert field.content["unavailable"].reshape(-1).tolist() == [1, 1, 0]
    assert field.content["membership"].sum(dim=-1).reshape(-1).tolist() == [1, 0, 0]


@pytest.mark.parametrize(
    "accelerator",
    ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_set_and_cluster_train_with_bfloat16_mixed_precision(accelerator):
    module = rf.Model(
        d_model=8,
        n_heads=2,
        n_layers=1,
        batch_size=4,
        x=rf.Number,
        tags=rf.Set(p_unavailable=0.5, mask=rf.Mask(rate=0.5, reconstruct=True)),
        group=rf.Cluster(bounds=(2, 4), p_unavailable=0.5, mask=True),
    )
    module.optimizer = rf.adamw(learning_rate=0.001)
    source = pa.table(
        {
            "x": [0.0, 1.0, 2.0, 3.0] * 4,
            "tags": [["a"], ["b", "new"], [], ["new"]] * 4,
            "group": ["a", "b", "new", "new"] * 4,
        }
    )
    trainer = lit.Trainer(
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
    trainer.fit(module, datamodule=rf.ArrowDataModule(model=module, train=source, validate=source, num_workers=0))
    assert trainer.global_step == 1
    for name in ("tags", "group"):
        scores = module.nodes[f"/{name}"].decoder.metrics["validate_metrics"].compute()
        assert torch.isfinite(scores["loss.content"])
