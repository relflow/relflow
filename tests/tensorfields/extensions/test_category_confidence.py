"""Probability objectives and coverage counts do not invent OOV supervision."""

from datetime import timedelta
from pathlib import Path

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from tensordict import TensorDict

import relflow as rf
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions.category import Metrics, Reconstruction, loss


def model():
    result = rf.Model(
        d_model=8,
        n_heads=2,
        n_layers=1,
        x=rf.Number,
        label=rf.Category(topk=[2], mask=True),
    )
    result.encode(pa.table({"x": [0.0, 0.0], "label": ["a", "b"]}), strata="train")
    result.track = lambda names, value: value
    return result


def evaluate(module, labels, logits):
    fields = module.encode(pa.table({"x": [0.0] * len(labels), "label": labels}), strata="test")
    batch = fields["/label"]
    prediction = Prediction(
        address="/label",
        payload=TensorDict(
            {
                rf.TensorKey.state: torch.zeros(len(labels), 1, len(rf.Tokens), requires_grad=True),
                rf.TensorKey.content: logits.reshape(len(labels), 1, -1),
            },
            batch_size=[len(labels)],
        ),
    )
    return loss(module, prediction, batch, rf.Strata.test)


def test_unavailable_simulation_preserves_reconstruction_targets():
    module = rf.Model(
        d_model=8,
        n_heads=2,
        n_layers=1,
        entity=rf.Category(p_unavailable=1.0),
        label=rf.Category(p_unavailable=1.0, mask=True),
    )
    encoded = module.encode(pa.table({"label": ["a", "b", None], "entity": ["a", "b", None]}), strata="train")
    field = encoded["/label"]
    assert encoded["/entity"].content.reshape(-1)[:2].tolist() == [-1, -1]
    assert field.targets[rf.TensorKey.content].reshape(-1)[:2].tolist() == [0, 1]
    assert field.targets[rf.TensorKey.state].reshape(-1).tolist() == [
        rf.Tokens.valued,
        rf.Tokens.valued,
        rf.Tokens.null,
    ]


def test_category_objective_is_unweighted_and_unknown_targets_have_no_content_gradient():
    module = model()
    module.nodes["/label"].embedder.counters["content"].counts[:2].copy_(torch.tensor([901, 101]))
    logits = torch.randn(4, 8, requires_grad=True)
    result = evaluate(module, ["a", "b", "unseen", None], logits)
    result.backward()
    expected = logits.detach().clone().requires_grad_()
    torch.nn.functional.cross_entropy(expected[:2, :2], torch.tensor([0, 1])).backward()
    torch.testing.assert_close(logits.grad, expected.grad)


def test_all_unknown_content_is_zero_for_backward_but_undefined_for_evaluation():
    module = model()
    logits = torch.randn(3, 8, requires_grad=True)
    result = evaluate(module, ["unseen"] * 3, logits)
    result.backward()
    assert torch.isfinite(result)
    assert torch.equal(logits.grad, torch.zeros_like(logits))
    metrics = module.nodes["/label"].decoder.metrics["test_metrics"].compute()
    assert metrics["targets.known"] == 0
    assert metrics["targets.unavailable"] == 3
    assert metrics["coverage.content"] == 0
    assert metrics["accuracy.all"] == 0
    for name in ("accuracy.content", "nll.content", "loss.content", "accuracy.top2"):
        assert torch.isnan(metrics[name])


def test_single_populated_label_probability_is_conditional_not_learned_competence():
    module = rf.Model(d_model=8, n_heads=2, n_layers=1, x=rf.Number, label=rf.Category(mask=True))
    module.encode(pa.table({"x": [0.0], "label": ["only-label"]}), strata="train")
    result = module.predict(pa.table({"x": [10.0]}))["predictions"].to_pylist()[0]["/label"]["content"]
    assert result["value"] == "only-label"
    assert result["probability"] == 1.0


def test_coverage_and_scores_are_invariant_to_batch_partition_and_unused_capacity():
    labels = ["a", "unseen", "unseen", "b", "a", None, "unseen"]
    logits = torch.tensor([[2.0, 0.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0]] * len(labels))
    joint, split = model(), model()
    evaluate(joint, labels, logits)
    for start, end in ((0, 1), (1, 3), (3, 7)):
        evaluate(split, labels[start:end], logits[start:end])
    first = joint.nodes["/label"].decoder.metrics["test_metrics"].compute()
    second = split.nodes["/label"].decoder.metrics["test_metrics"].compute()
    for name, value in first.items():
        torch.testing.assert_close(value, second[name])
    assert first["coverage.content"] == 0.5
    assert first["accuracy.content"] == pytest.approx(2 / 3)
    assert first["accuracy.all"] == pytest.approx(1 / 3)
    assert first["accuracy.top2"] == 1
    expected = torch.nn.functional.cross_entropy(logits[[0, 3, 4], :2], torch.tensor([0, 1, 0]))
    assert first["nll.content"] == pytest.approx(expected.item())
    assert first["loss.content"] == pytest.approx(expected.item())


def reduce_statistics(rank, directory):
    distributed.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        metric = Reconstruction([2])
        metric.update(
            torch.tensor([2, 3, 1, 2]) if rank == 0 else torch.tensor([0, 7, 0, 0]),
            torch.tensor([6.0, 4.0]) if rank == 0 else torch.zeros(2),
        )
        result = metric.compute()
        assert result["targets.known"] == 2
        assert result["targets.unavailable"] == 8
        assert result["coverage.content"] == pytest.approx(0.2)
        assert result["accuracy.content"] == 0.5
        assert result["accuracy.all"] == pytest.approx(0.1)
        assert result["nll.content"] == 2
        assert result["loss.content"] == 3
        metric.reset()
        assert not metric.counts.any()
        assert not metric.sums.any()
    finally:
        distributed.destroy_process_group()


def test_reconstruction_counts_reduce_globally_with_an_all_unknown_rank(tmp_path):
    multiprocessing.spawn(reduce_statistics, args=(str(tmp_path),), nprocs=2, join=True)


def test_epoch_start_resets_and_detaches_spawn_shared_metric_storage():
    module = model()
    metric = module.nodes["/label"].decoder.metrics["test_metrics"]
    metric.counts.share_memory_().fill_(5)
    metric.sums.share_memory_().fill_(5)
    previous = metric.counts
    assert previous.is_shared()
    Metrics().on_test_epoch_start(None, module)
    assert not metric.counts.is_shared()
    assert not metric.sums.is_shared()
    assert not metric.counts.any()
    assert not metric.sums.any()
    assert previous.eq(5).all()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_metrics_keep_integer_counts_and_accelerator_supported_float_sums(device):
    metric = Reconstruction([]).to(device)
    metric.update(torch.tensor([20_000_001, 40_000_002, 20_000_001], device=device), torch.ones(2, device=device))
    result = metric.compute()
    assert result["targets.known"].item() == 20_000_001
    assert result["coverage.content"].item() == 0.5
    assert metric.counts.dtype == torch.int64
    assert metric.sums.dtype == torch.float32


def test_lightning_validation_counts_include_an_objective_empty_rank():
    module = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
        x=rf.Number,
        label=rf.Category(mask=rf.Mask(query="selected", reconstruct=True)),
    )
    module.encode(pa.table({"x": [0.0, 1.0], "label": ["a", "b"], "selected": [True, True]}), strata="train")
    source = pa.table(
        {
            "x": [float(i) for i in range(16)],
            "label": ["a" if i % 4 == 0 else "unseen" for i in range(16)],
            "selected": [i % 2 == 0 for i in range(16)],
        }
    )
    data = rf.ArrowDataModule(model=module, validate=source, num_workers=0)
    trainer = lit.Trainer(
        accelerator="cpu",
        devices=2,
        strategy="ddp_spawn",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    measured = trainer.validate(module, datamodule=data, verbose=False)[0]
    assert measured[".label/validate.targets.known"] == 4
    assert measured[".label/validate.targets.unavailable"] == 4
    assert measured[".label/validate.coverage.content"] == 0.5
