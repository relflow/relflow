"""Real, single-fit DDP growth across optimizer steps and prefetched Arrow batches."""

from collections import Counter
from copy import deepcopy
from datetime import timedelta
from weakref import ref

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel

import relflow as rf
from relflow.structs.enums import TensorKey, Tokens

LABELS = (("rank0-a", "rank0-b", "rank0-b", "rank0-a"), ("rank1-a", "rank1-b", "rank1-late", "rank1-late"))
FIELDS = ("category", "tags", "cluster")
CAPACITIES = (2, 4, 8, 8)


def optimizer(model):
    """Construct the public AdamW configuration inside each spawned rank."""
    return rf.adamw(learning_rate=0.01, weight_decay=0, fused=False)(model)


def schedule(model, optimizer):
    """A picklable factory whose scheduler must survive every resource replacement."""
    return {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9), "interval": "step"}


def parameters(model):
    """Include growing rows, decoder biases, and Cluster's fixed latent decoder."""
    result = {}
    for name in FIELDS:
        node = model.nodes[f"/{name}"]
        key = "cluster" if name == "cluster" else "content"
        result[f"{name}/embedding"] = node.embedder.embeddings[key].weight
        result[f"{name}/decoder.weight"] = node.decoder.linears[key].weight
        if node.decoder.linears[key].bias is not None:
            result[f"{name}/decoder.bias"] = node.decoder.linears[key].bias
    return result


def snapshot(trainer, model):
    optimizer = trainer.optimizers[0]
    return {
        "step": trainer.global_step,
        "epoch": trainer.current_epoch,
        "scheduler": deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict()),
        "lr": [group["lr"] for group in optimizer.param_groups],
        "weights": {name: value.detach().cpu().clone() for name, value in parameters(model).items()},
        "gradients": {
            name: None if value.grad is None else value.grad.detach().cpu().clone()
            for name, value in parameters(model).items()
        },
        "moments": {name: deepcopy(optimizer.state.get(value, {})) for name, value in parameters(model).items()},
        "fields": {
            name: {
                "vocabulary": list(model.nodes[f"/{name}"].embedder.vocab.snapshot()),
                "capacity": model.nodes[f"/{name}"].embedder.vocab.size,
                "counts": model.nodes[f"/{name}"].embedder.counters["content"].counts.cpu().clone(),
                "pending": model.nodes[f"/{name}"].embedder.counters["content"]._pending_counts.cpu().clone(),
            }
            for name in FIELDS
        },
    }


def assert_migration(before, after):
    """No optimizer/scheduler work may occur while the next batch is bound."""
    assert after["step"] == before["step"]
    assert after["scheduler"] == before["scheduler"]
    assert after["lr"] == before["lr"]
    for name, old in before["weights"].items():
        new = after["weights"][name]
        destinations = torch.arange(old.shape[0])
        if name == "cluster/embedding":
            destinations[-1] = new.shape[0] - 1
        assert torch.equal(new[destinations], old), name
        introduced = torch.ones(new.shape[0], dtype=torch.bool)
        introduced[destinations] = False
        previous = before["moments"][name]
        current = after["moments"][name]
        assert current.keys() == previous.keys(), name
        for key, values in previous.items():
            if key == "step":
                assert torch.equal(current[key], values), name
            else:
                assert torch.equal(current[key][destinations], values), (name, key)
                assert torch.count_nonzero(current[key][introduced]) == 0, (name, key)


class GrowthAudit(lit.Callback):
    """Audit inside each spawned rank; persist evidence after epoch counter sync."""

    def __init__(self, directory, bucket_view):
        self.directory = directory
        self.bucket_view = bucket_view
        self.starts = []
        self.ends = []
        self.rewraps = []
        self.sources = []

    def on_train_start(self, trainer, pl_module):
        torch.set_num_threads(1)
        torch.manual_seed(811 + trainer.global_rank)
        assert trainer.world_size == 2
        assert torch.distributed.get_backend() == "gloo"
        self.trainer = trainer
        self.model = pl_module
        self.optimizer = trainer.optimizers[0]
        self.scheduler = trainer.lr_scheduler_configs[0].scheduler
        self.groups = list(self.optimizer.param_groups)
        self.parameter_lists = [group["params"] for group in self.groups]
        self.wrapped = ref(trainer.strategy.model)
        self.previous_parameters = parameters(pl_module)
        self.initial = snapshot(trainer, pl_module)
        assert all(field["capacity"] == 1 for field in self.initial["fields"].values())
        assert all(not field["vocabulary"] for field in self.initial["fields"].values())

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        assert trainer is self.trainer
        assert pl_module is self.model
        assert pl_module.trainer is self.trainer
        assert trainer.optimizers[0] is self.optimizer
        assert trainer.lr_scheduler_configs[0].scheduler is self.scheduler
        assert self.scheduler.optimizer is self.optimizer
        assert trainer.global_step == batch_idx // trainer.accumulate_grad_batches
        assert trainer.current_epoch == 0
        for index, group in enumerate(self.optimizer.param_groups):
            assert group is self.groups[index]
            assert group["params"] is self.parameter_lists[index]
        assert {id(value) for group in self.groups for value in group["params"]} == {
            id(value) for value in pl_module.parameters() if value.requires_grad
        }
        wrapped = trainer.strategy.model
        assert isinstance(wrapped, DistributedDataParallel)
        assert wrapped.module is pl_module
        assert wrapped.gradient_as_bucket_view == self.bucket_view
        assert wrapped.broadcast_buffers
        self.rewraps.append(wrapped is not self.wrapped())
        self.wrapped = ref(wrapped)

        current = snapshot(trainer, pl_module)
        assert_migration(self.ends[-1] if self.ends else self.initial, current)
        for name, parameter in parameters(pl_module).items():
            previous = self.previous_parameters[name]
            if previous.shape != parameter.shape:
                assert previous is not parameter
                assert previous not in self.optimizer.state
            else:
                assert previous is parameter

        assert not batch.bindings, "worker-local IDs must be bound before batch-start callbacks"
        rank = trainer.global_rank
        label = LABELS[rank][batch_idx]
        assert batch.source["row"].to_pylist() == [batch_idx * 4 + rank, batch_idx * 4 + rank + 2]
        self.sources.append(batch.source["row"].to_pylist())
        for name in FIELDS:
            address = f"/{name}"
            values = batch.source[name].to_pylist()
            assert values == ([[label], [label]] if name == "tags" else [label, label])
            field = batch.tensors[address]
            resource = current["fields"][name]
            assert resource["capacity"] == CAPACITIES[batch_idx]
            token = resource["vocabulary"].index(label)
            assert token < resource["capacity"]
            assert field.trainable.reshape(-1).tolist() == [False, True]
            assert field.state.reshape(-1)[0] == Tokens.valued.value
            if name == "tags":
                for content, row in ((field.content, 0), (field.targets[TensorKey.content], 1)):
                    membership = content["membership"].reshape(2, -1)
                    assert membership.shape[-1] == resource["capacity"]
                    assert membership[row, token] == 1
                    assert membership[row].sum() == 1
                    assert content["unavailable"].reshape(-1)[row] == 0
            else:
                assert field.content.reshape(-1)[0] == token
                assert field.targets[TensorKey.content].reshape(-1)[1] == token
            expected = torch.zeros_like(resource["counts"])
            expected[token] = 2
            assert torch.equal(batch.observations[address][TensorKey.content], expected), address
        self.starts.append(current)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert trainer.global_step == (batch_idx + 1) // trainer.accumulate_grad_batches
        assert torch.isfinite(outputs["loss"])
        if self.bucket_view and (batch_idx + 1) % trainer.accumulate_grad_batches == 0:
            assert all(
                value.grad is not None and value.grad._base is not None for value in parameters(pl_module).values()
            )
        self.ends.append(snapshot(trainer, pl_module))
        self.previous_parameters = parameters(pl_module)

    def on_train_end(self, trainer, pl_module):
        assert len(self.starts) == len(self.ends) == 4
        assert self.rewraps == [True, True, True, False]
        torch.save(
            {
                "rank": trainer.global_rank,
                "initial": self.initial,
                "starts": self.starts,
                "ends": self.ends,
                "final": snapshot(trainer, pl_module),
                "sources": self.sources,
                "rewraps": self.rewraps,
            },
            self.directory / f"rank-{trainer.global_rank}.pt",
        )


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="requires CPU Gloo"
)
@pytest.mark.parametrize(
    "bucket_view,accumulation,init_sync",
    [(False, 1, True), (True, 1, True), (False, 2, True), (True, 2, True), (True, 2, False)],
    ids=["ddp", "bucket_view", "accumulate", "bucket_view_accumulate", "no_initial_sync"],
)
def test_distributed_vocabulary_growth_in_one_fit(tmp_path, monkeypatch, bucket_view, accumulation, init_sync):
    # The two data workers also use spawn; keep this proof independent of host thread counts.
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    lit.seed_everything(31415, workers=True)
    selected = rf.Mask(query="selected", reconstruct=True)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
        category=rf.Category(p_unavailable=0, mask=selected),
        tags=rf.Set(p_unavailable=0, mask=selected),
        cluster=rf.Cluster(bounds=2, p_unavailable=0, revive_temperature=0, mask=selected),
        outcome=rf.Boolean(mask=True),
    )
    # Visible rows train the input embeddings through outcome; masked rows train each vocabulary decoder.
    model.optimizer = optimizer
    model.scheduler = schedule
    records = []
    # distribute() takes alternating rows within each global batch of four.
    for batch_idx in range(4):
        for selected_row in (False, True):
            for rank in range(2):
                label = LABELS[rank][batch_idx]
                records.append(
                    {
                        "row": len(records),
                        "category": label,
                        "tags": [label],
                        "cluster": label,
                        "selected": selected_row,
                        "outcome": bool(rank),
                    }
                )
    datamodule = rf.ArrowDataModule(
        model=model,
        train=pa.Table.from_pylist(records),
        shuffle=False,
        num_workers=1,
        persistent_workers=False,
        pin_memory=False,
        multiprocessing_context="spawn",
        prefetch_factor=2,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(
            start_method="spawn",
            process_group_backend="gloo",
            timeout=timedelta(seconds=60),
            find_unused_parameters=True,
            gradient_as_bucket_view=bucket_view,
            init_sync=init_sync,
        ),
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulation,
        callbacks=[GrowthAudit(tmp_path, bucket_view)],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=datamodule)

    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    expected_vocabulary = ["rank0-a", "rank1-a", "rank0-b", "rank1-b", "rank1-late"]
    global_counts = Counter(label for labels in LABELS for label in labels)
    for rank, result in enumerate(ranks):
        assert result["rank"] == rank
        assert result["rewraps"] == [True, True, True, False]
        assert result["final"]["step"] == 4 // accumulation
        assert result["final"]["scheduler"]["last_epoch"] == 4 // accumulation
        assert result["final"]["lr"] == pytest.approx([0.01 * 0.9 ** (4 // accumulation)] * 2)
        local_counts = Counter()
        for batch_idx, (start, end) in enumerate(zip(result["starts"], result["ends"], strict=True)):
            assert start["step"] == batch_idx // accumulation
            assert end["step"] == (batch_idx + 1) // accumulation
            assert start["epoch"] == end["epoch"] == 0
            assert start["scheduler"]["last_epoch"] == start["step"]
            assert end["scheduler"]["last_epoch"] == end["step"]
            if (batch_idx + 1) % accumulation:
                # A microbatch and any subsequent growth must not introduce an optimizer step.
                assert_migration(start, end)
            for point, counts in ((start, local_counts), (end, local_counts + Counter({LABELS[rank][batch_idx]: 1}))):
                for name in FIELDS:
                    resource = point["fields"][name]
                    assert resource["vocabulary"] == expected_vocabulary[: (2, 4, 5, 5)[batch_idx]]
                    expected = torch.zeros_like(resource["counts"])
                    for token, label in enumerate(resource["vocabulary"]):
                        expected[token] = counts[label] * 2
                    assert torch.equal(resource["pending"], expected), (rank, batch_idx, name)
                    assert torch.equal(resource["counts"], expected + 1), (rank, batch_idx, name)
            local_counts[LABELS[rank][batch_idx]] += 1
            if batch_idx in (1, 2):
                lower, upper = (2, 4) if batch_idx == 1 else (4, 5)
                trained = result["ends"][(batch_idx // accumulation + 1) * accumulation - 1]
                for name, weights in start["weights"].items():
                    if name == "cluster/decoder.weight":
                        continue
                    # No weight decay: changed rows plus nonzero Adam moments prove actual training.
                    assert (
                        (weights[lower:upper] != trained["weights"][name][lower:upper])
                        .reshape(upper - lower, -1)
                        .any(1)
                        .all()
                    ), name
                    state = trained["moments"][name]
                    assert state["step"].item() == trained["step"]
                    for key in ("exp_avg", "exp_avg_sq"):
                        assert state[key][lower:upper].reshape(upper - lower, -1).abs().sum(1).gt(0).all(), name
        for name in FIELDS:
            final = result["final"]["fields"][name]
            assert final["vocabulary"] == expected_vocabulary
            expected = torch.ones_like(final["counts"])
            for token, label in enumerate(expected_vocabulary):
                expected[token] += global_counts[label] * 2
            assert torch.equal(final["counts"], expected), (rank, name)
            assert torch.count_nonzero(final["pending"]) == 0

    assert set(sum(ranks[0]["sources"], [])).isdisjoint(sum(ranks[1]["sources"], []))
    assert sorted(sum(ranks[0]["sources"] + ranks[1]["sources"], [])) == list(range(16))
    for phase in ("starts", "ends"):
        for left, right in zip(ranks[0][phase], ranks[1][phase], strict=True):
            for name, weights in left["weights"].items():
                assert torch.equal(weights, right["weights"][name]), (phase, name)
                for key, values in left["moments"][name].items():
                    assert torch.equal(values, right["moments"][name][key]), (phase, name, key)

    if accumulation == 2:
        # Observe the flush without adding a test collective that could perform it itself.
        before = [result["ends"][0]["gradients"] for result in ranks]
        assert any(not torch.equal(before[0][name], before[1][name]) for name in before[0])
        for name, left in before[0].items():
            right = before[1][name]
            assert left is not None and right is not None, name
            expected = (left + right) / 2
            for rank, result in enumerate(ranks):
                after = result["starts"][1]["gradients"][name]
                assert after is not None, (rank, name)
                destinations = torch.arange(left.shape[0])
                if name == "cluster/embedding":
                    destinations[-1] = after.shape[0] - 1
                assert torch.equal(after[destinations], expected), (rank, name)
                introduced = torch.ones(after.shape[0], dtype=torch.bool)
                introduced[destinations] = False
                assert torch.count_nonzero(after[introduced]) == 0, (rank, name)
