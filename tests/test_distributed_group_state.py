"""Adaptive Group state incorporates every rank, including skipped objectives."""

from datetime import timedelta

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
from lightning.pytorch.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel

import relflow as rf
from relflow.tensorfields.extensions.group import GroupUsageCallback
from tests.distributed import launch

ADDRESS = "/identity"
COUNTS = ((0, 4), (1, 3), (0, 0), (4, 0))
PROBABILITIES = (
    ((0.8, 0.15, 0.05), (0.1, 0.2, 0.7)),
    ((0.98, 0.01, 0.01), (0.01, 0.49, 0.5)),
    ((0.05, 0.05, 0.9), (0.8, 0.1, 0.1)),
    ((0.05, 0.9, 0.05), (0.7, 0.2, 0.1)),
)


def snapshot(model):
    embedder = model.nodes[ADDRESS].embedder
    assert embedder.statistics.count_nonzero() == 0, "consumed evidence must not survive into the next batch"
    return {
        name: getattr(embedder, name).detach().cpu().clone()
        for name in ("usage_ema", "information_ema", "adherence_ema", "committed")
    }


class Audit(lit.Callback):
    """Inject known predictions while exercising the real loss and callback loop."""

    def __init__(self, directory):
        self.directory = directory
        self.states = []

    def on_train_start(self, trainer, pl_module):
        assert trainer.world_size == 2
        assert isinstance(trainer.strategy.model, DistributedDataParallel)
        assert trainer.strategy.model.broadcast_buffers
        assert sum(isinstance(callback, GroupUsageCallback) for callback in trainer.callbacks) == 1
        self.states.append(snapshot(pl_module))
        self.handle = pl_module.nodes[ADDRESS].decoder.linears["group"].register_forward_hook(self.predictions)

    def predictions(self, module, inputs, output):
        # Keep the decoder in autograd/DDP while fixing the probability evidence.
        return output * 0 + output.new_tensor(self.probabilities).log()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx:
            self.states.append(snapshot(pl_module))
        self.probabilities = PROBABILITIES[batch_idx][trainer.global_rank]
        field = batch.tensors[ADDRESS]
        assert int(field.trainable.sum()) == COUNTS[batch_idx][trainer.global_rank]
        assert rf.Group.vocabulary(pl_module, ADDRESS) == ("known",)

    def on_train_end(self, trainer, pl_module):
        self.handle.remove()
        self.states.append(snapshot(pl_module))
        assert len(self.states) == len(COUNTS) + 1
        state = pl_module.state_dict()
        assert any(name.endswith("information_ema") for name in state)
        assert not any(name.endswith("statistics") for name in state)
        torch.save(self.states, self.directory / f"rank-{trainer.global_rank}.pt")


def train(directory, accumulation):
    lit.seed_everything(8251, workers=True)
    model = rf.Model.xs(
        d_model=8,
        batch_size=4,
        identity=rf.Group(
            n_groups=(1, 3),
            ema_decay=0.5,
            revive_temperature=0,
            p_unavailable=0,
            mask=rf.Mask(query="selected", reconstruct=True),
        ),
        # The both-empty batch omits Group loss while retaining a valid model step.
        outcome=rf.Boolean(mask=True),
    )
    # Isolate statistics from dynamic vocabulary/resource replacement.
    model.encode(pa.table({"identity": ["known"], "selected": [False], "outcome": [False]}), strata="train")
    model.optimizer = rf.adamw(learning_rate=0.001, weight_decay=0, fused=False)
    records = []
    for counts in COUNTS:
        for row in range(4):
            for rank in range(2):
                records.append({"identity": "known", "selected": row < counts[rank], "outcome": bool(rank)})
    datamodule = rf.ArrowDataModule(
        model=model,
        train=pa.Table.from_pylist(records),
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        devices=2,
        strategy=DDPStrategy(
            start_method="popen",
            process_group_backend="gloo",
            timeout=timedelta(seconds=60),
            find_unused_parameters=True,
        ),
        max_epochs=1,
        limit_train_batches=len(COUNTS),
        limit_val_batches=0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=accumulation,
        callbacks=[Audit(directory)],
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=directory,
    )
    trainer.fit(model, datamodule=datamodule)


@pytest.mark.skipif(
    not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason="requires CPU Gloo"
)
@pytest.mark.parametrize("accumulation", [1, 2], ids=["each_step", "accumulated"])
def test_group_usage_is_globally_weighted_with_empty_and_unequal_ranks(tmp_path, monkeypatch, accumulation):
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    launch(train, tmp_path, accumulation)
    ranks = [torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)]
    for left, right in zip(*ranks, strict=True):
        for name in left:
            torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)

    usage = torch.tensor([1.0, 0.0, 0.0])
    information = torch.tensor(0.0)
    adherence = torch.tensor(0.0)
    committed = torch.tensor([True, False, False])
    for index, counts in enumerate(COUNTS):
        count = sum(counts)
        if count:
            probabilities = torch.tensor(PROBABILITIES[index])
            weights = torch.tensor(counts, dtype=torch.float32) / count
            mean = (probabilities * weights[:, None]).sum(0)
            entropy = -(mean * mean.log()).sum()
            conditional_entropy = -(probabilities * probabilities.log()).sum(1)
            evidence = (entropy - (conditional_entropy * weights).sum()).clamp_min(0)
            usage = (usage + mean) / 2
            information = (information + evidence) / 2
            committed = torch.zeros(3, dtype=torch.bool)
            committed[usage.topk(max(1, min(3, round(information.exp().item())))).indices] = True
            adherence = (adherence + mean[~committed].sum()) / 2
        for result in ranks:
            actual = result[index + 1]
            torch.testing.assert_close(actual["usage_ema"], usage, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(actual["information_ema"], information, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(actual["adherence_ema"], adherence, rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(actual["committed"], committed)
            if not count:
                for name, value in actual.items():
                    torch.testing.assert_close(value, result[index][name], rtol=0, atol=0)
