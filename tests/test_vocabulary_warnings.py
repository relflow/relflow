"""Loop-start diagnostics distinguish allocated vocabulary from learned exposure."""

from datetime import timedelta
from logging.handlers import BufferingHandler
from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

import relflow as rf
from relflow.logging import logger
from relflow.logging.config import CONTEXT
from relflow.tensorfields.shared.counter import Counter
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularySyncCallback


@pytest.fixture
def records():
    handler = BufferingHandler(capacity=1000)
    handler.setLevel("WARNING")
    logger.logger.addHandler(handler)
    try:
        yield handler.buffer
    finally:
        logger.logger.removeHandler(handler)


def model():
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=4,
        category=rf.Category(mask=True, p_unavailable=1.0),
        tags=rf.Set(mask=True, p_unavailable=1.0),
        cluster=rf.Cluster(bounds=2, mask=True, p_unavailable=1.0),
    )


def table(labels):
    return pa.table({"category": labels, "tags": [[label] for label in labels], "cluster": labels})


def warnings(records):
    return [getattr(record, CONTEXT) for record in records if "threshold" in getattr(record, CONTEXT, {})]


@pytest.mark.parametrize("stage", ["validate", "test", "predict"])
def test_warning_counts_only_allocated_pristine_training_exposures(records, stage):
    module = model()
    module.encode(table(["private-rare"] * 9 + ["private-warm"] * 10), strata="train")
    # Neither evaluation nor input augmentation may change training exposure.
    module.encode(table(["private-rare"] * 20 + ["unseen"]), strata=stage)
    callback = VocabularySyncCallback()
    hook = callback.on_validation_start if stage == "validate" else getattr(callback, f"on_{stage}_start")
    hook(None, module)

    events = warnings(records)
    assert len(events) == 3
    assert [event["address"] for event in events] == ["/category", "/cluster", "/tags"]
    for event in events:
        assert event == {
            "component": "vocabulary",
            "address": event["address"],
            "strata": stage,
            "underobserved": 1,
            "size": 2,
            "threshold": 10,
            "minimum_observations": 9,
        }
    assert all("private-" not in record.getMessage() for record in records)


def test_warning_excludes_empty_vocabularies_unused_slots_and_smoothing_prior(records):
    module = model()
    callback = VocabularySyncCallback()
    callback.on_test_start(None, module)
    assert not warnings(records)

    module.encode(table(["warm"] * 10), strata="train")
    callback.on_test_start(None, module)
    assert not warnings(records)

    # Allocation alone does not count as an observation.
    from relflow.architecture.binding import bind
    from relflow.data.iterables import encode

    bind(
        module,
        encode(table(["cold"]), module.schema, rf.Strata.train, module.interprocess_encoding_context),
        rf.Strata.train,
    )
    callback.on_test_start(None, module)
    assert len(warnings(records)) == 3
    assert all(event["minimum_observations"] == 0 for event in warnings(records))
    assert all(event["underobserved"] == 1 for event in warnings(records))


def test_warning_uses_checkpointed_counts_and_stops_after_more_training(records, tmp_path):
    module = model()
    module.encode(table(["rare"] * 9 + ["warm"] * 10), strata="train")
    checkpoint = tmp_path / "exposure.ckpt"
    module.save(checkpoint)
    restored = rf.Model.load(checkpoint)
    callback = VocabularySyncCallback()
    callback.on_predict_start(None, restored)
    assert len(warnings(records)) == 3
    assert all(event["minimum_observations"] == 9 for event in warnings(records))
    records.clear()

    restored.encode(table(["rare"]), strata="train")
    callback.on_predict_start(None, restored)
    assert not warnings(records)


def test_training_loop_start_does_not_emit_exposure_warnings(records):
    module = model()
    module.encode(table(["rare"]), strata="train")
    callback = VocabularySyncCallback()
    callback.on_fit_start(None, module)
    callback.on_train_start(None, module)
    assert not warnings(records)


def test_warning_uses_shared_resources_without_a_builtin_field_type(records):
    vocabulary = OnlineVocabularyModel(size=4)
    vocabulary.load_snapshot(["cold"])
    counter = Counter(address=rf.Address("third_party/value"), size=4)
    module = SimpleNamespace(
        nodes={
            "third_party/value": SimpleNamespace(
                embedder=SimpleNamespace(vocab=vocabulary, counters={"content": counter})
            ),
            "third_party/no_counts": SimpleNamespace(embedder=SimpleNamespace(vocab=vocabulary)),
            "third_party/no_vocabulary": SimpleNamespace(embedder=SimpleNamespace(counters={"content": counter})),
        }
    )
    VocabularySyncCallback().on_test_start(None, module)
    assert len(warnings(records)) == 1
    assert warnings(records)[0]["address"] == "third_party/value"
    assert warnings(records)[0]["minimum_observations"] == 0


@pytest.mark.parametrize("stage", ["validate", "test", "predict"])
def test_lightning_warns_once_per_field_not_per_batch(records, stage):
    module = model()
    module.encode(table(["rare"] * 9), strata="train")
    before = rf.Category.counts(module, "/category")
    data = rf.ArrowDataModule(model=module, **{stage: table(["rare"] * 8)})
    trainer = lit.Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    getattr(trainer, stage)(module, datamodule=data)
    assert len(warnings(records)) == 3
    assert all(event["strata"] == stage for event in warnings(records))
    assert rf.Category.counts(module, "/category") == before


def test_validation_during_fit_warns_using_only_training_exposures(records):
    module = model()
    for vocabulary in OnlineVocabularyModel.from_model(module).values():
        vocabulary.state.reserve(["rare"], learn=True)
    module.optimizer = rf.adamw(learning_rate=1e-3)
    data = rf.ArrowDataModule(model=module, train=table(["rare"] * 4), validate=table(["rare"] * 4))
    trainer = lit.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=data)
    events = warnings(records)
    assert len(events) == 6
    assert [event["minimum_observations"] for event in events] == [0, 0, 0, 4, 4, 4]
    assert all(event["strata"] == "validate" for event in events)
    assert rf.Category.counts(module, "/category") == {"rare": 4}


def test_distributed_warning_flushes_metrics_before_count_collectives(monkeypatch, records):
    module = model()
    module.encode(table(["warm"] * 5), strata="train")
    events = []

    class TrainerStub:
        @property
        def callback_metrics(self):
            events.append("metrics")
            return {}

    def reduce(counts):
        events.append("counts")
        return counts * 2

    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.is_distributed", lambda: True)
    monkeypatch.setattr("relflow.tensorfields.shared.counter.all_reduce_sum", reduce)
    VocabularySyncCallback().on_validation_start(TrainerStub(), module)
    assert events == ["metrics", "counts", "counts", "counts"]
    assert not warnings(records)
    assert rf.Category.counts(module, "/category") == {"warm": 10}


def distributed_warning(rank, directory):
    distributed.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    handler = BufferingHandler(capacity=100)
    handler.setLevel("WARNING")
    logger.logger.addHandler(handler)
    try:
        vocabulary = OnlineVocabularyModel(size=4)
        vocabulary.load_snapshot(["warm", "cold"] if rank == 0 else ["warm"])
        counter = Counter(address=rf.Address("third_party/value"), size=4)
        counter.learn(torch.tensor([5, 0, 0, 0]))
        module = SimpleNamespace(
            nodes={
                "third_party/value": SimpleNamespace(
                    embedder=SimpleNamespace(vocab=vocabulary, counters={"content": counter})
                )
            }
        )
        trainer = SimpleNamespace(callback_metrics={})
        callback = VocabularySyncCallback()
        callback.on_validation_start(trainer, module)
        assert counter.counts.tolist() == [11, 1, 1, 1]
        events = warnings(handler.buffer)
        assert len(events) == (1 if rank == 0 else 0)
        if rank == 0:
            assert events[0]["underobserved"] == 1
            assert events[0]["minimum_observations"] == 0
        handler.buffer.clear()

        # Empty local vocabularies must still participate in every collective.
        if rank == 1:
            vocabulary.load_snapshot([])
        callback.on_predict_start(trainer, module)
        assert counter.counts.tolist() == [11, 1, 1, 1]
        assert len(warnings(handler.buffer)) == (1 if rank == 0 else 0)
    finally:
        logger.logger.removeHandler(handler)
        distributed.destroy_process_group()


def test_distributed_warning_uses_global_counts_and_logs_only_on_rank_zero(tmp_path):
    multiprocessing.spawn(distributed_warning, args=(str(tmp_path),), nprocs=2, join=True)
