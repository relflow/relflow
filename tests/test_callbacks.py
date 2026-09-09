from types import SimpleNamespace

from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularySyncCallback


def test_vocabulary_sync_callback_gathers_rank_proposals(monkeypatch):
    events = []
    vocab = OnlineVocabularyModel(size=8)
    vocab.load_snapshot(["ALPHA"])
    vocab.proposals.append("BETA")

    class TrainerStub:
        strategy = SimpleNamespace(barriers=[])

        @property
        def callback_metrics(self):
            events.append("metrics")
            return {}

    trainer = TrainerStub()
    trainer.strategy.barrier = lambda name: trainer.strategy.barriers.append(name)
    module = SimpleNamespace(
        nodes={
            "root/category": SimpleNamespace(
                embedder=SimpleNamespace(vocab=vocab),
            ),
        },
    )

    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.is_distributed", lambda: True)
    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.is_rank_zero", lambda: True)
    monkeypatch.setattr(
        "relflow.tensorfields.shared.vocabulary.all_gather_object",
        lambda local: events.append("vocabulary") or [local, {"root/category": ["GAMMA"]}],
    )
    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.broadcast_object", lambda payload, src: payload)

    VocabularySyncCallback().on_train_epoch_end(trainer=trainer, pl_module=module)

    assert vocab.snapshot() == ["ALPHA", "BETA", "GAMMA"]
    assert list(vocab.proposals) == []
    assert events == ["metrics", "vocabulary"]
    assert trainer.strategy.barriers == ["vocabulary-sync-train_epoch_end"]


def test_vocabulary_sync_callback_merges_worker_proposals_without_ddp():
    vocabulary = OnlineVocabularyModel(size=8)
    vocabulary.load_snapshot(["ALPHA"])
    vocabulary.share()
    try:
        worker = vocabulary.state
        worker.configure_distributed(global_rank=1, world_size=2)
        worker.reserve(["BETA"], learn=True)
        assert vocabulary.snapshot() == ["ALPHA"]
        assert list(vocabulary.proposals) == ["BETA"]
        trainer = SimpleNamespace(callback_metrics={}, strategy=SimpleNamespace(barrier=lambda name: None))
        module = SimpleNamespace(nodes={"root/category": SimpleNamespace(embedder=SimpleNamespace(vocab=vocabulary))})

        VocabularySyncCallback().on_train_epoch_end(trainer=trainer, pl_module=module)

        assert vocabulary.snapshot() == ["ALPHA", "BETA"]
        worker.refresh()
        assert worker.encode("BETA") == 1
        assert list(vocabulary.proposals) == []
    finally:
        vocabulary.freeze()
