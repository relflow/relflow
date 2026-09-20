from types import SimpleNamespace

import pyarrow as pa

from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularySyncCallback


def test_vocabulary_fit_start_matches_rank_zero_metadata(monkeypatch):
    vocabulary = OnlineVocabularyModel(size=8)
    vocabulary.load_snapshot(["local-label"])
    module = SimpleNamespace(nodes={"root/category": SimpleNamespace(embedder=SimpleNamespace(vocab=vocabulary))})
    calls = []

    def broadcast(payload, src):
        calls.append((payload, src))
        return {"root/category": ["rank-zero-label"]}

    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.is_distributed", lambda: True)
    monkeypatch.setattr("relflow.tensorfields.shared.vocabulary.broadcast_object", broadcast)
    VocabularySyncCallback().on_fit_start(trainer=None, pl_module=module)

    assert vocabulary.snapshot() == ["rank-zero-label"]
    assert calls == [({"root/category": ["local-label"]}, 0)]


def test_prefetching_does_not_admit_labels_at_epoch_end():
    vocabulary = OnlineVocabularyModel(size=1)
    vocabulary.load_snapshot(["ALPHA"])
    local, binding = vocabulary.state.batch(pa.array(["BETA", "GAMMA", "BETA"]))
    assert local.encode("BETA") == 0
    assert local.encode("GAMMA") == 1
    assert binding.labels.to_pylist() == ["BETA", "GAMMA"]
    module = SimpleNamespace(nodes={"root/category": SimpleNamespace(embedder=SimpleNamespace(vocab=vocabulary))})

    VocabularySyncCallback().on_train_epoch_end(trainer=None, pl_module=module)

    assert vocabulary.snapshot() == ["ALPHA"]
    assert vocabulary.size == 1
