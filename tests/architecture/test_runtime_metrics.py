"""The loss metric is emitted on every rank, weighted by the rows it scored."""

import pyarrow as pa
import torch

import relflow as rf
from relflow.architecture import runtime


def logged(monkeypatch) -> list[dict]:
    entries: list[dict] = []
    monkeypatch.setattr(rf.Model, "log", lambda self, **options: entries.append(options))
    return entries


def batch(model: rf.Model):
    return model.encode(pa.table({"amount": [1.0, 2.0, 3.0], "label": [True, False, True]}), strata="train")


def test_loss_metric_is_uniform_and_weighted_by_batch_rows(monkeypatch):
    model = rf.Model.xs(d_model=8, batch_size=8, amount=rf.Number, label=rf.Boolean(mask=True))
    entries = logged(monkeypatch)

    output = runtime.step(model, batch(model), 0, strata=rf.Strata.train)

    assert output is not None and output["loss"].requires_grad
    by_name = {entry["name"]: entry for entry in entries}
    for name in ("loss/train", "loss/train_epoch"):
        assert by_name[name]["batch_size"] == 3
        assert by_name[name]["sync_dist"] is True
        assert by_name[name]["rank_zero_only"] is False
    extension = [entry for entry in entries if not entry["name"].startswith("loss/")]
    assert extension
    assert all(entry["rank_zero_only"] is True and entry["batch_size"] == 8 for entry in extension)


def test_rank_without_local_targets_still_emits_the_anchored_loss(monkeypatch):
    model = rf.Model.xs(d_model=8, batch_size=8, amount=rf.Number, label=rf.Boolean(mask=True))
    entries = logged(monkeypatch)
    monkeypatch.setattr(runtime, "participation", lambda module, inputs, strata: (set(), {rf.Address("/label")}))

    output = runtime.step(model, batch(model), 1, strata=rf.Strata.train)

    assert output is not None and output["loss"].requires_grad and output["loss"].item() == 0.0
    by_name = {entry["name"]: entry for entry in entries}
    assert set(by_name) == {"loss/train", "loss/train_epoch"}
    for entry in by_name.values():
        assert entry["batch_size"] == 0
        assert entry["rank_zero_only"] is False
        assert entry["sync_dist"] is True
    output["loss"].backward()
    assert all(parameter.grad is not None for parameter in model.nodes["/label"].decoder.parameters())

    entries.clear()
    validated = runtime.step(model, batch(model), 2, strata=rf.Strata.validate)
    assert validated["loss"].item() == 0.0 and not validated["loss"].requires_grad
    assert {entry["name"] for entry in entries} == {"loss/validate"}
    assert entries[0]["batch_size"] == 0 and torch.equal(entries[0]["value"], torch.zeros(()))
