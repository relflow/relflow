"""Model.load builds the saved graph once and rejects incomplete checkpoints by name."""

import pytest
import torch

import relflow as rf
from relflow.architecture.graph import ModelGraph


def test_load_installs_the_graph_once_and_restores_tensors_and_provenance(tmp_path, monkeypatch):
    torch.manual_seed(11)
    model = rf.Model.xs(d_model=8, batch_size=3, amount=rf.Number, label=rf.Boolean(mask=True))
    model.save(tmp_path / "model.pt")
    installs = []
    install = ModelGraph.install
    monkeypatch.setattr(ModelGraph, "install", lambda module: (installs.append(module), install(module))[1])

    loaded = rf.Model.load(tmp_path / "model.pt")

    assert len(installs) == 1
    assert loaded.version == model.version
    assert loaded.batch_size == 3
    assert loaded.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, rtol=0, atol=0)
    assert loaded.schema.model_dump() == model.schema.model_dump()


def test_load_names_every_missing_checkpoint_field(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "partial.pt")
    with pytest.raises(ValueError, match="missing checkpoint fields: batch_size, schema, version"):
        rf.Model.load(tmp_path / "partial.pt")
