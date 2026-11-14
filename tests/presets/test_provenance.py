"""Published defaults retain their exact mapping to the frozen HPO recipes."""

import json
from pathlib import Path

import pytest

import relflow as rf


@pytest.mark.parametrize("name,index,tasks", [("small", 33, 8), ("medium", 8, 13)])
def test_bundled_architecture_and_fit_defaults_match_the_retained_source(name, index, tasks):
    data = json.loads((Path(rf.presets.__file__).parent / "provenance.json").read_text())
    entry = data["presets"][name]
    config = entry["recipe"]
    profile = rf.presets.get(name)
    assert entry["source"]["candidate_index"] == index
    assert entry["selected_tasks"] == tasks
    assert profile.d_model == config["d_model"]
    for key in ("n_layers", "n_heads", "ffnx", "dropout"):
        assert getattr(profile.root, key) == config[key]
    for source, target in (
        ("branch_layers", "n_layers"),
        ("branch_heads", "n_heads"),
        ("branch_ffnx", "ffnx"),
        ("branch_dropout", "dropout"),
    ):
        assert getattr(profile.branch, target) == config[source]
    assert profile.branch.reduction.n_outputs == config["branch_outputs"]
    for source, target in (
        ("decoder_layers", "n_linear"),
        ("decoder_heads", "n_heads"),
        ("decoder_ffnx", "ffnx"),
        ("decoder_dropout", "dropout"),
    ):
        assert getattr(profile.leaf, target) == config[source]
    for key in ("batch_size", "learning_rate", "weight_decay", "ema_decay"):
        assert getattr(profile.fit, key) == config[key]
    assert profile.fit.max_epochs == config["passes"]
    assert profile.fit.gradient_clip_val == 0.25
    assert profile.fit.early_stopping is False
    assert profile.fit.rollback is False
