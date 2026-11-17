"""Shared primitives and experiment-owned components render recorded values without model imports."""

import ast
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from evidence.diagnostics.plots import (
    group_composition,
    group_learning,
    group_predictions,
    trajectory,
)


@pytest.mark.parametrize(
    ("point", "expected"),
    [
        ({"epoch": 1, "n_committed": 2, "adherence": 0.2}, ("Committed groups", "Input adherence")),
        ({"epoch": 1, "n_assigned_groups": 3, "perplexity": 2.1}, ("Assigned groups", "Usage perplexity")),
        ({"epoch": 1, "assigned_groups": 3, "adjusted_rand": 0.4}, ("Assigned groups", "Adjusted Rand index")),
    ],
)
def test_trajectory_uses_recorded_metric_names(point, expected):
    fig, _, _ = trajectory(SimpleNamespace(record={"metrics": {"trajectory": [point]}}), {})
    assert tuple(axis.get_ylabel() for axis in fig.axes) == expected
    assert all(len(axis.lines) == 1 for axis in fig.axes)


def test_regime_prediction_labels_distinguish_the_shuffled_control():
    source = "measured = measure(model, test)\nmeasured['shuffled'] = measure(model, shuffled)\n"
    frames = [
        {"ordinal": 0, "context": [{"function": "run", "line": 1}]},
        {"ordinal": 1, "context": [{"function": "run", "line": 2}]},
        {"ordinal": 2, "context": [{"function": "run", "line": 1}]},
    ]
    pairs = [
        {
            "frame": index,
            "population": "test",
            "context": {"arm": arm},
            "field": "y",
            "actual": np.array([0.0, 1.0]),
            "predicted": np.array([0.1, 0.9]),
            "coordinates": [],
            "rows": None,
        }
        for index, arm in enumerate(["stable", "stable", "no_regime"])
    ]
    bundle = SimpleNamespace(
        manifest={"id": "P078"},
        record={"metrics": {"group_counts": {}}},
        syntax=ast.parse(source),
        frames=frames,
        pairs=lambda target: pairs,
    )
    fig, _, _ = group_predictions(bundle, {"target": "y"})
    labels = [line.get_label().replace("\n", " ") for line in fig.axes[1].lines]
    assert labels == ["Stable identities", "Stable model, shuffled test identities", "No stable identity regimes"]


def test_group_learning_uses_recorded_epoch_trajectories():
    stable = {"ari_by_step": [{"step": 32, "adjusted_rand": 0.2, "validation_rmse": 3.0}]}
    control = {"ari_by_step": [{"step": 32, "adjusted_rand": 0.0, "validation_rmse": 4.0}]}
    bundle = SimpleNamespace(
        manifest={"id": "P111"}, record={"metrics": {"group_counts": {"4": {"stable": stable, "no_regime": control}}}}
    )
    fig, note, _ = group_learning(bundle, {})
    assert len(fig.axes[0].lines) == 3
    assert fig.axes[1].lines[0].get_ydata().tolist() == [3.0]
    assert "validation observations" in note


def test_group_composition_retains_unassigned_identities():
    stable = {"expected_partition": [0, 0, 1, 1], "assignments": [0, -1, 1, 1]}
    bundle = SimpleNamespace(manifest={"id": "P111"}, record={"metrics": {"group_counts": {"4": {"stable": stable}}}})
    fig, _, _ = group_composition(bundle, {})
    assert "Unassigned" in [label.get_text() for label in fig.axes[0].get_yticklabels()]
    assert sum(patch.get_width() for patch in fig.axes[0].patches) == 4


@pytest.mark.parametrize("mode", ["regression", "classification", "group-learning"])
def test_rendering_resolves_declared_modes_without_experiment_ids(monkeypatch, tmp_path, mode):
    import matplotlib.pyplot as plt

    from evidence.diagnostics import plots

    calls = []

    def draw(bundle, spec):
        calls.append(mode)
        return plt.figure(), mode, {}

    monkeypatch.setitem(plots.HANDLERS, mode, draw)
    bundle = SimpleNamespace(manifest={"id": "P999"}, record={"metrics": {}})
    plots.render(bundle, {"mode": mode, "focus": "focus", "comparison": "comparison", "limit": "limit"}, tmp_path)
    assert calls == [mode, mode]


def draw(identifier, metrics):
    from pathlib import Path

    from evidence.diagnostics.data import Bundle
    from evidence.experiments.source import registry, view

    root = Path(__file__).resolve().parents[2]
    entry = registry(root, measurements=False)[identifier]
    source = (root / entry["script"]).read_text()
    plan = view(source)
    bundle = object.__new__(Bundle)
    bundle.source, bundle.directory, bundle.components = source, Path("/unused"), {}
    bundle.manifest = {"id": "P999"}
    bundle.record = {"metrics": metrics}
    return bundle.component(plan["renderer"])(bundle, plan)


@pytest.mark.parametrize(
    ("identifier", "metrics", "values"),
    [
        ("P017", {"signal_auc": 0.9, "noise_auc": 0.5}, [0.9, 0.5]),
        ("P021", {"hash_auc": 0.94, "category_oov_auc": 0.51, "shuffled_auc": 0.5}, [0.94, 0.51, 0.5]),
    ],
)
def test_owned_components_render_declared_scores_without_identifier_routing(identifier, metrics, values):
    import matplotlib.pyplot as plt

    fig, note, _ = draw(identifier, metrics)
    np.testing.assert_allclose(fig.axes[0].collections[0].get_offsets()[:, 0], values)
    assert note
    plt.close(fig)


def test_owned_group_plot_uses_current_group_arms():
    import matplotlib.pyplot as plt

    arm = {"nrmse": 0.5, "rare_identity_nrmse": 0.7}
    arms = {name: arm for name in ("group", "category", "oracle", "shuffled")}
    fig, _, _ = draw("P112", {"conditions": {"dense": {"arms": arms}}})
    assert len(fig.axes[0].collections) == 4
    plt.close(fig)
