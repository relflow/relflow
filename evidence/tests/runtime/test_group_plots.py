"""Group and historical Cluster records retain their measured plot shapes."""

import ast
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from evidence.diagnostics.plots import (
    group_composition,
    group_learning,
    group_predictions,
    grouping,
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


def test_group_partition_plot_uses_fixed_width():
    arm = {"expected_partition": [0, 0, 1, 1], "assignments": [0, 0, 1, 1], "adjusted_rand": 1.0}
    bundle = SimpleNamespace(
        manifest={"id": "P078"}, record={"metrics": {"group_counts": {"4": {"stable": arm, "no_regime": arm}}}}
    )
    fig, _, _ = grouping(bundle, {})
    assert fig.axes[0].images[0].get_array().shape == (2, 4)


@pytest.mark.parametrize("identity", ["cluster", "group"])
@pytest.mark.parametrize("identifier", ["P095", "P112"])
def test_tradeoff_plot_supports_historical_and_current_arms(identity, identifier):
    arm = {"nrmse": 0.5, "rare_identity_nrmse": 0.7}
    arms = {name: arm for name in (identity, "category", "oracle", "shuffled")}
    bundle = SimpleNamespace(manifest={"id": identifier}, record={"metrics": {"conditions": {"dense": {"arms": arms}}}})
    fig, _, _ = grouping(bundle, {})
    assert len(fig.axes[0].collections) == 4


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


@pytest.mark.parametrize(
    ("identifier", "measured_mode", "expected"),
    [("P099", "duration", "duration"), ("P100", "ordinal", "ordinal"), ("P111", None, "group")],
)
def test_measured_research_uses_the_datatype_or_group_renderer(
    monkeypatch, tmp_path, identifier, measured_mode, expected
):
    import matplotlib.pyplot as plt

    from evidence.diagnostics import plots

    calls = []

    def handler(name):
        def draw(bundle, spec):
            calls.append(name)
            return plt.figure(), name, {}

        return draw

    for name in ("duration", "ordinal", "research"):
        monkeypatch.setitem(plots.HANDLERS, name, handler(name))
    monkeypatch.setattr(plots, "group_learning", handler("group"))
    monkeypatch.setattr(plots, "group_composition", handler("composition"))
    monkeypatch.setattr(plots, "group_predictions", handler("predictions"))
    bundle = SimpleNamespace(manifest={"id": identifier}, record={"metrics": {"group_counts": {}}})
    spec = {"mode": "research", "focus": "focus", "comparison": "comparison", "limit": "limit"}
    if measured_mode:
        spec["measured_mode"] = measured_mode
    plots.render(bundle, spec, tmp_path)
    assert calls.count(expected) == 2
    assert calls[0] == expected
