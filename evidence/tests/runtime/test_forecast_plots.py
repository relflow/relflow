"""Forecast records render their measured errors instead of falling back to a generic summary."""

from types import SimpleNamespace

import pytest

pytest.importorskip("matplotlib")

from evidence.diagnostics.plots import HANDLERS


def test_forecast_plot_compares_ordered_history_with_both_controls():
    metrics = {"ordered_mae": 0.145, "last_step_mae": 0.866, "shuffled_mae": 0.650}
    bundle = SimpleNamespace(manifest={"id": "P101"}, record={"metrics": metrics})
    fig, _, _ = HANDLERS["forecast"](bundle, {})
    values = sorted(float(x) for x in fig.axes[0].collections[0].get_offsets()[:, 0])
    assert values == sorted(metrics.values())
