import numpy as np
import pyarrow as pa
import pytest
from lightning.pytorch import seed_everything

import relflow as rf


def table(values, *, length=6):
    """One record per entry of ``values``, each with ``length`` rows of input ``a`` and a null target ``y``."""
    return pa.table({"ett": [[{"a": float(a), "y": None} for a in row] for row in values]})


def model_for(mixer, *, direction):
    seed_everything(0, workers=True)
    return rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        ett=rf.Branch(
            length=6,
            grain=rf.Steps(),
            mixer=mixer,
            direction=direction,
            reduction=rf.Mean(),
            a=rf.Number,
            y=rf.Number(mask=True),
        ),
    ).eval()


def forecasts(model, values):
    frame = model.predict(table(values))
    rows = frame["predictions"].struct.field("ett").to_list()
    return np.array([[row["y"]["content"] for row in record] for record in rows], dtype=float)


BASE = [[0.1, 0.5, -0.3, 0.8, 0.2, -0.6]]
CHANGED = [[0.1, 0.5, -0.3, 0.8, 9.0, -7.0]]


@pytest.mark.parametrize("mixer", [rf.Transformer(), rf.Conv(n_blocks=2)])
def test_causal_steps_predictions_ignore_future_rows(mixer):
    model = model_for(mixer, direction=rf.Direction.causal)
    first, second = forecasts(model, BASE), forecasts(model, CHANGED)
    assert np.array_equal(first[:, :4], second[:, :4])


@pytest.mark.parametrize("mixer", [rf.Transformer(), rf.Conv(n_blocks=2)])
def test_bidirectional_steps_predictions_do_see_future_rows(mixer):
    model = model_for(mixer, direction=rf.Direction.bidirectional)
    first, second = forecasts(model, BASE), forecasts(model, CHANGED)
    assert not np.array_equal(first[:, 0], second[:, 0])


def test_series_leaf_inside_a_chronological_row():
    seed_everything(0, workers=True)
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        ett=rf.Branch(
            length=5,
            grain=rf.Steps(),
            mixer=rf.Conv(n_blocks=1),
            reduction=rf.Mean(),
            wave=rf.Series(query="wave", n_dim=1, horizon=2, mask=True),
            x=rf.Number,
        ),
    ).eval()
    records = [[{"wave": [0.1 * step, 0.2 * step, 0.3 * step], "x": float(row)} for row in range(5)] for step in (1, 2)]
    frame = model.predict(pa.table({"ett": records}))
    rows = frame["predictions"].struct.field("ett").to_list()
    assert [len(record) for record in rows] == [5, 5]
    assert all(len(row["wave"]["content"]) == 2 for record in rows for row in record)


@pytest.mark.integration
def test_chronological_inside_repeated_parent_trains_and_predicts(tmp_path):
    seed_everything(0, workers=True)
    model = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        stations=rf.Branch(
            length=2,
            reduction=rf.Mean(),
            readings=rf.Branch(
                length=6,
                time_index="ts",
                time_step=1.0,
                grain=rf.Steps(),
                mixer=rf.Conv(n_blocks=1),
                direction=rf.Direction.causal,
                reduction=rf.Mean(),
                ts=rf.Number(mask=rf.Mask(skip=True)),
                value=rf.Number(mask=True),
                signal=rf.Number,
            ),
        ),
    )

    def records(signals):
        stations = [
            {"readings": [{"ts": float(row), "value": float(row), "signal": signal[row]} for row in range(6)]}
            for signal in signals
        ]
        return pa.table({"stations": [stations]})

    base = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    changed = [base[0], [1.0, 2.0, 3.0, 4.0, 90.0, 80.0]]
    datamodule = rf.ArrowDataModule(
        model=model,
        train=records(base),
        validate=records(base),
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        datamodule,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    model.eval()

    def values(signals):
        frame = model.predict(records(signals))
        stations = frame["predictions"].struct.field("stations").to_list()[0]
        return np.array([[row["value"]["content"] for row in station["readings"]] for station in stations])

    first, second = values(base), values(changed)
    assert first.shape == (2, 6)
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1, :4], second[1, :4])
