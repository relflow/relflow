"""An Index rotates its branch's attention by record index instead of record ordinal."""

import datetime as dt

import numpy as np
import polars as pl
import pytest
import torch

import relflow as rf
from relflow.architecture.contracts import ForwardContractError
from relflow.architecture.encoder import spread
from relflow.structs.packages import Parcel

EVENTS = pl.List(pl.Struct({"at": pl.Datetime("us", "UTC"), "amount": pl.Float64}))
START = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)


def frame(histories: list[list[tuple[float, float]]]) -> pl.DataFrame:
    """Build events from ``(hours after START, amount)`` pairs."""
    rows = [
        [{"at": START + dt.timedelta(hours=hours), "amount": amount} for hours, amount in history]
        for history in histories
    ]
    return pl.DataFrame({"events": rows, "score": [None] * len(rows)}, schema={"events": EVENTS, "score": pl.Float64})


def model(*, index: bool = True, **options) -> rf.Model:
    torch.manual_seed(5)
    children = {"at": rf.Index(granularity="hours")} if index else {}
    built = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(length=6, **children, amount=rf.Number, **options),
        score=rf.Number(predict=True),
    )
    return built.eval()


def scores(built: rf.Model, data: pl.DataFrame) -> np.ndarray:
    return built.predict(data)["predictions"].struct.field("score").struct.field("content").to_numpy()


HISTORIES = [[(0, 1.0), (5, -2.0), (30, 0.5)], [(2, 0.3), (3, 0.9)], [(48, 1.5), (0, -1.0), (1, 0.0), (7, 2.0)]]


def test_shifting_every_index_by_a_constant_changes_nothing():
    built = model()
    shifted = [[(hours + 1000, amount) for hours, amount in history] for history in HISTORIES]

    np.testing.assert_array_equal(scores(built, frame(HISTORIES)), scores(built, frame(shifted)))


@pytest.mark.parametrize("options", [{}, {"n_layers": 2, "reduction": rf.Attention(n_outputs=2)}])
def test_reordering_records_with_their_indexes_keeps_a_reduced_branch_unchanged(options):
    reordered = [list(reversed(history)) for history in HISTORIES]

    indexed = model(**options)
    np.testing.assert_allclose(scores(indexed, frame(HISTORIES)), scores(indexed, frame(reordered)), atol=1e-5)

    ordinal = model(index=False, **options)
    assert not np.allclose(scores(ordinal, frame(HISTORIES)), scores(ordinal, frame(reordered)), atol=1e-5)


def test_index_gaps_reach_attention():
    built = model()
    stretched = [[(hours * 3, amount) for hours, amount in history] for history in HISTORIES]

    assert not np.allclose(scores(built, frame(HISTORIES)), scores(built, frame(stretched)), atol=1e-5)
    ordinal = model(index=False)
    np.testing.assert_array_equal(scores(ordinal, frame(HISTORIES)), scores(ordinal, frame(stretched)))


def test_spread_gives_every_token_of_an_item_its_index_position():
    def parcel(tokens: int) -> Parcel:
        return Parcel(
            payload=torch.zeros(1, tokens, 4),
            present=torch.ones(1, tokens, dtype=torch.bool),
            origin=rf.Address("/x"),
            destination=rf.Address("/"),
            batch_size=[1],
        )

    positions = torch.tensor([[2.5, 0.0, 7.0]])

    assert spread([parcel(3), parcel(6)], positions).tolist() == [[2.5, 0.0, 7.0, 2.5, 2.5, 0.0, 0.0, 7.0, 7.0]]
    assert spread([parcel(0)], torch.zeros(1, 0)).shape == (1, 0)


def test_unreduced_indexed_branch_decodes_every_record_by_slot_and_ignores_index_offsets():
    """Decoders keep structural positions, so each finds its own record even when indexes tie."""
    torch.manual_seed(5)
    built = rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            length=6,
            reduction=None,
            at=rf.Index(granularity="hours"),
            amount=rf.Number,
            doubled=rf.Number(predict=True),
        ),
        label=rf.Number,
    ).eval()
    tied = [[(0, 1.0), (0, -2.0), (0, 0.5)]]

    def doubled(histories: list[list[tuple[float, float]]]) -> list[list[float]]:
        events = built.predict(frame(histories).select("events", label=pl.lit(1.0)))["predictions"]
        records = events.struct.field("events").list.eval(pl.element().struct.field("doubled").struct.field("content"))
        return records.to_list()

    assert [len(row) for row in doubled(HISTORIES)] == [len(history) for history in HISTORIES]
    shifted = [[(hours + 500, amount) for hours, amount in history] for history in HISTORIES]
    assert doubled(shifted) == doubled(HISTORIES)
    assert len(set(doubled(tied)[0])) == 3


def test_forward_contract_requires_index_positions():
    built = model()
    encoded = built.encode(frame(HISTORIES))
    geometry = encoded[rf.Address()]
    geometry["/events"].positions = None

    with pytest.raises(ForwardContractError, match="needs positions for Index '/events/at'"):
        built(encoded, strata=rf.Strata.predict)

    del encoded[rf.Address()]
    with pytest.raises(ForwardContractError):
        built(encoded, strata=rf.Strata.predict)


def test_saved_index_models_reload_with_identical_predictions(tmp_path):
    built = model()
    built.save(tmp_path / "model.pt")

    loaded = rf.Model.load(tmp_path / "model.pt").eval()

    assert loaded.schema.model_dump() == built.schema.model_dump()
    np.testing.assert_array_equal(scores(loaded, frame(HISTORIES)), scores(built, frame(HISTORIES)))


@pytest.mark.integration
def test_index_model_trains_and_predicts():
    data = frame(HISTORIES * 8).with_columns(score=pl.Series([1.0, 2.0, 3.0] * 8))
    built = model().train()
    datamodule = rf.PolarsDataModule(
        model=built, train=data, validate=data, num_workers=0, persistent_workers=False, pin_memory=False
    )

    built.fit(datamodule, max_epochs=1, limit_train_batches=1, limit_val_batches=1, batch_size=8)

    assert np.isfinite(scores(built.eval(), frame(HISTORIES))).all()
