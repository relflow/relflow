import numpy as np
import pyarrow as pa
import torch

import relflow as rf
from relflow.data.crops import starts

CROPS = rf.Crops(width=4, overlap=0.5)


def test_starts_follow_the_record_length():
    drawn = starts(np.array([10, 5, 3, 0]), crops=CROPS, seed=1, epoch=0, ordinal=0, address=rf.Address("/ett"))
    assert drawn.dtype == np.int64 and drawn.shape == (4, 2)
    assert 0 <= drawn[0, 0] <= 4 and drawn[0, 1] == drawn[0, 0] + 2
    assert drawn[1].tolist() == [0, 1]
    assert drawn[2].tolist() == [0, 0] and drawn[3].tolist() == [0, 0]


def test_starts_replay_for_one_seed_and_vary_with_the_ordinal():
    lengths = np.full(64, 40)
    first = starts(lengths, crops=CROPS, seed=7, epoch=2, ordinal=3, address=rf.Address("/ett"))
    again = starts(lengths, crops=CROPS, seed=7, epoch=2, ordinal=3, address=rf.Address("/ett"))
    later = starts(lengths, crops=CROPS, seed=7, epoch=2, ordinal=4, address=rf.Address("/ett"))
    assert np.array_equal(first, again)
    assert not np.array_equal(first, later)


def test_starts_keep_the_parent_shape():
    drawn = starts(np.full((3, 2), 10), crops=CROPS, seed=0, epoch=0, ordinal=0, address=rf.Address("/a/b"))
    assert drawn.shape == (3, 2, 2)


def model() -> rf.Model:
    return rf.Model.xs(d_model=8, ett=rf.Branch(length=8, x=rf.Number, contrastive=rf.Contrastive(views=CROPS)))


def table() -> pa.Table:
    ett = pa.list_(pa.struct([("x", pa.float64())]))
    collections = [[{"x": float(i)} for i in range(8)], [{"x": 1.0}], [], None]
    return pa.table({"ett": pa.array(collections, type=ett)})


def test_training_geometry_carries_views_and_other_strata_carry_none():
    candidate = model()
    views = candidate.encode(table(), strata="train", seed=5)[rf.Address()][rf.Address("/ett")].views
    assert views.shape == (4, 1, 2) and views.dtype == torch.int64
    assert views[1:].tolist() == [[[0, 0]]] * 3
    for strata in ("validate", "test", "predict"):
        assert candidate.encode(table(), strata=strata)[rf.Address()][rf.Address("/ett")].views is None


def test_plain_branch_geometry_has_no_views():
    plain = rf.Model.xs(d_model=8, ett=rf.Branch(length=8, x=rf.Number))
    assert plain.encode(table(), strata="train")[rf.Address()][rf.Address("/ett")].views is None


def test_views_survive_row_selection():
    encoded = model().encode(table(), strata="train", seed=5)
    geometry = encoded[rf.Address()][rf.Address("/ett")]
    chosen = encoded[torch.tensor([2, 0])][rf.Address()][rf.Address("/ett")]
    assert torch.equal(chosen.views, geometry.views[[2, 0]])


def test_views_survive_shipping():
    from relflow.data.datasets.arrow import ship
    from relflow.data.iterables import encode

    encoded = encode(table(), model().schema, rf.Strata.train, {}, seed=5)
    shipped = ship(encoded)
    assert torch.equal(
        shipped.tensors[rf.Address()][rf.Address("/ett")].views,
        encoded.tensors[rf.Address()][rf.Address("/ett")].views,
    )
