"""Hidden targets never carry a present token, so adding one must not rescale a pool or move predictions."""

import pyarrow as pa
import pytest
import torch

import relflow as rf

ROOT = rf.Address("/")


def table() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {"items": [{"amount": 1.0}, {"amount": 2.0}, {"amount": 4.0}], "size": 3.0, "total": 7.0},
            {"items": [{"amount": 5.0}], "size": 1.0, "total": 5.0},
        ]
    )


def totals(model: rf.Model) -> list[float]:
    return model.predict(table())["predictions"].combine_chunks().field("total").field("content").to_pylist()


@pytest.mark.parametrize("reduction", [rf.Attention(), rf.Mean()], ids=["attention", "mean"])
def test_extending_a_root_target_leaves_capacity_and_predictions_unchanged(reduction):
    torch.manual_seed(0)
    model = rf.Model.xs(
        d_model=8,
        batch_size=2,
        reduction=reduction,
        items=rf.Branch(length=4, amount=rf.Number),
        size=rf.Number,
        total=rf.Number(mask=True),
    )
    before = totals(model)
    encoder = model.nodes[ROOT].encoder
    assert encoder.visible_width == 2
    capacity = encoder.pool.mass_capacity

    model.extend(rf.where("address") == str(ROOT), flag=rf.Boolean(mask=True))

    encoder = model.nodes[ROOT].encoder
    assert encoder.visible_width == 2
    assert encoder.pool.mass_capacity == capacity
    assert not model.schema.requests[rf.Address("/flag")].embedded
    # The new target appends one absent slot, so predictions agree up to float reassociation.
    assert totals(model) == pytest.approx(before, rel=1e-5, abs=1e-6)
