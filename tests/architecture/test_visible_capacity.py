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
    return pa.table(model.predict(table()))["predictions"].combine_chunks().field("total").field("content").to_pylist()


def test_projected_context_field_contributes_to_crosses_capacity_and_child_context():
    def configured(amount):
        torch.manual_seed(17)
        return rf.Model(
            preset="small",
            d_model=8,
            cross=True,
            size=amount,
            other=rf.Number,
            items=rf.Branch(length=4, amount=rf.Number),
            total=rf.Number(predict=True),
        )

    plain = configured(rf.Number)
    gated = configured(rf.Number(query="available.size"))
    source = (
        table()
        .append_column("other", pa.array([1.0, 4.0]))
        .append_column("available", pa.array([{"size": 3.0}, {"size": 1.0}]))
    )
    assert gated.nodes[ROOT].encoder.visible_width == plain.nodes[ROOT].encoder.visible_width
    assert "/size" in gated.nodes[ROOT].encoder.coordinate_origins
    assert "/size" in gated.nodes["/items"].encoder.context_origins
    assert gated.predict(source).equals(plain.predict(source))


@pytest.mark.parametrize("reduction", [rf.Attention(), rf.Mean()], ids=["attention", "mean"])
@pytest.mark.parametrize("cross", [False, True])
def test_extending_a_root_target_leaves_capacity_and_predictions_unchanged(reduction, cross):
    torch.manual_seed(0)
    model = rf.Model(
        preset="small",
        d_model=8,
        batch_size=2,
        reduction=reduction,
        cross=cross,
        items=rf.Branch(length=4, amount=rf.Number),
        size=rf.Number,
        total=rf.Number(predict=True),
    )
    before = totals(model)
    encoder = model.nodes[ROOT].encoder
    assert encoder.visible_width == 5
    capacity = encoder.pool.mass_capacity

    model.extend(rf.where("address") == str(ROOT), flag=rf.Boolean(predict=True))

    encoder = model.nodes[ROOT].encoder
    assert encoder.visible_width == 5
    assert encoder.pool.mass_capacity == capacity
    assert not model.schema.requests[rf.Address("/flag")].embedded
    # The new target appends one absent slot, so predictions agree up to float reassociation.
    assert totals(model) == pytest.approx(before, rel=1e-5, abs=1e-6)
