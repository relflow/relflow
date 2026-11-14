import pyarrow as pa
import torch

import relflow as rf
from relflow.architecture import runtime
from relflow.architecture.runtime import execution
from relflow.structs.enums import Strata
from relflow.tensorfields.base import TENSORFIELDS


def model() -> rf.Model:
    return rf.Model.xs(
        d_model=8,
        dropout=0.0,
        context=rf.Number,
        first=rf.Boolean(mask=True, weight=3.0),
        history=rf.Branch(length=2, second=rf.Boolean(mask=True, weight=0.5)),
    )


def test_execution_plan_lists_leaf_objectives_as_weighted_entries():
    candidate = model()
    entries = execution(candidate).objectives
    assert [entry.address for entry in entries] == candidate.schema.objectives == ["/first", "/history/second"]
    assert [entry.weight for entry in entries] == [3.0, 0.5]


def test_training_loss_is_the_exact_weighted_sum_of_leaf_losses(monkeypatch):
    candidate = model()
    monkeypatch.setattr(candidate, "track", lambda names, value, **kwargs: value)
    table = pa.table(
        {
            "context": [1.0, 2.0],
            "first": [True, False],
            "history": [[{"second": True}], [{"second": False}, {"second": True}]],
        }
    )
    inputs = candidate.encode(table, strata=Strata.train)
    expected = torch.stack(
        [
            TENSORFIELDS[candidate.schema.requests[p.address].type].loss(
                module=candidate, prediction=p, batch=inputs[p.address], strata=Strata.train
            )
            * torch.tensor(candidate.schema.requests[p.address].weight)
            for p in candidate(inputs, strata=Strata.train)
        ]
    ).sum()
    actual = runtime.step(candidate, inputs, 0, strata=Strata.train)["loss"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
