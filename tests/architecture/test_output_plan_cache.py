"""An evaluation model reuses its compiled output plan across prediction calls."""

import pyarrow as pa

import relflow as rf
from relflow.architecture import output


def test_evaluation_prediction_reuses_the_plan_and_training_refreshes_it(monkeypatch):
    model = rf.Model.xs(d_model=8, amount=rf.Number, label=rf.Boolean(mask=True)).eval()
    table = pa.table({"amount": [1.0, 2.0], "label": [True, False]})
    refreshes = []
    plan = output.plan
    monkeypatch.setattr(
        output,
        "plan",
        lambda module, retain=(), *, refresh=False: (refreshes.append(refresh), plan(module, retain, refresh=refresh))[
            1
        ],
    )

    first = model.predict(table)
    compiled = model.output_plans[(model._contract_generation, ())]
    second = model.predict(table, batch_size=1)

    assert refreshes == [False, False]
    assert model.output_plans[(model._contract_generation, ())] is compiled
    assert first.schema == second.schema

    model.train()
    model.predict(table)
    assert refreshes[-1] is True

    model.eval()
    model.extend(rf.where("address") == "/", flag=rf.Boolean(mask=True))
    model.predict(table)
    assert model.output_plans[(model._contract_generation, ())] is not compiled
