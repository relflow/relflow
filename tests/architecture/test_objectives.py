import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture import runtime
from relflow.architecture.runtime import execution
from relflow.structs.enums import Strata, TensorKey
from relflow.tensorfields.base import TENSORFIELDS


def model() -> rf.Model:
    return rf.Model(
        preset="small",
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


@pytest.fixture
def descent():
    pytest.importorskip("torchjd")
    pytest.importorskip("quadprog")
    from relflow.architecture import jacobian

    return jacobian


def contrastive(*, label: bool = False, weight: float = 1.0) -> rf.Model:
    fields = {"label": rf.Boolean(mask=True)} if label else {}
    return rf.Model(
        preset="small",
        d_model=8,
        dropout=0.0,
        **fields,
        ett=rf.Branch.chronological(
            length=8,
            reduction=rf.Mean(),
            x=rf.Number,
            contrastive=rf.Contrastive(views=rf.Crops(width=4, overlap=0.5), weight=weight),
        ),
    )


def rows(count: int) -> list[dict]:
    return [{"x": float(i)} for i in range(count)]


ETT = pa.list_(pa.struct([("x", pa.float64())]))  # explicit: all-empty or null batches infer no type


def batch(model, collections, strata="train"):
    return model.encode(pa.table({"ett": pa.array(collections, type=ETT)}), strata=strata, seed=0)


def test_contrastive_only_model_has_an_objective_and_steps(monkeypatch):
    model = contrastive(weight=2.0)
    model.on_fit_start()
    entries = []
    monkeypatch.setattr(rf.Model, "log", lambda self, **options: entries.append(options))
    output = runtime.step(model, batch(model, [rows(8), rows(6)]), 0, strata=Strata.train)
    names = {entry["name"]: entry["value"] for entry in entries}
    torch.testing.assert_close(names["loss/train"], 2.0 * names[".ett/train.loss"])
    output["loss"].backward()
    assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in model.nodes["/ett"].encoder.parameters())


def test_short_empty_and_null_records_give_a_finite_contrastive_loss():
    model = contrastive()
    output = runtime.step(model, batch(model, [rows(2), rows(1), [], None]), 0, strata=Strata.train)
    assert torch.isfinite(output["loss"]) and output["loss"].requires_grad


def test_rank_without_contrastive_rows_still_returns_an_anchored_loss(monkeypatch):
    model = contrastive()
    monkeypatch.setattr(runtime, "participation", lambda module, inputs, strata: (set(), {rf.Address("/ett")}))
    output = runtime.step(model, batch(model, [[], None]), 0, strata=Strata.train)
    assert output["loss"].item() == 0.0 and output["loss"].requires_grad
    output["loss"].backward()
    assert all(p.grad is not None for p in model.nodes["/ett"].encoder.parameters() if p.requires_grad)


@pytest.mark.parametrize(
    "options",
    [
        {"mixer": None},
        {"grain": rf.Coordinates(), "mixer": rf.Transformer()},
        {"direction": rf.Direction.bidirectional, "grain": rf.Coordinates(), "mixer": rf.Transformer()},
    ],
    ids=["idle", "coordinates", "bidirectional"],
)
def test_contrastive_runs_without_a_mixer_under_coordinates_and_bidirectional(options):
    model = rf.Model(
        preset="small",
        d_model=8,
        dropout=0.0,
        ett=rf.Branch.chronological(
            length=8, reduction=rf.Mean(), x=rf.Number, contrastive=rf.Contrastive(views=rf.Crops(width=4)), **options
        ),
    )
    output = runtime.step(model, batch(model, [rows(8), rows(3)]), 0, strata=Strata.train)
    assert torch.isfinite(output["loss"]) and output["loss"].requires_grad


@pytest.mark.parametrize("strata", [Strata.validate, Strata.predict])
def test_contrastive_never_runs_outside_training(strata):
    model = contrastive(label=True)
    table = pa.Table.from_pylist([{"label": True, "ett": rows(8)}])
    inputs = model.encode(table, strata=strata)
    assert all(TensorKey.objective not in p.payload.keys() for p in model(inputs, strata=strata))
    assert rf.Address("/ett") not in runtime.participation(model, inputs, strata)[1]


def test_objectives_follow_the_schema_order_of_leaves_and_branches():
    model = contrastive(label=True)
    assert [entry.address for entry in execution(model).objectives] == ["/label", "/ett"]
    assert [entry.weight for entry in execution(contrastive(weight=3.0)).objectives] == [3.0]


def test_jacobian_combines_the_contrastive_objective_with_leaf_losses(descent, monkeypatch):
    model = contrastive(label=True)
    model.jacobian = True
    monkeypatch.setattr(model, "track", lambda names, value, **kwargs: value)
    captured = []
    original = descent.combine
    monkeypatch.setattr(
        descent, "combine", lambda module, losses, anchor: captured.extend(losses) or original(module, losses, anchor)
    )
    table = pa.Table.from_pylist([{"label": True, "ett": rows(8)}, {"label": False, "ett": rows(6)}])
    model.training_step(model.encode(table, strata="train"), 0)
    assert len(captured) == 2 and all(loss.requires_grad for loss in captured)


@pytest.mark.integration
def test_contrastive_only_model_trains(tmp_path):
    model = contrastive()
    table = pa.Table.from_pylist([{"ett": rows(count)} for count in (8, 6, 3, 8)])
    datamodule = rf.ArrowDataModule(
        model=model, train=table, validate=table, num_workers=0, persistent_workers=False, pin_memory=False
    )
    # A contrastive-only model logs no validation loss, so nothing may monitor it.
    model.fit(
        datamodule,
        early_stopping=False,
        rollback=False,
        batch_size=4,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        default_root_dir=tmp_path,
    )
    assert model.trainer.global_step == 1
