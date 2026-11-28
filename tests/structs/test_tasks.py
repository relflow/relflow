"""Task declarations own compatibility, independent of input visibility or decoder routing."""

from typing import Literal, get_type_hints

import polars as pl
import pytest

import relflow as rf
from relflow.structs import tasks


def test_supervision_groups_declarations_and_reconstruction_owns_its_remaining_leaves(tmp_path):
    assert get_type_hints(rf.Task.conflict)["shared"] == frozenset[str]
    rule: rf.Task = rf.reconstruct(query="review", ignore="labels", ablate=True)
    model = rf.Model(
        preset="small",
        task=rule,
        reading=rf.Number(hide=0.2),
        labels=rf.Branch(predict=True, grade=rf.Number(predict=True, ablate=0.4), valid=rf.Boolean),
        unused=rf.Number(active=False, predict=True),
    )
    supervised, reconstructed = model.schema.tasks
    assert supervised.task.kind == "supervised"
    assert supervised.owners == ("/labels", "/labels/grade")
    assert supervised.targets == ("/labels/grade", "/labels/valid")
    assert reconstructed.task == rule and reconstructed.owners == ("/",)
    assert reconstructed.targets == ("/reading",)
    assert model.schema.reconstruct == ["/reading"]
    checkpoint = model.save(tmp_path / "tasks.ckpt")
    assert rf.Model.load(checkpoint).schema.tasks == model.schema.tasks
    assert rf.inspect(checkpoint)["tasks"] == [
        {"kind": b.task.kind, "owners": list(b.owners), "targets": list(b.targets)} for b in model.schema.tasks
    ]


@pytest.mark.parametrize("scope", ["leaf", "branch", "root"])
def test_conflicting_task_declarations_report_their_owners_and_shared_answer(scope):
    options = {"task": rf.reconstruct()}
    reading = rf.Number(predict=True, **(options if scope == "leaf" else {}))
    history = rf.Branch(reading=reading, **(options if scope == "branch" else {}))
    with pytest.raises(ValueError, match="task conflict.*supervised.*reconstruct.*same answers at /history/reading"):
        rf.Model(preset="small", history=history, **(options if scope == "root" else {}))


def test_tree_edits_recheck_conflicts_and_restore_the_previous_graph():
    model = rf.Model(preset="small", history=rf.Branch(reading=rf.Number(task=rf.reconstruct(query="review"))))
    history = rf.where("address") == "/history"
    graph = model.nodes["/history/reading"]
    tasks = model.schema.tasks
    with pytest.raises(ValueError, match=r"supervised on /history.*reconstruct on /history/reading"):
        model.update(history, predict=True)
    with pytest.raises(ValueError, match="task conflict"), model.override(history, predict=True):
        pytest.fail("a conflicting override must fail on entry")
    assert model.schema.tasks == tasks and model.nodes["/history/reading"] is graph
    model.update(history, task=rf.reconstruct(rate=0.25))
    with pytest.raises(ValueError, match=r"same answers at /history/new"):
        model.extend(history, new=rf.Number(predict=True))
    assert "/history/new" not in model.schema.requests


def test_each_task_can_veto_disjoint_tasks_in_either_order(monkeypatch):
    monkeypatch.setattr(tasks, "DECLARATIONS", tasks.DECLARATIONS.copy())

    @rf.Task.register
    class Exclusive(rf.Reconstruction):
        kind: Literal["exclusive_test"] = "exclusive_test"

        def conflict(self, other, *, shared):
            return "uses the entire model"

    rule = Exclusive(rate=0.25)
    for fields in (
        {"a": rf.Number(task=rule), "b": rf.Number(task=rf.reconstruct())},
        {"b": rf.Number(task=rf.reconstruct()), "a": rf.Number(task=rule)},
        {"a": rf.Number(task=rule), "b": rf.Number(task=rule)},
    ):
        with pytest.raises(ValueError, match="task conflict.*uses the entire model"):
            rf.Model(preset="small", fields=fields)
    model = rf.Model(preset="small", a=rf.Number(task=rule), context=rf.Number(hide=0.2, ablate=0.4))
    restored = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert type(restored.tasks[0].task) is Exclusive
    assert restored.tasks[0].targets == ("/a",)


def test_leaf_task_validates_its_bound_address_on_build_restore_and_edit(monkeypatch):
    monkeypatch.setattr(tasks, "DECLARATIONS", tasks.DECLARATIONS.copy())

    @rf.Task.register
    class HistoryOnly(rf.Reconstruction):
        kind: Literal["history_only"] = "history_only"

        def validate_placement(self, *, site, address):
            super().validate_placement(site=site, address=address)
            if address is not None and address != "/history/amount":
                raise ValueError(f"{self.kind} requires /history/amount, got {address}")

    amount = rf.Number(task=HistoryOnly(rate=0.2))
    model = rf.Model(preset="small", history=rf.Branch(length=4, amount=amount), amount=rf.Number)
    schema = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert schema.tasks[0].targets == ("/history/amount",)

    with pytest.raises(ValueError, match=r"history_only requires /history/amount, got /orders/amount"):
        rf.Model(preset="small", orders=rf.Branch(length=4, amount=amount))
    payload = schema.model_dump()
    payload["fields"]["fields"][0]["name"] = "orders"
    with pytest.raises(ValueError, match=r"history_only requires /history/amount, got /orders/amount"):
        rf.Schema.model_validate(payload)
    with pytest.raises(ValueError, match=r"history_only requires /history/amount, got /amount"):
        model.update(rf.where("address") == "/amount", task=amount.task)
    assert model.schema.tasks == schema.tasks
    assert model.schema.requests["/amount"].task is None


def test_disabled_tasks_do_not_claim_answers_and_unknown_kinds_fail_before_training():
    model = rf.Model(preset="small", amount=rf.Number(predict=True, task=rf.reconstruct(rate=0.0)))
    assert [b.task.kind for b in model.schema.tasks] == ["supervised"]
    payload = model.schema.model_dump()
    payload["fields"]["fields"][0]["task"] = {"kind": "unknown"}
    with pytest.raises(ValueError, match="unknown explicit task kind 'unknown'"):
        rf.Schema.model_validate(payload)


def test_only_unconditionally_ablated_reconstruction_removes_static_encoder_inputs():
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        task=rf.reconstruct(ablate=True, ignore=("context", "sometimes")),
        context=rf.Branch(value=rf.Number),
        sometimes=rf.Number(task=rf.reconstruct(rate=0.5, ablate=True)),
        excluded=rf.Branch(value=rf.Number),
    )
    assert schema.requests["/context/value"].embedded
    assert schema.requests["/sometimes"].embedded
    assert not schema.requests["/excluded/value"].embedded
    with pytest.raises(ValueError, match="boolean"):
        rf.reconstruct(ablate=0.5)


def test_field_sampling_checks_placement_and_refreshes_candidates_after_tree_edits(tmp_path):
    for value in (0, -1, 0.5, True):
        with pytest.raises(ValueError):
            rf.reconstruct(fields=value)
    with pytest.raises(ValueError, match=r"reconstruct\(fields=.*branch or model.*leaf"):
        rf.Model(preset="small", value=rf.Number(task=rf.reconstruct(fields=1)))
    model = rf.Model(
        preset="small",
        task=rf.reconstruct(fields=8, ignore="context", ablate=True),
        context=rf.Branch(key=rf.Number),
        sample=rf.Branch(x=rf.Number),
    )
    assert list(model.schema.field_pools.values()) == [("/sample/x",)]
    model.extend(rf.where("address") == "/sample", y=rf.Number)
    assert list(model.schema.field_pools.values()) == [("/sample/x", "/sample/y")]
    assert all(r.embedded for r in model.schema.requests.values())
    loaded = rf.Model.load(model.save(tmp_path / "fields.ckpt"))
    assert loaded.schema.field_pools == model.schema.field_pools
    rows = pl.DataFrame({"context": [[{"key": 0}]], "sample": [[{"x": 1, "y": 2}]]})
    batch = loaded.encode(rows, strata="validate")
    assert batch["/sample/x"].trainable.all() and batch["/sample/y"].trainable.all()
    assert batch["/context/key"].targets.is_empty()


def test_registered_kinds_need_unique_names_and_an_execution_implementation(monkeypatch):
    monkeypatch.setattr(tasks, "DECLARATIONS", tasks.DECLARATIONS.copy())

    @rf.Task.register
    class Pending(rf.Task):
        kind: Literal["test_pending"] = "test_pending"

    class Duplicate(Pending):
        pass

    class Reserved(rf.Task):
        kind: Literal["supervised"] = "supervised"

    assert rf.Task.register(Pending) is Pending
    with pytest.raises(ValueError, match="already registered"):
        rf.Task.register(Duplicate)
    with pytest.raises(ValueError, match="supervised is reserved"):
        rf.Task.register(Reserved)
    with pytest.raises(ValueError, match="test_pending.*no execution implementation"):
        rf.Model(preset="small", signal=rf.Number(task=Pending()))


def test_delete_rolls_back_when_it_removes_a_branchs_last_active_input():
    model = rf.Model(
        preset="small",
        history=rf.Branch(hide=0.2, unused=rf.Number(active=False), reading=rf.Number),
        label=rf.Boolean(predict=True),
    )
    schema = model.schema
    original = schema.model_dump_json()
    with pytest.raises(ValueError, match="requires active descendant"):
        schema.delete(rf.where("address") == "/history/reading")
    assert schema.model_dump_json() == original
    assert tuple(schema.branches["/history"].children) == tuple(schema.branches["/history"].fields)
    assert schema.requests["/history/reading"].parent is schema.branches["/history"]


def test_equal_hide_and_reconstruction_rates_draw_separately():
    fields = rf.Model(preset="small", reading=rf.Number(hide=0.5, task=rf.reconstruct(rate=0.5)))
    source = pl.DataFrame({"reading": range(512)})
    train = fields.encode(source, strata="train", seed=83)["/reading"]
    hidden = train.state == rf.Tokens.masked
    assert (hidden & ~train.trainable).any()
    assert (hidden & train.trainable).any() and (~hidden).any()
    assert not (train.trainable & ~hidden).any()
    val = fields.encode(source, strata="validate", seed=83)["/reading"]
    assert ((val.state == rf.Tokens.masked) == val.trainable).all()
