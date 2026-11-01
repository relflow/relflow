from __future__ import annotations

from collections.abc import Generator
from copy import deepcopy
from logging.handlers import BufferingHandler

import pytest
import torch
from lightning.pytorch.utilities.model_summary.model_summary import summarize

import relflow as rf
from relflow.architecture.graph import ModelGraph
from relflow.logging import logger
from relflow.logging.config import CONTEXT


def configured() -> rf.Model:
    return rf.Model.xs(amount=rf.Number, label=rf.Category(mask=True), d_model=8)


@pytest.fixture
def mutation_records() -> Generator[BufferingHandler, None, None]:
    records = BufferingHandler(capacity=100)
    records.addFilter(lambda record: getattr(record, CONTEXT, {}).get("component") == "schema_mutation")
    logger.logger.addHandler(records)
    try:
        yield records
    finally:
        logger.logger.removeHandler(records)


def test_model_mutations_emit_one_concise_event_per_operation(mutation_records: BufferingHandler) -> None:
    model = configured()
    selected = rf.where("name").is_in(("amount", "label"))

    model.update(selected, weight=2.0, description="Transaction fields")
    model.extend(rf.where("address") == "/", extra=rf.Number, other=rf.Number)
    model.reset(rf.where("address") == "/", descendants=True)
    with model.override(selected, weight=3.0, description="Temporary fields"):
        pass
    model.delete(rf.where("name").is_in(("extra", "other")))

    assert [getattr(record, CONTEXT) for record in mutation_records.buffer] == [
        {"component": "schema_mutation", "action": action}
        for action in ("update", "extend", "reset", "override", "override_restore", "delete")
    ]


@pytest.mark.parametrize(
    "action,address,options",
    [
        ("update", "/amount", {"weight": -1.0}),
        ("extend", "/", {"amount": rf.Number}),
        ("delete", "/", {"include_root": True}),
        ("reset", "/missing", {}),
        ("override", "/amount", {"weight": -1.0}),
    ],
)
def test_rejected_mutations_do_not_emit_success_events(mutation_records, action, address, options) -> None:
    model = configured()
    before = model.schema.model_dump()
    operation = getattr(model, action)

    with pytest.raises(ValueError):
        if action == "override":
            with operation(rf.where("address") == address, **options):
                pytest.fail("an invalid override entered its body")
        else:
            operation(rf.where("address") == address, **options)

    assert model.schema.model_dump() == before
    assert not mutation_records.buffer


@pytest.mark.parametrize(
    "action,address,options",
    [
        ("update", "/amount", {"weight": 2.0}),
        ("extend", "/", {"extra": rf.Number}),
        ("delete", "/amount", {}),
    ],
)
def test_failed_rebuild_rolls_back_without_success_events(mutation_records, monkeypatch, action, address, options):
    model = configured()
    before = model.schema.model_dump()
    nodes = model.nodes

    def fail(module):
        raise RuntimeError("rebuild failed")

    monkeypatch.setattr(ModelGraph, "rebuild", fail)
    with pytest.raises(RuntimeError, match="rebuild failed"):
        getattr(model, action)(rf.where("address") == address, **options)

    assert model.schema.model_dump() == before
    assert model.nodes is nodes
    assert not mutation_records.buffer


@pytest.mark.parametrize(
    "failure,expected",
    [
        ("entry", ["override_restore"]),
        ("body", ["override", "override_restore"]),
        ("restore", ["override"]),
    ],
)
def test_override_logs_only_completed_transitions(mutation_records, monkeypatch, failure, expected):
    model = configured()
    before = model.schema.model_dump()
    rebuild = ModelGraph.rebuild
    calls = 0

    def rebuild_or_fail(module):
        nonlocal calls
        calls += 1
        if (failure == "entry" and calls == 1) or (failure == "restore" and calls == 2):
            raise RuntimeError(f"{failure} failed")
        rebuild(module)

    monkeypatch.setattr(ModelGraph, "rebuild", rebuild_or_fail)
    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        with model.override(rf.where("name") == "amount", weight=3.0):
            assert model.schema.requests["/amount"].weight == 3.0
            if failure == "body":
                raise RuntimeError("body failed")

    assert model.schema.model_dump() == before
    assert [getattr(record, CONTEXT)["action"] for record in mutation_records.buffer] == expected


@pytest.mark.parametrize(
    "action,address,options",
    [
        ("update", "/amount", {"weight": 2.0}),
        ("extend", "/", {"extra": rf.Number}),
        ("delete", "/amount", {}),
    ],
)
def test_mutation_logging_does_not_evaluate_predicates_again(action, address, options):
    model = configured()
    seen: set[int] = set()

    def select_once(node):
        assert id(node) not in seen, "mutation evaluated the same node twice"
        seen.add(id(node))
        return node.address == address

    getattr(model, action)(select_once, **options)
    assert seen


def test_model_graph_rebuild_preserves_compatible_state() -> None:
    model = configured()
    before = deepcopy(model.state_dict())

    ModelGraph.rebuild(model)

    assert model.state_dict().keys() == before.keys()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_model_summary_uses_forward_kwargs_example_input() -> None:
    model = configured()

    summary = summarize(model, max_depth=1)

    assert summary.total_parameters > 0
