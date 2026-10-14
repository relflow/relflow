from __future__ import annotations

import ast
from collections.abc import Generator
from logging.handlers import BufferingHandler
from pathlib import Path

import pytest
import torch
from lightning.pytorch.utilities.model_summary.model_summary import summarize

import relflow as rf
from relflow.architecture.checkpoint import CheckpointState
from relflow.architecture.graph import ModelGraph
from relflow.architecture.mutations import SchemaEditor
from relflow.logging import logger
from relflow.logging.config import CONTEXT
from relflow.structs import experiment, selectors

FRAMEWORK_PROTOCOL_METHODS = frozenset(
    {
        "_load_from_state_dict",
        "_missing_",
        "_save_to_state_dict",
    }
)


def _model() -> rf.Model:
    return rf.Model(amount=rf.Number(), label=rf.Category(mask=True), d_model=8, n_layers=1, n_heads=2)


@pytest.fixture
def mutation_records() -> Generator[BufferingHandler, None, None]:
    records = BufferingHandler(capacity=100)
    records.addFilter(lambda record: getattr(record, CONTEXT, {}).get("component") == "schema_mutation")
    logger.logger.addHandler(records)
    try:
        yield records
    finally:
        logger.logger.removeHandler(records)


def test_model_uses_mutation_facade() -> None:
    model = _model()

    assert isinstance(model.schema, rf.Schema)
    assert isinstance(model._schema_editor, SchemaEditor)
    assert model.schema.select(rf.where("name") == "amount") == model.select(rf.where("name") == "amount")


def test_model_mutations_emit_one_concise_event_per_operation(mutation_records: BufferingHandler) -> None:
    model = _model()
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
    model = _model()
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
    model = _model()
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
    model = _model()
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
    model = _model()
    seen: set[int] = set()

    def select_once(node):
        assert id(node) not in seen, "mutation evaluated the same node twice"
        seen.add(id(node))
        return node.address == address

    getattr(model, action)(select_once, use_cache=False, **options)
    assert seen


def test_model_graph_rebuild_preserves_compatible_state() -> None:
    model = _model()
    name, before = next(iter(model.state_dict().items()))

    ModelGraph.rebuild(model)

    assert torch.equal(model.state_dict()[name], before)


def test_model_summary_uses_forward_kwargs_example_input() -> None:
    model = _model()

    summary = summarize(model, max_depth=1)

    assert summary.total_parameters > 0


def test_checkpoint_state_round_trip(tmp_path) -> None:
    model = _model()
    path = tmp_path / "model.ckpt"

    CheckpointState.save(model, path)
    restored = CheckpointState.load(rf.Model, path)

    assert restored.schema.model_dump(mode="python") == model.schema.model_dump(mode="python")
    assert restored.batch_size == model.batch_size


def test_experiment_reexports_selector_api() -> None:
    assert experiment.where is selectors.where
    assert experiment.NodePredicate is selectors.NodePredicate
    assert experiment.NodeAttribute is selectors.NodeAttribute


def test_source_functions_and_classes_do_not_use_private_names() -> None:
    source = Path(__file__).parents[2] / "src" / "relflow"
    violations: list[str] = []

    for path in sorted(source.rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef, ast.FunctionDef)):
                continue
            if not node.name.startswith("_") or node.name.startswith("__"):
                continue
            if node.name in FRAMEWORK_PROTOCOL_METHODS:
                continue
            violations.append(f"{path.relative_to(source)}:{node.lineno}: {node.name}")

    assert not violations, "private function/class names:\n" + "\n".join(violations)
