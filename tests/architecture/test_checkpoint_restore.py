"""Transactional checkpoint restoration contracts."""

from __future__ import annotations

import pytest
import torch

import relflow as rf


def model() -> rf.Model:
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=4,
        attention="none",
        reduction=rf.Attention(n_layers=2),
        nested=rf.Branch(
            length=2,
            n_layers=1,
            n_heads=4,
            attention="none",
            reduction=rf.Attention(n_layers=2),
            value=rf.Number,
        ),
        answer=rf.Number(mask=True),
    )


def checkpoint(configured: rf.Model, *, schema: dict | None = None, state_dict: dict | None = None) -> dict:
    return {
        "state_dict": configured.state_dict() if state_dict is None else state_dict,
        "schema": configured.schema.model_dump(mode="python") if schema is None else schema,
        "batch_size": configured.batch_size,
        "version": configured.version,
    }


def target() -> rf.Model:
    return rf.Model(
        original=rf.Number,
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=3,
    ).eval()


def snapshot(configured: rf.Model) -> tuple[object, object, object, dict[str, torch.Tensor]]:
    return (
        configured.schema,
        configured.nodes,
        configured.example_input_array,
        {name: value.detach().clone() for name, value in configured.state_dict().items()},
    )


def assert_unchanged(
    configured: rf.Model,
    before: tuple[object, object, object, dict[str, torch.Tensor]],
) -> None:
    schema, nodes, example, state = before
    assert configured.schema is schema
    assert configured.nodes is nodes
    assert configured.example_input_array is example
    assert configured.batch_size == 3
    assert not configured.training
    assert all(not child.training for child in configured.modules())
    assert configured.state_dict().keys() == state.keys()
    for name, value in configured.state_dict().items():
        assert torch.equal(value, state[name])


def test_failed_in_place_state_restore_is_transactional() -> None:
    source = model()
    state_dict = dict(source.state_dict())
    state_dict.pop("nodes.record.encoder.pool.mass_direction")
    configured = target()
    before = snapshot(configured)

    with pytest.raises(RuntimeError, match="Missing key"):
        configured.restore_checkpoint_state(checkpoint(source, state_dict=state_dict))

    assert_unchanged(configured, before)


def test_failed_in_place_graph_install_is_transactional() -> None:
    configured = target()
    schema = configured.schema.model_dump(mode="python")
    schema["fields"]["n_heads"] = 6
    schema["fields"]["reduction"] = {"type": "mean"}
    before = snapshot(configured)

    with pytest.raises(ValueError, match="d_model must be divisible by nhead"):
        configured.restore_checkpoint_state(checkpoint(configured, schema=schema))

    assert_unchanged(configured, before)
