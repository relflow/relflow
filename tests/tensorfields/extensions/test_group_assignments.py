"""Tests for persistent Group assignments and scoped overrides."""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.enums import Strata, Tokens
from relflow.structs.tree import Address
from relflow.tensorfields.extensions.group import Embedder, TensorField

CAPACITY = 8
K = 4
ADDRESS = "/merchant_id"
NON_INTEGER_ASSIGNMENTS = [
    pytest.param([0.05, 0.05, 0.6, 0.3], id="list"),
    pytest.param((0.0, 0.0, 1.0, 0.0), id="tuple"),
    pytest.param(torch.tensor([0.0, 0.0, 1.0, 0.0]), id="tensor"),
    pytest.param(1.5, id="float"),
    pytest.param(True, id="bool"),
]


def build_model(*, embed: bool = False) -> rf.Model:
    torch.manual_seed(0)
    return rf.Model.xs(
        d_model=16,
        batch_size=4,
        embed=embed,
        merchant_id=rf.Group(n_groups=K, p_unavailable=0.0),
    )


def group_embedder(model: rf.Model) -> Embedder:
    embedder = model.nodes[Address(ADDRESS)].embedder
    assert isinstance(embedder, Embedder)
    return embedder


def assignment(model: rf.Model, token: str) -> int:
    return rf.Group.assignments(model, ADDRESS)[token]["group"]


def valued_token(index: int) -> TensorField:
    return TensorField(
        state=torch.tensor([[Tokens.valued.value]], dtype=torch.int64),
        content=torch.tensor([[index]], dtype=torch.int64),
        present=torch.ones((1, 1), dtype=torch.bool),
        trainable=torch.zeros((1, 1), dtype=torch.bool),
        inferred=torch.zeros((1, 1), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )


# ---------- persistent assignments ----------


def test_assign_grows_vocab_and_writes_row_for_int_group():
    model = build_model()
    embedder = group_embedder(model)
    assert len(embedder.vocab.master) == 0

    index = rf.Group.assign(model, ADDRESS, "brand-new", 2)

    assert index == 0
    assert list(embedder.vocab.master) == ["brand-new"]
    row = embedder.embeddings["group"].weight[index]
    assert row.shape == (1,)
    assert row.item() == pytest.approx(math.pi)
    assert assignment(model, "brand-new") == 2


@pytest.mark.parametrize("value", NON_INTEGER_ASSIGNMENTS)
def test_assign_rejects_non_integer_assignments(value):
    model = build_model()
    embedder = group_embedder(model)

    with pytest.raises((TypeError, ValueError)):
        rf.Group.assign(model, ADDRESS, "invalid", value)
    assert list(embedder.vocab.master) == []


def test_assign_overwrites_existing_token():
    model = build_model()
    embedder = group_embedder(model)
    idx = rf.Group.assign(model, ADDRESS, "shared-token", 1)
    assert assignment(model, "shared-token") == 1

    idx_again = rf.Group.assign(model, ADDRESS, "shared-token", 3)

    assert idx_again == idx
    assert len(embedder.vocab.master) == 1
    assert assignment(model, "shared-token") == 3


def test_assign_grows_when_capacity_full():
    model = build_model()
    embedder = group_embedder(model)
    for i in range(CAPACITY):
        rf.Group.assign(model, ADDRESS, f"tok-{i}", i % K)
    assert len(embedder.vocab.master) == CAPACITY

    before = embedder.embeddings["group"].weight.detach().clone()
    assert rf.Group.assign(model, ADDRESS, "one-more", 0) == CAPACITY
    assert embedder.capacity == CAPACITY * 2
    torch.testing.assert_close(embedder.embeddings["group"].weight[:CAPACITY], before[:CAPACITY])
    torch.testing.assert_close(embedder.embeddings["group"].weight[-1], before[-1])


@pytest.mark.parametrize("value", [-1, K])
def test_assign_rejects_out_of_range_group(value):
    model = build_model()

    with pytest.raises(ValueError, match=r"group must be in \[0, 4\)"):
        rf.Group.assign(model, ADDRESS, "tok", value)


def test_assign_requires_exactly_one_assignment():
    model = build_model()

    with pytest.raises(TypeError):
        rf.Group.assign(model, ADDRESS, "tok")
    with pytest.raises(TypeError):
        rf.Group.assign(model, ADDRESS, "tok", 0, 1)


def test_assign_raises_for_unknown_address():
    model = build_model()

    with pytest.raises(KeyError):
        rf.Group.assign(model, "/nonexistent", "tok", 0)


def test_assign_raises_for_non_group_field():
    torch.manual_seed(0)
    model = rf.Model.xs(
        d_model=16,
        batch_size=4,
        amount=rf.Number,
    )

    with pytest.raises(TypeError, match="not a Group field"):
        rf.Group.assign(model, "/amount", "tok", 0)


# ---------- scoped overrides ----------


def test_override_applies_only_within_prediction_scope():
    model = build_model(embed=True)
    embedder = group_embedder(model)
    initial_vocab_size = len(embedder.vocab.master)
    initial_row_2 = embedder.embeddings["group"].weight[min(initial_vocab_size, CAPACITY)].detach().clone()

    with rf.Group.override(model, ADDRESS, {"overridden": 3}):
        assert assignment(model, "overridden") == 3
        model.predict(pa.table({"merchant_id": ["overridden"]}))

    assert len(embedder.vocab.master) == initial_vocab_size
    # The row that got temporarily written must be reverted to its original values.
    reverted_row = embedder.embeddings["group"].weight[min(initial_vocab_size, CAPACITY)]
    assert torch.equal(reverted_row, initial_row_2)


def test_override_is_used_by_tensorization():
    model = build_model()

    with rf.Group.override(model, ADDRESS, {"overridden": 3}):
        overridden = model.encode(pa.table({"merchant_id": ["overridden"]}))[Address(ADDRESS)]
        assert int(overridden.content.item()) == 0
        assert int(overridden.state.item()) == Tokens.valued.value

    unavailable = model.encode(pa.table({"merchant_id": ["overridden"]}))[Address(ADDRESS)]
    assert int(unavailable.content.item()) == -1


def test_override_rolls_back_when_prediction_scope_raises():
    model = build_model()
    embedder = group_embedder(model)
    original = embedder.embeddings["group"].weight[0].detach().clone()

    with pytest.raises(RuntimeError, match="prediction failed"):
        with rf.Group.override(model, ADDRESS, {"overridden": 3}):
            assert embedder.vocab.state.index["overridden"] == 0
            raise RuntimeError("prediction failed")

    assert "overridden" not in embedder.vocab.state.index
    assert torch.equal(embedder.embeddings["group"].weight[0], original)


def test_override_rolls_back_when_entering_scope_raises():
    model = build_model()
    embedder = group_embedder(model)
    original = embedder.embeddings["group"].weight[0].detach().clone()

    with pytest.raises(ValueError, match=r"group must be in \[0, 4\)"):
        with rf.Group.override(model, ADDRESS, {"valid": 2, "invalid": K}):
            pytest.fail("invalid overrides should fail before entering the scope")

    assert list(embedder.vocab.master) == []
    assert torch.equal(embedder.embeddings["group"].weight[0], original)


def test_override_restores_an_existing_assignment():
    model = build_model()
    embedder = group_embedder(model)
    index = rf.Group.assign(model, ADDRESS, "existing", 1)
    original = embedder.embeddings["group"].weight[index].detach().clone()

    with rf.Group.override(model, ADDRESS, {"existing": 3}):
        assert assignment(model, "existing") == 3

    assert list(embedder.vocab.master) == ["existing"]
    assert embedder.vocab.state.index["existing"] == index
    assert torch.equal(embedder.embeddings["group"].weight[index], original)
    assert assignment(model, "existing") == 1


def test_override_blocks_checkpointing_and_model_mutation(tmp_path: Path):
    model = build_model()

    with rf.Group.override(model, ADDRESS, {"temporary": 2}):
        with pytest.raises(RuntimeError, match="assignment overrides are active"):
            model.save(tmp_path / "overridden.rf")
        with pytest.raises(RuntimeError, match="active loop"):
            model.reset(ADDRESS)

    assert "temporary" not in group_embedder(model).vocab.state.index


def test_runtime_assignments_reject_active_models_and_shared_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
):
    model = build_model()
    embedder = group_embedder(model)

    model.locks[Strata.train] += 1
    try:
        with pytest.raises(RuntimeError, match="model is active"):
            rf.Group.assign(model, ADDRESS, "active", 1)
    finally:
        model.locks.pop(Strata.train)

    monkeypatch.setattr(type(embedder.vocab), "is_shared", property(lambda self: True))
    with pytest.raises(RuntimeError, match="vocabulary state is shared"):
        with rf.Group.override(model, ADDRESS, {"shared": 1}):
            pytest.fail("shared vocabulary should fail before entering the scope")


def test_override_replaces_encoder_row_for_oov_token():
    """The encoder consumes the override row instead of the OOV sentinel.

    Overrides act on the encoder path; the write() group.value comes from the
    decoder, so probe the embedder's forward output directly.
    """
    model = build_model()
    embedder = group_embedder(model)
    embedder.eval()

    with torch.no_grad():
        embedder.embeddings["group"].weight[embedder.capacity].zero_()

    with rf.Group.override(model, ADDRESS, {"overridden-oov": 2}):
        # During the override, this token is encoded as valued/index 0.
        overridden_index = embedder.vocab.state.index["overridden-oov"]
        overridden_output = embedder.embed(valued_token(overridden_index)).payload.clone()

    # The reference angle is the center of group 2.
    with torch.no_grad():
        embedder.embeddings["group"].weight[0].fill_(2 * math.pi * 2 / K)
    reference_output = embedder.embed(valued_token(0)).payload.clone()

    assert torch.allclose(overridden_output, reference_output)


@pytest.mark.parametrize("value", NON_INTEGER_ASSIGNMENTS)
def test_override_rejects_non_integer_assignments(value):
    model = build_model()
    embedder = group_embedder(model)

    with pytest.raises((TypeError, ValueError)):
        with rf.Group.override(model, ADDRESS, {"invalid": value}):
            pytest.fail("a non-integer assignment must not enter the override scope")
    assert list(embedder.vocab.master) == []


def test_persistent_assignment_survives_predict_calls():
    model = build_model()
    embedder = group_embedder(model)
    embedder.eval()

    rf.Group.assign(model, ADDRESS, "returning-token", 3)
    assigned_index = embedder.vocab.state.index["returning-token"]
    first_row = embedder.embeddings["group"].weight[assigned_index].detach().clone()

    # Simulate a subsequent predict call by running the embedder again.
    _ = embedder.embed(valued_token(assigned_index))
    second_row = embedder.embeddings["group"].weight[assigned_index].detach().clone()

    assert torch.equal(first_row, second_row)
    assert assignment(model, "returning-token") == 3


def test_persistent_assignment_round_trips_through_checkpoint(tmp_path: Path):
    model = build_model()
    index = rf.Group.assign(model, ADDRESS, "persistent", 2)
    pathname = tmp_path / "group.rf"

    model.save(pathname)
    restored = rf.Model.load(pathname)
    embedder = group_embedder(restored)

    assert embedder.vocab.state.index["persistent"] == index
    assert assignment(restored, "persistent") == 2


def test_assign_resolves_the_live_embedder_after_reset():
    model = build_model()
    original = group_embedder(model)

    model.reset(rf.where("address") == ADDRESS)
    rebuilt = group_embedder(model)
    assert rebuilt is not original

    index = rf.Group.assign(model, ADDRESS, "after-reset", 2)
    assert rebuilt.vocab.master[index] == "after-reset"
    assert assignment(model, "after-reset") == 2


def test_group_api_does_not_leak_into_model_or_runtime_signatures():
    assert not hasattr(rf.Group, "bind")
    assert "group_hints" not in inspect.signature(rf.Model.predict).parameters
    assert not hasattr(rf.Model, "commit_group")
    assert not hasattr(rf.Model, "_group_embedder")
