from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.contracts import ForwardContractError, same_device
from relflow.structs.enums import Strata, Tokens
from relflow.structs.tree import Address
from tests.arrow import batch as arrow_batch


def build(**fields: rf.TreeFieldInput) -> rf.Model:
    return rf.Model(
        **fields,
        d_model=8,
        n_layers=1,
        n_heads=4,
        batch_size=2,
        attention=None,
    )


def prepared(model: rf.Model, rows: list[dict] | None = None, strata: Strata = Strata.train) -> TensorDict:
    return model.encode(
        arrow_batch(
            rows
            or [
                {"color": "red", "amount": 1.0, "label": "warm"},
                {"color": "blue", "amount": 2.0, "label": "cool"},
            ]
        ),
        strata=strata,
    )


def test_forward_contract_canonicalizes_default_accelerator_device_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    assert same_device(torch.device("mps"), torch.device("mps:0"))

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    assert same_device(torch.device("cuda"), torch.device("cuda:0"))
    assert not same_device(torch.device("cuda"), torch.device("cuda:1"))


def test_forward_contract_rejects_missing_active_field() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)

    del inputs[Address("/color")]

    with pytest.raises(ForwardContractError, match="missing active request"):
        model(inputs, strata=Strata.train)


def test_forward_contract_requires_strata() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)

    with pytest.raises(TypeError, match="strata"):
        model(inputs)  # ty: ignore[missing-argument]


def test_forward_contract_rejects_unknown_extra_field() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)
    inputs[Address("/extra")] = inputs[Address("/color")].clone()

    with pytest.raises(ForwardContractError, match="unknown address"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_inactive_request_field() -> None:
    model = build(color=rf.Category(), ignored=rf.Category(active=False))
    inputs = prepared(model)
    inputs[Address("/ignored")] = inputs[Address("/color")].clone()

    with pytest.raises(ForwardContractError, match="inactive request"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_branch_address_field() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)
    inputs[Address("/")] = inputs[Address("/color")].clone()

    with pytest.raises(ForwardContractError, match="branch address"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_tensorfield_class() -> None:
    model = build(color=rf.Category(), amount=rf.Number())
    inputs = prepared(model)
    inputs[Address("/color")] = inputs[Address("/amount")].clone()

    with pytest.raises(TypeError, match="must use tensorfield class"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_state_shape() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)
    inputs[Address("/color")].state = inputs[Address("/color")].state[:, :0]

    with pytest.raises(ForwardContractError, match="state must have shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_wrong_state_dtype() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)
    inputs[Address("/color")].state = inputs[Address("/color")].state.to(torch.float32)

    with pytest.raises(TypeError, match="state must use an integer dtype"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_content_without_state_shape_prefix() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)
    inputs[Address("/color")].content = inputs[Address("/color")].content[:, :0]

    with pytest.raises(ForwardContractError, match="content.*state shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_allows_masked_non_trainable_input() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)
    field = inputs[Address("/color")]
    field.state[0, 0] = Tokens.masked.value

    model(inputs, strata=Strata.train)


def test_forward_contract_allows_masked_non_trainable_predict_input() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model, strata=Strata.predict)
    field = inputs[Address("/color")]
    field.state[0, 0] = Tokens.masked.value

    model(inputs, strata=Strata.predict)


def test_forward_contract_rejects_trainable_input_without_targets() -> None:
    model = build(color=rf.Category(mask=rf.Mask(dropout=False, reconstruct=True)))
    inputs = prepared(model)
    field = inputs[Address("/color")]
    field.targets = TensorDict({}, batch_size=field.state.shape)

    with pytest.raises(ForwardContractError, match=r"lacks targets\[state\]"):
        model(inputs, strata=Strata.train)


def test_forward_contract_rejects_presence_state_disagreement() -> None:
    model = build(color=rf.Category(), label=rf.Category(mask=True))
    inputs = prepared(model)
    inputs[Address("/label")].state[0, 0] = Tokens.valued.value

    with pytest.raises(ForwardContractError, match="present must be true exactly"):
        model(inputs, strata=Strata.train)


def test_forward_contract_allows_predict_target_placeholder() -> None:
    model = build(color=rf.Category(), label=rf.Category(mask=True))
    inputs = prepared(
        model,
        rows=[
            {"color": "red"},
            {"color": "blue"},
        ],
        strata=Strata.predict,
    )

    predictions = model(inputs, strata=Strata.predict)

    assert any(prediction.address == Address("/label") for prediction in predictions)


def test_forward_contract_rejects_predict_placeholder_in_train_strata() -> None:
    model = build(color=rf.Category(), label=rf.Category(mask=True))
    inputs = prepared(
        model,
        rows=[
            {"color": "red"},
            {"color": "blue"},
        ],
        strata=Strata.predict,
    )
    with pytest.raises(ForwardContractError, match="cannot be inferred during train"):
        model(inputs, strata=Strata.train)


def test_forward_contract_uses_deterministic_backoff_schedule() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("/color")].present[0, 0] = False

    model(inputs, strata=Strata.train)
    with pytest.raises(ForwardContractError, match="present must be true exactly"):
        model(inputs, strata=Strata.train)


def test_forward_contract_runs_when_batch_signature_changes() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("/color")].content = inputs[Address("/color")].content[:, :0]

    with pytest.raises(ForwardContractError, match="content.*state shape"):
        model(inputs, strata=Strata.train)


def test_forward_contract_runs_when_dataloader_index_changes() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)

    for _ in range(3):
        model(inputs, strata=Strata.train, dataloader_idx=0)

    inputs[Address("/color")].present[0, 0] = False

    with pytest.raises(ForwardContractError, match="present must be true exactly"):
        model(inputs, strata=Strata.train, dataloader_idx=1)


def test_forward_contract_resets_after_schema_mutation() -> None:
    model = build(color=rf.Category())
    inputs = prepared(model)

    for _ in range(3):
        model(inputs, strata=Strata.train)

    inputs[Address("/color")].present[0, 0] = False

    model(inputs, strata=Strata.train)
    model.reset(rf.where("name") == "color")

    with pytest.raises(ForwardContractError, match="present must be true exactly"):
        model(inputs, strata=Strata.train)
