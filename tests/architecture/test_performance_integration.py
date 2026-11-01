"""Optimized execution preserves objectives across mixed ragged field types."""

from datetime import datetime

import pyarrow as pa
import pytest
import torch

import relflow as rf
import relflow.architecture.encoder as encoder
from relflow.architecture import runtime
from relflow.tensorfields.base import TENSORFIELDS, DecoderBase
from tests.architecture.test_extension_contract import build_extension
from tests.tensorfields.extensions.test_text import patch_hf


@pytest.mark.parametrize("embed", [False, True])
def test_ragged_text_and_input_only_fields_preserve_dense_objective_and_gradients(monkeypatch, embed):
    patch_hf(monkeypatch)
    torch.manual_seed(107)
    model = rf.Model(
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        transactions=rf.Branch(
            length=None,
            note=rf.Text(model="local-test", max_length=4, cache=True),
            tags=rf.Set(p_unavailable=0.0),
            occurred=rf.DateParts(dateparts=("day_of_week",)),
            amount=rf.Number,
            risk=rf.Number(
                mask=rf.Mask(query="selected", skip=True, dropout=False, reconstruct=True),
                embed=embed,
            ),
        ),
    )
    records = []
    for length in (5, 3, 0):
        records.append(
            {
                "transactions": [
                    {
                        "note": "purchase" if index % 2 else "refund",
                        "tags": ["card", "online"] if index % 2 else [],
                        "occurred": datetime(2026, 1, index + 1),
                        "amount": float(index + 1),
                        "risk": float(index % 2),
                        "selected": index == 0,
                    }
                    if index != 1
                    else None
                    for index in range(length)
                ]
            }
        )
    source = pa.Table.from_pylist(records)
    fields = model.encode(source, strata="train")
    for address in ("/transactions/tags", "/transactions/occurred"):
        assert fields[address].targets.is_empty()
    monkeypatch.setattr(model, "track", lambda *args, value, **kwargs: value)

    losses, gradients = [], []
    for optimized in (False, True):
        model.zero_grad(set_to_none=True)
        with monkeypatch.context() as mode:
            if not optimized:
                mode.setattr(encoder, "customized", lambda *args, **kwargs: True)
                mode.setattr(type(model.nodes["/transactions/risk"].decoder), "coordinatewise", False)
            result = runtime.step(model, fields, 0, strata=rf.Strata.validate)
            assert result is not None
            loss = result["loss"]
            assert torch.isfinite(loss)
            loss.backward()
        losses.append(loss.detach())
        gradients.append(
            {name: None if value.grad is None else value.grad.clone() for name, value in model.named_parameters()}
        )

    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
    assert gradients[0].keys() == gradients[1].keys()
    for name, expected in gradients[0].items():
        actual = gradients[1][name]
        if expected is None:
            assert actual is None, name
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6, msg=lambda error: f"{name}: {error}")


def test_coordinatewise_extension_with_only_nested_output_can_train(monkeypatch):
    from tensordict import TensorDict

    extension, Request = build_extension(decoder=False, loss=False)

    @extension.register(rf.Component.Decoder)
    class Decoder(DecoderBase):
        coordinatewise = True

        def __init__(self, schema, address):
            super().__init__(schema, address)
            self.state = torch.nn.Linear(schema.d_model, len(rf.Tokens))

        def decode(self, pooled):
            return TensorDict({"nested": TensorDict({"state": self.state(pooled)}, batch_size=pooled.shape[:2])})

    @extension.register(rf.Component.loss)
    def objective(module, prediction, batch, strata):
        selected = batch.trainable.reshape(-1)
        logits = prediction.payload["nested", "state"].reshape(-1, len(rf.Tokens))[selected]
        targets = batch.targets[rf.TensorKey.state].reshape(-1)[selected]
        return torch.nn.functional.cross_entropy(logits, targets)

    try:
        model = rf.Model(
            d_model=8,
            n_heads=2,
            n_layers=1,
            dropout=0.0,
            events=rf.Branch(
                length=None,
                value=rf.Number,
                target=Request(mask=rf.Mask(query="selected", skip=True, dropout=False, reconstruct=True)),
            ),
        )
        fields = model.encode(
            pa.Table.from_pylist(
                [
                    {"events": [{"value": 1.0, "target": "a", "selected": True}, {"value": 2.0, "selected": False}]},
                    {"events": [{"value": 3.0, "target": "b", "selected": True}]},
                ]
            ),
            strata="train",
        )
        monkeypatch.setattr(model, "track", lambda *args, value, **kwargs: value)
        output = model.training_step(fields, 0)
        assert output is not None and torch.isfinite(output["loss"])
        output["loss"].backward()
        gradient = model.nodes["/events/target"].decoder.state.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    finally:
        TENSORFIELDS.pop(extension.name, None)
