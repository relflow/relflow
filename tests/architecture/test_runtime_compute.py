import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture.runtime import ModelRuntime, execution
from relflow.structs.enums import Strata, TensorKey
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS
from tests.architecture.test_extension_contract import build_extension


@pytest.mark.parametrize("entry", ["forward", "training_step"])
def test_forward_and_training_step_share_compute_after_host_preparation(monkeypatch, entry):
    model = rf.Model(
        value=rf.Number,
        label=rf.Boolean(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
        batch_size=2,
    )
    data = rf.ArrowDataModule(
        model=model,
        train=pa.table({"value": [1.0, 5.0], "label": [False, True]}),
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    batch = next(iter(data.train_dataloader()))
    assert rf.Number.normalization(model, "/value")["count"] == 0
    calls = []
    original = ModelRuntime.compute

    def compute(module, inputs, *, plan, decoders, predict):
        calls.append((module, inputs, decoders, predict, rf.Number.normalization(module, "/value")))
        return original(module, inputs, plan=plan, decoders=decoders, predict=predict)

    monkeypatch.setattr(ModelRuntime, "compute", staticmethod(compute))
    monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
    if entry == "forward":
        predictions = model(batch.tensors, strata=" TrAiN ")
        prediction = next(value for value in predictions if value.address == "/label")
        loss = prediction.payload[TensorKey.state].square().mean()
    else:
        result = model.training_step(batch, 0)
        assert result is not None
        loss = result["loss"]

    assert len(calls) == 1
    module, inputs, decoders, predict, normalization = calls[0]
    assert module is model and inputs is batch.tensors
    assert not predict
    assert tuple(route.address for route in decoders) == (Address("/label"),)
    assert normalization["count"] == (2 if entry == "training_step" else 0)
    assert torch.isfinite(loss)
    loss.backward()
    gradient = model.nodes["/label"].decoder.state.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


@pytest.mark.parametrize("entry", ["forward", "training_step"])
def test_host_validation_rejects_invalid_inputs_before_compute(monkeypatch, entry):
    model = rf.Model(value=rf.Number, label=rf.Boolean(mask=True), d_model=8, n_layers=1, n_heads=2)
    inputs = model.encode(pa.table({"value": [1.0], "label": [True]}), strata=Strata.train)
    inputs["/value"].state = inputs["/value"].state.float()

    def compute(*args, **kwargs):
        pytest.fail("invalid inputs reached the tensor computation")

    monkeypatch.setattr(ModelRuntime, "compute", staticmethod(compute))
    with pytest.raises(TypeError, match="state must use an integer dtype"):
        if entry == "forward":
            model(inputs, strata="train")
        else:
            model.training_step(inputs, 0)


def test_compute_preserves_late_extension_nested_content_and_module_hooks():
    extension, Request = build_extension(decoder=False, loss=False)
    try:
        model = rf.Model(
            value=Request(mask=rf.Mask(query="skip", skip=True, dropout=False)),
            d_model=8,
            n_layers=1,
            n_heads=2,
            embed=True,
        )
        inputs = model.encode(
            pa.table({"value": ["a", "bb", "ccc"], "skip": [False, True, False]}),
            strata=Strata.train,
        )
        embedder = model.nodes["/value"].embedder
        seen = []
        handle = embedder.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
        try:
            public = model(inputs, strata=Strata.train)
            computed = ModelRuntime.compute(model, inputs, plan=execution(model), decoders=(), predict=False)
        finally:
            handle.remove()

        assert len(seen) == 2
        for compact in seen:
            assert compact.content["matrix"].shape == (2, 2, 3)
            assert compact.content["nested", "cube"].shape == (2, 2, 2, 2)
        assert [prediction.address for prediction in computed] == [Address("/")]
        actual = computed[0].payload[TensorKey.embedding]
        torch.testing.assert_close(actual, public[0].payload[TensorKey.embedding], rtol=0, atol=0)
        assert actual.shape == (3, 8)
        assert torch.count_nonzero(actual[1]) == 0
        (actual * torch.arange(8.0)).sum().backward()
        gradient = embedder.projection.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    finally:
        TENSORFIELDS.pop(extension.name, None)


def test_execution_plan_follows_schema_updates_resets_and_checkpoint_restore():
    model = rf.Model(value=rf.Number, d_model=8, n_layers=1, n_heads=2, embed=True)
    source = pa.table({"value": [1.0, 2.0]})
    inputs = model.encode(source)
    state = {name: value.clone() for name, value in model.state_dict().items()}
    checkpoint = {
        "schema": model.schema.model_dump(mode="python"),
        "state_dict": state,
        "batch_size": model.batch_size,
        "version": model.version,
    }
    original = model(inputs, strata=Strata.predict)
    cached = execution(model)
    assert execution(model) is cached
    assert model.state_dict().keys() == state.keys()

    model.update(rf.where("name") == "value", embed=True)
    assert model.execution_plan is None
    updated = model(inputs, strata=Strata.predict)
    assert [prediction.address for prediction in updated] == ["/", "/value"]
    assert execution(model) is not cached

    model.reset(rf.where("name") == "value")
    assert model.execution_plan is None
    assert [prediction.address for prediction in model(inputs, strata=Strata.predict)] == ["/", "/value"]

    model.restore_checkpoint_state(checkpoint)
    assert model.execution_plan is None
    restored = model(inputs, strata=Strata.predict)
    assert [prediction.address for prediction in restored] == ["/"]
    torch.testing.assert_close(
        restored[0].payload[TensorKey.embedding], original[0].payload[TensorKey.embedding], rtol=0, atol=0
    )
    assert model.state_dict().keys() == state.keys()


def test_failed_checkpoint_restore_keeps_original_execution_plan():
    model = rf.Model(value=rf.Number, d_model=8, n_layers=1, n_heads=2, embed=True)
    inputs = model.encode(pa.table({"value": [1.0]}))
    original = model(inputs, strata=Strata.predict)
    cached = execution(model)
    incomplete = dict(model.state_dict())
    incomplete.pop(next(iter(incomplete)))
    checkpoint = {
        "schema": model.schema.model_dump(mode="python"),
        "state_dict": incomplete,
        "batch_size": model.batch_size,
        "version": model.version,
    }

    with pytest.raises(RuntimeError, match="Missing key"):
        model.restore_checkpoint_state(checkpoint)

    assert execution(model) is cached
    restored = model(inputs, strata=Strata.predict)
    torch.testing.assert_close(
        restored[0].payload[TensorKey.embedding], original[0].payload[TensorKey.embedding], rtol=0, atol=0
    )


def test_cached_schema_plan_preserves_live_decoder_context_routes():
    model = rf.Model(value=rf.Number, label=rf.Number(mask=True), d_model=8, n_layers=1, n_heads=2)
    inputs = model.encode(pa.table({"value": [1.0]}))
    decoder = model.nodes["/label"].decoder
    seen = []
    handle = decoder.register_forward_pre_hook(
        lambda module, args, kwargs: seen.append(tuple(parcel.origin for parcel in kwargs["contexts"])),
        with_kwargs=True,
    )
    try:
        model(inputs, strata=Strata.predict)
        cached = execution(model)
        decoder.context_addresses = ()
        model(inputs, strata=Strata.predict)
    finally:
        handle.remove()

    assert execution(model) is cached
    assert seen == [(Address("/value"),), ()]
