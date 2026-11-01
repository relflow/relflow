"""Objective query selection preserves extension contracts and ragged coordinates."""

from datetime import datetime
from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.runtime import ModelRuntime
from relflow.structs.packages import Parcel
from relflow.tensorfields.base import DecoderBase
from relflow.tensorfields.extensions.number import Decoder as NumberDecoder


def decoder(*, length=8, dropout=0.0, pooling="query", decoder_type=NumberDecoder):
    schema = rf.Schema.from_tree(
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=dropout,
        events=rf.Branch(
            length=length, evidence=rf.Number, target=rf.Number(mask=True, pooling=pooling, dropout=dropout)
        ),
    )
    return decoder_type(schema=schema, address="/events/target")


def arguments():
    torch.manual_seed(109)
    memory = torch.randn(3, 11, 16, requires_grad=True)
    present = torch.ones(3, 11, dtype=torch.bool)
    present[0] = False
    context = torch.randn(3, 1, 8, 16, requires_grad=True)
    parcels = [Parcel(payload=memory, present=present, origin="/", destination=None, batch_size=3)]
    contexts = [
        Parcel(
            payload=context,
            present=torch.ones(3, 1, 8, dtype=torch.bool),
            origin="/events/evidence",
            destination=None,
            batch_size=3,
        )
    ]
    selection = torch.tensor(
        [[False] * 8, [False, True, False, False, False, False, True, False], [False] * 7 + [True]]
    )
    return parcels, dict(contexts=contexts, batch_size=3, device=torch.device("cpu"), target_shape=(1, 8)), selection


@pytest.mark.parametrize("pooling", ["query", "mean"])
@pytest.mark.parametrize("empty_memory", [False, True])
def test_objective_decodes_only_selected_coordinates_and_preserves_gradients(monkeypatch, pooling, empty_memory):
    configured = decoder(pooling=pooling)
    parcels, kwargs, selection = arguments()
    if empty_memory:
        parcels = []
    original = configured.decode
    widths = []

    def record(pooled):
        widths.append(pooled.shape[1])
        return original(pooled)

    monkeypatch.setattr(configured, "decode", record)
    dense = configured(parcels, **kwargs)
    compact = configured.objective(parcels, selection=selection, **kwargs)
    assert widths == [8, 2]
    for key in dense.payload.keys():
        torch.testing.assert_close(dense.payload[key][selection], compact.payload[key][selection], rtol=1e-5, atol=1e-6)
        assert not compact.payload[key][~selection].any()
    losses = [sum(value[selection].square().mean() for value in output.payload.values()) for output in (dense, compact)]
    gradients = []
    for loss in losses:
        configured.zero_grad(set_to_none=True)
        loss.backward()
        gradients.append(
            {name: None if p.grad is None else p.grad.clone() for name, p in configured.named_parameters()}
        )
    for name, expected in gradients[0].items():
        actual = gradients[1][name]
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6, msg=lambda error: f"{name}: {error}")


@pytest.mark.parametrize("case", ["all", "mostly", "empty", "embed", "hook", "custom_pool"])
def test_objective_retains_dense_execution_when_selection_is_unprofitable_or_observable(monkeypatch, case):
    configured = decoder()
    parcels, kwargs, selection = arguments()
    original = configured.decode
    widths = []

    def record(pooled):
        widths.append(pooled.shape[1])
        return original(pooled)

    monkeypatch.setattr(configured, "decode", record)
    if case == "all":
        selection[:] = True
    elif case == "mostly":
        selection[:, :6] = True
    elif case == "empty":
        selection[:] = False
    elif case == "embed":
        kwargs["embed"] = True
    elif case == "hook":
        configured.regression.register_forward_hook(lambda *args: None)
    elif case == "custom_pool":
        original_compute = configured.pool.compute
        monkeypatch.setattr(configured.pool, "compute", lambda *args, **kwargs: original_compute(*args, **kwargs))
    torch.manual_seed(717)
    reference = configured(parcels, **kwargs)
    torch.manual_seed(717)
    actual = configured.objective(parcels, selection=selection, **kwargs)
    if case == "empty":
        assert widths == [8, 0]
        assert all(value.shape[:2] == (3, 8) and not value.any() for value in actual.payload.values())
    else:
        assert widths == [8, 8]
        for key in reference.payload.keys():
            torch.testing.assert_close(actual.payload[key], reference.payload[key], rtol=0, atol=0)


@pytest.mark.parametrize("site", ["attention", "ffn", "all"])
def test_selected_objectives_keep_dropout_compact_repeatable_and_deterministic_in_eval(monkeypatch, site):
    configured = decoder(dropout=0.25)
    for block in configured.pool.blocks:
        if site == "ffn":
            block.attention.dropout_p = 0.0
        if site == "attention":
            for module in block.ffn.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
    parcels, kwargs, selection = arguments()
    memory = parcels[0].payload
    context = kwargs["contexts"][0].payload
    attention_calls, dropout_calls = [], []
    sdpa, dropout = F.scaled_dot_product_attention, F.dropout

    def attend(*args, **options):
        attention_calls.append((options["query"].shape[-2], options["dropout_p"]))
        return sdpa(*args, **options)

    def drop(inputs, p=0.5, training=True, inplace=False):
        dropout_calls.append((tuple(inputs.shape), p, training))
        return dropout(inputs, p=p, training=training, inplace=inplace)

    monkeypatch.setattr(F, "scaled_dot_product_attention", attend)
    monkeypatch.setattr(F, "dropout", drop)
    outputs, gradients, random_states = [], [], []
    for seed in (193, 193, 197):
        configured.zero_grad(set_to_none=True)
        memory.grad = context.grad = None
        torch.manual_seed(seed)
        output = configured.objective(parcels, selection=selection, **kwargs)
        outputs.append({key: value.detach().clone() for key, value in output.payload.items()})
        random_states.append(torch.get_rng_state())
        assert all(torch.isfinite(value).all() and not value[~selection].any() for value in output.payload.values())
        sum(value[selection].square().sum() for value in output.payload.values()).backward()
        gradients.append({name: parameter.grad.clone() for name, parameter in configured.named_parameters()})
        assert all(torch.isfinite(gradient).all() for gradient in gradients[-1].values())
        assert torch.isfinite(memory.grad).all() and not memory.grad[0].any()
        assert torch.isfinite(context.grad).all() and not context.grad.reshape(3, 8, 16)[~selection].any()
    assert attention_calls == [(2, 0.0 if site == "ffn" else 0.25)] * 3
    assert dropout_calls == [
        ((2, 2, width), 0.0 if site == "attention" else 0.25, True) for _ in range(3) for width in (128, 16)
    ]
    for key in outputs[0]:
        torch.testing.assert_close(outputs[0][key], outputs[1][key], rtol=0, atol=0)
    for name in gradients[0]:
        torch.testing.assert_close(gradients[0][name], gradients[1][name], rtol=0, atol=0)
    assert torch.equal(random_states[0], random_states[1])
    assert any(not torch.equal(outputs[0][key][selection], outputs[2][key][selection]) for key in outputs[0])

    configured.eval()
    attention_calls.clear()
    dropout_calls.clear()
    before = torch.get_rng_state()
    with torch.no_grad():
        first = configured.objective(parcels, selection=selection, **kwargs)
        second = configured.objective(parcels, selection=selection, **kwargs)
    assert torch.equal(torch.get_rng_state(), before)
    assert attention_calls == [(2, 0.0)] * 2
    assert all(not training for _, _, training in dropout_calls)
    for key in first.payload.keys():
        torch.testing.assert_close(first.payload[key], second.payload[key], rtol=0, atol=0)
    with torch.no_grad():
        dense = configured(parcels, **kwargs)
    for key in dense.payload.keys():
        torch.testing.assert_close(first.payload[key][selection], dense.payload[key][selection], rtol=1e-5, atol=1e-6)
        assert not first.payload[key][~selection].any()


def test_small_cuda_query_volume_bypasses_selection_without_device_reductions():
    configured = decoder()
    parcels, kwargs, _ = arguments()
    selection = SimpleNamespace()
    kwargs["device"] = torch.device("cuda")
    # The dense call does not inspect the selection object or touch CUDA data.
    # No CUDA runtime is needed to verify the host-side volume guard.
    configured.forward = lambda *args, **kwargs: kwargs
    result = configured.objective(parcels, selection=selection, **kwargs)
    assert "selection" not in result


def test_custom_forward_keeps_legacy_signature_and_custom_decode_requires_explicit_opt_in():
    class Legacy(NumberDecoder):
        coordinatewise = True

        def forward(self, parcels, *, contexts, batch_size, device, embed, target_shape, geometry):
            return super().forward(
                parcels,
                contexts=contexts,
                batch_size=batch_size,
                device=device,
                embed=embed,
                target_shape=target_shape,
                geometry=geometry,
            )

    class LegacyContext(NumberDecoder):
        coordinatewise = True

        def context(self, parcels, *, batch_size, n_context=None):
            return super().context(parcels, batch_size=batch_size, n_context=n_context)

    class Coupled(NumberDecoder):
        def decode(self, pooled):
            return super().decode(pooled.mean(dim=1, keepdim=True).expand_as(pooled))

    parcels, kwargs, selection = arguments()
    for implementation in (Legacy, LegacyContext, Coupled):
        configured = decoder(decoder_type=implementation)
        expected = configured(parcels, embed=False, geometry=None, **kwargs)
        actual = configured.objective(parcels, selection=selection, **kwargs)
        for key in expected.payload.keys():
            torch.testing.assert_close(actual.payload[key], expected.payload[key], rtol=0, atol=0)


@pytest.mark.parametrize("batch_axes", [0, 1, 2, 3])
def test_extension_can_opt_in_to_nested_output_selection_without_shared_changes(batch_axes):
    class Independent(DecoderBase):
        coordinatewise = True

        def decode(self, pooled):
            content = pooled[..., :4].reshape(*pooled.shape[:2], 2, 2)
            batch = (*pooled.shape[:2], 2)[:batch_axes]
            return TensorDict(
                {"custom": TensorDict({"matrix": content}, batch_size=batch), "empty": pooled[..., :0]},
                batch_size=batch[:2],
            )

    configured = decoder(decoder_type=Independent)
    parcels, kwargs, selection = arguments()
    expected = configured(parcels, **kwargs)
    actual = configured.objective(parcels, selection=selection, **kwargs)
    assert actual.payload["custom"].batch_size == expected.payload["custom"].batch_size
    assert actual.payload["custom", "matrix"].shape == (3, 8, 2, 2)
    assert actual.payload["empty"].shape == (3, 8, 0)
    torch.testing.assert_close(
        actual.payload["custom", "matrix"][selection],
        expected.payload["custom", "matrix"][selection],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "kind", ["number", "category", "set", "dateparts", "vector", "enum", "boolean", "hash", "quantile"]
)
def test_ragged_extension_losses_and_full_model_gradients_match_dense_execution(monkeypatch, kind):
    constructors = {
        "number": (rf.Number, {}, lambda i: float(i + 1)),
        "category": (rf.Category, {"p_unavailable": 0.0}, lambda i: f"c{i % 3}"),
        "set": (rf.Set, {"p_unavailable": 0.0}, lambda i: [f"t{i % 3}", "shared"]),
        "dateparts": (rf.DateParts, {"dateparts": ["day_of_week", "day_of_month"]}, lambda i: datetime(2026, 1, i + 1)),
        "vector": (rf.Vector, {"n_dim": 3}, lambda i: [float(i), float(i + 1), float(i + 2)]),
        "enum": (rf.Enum, {"values": ("a", "b"), "p_unavailable": 0.0}, lambda i: "a" if i % 2 else "b"),
        "boolean": (rf.Boolean, {}, lambda i: bool(i % 2)),
        "hash": (rf.Hash, {}, lambda i: f"id{i}"),
        "quantile": (rf.Quantile, {}, lambda i: float(i + 1)),
    }
    constructor, options, value = constructors[kind]
    torch.manual_seed(192)
    model = rf.Model(
        d_model=16,
        n_heads=4,
        n_layers=1,
        dropout=0.0,
        events=rf.Branch(
            length=None,
            reduction=None,
            evidence=rf.Number,
            target=constructor(mask=rf.Mask(query="selected", skip=True, dropout=False, reconstruct=True), **options),
        ),
    )
    source = pa.Table.from_pylist(
        [
            {
                "events": [
                    {"evidence": float(i), "target": value(i), "selected": i in (0, 5)} if i != 2 else None
                    for i in range(length)
                ]
            }
            for length in (8, 3, 0)
        ]
    )
    fields = model.encode(source, strata="train")
    monkeypatch.setattr(model, "track", lambda key, value, **kwargs: value)
    configured = model.nodes["/events/target"].decoder
    original = configured.decode
    widths = []

    def record(pooled):
        widths.append(pooled.shape[1])
        return original(pooled)

    monkeypatch.setattr(configured, "decode", record)
    losses, gradients = [], []
    for optimized in (False, True):
        model.zero_grad(set_to_none=True)
        with monkeypatch.context() as mode:
            if not optimized:
                mode.setattr(type(configured), "coordinatewise", False)
            output = ModelRuntime.step(model, fields, 0, strata=rf.Strata.validate)
            assert output is not None
            loss = output["loss"]
            assert torch.isfinite(loss)
            loss.backward()
        losses.append(loss.detach())
        gradients.append({name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()})
    assert widths == [8, 2]
    torch.testing.assert_close(losses[0], losses[1], rtol=1e-5, atol=1e-6)
    for name, expected in gradients[0].items():
        actual = gradients[1][name]
        if expected is None:
            assert actual is None, name
        else:
            torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-6, msg=lambda error: f"{name}: {error}")
