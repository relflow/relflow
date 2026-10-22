import math
from datetime import timedelta
from logging.handlers import BufferingHandler
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pydantic
import pytest
import torch
import torch.distributed as distributed
import torch.multiprocessing as multiprocessing
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.helpers import Jitter
from relflow.logging import logger
from relflow.logging.config import CONTEXT
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, TensorInput
from relflow.tensorfields.extensions.number import (
    Decoder,
    Embedder,
    GlobalOnlineNormalizer,
    TensorField,
    loss,
    moments,
    write,
)
from relflow.tensorfields.extensions.number import (
    output as output_type,
)
from tests.arrow import table as arrow_table
from tests.tensorfields.helpers import tensorize

ADDRESS = "/items/amount"


def build_schema(*, mask: bool | Mask = False, jitter: Jitter | dict[str, object] | None = None) -> Schema:
    options = {} if jitter is None else {"jitter": jitter}
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        items=rf.Branch(length=2, amount=rf.Number(mask=mask, **options)),
    )


def tensorfield(rows: list[list[Any]], *, schema: Schema, strata: Strata) -> TensorField:
    batch = arrow_table([{"items": [{"amount": value} for value in row]} for row in rows])
    projection = coalesce(batch, schema=schema, strata=strata).fields[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["number"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
    )


def test_number_request_hydrates_jitter_from_a_mapping():
    schema = build_schema(jitter={"add": 1.5, "multiply": 0.25, "normalize": False})
    jitter = schema.requests[ADDRESS].jitter

    assert isinstance(jitter, Jitter)
    assert jitter == Jitter(add=1.5, multiply=0.25, normalize=False)


def test_number_jitter_round_trips_through_schema_serialization():
    schema = build_schema(jitter=Jitter(add=1.5, multiply=0.25, normalize=False))

    restored = Schema.model_validate_json(schema.model_dump_json(round_trip=True))

    assert restored.requests[ADDRESS].jitter == Jitter(add=1.5, multiply=0.25, normalize=False)


def test_number_jitter_round_trips_through_model_checkpoint(tmp_path: Path):
    configured = Jitter(add=1.5, multiply=0.25, normalize=False)
    model = rf.Model.xs(amount=rf.Number(jitter=configured), d_model=8)
    pathname = tmp_path / "jitter.ckpt"

    model.save(pathname)
    restored = rf.Model.load(pathname)

    request = restored.schema.requests[rf.Address("/", "amount")]
    embedder = restored.nodes[rf.Address("/", "amount")].embedder
    assert request.jitter == configured
    assert embedder.jitter == configured


@pytest.mark.parametrize("value", [None, 0.0, 0.2, 1, True])
def test_number_requires_a_jitter_configuration(value: object):
    with pytest.raises(pydantic.ValidationError):
        rf.Number(jitter=value)


def test_default_jitter_does_not_advance_rng_during_training():
    schema = build_schema()
    field = tensorfield(rows=[[2.0, 4.0]], schema=schema, strata=Strata.train)
    embedder = Embedder(schema=schema, address=ADDRESS)
    embedder.train()
    before = torch.random.get_rng_state().clone()

    embedder.embed(field)

    assert torch.equal(torch.random.get_rng_state(), before)


def test_number_embedding_retains_standardized_value_in_monotone_lane():
    schema = build_schema()
    field = tensorfield(rows=[[8.0, 12.0]], schema=schema, strata=Strata.predict)
    embedder = Embedder(schema=schema, address=ADDRESS).eval()
    embedder.normalizers["raw"].mean.fill_(10.0)
    embedder.normalizers["raw"].var.fill_(4.0)

    parcel = embedder.embed(field)

    # The raw lane's monotone channel precedes the log lane's; both are standardized.
    assert parcel.payload[0, 0, 0, -2].item() == pytest.approx(-1.0, abs=1e-4)
    assert parcel.payload[0, 0, 1, -2].item() == pytest.approx(1.0, abs=1e-4)
    assert parcel.payload[0, 0, 0, -1].item() == pytest.approx(math.log1p(8.0), abs=1e-3)
    assert parcel.payload[0, 0, 1, -1].item() == pytest.approx(math.log1p(12.0), abs=1e-3)


def test_number_monotone_lanes_retain_autograd_at_model_width_two():
    schema = rf.Schema.from_tree(
        amount=rf.Number, d_model=2, n_layers=1, n_heads=2, attention=None, reduction=rf.Mean()
    )
    embedder = Embedder(schema=schema, address="/amount")
    inputs = TensorInput(
        state=torch.tensor([Tokens.valued, Tokens.valued]),
        content=torch.tensor([1.0, 2.0]),
        batch_size=[2],
    )

    parcel = embedder(inputs)
    parcel.payload.sum().backward()

    assert all(parameter.grad is not None for parameter in embedder.parameters())


def test_number_jitters_only_finite_valued_coordinates(monkeypatch: pytest.MonkeyPatch):
    schema = build_schema(jitter=Jitter(add=0.5))
    embedder = Embedder(schema=schema, address=ADDRESS)
    captured: list[torch.Tensor] = []

    def capture(
        self: Jitter,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self == Jitter(add=0.5)
        captured.append(mask.clone())
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)
    inputs = TensorInput(
        state=torch.tensor([Tokens.valued, Tokens.null, Tokens.masked, Tokens.valued]),
        content=torch.tensor([2.0, 0.0, 0.0, float("inf")]),
        batch_size=[4],
    )

    embedder.train()
    embedder(inputs)

    assert len(captured) == 1
    assert captured[0].tolist() == [True, False, False, False]


def test_number_jitter_runs_only_in_training_mode(monkeypatch: pytest.MonkeyPatch):
    schema = build_schema(jitter=Jitter(add=0.5))
    embedder = Embedder(schema=schema, address=ADDRESS)
    field = tensorfield(rows=[[2.0]], schema=schema, strata=Strata.train)
    calls: list[torch.Tensor] = []

    def capture(
        self: Jitter,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self == Jitter(add=0.5)
        calls.append(inputs.clone())
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)

    embedder.train()
    embedder.embed(field)
    embedder.eval()
    embedder.embed(field)

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("normalize", "expected_perturb", "expected_clamp"),
    [
        (True, 2.0, 5.0),
        (False, 14.0, 3.5),
    ],
)
def test_number_jitter_respects_normalization_order(
    monkeypatch: pytest.MonkeyPatch,
    normalize: bool,
    expected_perturb: float,
    expected_clamp: float,
):
    schema = build_schema(jitter=Jitter(add=3.0, normalize=normalize))
    embedder = Embedder(schema=schema, address=ADDRESS)
    embedder.normalizers["raw"].mean.fill_(10.0)
    embedder.normalizers["raw"].var.fill_(4.0)
    field = tensorfield(rows=[[14.0]], schema=schema, strata=Strata.train)
    observed: dict[str, torch.Tensor] = {}

    def add_three(
        self: Jitter,
        inputs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        assert self == Jitter(add=3.0, normalize=normalize)
        observed["apply"] = inputs.clone()
        return inputs + mask.to(dtype=inputs.dtype).mul(3.0)

    def capture_clamp(self: Embedder, content: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # The raw lane is clamped first; the log lane follows from the same perturbed value.
        observed.setdefault("clamp", content.clone())
        return content

    monkeypatch.setattr(Jitter, "apply", add_three)
    monkeypatch.setattr(Embedder, "clamp", capture_clamp)

    embedder.train()
    embedder.embed(field)

    assert observed["apply"].item() == pytest.approx(expected_perturb, abs=1e-4)
    assert observed["clamp"].item() == pytest.approx(expected_clamp, abs=1e-4)


def test_number_jitter_is_configuration_not_checkpoint_state():
    schema = build_schema(jitter=Jitter(add=1.5, multiply=0.25, normalize=False))
    embedder = Embedder(schema=schema, address=ADDRESS)

    assert embedder.jitter == Jitter(add=1.5, multiply=0.25, normalize=False)
    assert not any("jitter" in name for name in embedder.state_dict())
    assert not any("jitter" in name for name, _ in embedder.named_buffers())


def test_number_jitter_mutation_rebuilds_runtime_configuration():
    model = rf.Model.xs(amount=rf.Number(jitter=rf.Jitter(add=0.1)), d_model=8)
    address = rf.Address("/", "amount")

    model.update(
        lambda node: node.address == address,
        jitter=rf.Jitter(add=0.8, multiply=0.3, normalize=False),
    )

    request = model.schema.requests[address]
    embedder = model.nodes[address].embedder
    assert request.jitter == rf.Jitter(add=0.8, multiply=0.3, normalize=False)
    assert embedder.jitter == request.jitter


class TrackingModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_number_loss_does_not_mutate_counter():
    schema = build_schema(mask=Mask(reconstruct=True))

    field = tensorfield(
        rows=[[1.0, None], [2.0]],
        schema=schema,
        strata=Strata.train,
    )
    embedder = Embedder(schema=schema, address=ADDRESS)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = TrackingModule(schema=schema, embedder=embedder, decoder=decoder)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.content.shape, 1),
                "scaled": torch.zeros(*field.content.shape, 1),
            },
            batch_size=field.batch_size,
        ),
    )

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    expected_counts = torch.ones(len(Tokens), dtype=torch.int64)
    assert torch.equal(embedder.counter.counts, expected_counts)


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_number_evaluation_metrics_count_valued_targets_across_unequal_batches(batch_size, tmp_path):
    model = rf.Model.xs(d_model=8, target=rf.Number(mask=True))
    decoder = model.nodes["/target"].decoder
    with torch.no_grad():
        decoder.regression.weight.zero_()
        decoder.regression.bias.zero_()
    rows = pa.table({"target": pa.array([3.0, 4.0, None], type=pa.float32())})
    data = rf.ArrowDataModule(model, validate=rows)
    values = model.validate(
        data,
        batch_size=batch_size,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )[0]
    assert values[".target/validate.mae.content"] == pytest.approx(3.5)
    assert values[".target/validate.rmse.content"] == pytest.approx((25 / 2) ** 0.5)


def test_number_null_targets_train_state_without_fabricating_numeric_errors():
    schema = build_schema(mask=Mask(reconstruct=True))
    field = tensorfield(rows=[[None]], schema=schema, strata=Strata.train)
    embedder = Embedder(schema=schema, address=ADDRESS)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = TrackingModule(schema=schema, embedder=embedder, decoder=decoder)
    scaled = torch.full((*field.state.shape, 1), math.log1p(100.0), requires_grad=True)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens), requires_grad=True),
                TensorKey.content: torch.full((*field.state.shape, 1), 100.0),
                "scaled": scaled,
            },
            batch_size=field.batch_size,
        ),
    )
    value = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    assert torch.isfinite(value)
    value.backward()
    assert scaled.grad is not None
    assert torch.count_nonzero(scaled.grad) == 0


def test_number_normalizer_ignores_nonfinite_values_when_updating():
    normalizer = GlobalOnlineNormalizer()
    normalizer.train()
    inputs = torch.tensor([1.0, float("inf"), float("-inf"), float("nan")])
    mask = torch.ones_like(inputs, dtype=torch.bool)

    output = normalizer(inputs=inputs, mask=mask)

    assert torch.isfinite(normalizer.mean).all()
    assert torch.isfinite(normalizer.var).all()
    assert normalizer.count.item() == 1
    assert torch.isfinite(output[0])
    assert torch.isinf(output[1])
    assert torch.isinf(output[2])
    assert torch.isnan(output[3])


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("layout", ["flat", "transposed", "scalar", "empty"])
@pytest.mark.parametrize(
    ("dtype", "buffer_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.float64, torch.float32),
        (torch.float16, torch.float16),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_number_normalizer_preserves_masked_values_and_input_gradients_without_updates(
    device: str, layout: str, dtype: torch.dtype, buffer_dtype: torch.dtype
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    normalizer = GlobalOnlineNormalizer(epsilon=0.0).to(device=device, dtype=buffer_dtype).train()
    normalizer.mean.fill_(2.0)
    normalizer.var.fill_(9.0)
    inputs = torch.tensor([8.0, -3.0, float("nan"), float("inf"), float("-inf"), 5.0], device=device, dtype=dtype)
    mask = torch.tensor([True, False, True, True, False, True], device=device)
    expected = torch.tensor([2.0, -3.0, float("nan"), float("inf"), float("-inf"), 1.0], device=device, dtype=dtype)
    slopes = torch.tensor([1 / 3, 1.0, 1.0, 1.0, 1.0, 1 / 3], device=device, dtype=dtype)
    if layout == "transposed":
        inputs, mask, expected, slopes = [value.reshape(2, 3).T for value in (inputs, mask, expected, slopes)]
        assert not inputs.is_contiguous()
    elif layout == "empty":
        inputs, mask, expected, slopes = [value[:0] for value in (inputs, mask, expected, slopes)]
    elif layout == "scalar":
        inputs, mask, expected, slopes = [value[0] for value in (inputs, mask, expected, slopes)]
    original = inputs.clone()
    inputs.requires_grad_()

    output = normalizer(inputs=inputs, mask=mask, update=False)
    weights = torch.arange(1, inputs.numel() + 1, dtype=inputs.dtype, device=device).reshape(inputs.shape)
    (gradient,) = torch.autograd.grad(output, inputs, grad_outputs=weights)

    torch.testing.assert_close(output, expected, equal_nan=True)
    torch.testing.assert_close(inputs, original, equal_nan=True)
    torch.testing.assert_close(gradient, weights * slopes)
    assert torch.isfinite(gradient).all()
    assert normalizer.mean.item() == 2.0
    assert normalizer.var.item() == 9.0
    assert normalizer.count.item() == 0


def test_number_normalizer_learns_precomputed_finite_moments():
    normalizer = GlobalOnlineNormalizer()
    observation = moments(torch.tensor([1.0, 3.0, float("nan"), float("inf")]))

    normalizer.learn(observation)

    assert normalizer.count.item() == 2
    assert normalizer.mean.item() == pytest.approx(2.0)
    assert normalizer.var.item() == pytest.approx(1.0)


@pytest.mark.parametrize("values", [[1.0, 3.0], [], [float("nan"), float("inf")]])
def test_number_moments_keep_float64_statistics_and_int64_counts(values):
    observation = moments(torch.tensor(values, dtype=torch.float64))
    assert observation["count"].dtype == torch.int64
    assert observation["mean"].dtype == observation["var"].dtype == torch.float64
    assert observation.batch_size == torch.Size([])
    assert all(torch.isfinite(observation[key]).all() for key in ("count", "mean", "var"))


def test_number_float64_variance_preserves_small_differences_at_large_offsets():
    values = torch.tensor([1_000_000.0, 1_000_002.0, 1_000_004.0, 1_000_006.0])
    normalizer = GlobalOnlineNormalizer()
    normalizer.learn(moments(values[:2]))
    normalizer.learn(moments(values[2:]))
    assert normalizer.mean.item() == 1_000_003.0
    assert normalizer.var.item() == pytest.approx(5.0)
    assert normalizer.count.item() == 4
    assert normalizer.mean.dtype == normalizer.var.dtype == torch.float64


def test_number_constant_large_values_do_not_overflow_squared_sums():
    normalizer = GlobalOnlineNormalizer()
    normalizer.update(torch.full((4,), 1e30))
    assert torch.isfinite(normalizer.mean).all()
    assert normalizer.var.item() == 0.0


def test_number_observes_large_integers_in_float64():
    configured = rf.Model.xs(amount=rf.Number, d_model=8, batch_size=2)
    records = pa.table({"amount": [16_777_217, 16_777_219]})
    datamodule = rf.ArrowDataModule(configured, train=records, shuffle=False)
    encoded = next(iter(datamodule.train_dataloader()))
    observation = encoded.observations["/amount"][TensorKey.content]
    assert observation["mean"].dtype == torch.float64
    assert observation["count"].dtype == torch.int64
    assert observation["mean"].item() == records["amount"].to_numpy().astype("float64").mean().item()


def distributed_normalization(rank: int, directory: str, empty: bool) -> None:
    distributed.init_process_group(
        "gloo",
        init_method=(Path(directory) / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        normalizer = GlobalOnlineNormalizer()
        local = [1_000_000.0] if rank == 0 else [1_000_002.0, 1_000_004.0, 1_000_006.0]
        if empty and rank == 0:
            local = []
        normalizer.learn(moments(torch.tensor(local, dtype=torch.float32)))
        assert normalizer.count.item() == (3 if empty else 4)
        assert normalizer.mean.item() == (1_000_004.0 if empty else 1_000_003.0)
        assert normalizer.var.item() == pytest.approx(8 / 3 if empty else 5.0)
        assert normalizer.mean.dtype == normalizer.var.dtype == torch.float64
        normalizer.learn(moments(torch.empty(0)))
        assert normalizer.count.item() == (3 if empty else 4)
    finally:
        distributed.destroy_process_group()


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.skipif(not distributed.is_available() or not distributed.is_gloo_available(), reason="requires CPU Gloo")
def test_number_float64_moments_merge_across_ranks(tmp_path, empty):
    multiprocessing.spawn(distributed_normalization, args=(str(tmp_path), empty), nprocs=2, join=True)


def test_number_embedder_clamps_unsafe_fourier_inputs_and_warns():
    schema = build_schema()
    embedder = Embedder(schema=schema, address=ADDRESS)
    bound = embedder.max_fourier_input.detach()
    content = torch.stack(
        [
            bound.mul(2),
            bound.mul(-3),
            torch.tensor(float("inf")),
            torch.tensor(float("nan")),
            bound.mul(4),
        ]
    )
    state = torch.tensor(
        [
            Tokens.valued,
            Tokens.valued,
            Tokens.valued,
            Tokens.valued,
            Tokens.padded,
        ],
        dtype=torch.int64,
    )
    records = BufferingHandler(capacity=10)
    records.setLevel("WARNING")
    logger.logger.addHandler(records)

    try:
        clamped = embedder.clamp(content=content, state=state)
    finally:
        logger.logger.removeHandler(records)

    events = [getattr(record, CONTEXT) for record in records.buffer]
    messages = [record.getMessage() for record in records.buffer]

    assert torch.isfinite(clamped).all()
    assert torch.allclose(clamped[:3], torch.stack([bound, -bound, bound]))
    assert clamped[3].item() == 0.0
    assert clamped[4].item() == bound.item()
    assert any("number Fourier inputs exceed safe range" in message for message in messages)
    assert any(
        event.get("component") == "tensorfield"
        and event.get("field_type") == "number"
        and event.get("address") == ADDRESS
        and event.get("count") == 5
        and event.get("valued_count") == 4
        and event.get("nonfinite_count") == 2
        for event in events
    )


def test_number_embedder_outputs_finite_payload_for_extreme_outliers():
    schema = build_schema()
    field = tensorfield(
        rows=[[1.0, 2.0]],
        schema=schema,
        strata=Strata.train,
    )
    field.content[0, 0, 0] = float("inf")

    embedder = Embedder(schema=schema, address=ADDRESS)
    embedder.train()
    parcel = embedder.embed(field)

    for normalizer in embedder.normalizers.values():
        assert torch.isfinite(normalizer.mean).all()
        assert torch.isfinite(normalizer.var).all()
    assert torch.isfinite(parcel.payload).all()


def test_number_write_emits_flat_arrow_content():
    schema = build_schema()
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.null.value] = 10.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: torch.tensor([[[1.5]], [[2.5]]]),
            },
            batch_size=[2],
        ),
    )

    module = SimpleNamespace(schema=schema)
    datatype = output_type(module, ADDRESS)
    output = write(module=module, prediction=prediction, datatype=datatype)

    assert output.type == datatype
    assert output.field(TensorKey.content.name).to_pylist() == [1.5, 2.5]


def log_model(**options) -> rf.Model:
    return rf.Model.xs(d_model=8, batch_size=2, amount=rf.Number(**options))


def log_prediction(field, scaled: torch.Tensor) -> Prediction:
    """A decoder-shaped payload: source-unit content plus the log-unit lane the loss reads."""
    state = torch.full((*field.state.shape, len(Tokens)), -50.0)
    state[..., Tokens.valued.value] = 50.0
    content = scaled.sign() * scaled.abs().expm1()
    return Prediction(
        address=rf.Address("amount"),
        payload=TensorDict(
            {TensorKey.state: state, TensorKey.content: content, "scaled": scaled}, batch_size=field.batch_size
        ),
    )


@pytest.mark.parametrize("mask", [False, True], ids=["input", "target"])
def test_number_values_stay_in_source_units_and_the_readout_learns_the_log_scale(mask):
    model = log_model(mask=mask)
    table = arrow_table([{"amount": 0.0}, {"amount": 99.0}, {"amount": -99.0}])

    field = model.encode(table, strata="train")[rf.Address("amount")]

    source = torch.tensor([0.0, 99.0, -99.0], dtype=torch.float64)
    torch.testing.assert_close((field.targets[TensorKey.content] if mask else field.content).flatten(), source)
    stats = rf.Number.normalization(model, "amount")
    assert stats["count"] == 3 and stats["mean"] == pytest.approx(0.0, abs=1e-6)
    assert stats["std"] == pytest.approx(math.sqrt(2 * 99.0**2 / 3), rel=1e-4)
    assert model.schema.requests[rf.Address("amount")].log is True
    if mask:
        decoder = model.nodes[rf.Address("amount")].decoder
        assert decoder.normalizer.count.item() == 3
        assert decoder.normalizer.var.item() == pytest.approx(2 * math.log1p(99.0) ** 2 / 3, rel=1e-5)


def test_number_rejects_the_removed_lanes_and_log_options():
    with pytest.raises(pydantic.ValidationError):
        rf.Number(log=True)
    with pytest.raises(pydantic.ValidationError):
        rf.Number(lanes=("raw",))


def test_number_mse_and_huber_train_in_source_units():
    assert rf.Number(objective="mse").log is False and rf.Number(objective="huber").log is False
    model = rf.Model.xs(d_model=8, batch_size=2, amount=rf.Number(objective="mse", mask=True))
    decoder = model.nodes[rf.Address("amount")].decoder
    assert decoder.log is False
    with torch.no_grad():
        decoder.regression.weight.zero_()
        decoder.regression.bias.zero_()
        decoder.normalizer.mean.fill_(10.0)

    payload = decoder.decode(torch.zeros(3, model.schema.d_model))

    torch.testing.assert_close(payload[TensorKey.content].flatten(), torch.full((3,), 10.0, dtype=torch.float64))
    torch.testing.assert_close(payload["scaled"].flatten(), torch.full((3,), 10.0, dtype=torch.float64))


def test_number_readout_starts_at_the_training_mean():
    model = log_model(mask=True)
    model.encode(arrow_table([{"amount": 99.0}, {"amount": 99.0}]), strata="train")
    decoder = model.nodes[rf.Address("amount")].decoder
    with torch.no_grad():
        decoder.regression.weight.zero_()
        decoder.regression.bias.zero_()

    payload = decoder.decode(torch.zeros(2, model.schema.d_model))

    assert decoder.normalizer.mean.item() == pytest.approx(math.log1p(99.0), rel=1e-6)
    torch.testing.assert_close(
        payload[TensorKey.content].flatten(), torch.full((2,), 99.0, dtype=torch.float64), rtol=1e-5, atol=1e-3
    )


def test_number_mae_loss_adds_the_source_unit_error(monkeypatch):
    model = log_model(mask=True)
    field = model.encode(arrow_table([{"amount": 9.0}, {"amount": 99.0}]), strata="train")[rf.Address("amount")]
    tracked = {}
    monkeypatch.setattr(model, "track", lambda names, value: tracked.__setitem__(names[-2:], value) or value)
    # The first row is exact; the second overshoots by one log unit, inside the bounded source-unit term.
    predicted = log_prediction(field, torch.tensor([[math.log1p(9.0)], [math.log1p(99.0) + 1.0]]))

    rf.TENSORFIELDS["number"].loss(model, predicted, field, rf.Strata.train)

    decoder = model.nodes[rf.Address("amount")].decoder
    std = decoder.normalizer.scale.item()
    raw = model.nodes[rf.Address("amount")].embedder.normalizers["raw"]
    raw_std = raw.scale.item()
    expected = (1.0 / std + abs(math.expm1(math.log1p(99.0) + 1.0) - 99.0) / raw_std) / 2
    assert float(tracked[Metric.loss, TensorKey.content]) == pytest.approx(expected, rel=1e-4)
    assert float(tracked[Metric.mae, TensorKey.content]) == pytest.approx(
        abs(math.expm1(math.log1p(99.0) + 1.0) - 99.0) / 2, rel=1e-4
    )


def test_number_embedder_and_readout_stay_wide_under_autocast():
    schema = build_schema()
    embedder = Embedder(schema=schema, address=ADDRESS).eval()
    field = tensorfield(rows=[[8.0, 12.0]], schema=schema, strata=Strata.predict)
    model = log_model(mask=True)
    decoder = model.nodes[rf.Address("amount")].decoder

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        parcel = embedder.embed(field)
        payload = decoder.decode(torch.zeros(3, model.schema.d_model))

    assert parcel.payload.dtype == torch.float32
    assert payload["scaled"].dtype == torch.float64 and payload[TensorKey.content].dtype == torch.float64


def test_number_observes_moments_for_both_scales_and_the_readout_scale():
    model = rf.Model.xs(d_model=8, batch_size=2, amount=rf.Number)
    table = arrow_table([{"amount": 0.0}, {"amount": 99.0}, {"amount": -99.0}])

    model.encode(table, strata="train")

    embedder = model.nodes[rf.Address("amount")].embedder
    raw, log = embedder.normalizers["raw"], embedder.normalizers["log"]
    assert raw.count.item() == log.count.item() == 3
    assert raw.mean.item() == pytest.approx(0.0, abs=1e-6) and log.mean.item() == pytest.approx(0.0, abs=1e-6)
    assert raw.var.item() == pytest.approx(2 * 99.0**2 / 3, rel=1e-5)
    assert log.var.item() == pytest.approx(2 * math.log1p(99.0) ** 2 / 3, rel=1e-5)
    assert rf.Number.normalization(model, "amount")["variance"] == pytest.approx(raw.var.item(), rel=1e-6)


def test_number_log_lane_embeds_standardized_compressed_values():
    schema = build_schema()
    field = tensorfield(rows=[[8.0, 12.0]], schema=schema, strata=Strata.predict)
    embedder = Embedder(schema=schema, address=ADDRESS).eval()
    embedder.normalizers["log"].mean.fill_(2.0)
    embedder.normalizers["log"].var.fill_(0.25)

    parcel = embedder.embed(field)

    assert embedder.linear.in_features == 4 * embedder.weights.numel()
    assert parcel.payload[0, 0, 0, -1].item() == pytest.approx((math.log1p(8.0) - 2.0) / 0.5, abs=1e-3)
    assert parcel.payload[0, 0, 1, -1].item() == pytest.approx((math.log1p(12.0) - 2.0) / 0.5, abs=1e-3)


def test_number_rejects_more_lanes_than_model_width():
    schema = rf.Schema.from_tree(
        amount=rf.Number, d_model=1, n_layers=1, n_heads=2, attention=None, reduction=rf.Mean()
    )
    with pytest.raises(ValueError, match="one payload channel per lane"):
        Embedder(schema=schema, address="/amount")


def test_number_jitter_perturbs_the_source_value_once_for_every_lane(monkeypatch: pytest.MonkeyPatch):
    schema = build_schema(jitter=Jitter(add=3.0, normalize=True))
    embedder = Embedder(schema=schema, address=ADDRESS)
    embedder.normalizers["raw"].mean.fill_(10.0)
    embedder.normalizers["raw"].var.fill_(4.0)
    field = tensorfield(rows=[[14.0]], schema=schema, strata=Strata.train)
    calls: list[torch.Tensor] = []

    def add_three(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        calls.append(inputs.clone())
        return inputs + mask.to(dtype=inputs.dtype).mul(3.0)

    monkeypatch.setattr(Jitter, "apply", add_three)
    embedder.train()
    parcel = embedder.embed(field)

    # One perturbation of the standardized raw value (2.0 -> 5.0, source 20.0) feeds both lanes.
    assert len(calls) == 1 and calls[0].item() == pytest.approx(2.0, abs=1e-5)
    assert parcel.payload[0, 0, 0, -2].item() == pytest.approx(5.0, abs=1e-4)
    assert parcel.payload[0, 0, 0, -1].item() == pytest.approx(math.log1p(20.0), abs=1e-3)


def test_number_log_scale_metrics_and_output_return_to_source_units(monkeypatch):
    model = log_model(mask=True)
    field = model.encode(arrow_table([{"amount": 99.0}, {"amount": 9.0}]), strata="train")[rf.Address("amount")]
    tracked = {}
    monkeypatch.setattr(model, "track", lambda names, value: tracked.__setitem__(names[-2:], value) or value)
    predicted = log_prediction(field, torch.tensor([[math.log1p(199.0)], [math.log1p(9.0)]]))

    rf.TENSORFIELDS["number"].loss(model, predicted, field, rf.Strata.train)

    assert float(tracked[Metric.mae, TensorKey.content]) == pytest.approx(50.0, rel=1e-4)
    output = rf.TENSORFIELDS["number"].write(
        model, predicted, rf.TENSORFIELDS["number"].output(model, rf.Address("amount"))
    )
    assert output.field("content").to_pylist() == pytest.approx([199.0, 9.0], rel=1e-5)


def test_number_decoder_emits_source_units_and_a_training_scale_lane():
    model = log_model(mask=True)
    decoder = model.nodes[rf.Address("amount")].decoder
    with torch.no_grad():
        decoder.regression.weight.zero_()
        decoder.regression.bias.zero_()
        decoder.normalizer.mean.fill_(math.log1p(99.0))
        decoder.normalizer.var.fill_(0.0)

    payload = decoder.decode(torch.zeros(3, model.schema.d_model))

    torch.testing.assert_close(
        payload[TensorKey.content].flatten(), torch.full((3,), 99.0, dtype=torch.float64), rtol=1e-5, atol=1e-3
    )
    torch.testing.assert_close(payload["scaled"].flatten(), torch.full((3,), math.log1p(99.0), dtype=torch.float64))
    assert not payload[TensorKey.content].requires_grad and payload["scaled"].requires_grad


def test_number_nan_tensorizes_as_the_other_state():
    model = log_model(mask=False)
    rows = arrow_table([{"amount": 9.0}, {"amount": float("nan")}, {"amount": None}, {"amount": float("inf")}])
    field = model.encode(rows, strata="train")[rf.Address("amount")]

    assert field.state.flatten().tolist() == [Tokens.valued, Tokens.other, Tokens.null, Tokens.valued]
    assert field.content.dtype == torch.float64
    assert field.content.flatten().tolist() == [9.0, 0.0, 0.0, math.inf]


def test_number_nan_and_infinite_targets_are_not_regressed(monkeypatch):
    model = log_model(mask=True)
    rows = arrow_table([{"amount": 9.0}, {"amount": 99.0}, {"amount": float("nan")}, {"amount": float("inf")}])
    field = model.encode(rows, strata="train")[rf.Address("amount")]
    assert field.targets[TensorKey.state].flatten().tolist() == [Tokens.valued] * 2 + [Tokens.other, Tokens.valued]
    tracked = {}
    monkeypatch.setattr(model, "track", lambda names, value: tracked.__setitem__(names[-2:], value) or value)
    # Only the exact finite rows are regression targets; the others would otherwise contribute enormous errors.
    scaled = torch.tensor([[math.log1p(9.0)], [math.log1p(99.0)], [50.0], [-50.0]], dtype=torch.float64)
    predicted = log_prediction(field, scaled)

    rf.TENSORFIELDS["number"].loss(model, predicted, field, rf.Strata.train)

    assert float(tracked[Metric.loss, TensorKey.content]) == pytest.approx(0.0, abs=1e-9)
    assert float(tracked[Metric.mae, TensorKey.content]) == pytest.approx(0.0, abs=1e-9)


def test_number_observation_counts_nan_as_other_and_excludes_it_from_moments():
    configured = rf.Model.xs(amount=rf.Number, d_model=8, batch_size=3)
    records = pa.table({"amount": pa.array([1.0, float("nan"), 3.0])})
    datamodule = rf.ArrowDataModule(configured, train=records, shuffle=False)
    encoded = next(iter(datamodule.train_dataloader()))
    observation = encoded.observations["/amount"]

    counts = observation[TensorKey.state].tolist()
    assert counts[Tokens.valued] == 2 and counts[Tokens.other] == 1
    assert observation[TensorKey.content]["count"].item() == 2
    assert observation[TensorKey.content]["mean"].item() == pytest.approx(2.0)


def test_number_keeps_float64_source_values_at_large_offsets():
    model = log_model(mask=True)
    rows = arrow_table([{"amount": 1e9 + 3.0}, {"amount": 1e9 + 5.0}])
    field = model.encode(rows, strata="train")[rf.Address("amount")]

    assert field.targets[TensorKey.content].dtype == torch.float64
    assert field.targets[TensorKey.content].flatten().tolist() == [1e9 + 3.0, 1e9 + 5.0]


def test_number_rejects_integers_beyond_float64_exactness():
    model = log_model(mask=False)
    with pytest.raises(ValueError, match=r"number field at '/amount' holds int64 values.*2\^53"):
        model.encode(arrow_table([{"amount": 2**53 + 1}]), strata="train")


def test_number_moments_stay_finite_beyond_float32_range():
    values = torch.tensor([1e20, 3e20, -2e20], dtype=torch.float64)
    observation = moments(values)

    assert torch.isfinite(observation["var"]).all()
    assert observation["var"].item() == pytest.approx(values.var(correction=0).item(), rel=1e-12)


def test_number_scale_floor_is_relative_to_the_mean():
    constant = GlobalOnlineNormalizer()
    constant.learn(moments(torch.tensor([1e6, 1e6, 1e6])))
    assert constant.var.item() == 0.0
    assert constant.scale.item() == pytest.approx(1e-12 * (1e6 + 1))
    assert constant(torch.tensor([1e6]), mask=torch.tensor([True]), update=False).item() == 0.0

    narrow = GlobalOnlineNormalizer()
    narrow.learn(moments(torch.tensor([1e6 - 3.0, 1e6 + 3.0])))
    assert narrow.scale.item() == pytest.approx(3.0)
    assert narrow(torch.tensor([1e6 + 3.0]), mask=torch.tensor([True]), update=False).item() == pytest.approx(1.0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_number_normalizer_keeps_float32_statistics_on_mps():
    normalizer = GlobalOnlineNormalizer().to("mps")
    assert normalizer.mean.dtype == normalizer.var.dtype == torch.float32
    assert normalizer.mean.device.type == "mps"

    normalizer.learn(moments(torch.tensor([1.0, 3.0], dtype=torch.float64)))

    assert normalizer.count.item() == 2
    assert normalizer.mean.item() == pytest.approx(2.0)
