"""Percentile coordinates preserve numeric state, units, and training ownership."""

from pathlib import Path

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.binding import bind
from relflow.data.iterables import encode
from relflow.structs.enums import Metric, TensorKey, Tokens
from relflow.structs.packages import Prediction

ADDRESS = rf.Address("amount")


def model(**options) -> rf.Model:
    return rf.Model.xs(d_model=8, batch_size=2, amount=rf.Quantile(**options))


def values(*content: float | None) -> pa.Table:
    return pa.table({"amount": pa.array(content, type=pa.float64())})


def normalization(configured: rf.Model) -> dict:
    return rf.Quantile.normalization(configured, ADDRESS)


def prediction(field, content: torch.Tensor) -> Prediction:
    state = torch.full((*field.state.shape, len(Tokens)), -50.0)
    state[..., Tokens.valued.value] = 50.0
    return Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state,
                "percentile": content,
                TensorKey.content: torch.zeros_like(content, dtype=torch.float64),
            },
            batch_size=field.batch_size,
        ),
    )


def decoded(configured: rf.Model, field, percentile: float = 0.5) -> Prediction:
    decoder = configured.nodes[ADDRESS].decoder
    with torch.no_grad():
        decoder.regression.weight.zero_()
        decoder.regression.bias.fill_(percentile)
    pooled = torch.zeros(*field.state.shape, configured.schema.d_model)
    return Prediction(address=ADDRESS, payload=decoder.decode(pooled))


def write(configured: rf.Model, predicted: Prediction) -> pa.StructArray:
    extension = rf.TENSORFIELDS["quantile"]
    return extension.write(configured, predicted, extension.output(configured, ADDRESS))


def test_quantile_defaults_and_schema_round_trip():
    configured = rf.Model.xs(amount=rf.Quantile, target=rf.Quantile(compression=150, objective="huber", mask=True))
    restored = rf.Schema.model_validate_json(configured.schema.model_dump_json(round_trip=True))
    request = restored.requests[ADDRESS]

    assert request.type == "quantile"
    assert request.compression == 100
    assert request.n_bands == 8
    assert request.objective == "mae"
    assert "jitter" not in rf.Quantile.model_fields
    assert restored.requests[rf.Address("target")].compression == 150
    assert restored.requests[rf.Address("target")].objective == "huber"


@pytest.mark.parametrize("options", [{"compression": 9}, {"n_bands": 0}, {"objective": "l2"}, {"jitter": rf.Jitter()}])
def test_quantile_rejects_invalid_options(options):
    with pytest.raises(pydantic.ValidationError):
        rf.Quantile(**options)


@pytest.mark.parametrize(
    "column",
    [pa.array([True]), pa.array(["1"]), pa.array([[1.0]])],
    ids=["boolean", "string", "list"],
)
def test_quantile_rejects_incompatible_values_before_learning(column):
    configured = model()
    with pytest.raises((TypeError, ValueError), match="amount"):
        configured.encode(pa.table({"amount": column}), strata=rf.Strata.train)
    assert normalization(configured)["count"] == 0


@pytest.mark.parametrize("strata", [rf.Strata.train, rf.Strata.predict])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=["nan", "inf", "-inf"])
def test_quantile_nonfinite_values_take_the_other_state(strata, value):
    expected = [Tokens.other.value, Tokens.valued.value, Tokens.null.value]
    embedded = model()
    embedded.encode(values(10, 20, 30), strata=rf.Strata.train)
    field = embedded.encode(values(value, 20, None), strata=strata)[ADDRESS]

    assert field.state.flatten().tolist() == expected
    assert field.raw.flatten()[0] == 0
    torch.testing.assert_close(field.content.reshape(3, 2)[0], torch.tensor([0.5, 0.0]))
    assert normalization(embedded)["count"] == 3 + (strata == rf.Strata.train)

    # A supervised field carries the same states in its training targets.
    supervised = model(mask=True)
    supervised.encode(values(10, 20, 30), strata=rf.Strata.train)
    targets = supervised.encode(values(value, 20, None), strata=rf.Strata.train)[ADDRESS].targets

    assert targets[TensorKey.state].flatten().tolist() == expected
    assert targets["raw"].flatten()[0] == 0
    torch.testing.assert_close(targets[TensorKey.content].reshape(3, 2)[0], torch.tensor([0.5, 0.0]))
    assert normalization(supervised)["count"] == 4


def test_quantile_accepts_chunked_integers_without_losing_float64_distinctions():
    configured = model()
    content = [1_000_000_000.0, 1_000_000_001.0, 1_000_000_002.0]
    source = pa.table(
        {"amount": pa.chunked_array([[int(content[0])], [int(content[1]), int(content[2])]], type=pa.int64())}
    )
    field = configured.encode(source, strata=rf.Strata.train)[ADDRESS]

    assert field.content.dtype == torch.float32
    assert field.raw.dtype == torch.int64
    torch.testing.assert_close(field.raw.view(torch.float64).flatten(), torch.tensor(content, dtype=torch.float64))
    assert torch.all(field.content[..., 0].flatten().diff() > 0)
    assert field.targets.is_empty()
    assert normalization(configured)["count"] == 3


def test_quantile_empty_constant_and_tail_coordinates():
    configured = model(embed=True)
    fresh = configured.encode(values(-10, 20))[ADDRESS]
    torch.testing.assert_close(fresh.content[..., 0], torch.full_like(fresh.content[..., 0], 0.5))
    assert torch.equal(fresh.content[..., 1], torch.zeros_like(fresh.content[..., 1]))
    assert normalization(configured) == {
        "count": 0,
        "compression": 100,
        "minimum": None,
        "median": None,
        "maximum": None,
    }
    assert write(configured, decoded(configured, fresh)).field("content").to_pylist() == [None, None]

    configured.encode(values(7, 7, None), strata=rf.Strata.train)
    field = configured.encode(values(-100, 7, 100))[ADDRESS]
    torch.testing.assert_close(field.content[..., 0].flatten(), torch.tensor([0.0, 0.5, 1.0]))
    assert normalization(configured) == {"count": 2, "compression": 100, "minimum": 7.0, "median": 7.0, "maximum": 7.0}
    assert write(configured, decoded(configured, field)).field("content").to_pylist() == [7.0, 7.0, 7.0]


def test_quantile_repeated_null_and_empty_rows_keep_geometry():
    address = rf.Address("items", "amount")
    configured = rf.Model.xs(
        d_model=8,
        items=rf.Branch(length=3, amount=rf.Quantile(mask=rf.Mask(reconstruct=True))),
    )
    source = pa.Table.from_pylist(
        [
            {"items": []},
            {"items": [{"amount": 0.0}, {"amount": None}, {"amount": 10.0}]},
        ]
    )
    field = configured.encode(source, strata=rf.Strata.train)[address]

    assert field.state.shape == (2, 1, 3)
    assert field.targets.batch_size == field.state.shape
    assert field.raw.dtype == field.targets["raw"].dtype == torch.int64
    assert field.targets[TensorKey.state][0].eq(Tokens.padded).all()
    assert field.targets[TensorKey.state][1].flatten().tolist() == [Tokens.valued, Tokens.null, Tokens.valued]
    torch.testing.assert_close(
        field.targets["raw"][1].view(torch.float64).flatten(), torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64)
    )
    assert not field.present[0].any()
    assert rf.Quantile.normalization(configured, address)["count"] == 2


@pytest.mark.parametrize("owner", ["leaf", "branch", "root"])
def test_quantile_observes_skipped_supervised_targets_once(owner):
    address = rf.Address("items", "amount")
    leaf = rf.Quantile(mask=owner == "leaf")
    branch = rf.Branch(length=3, mask=owner == "branch", amount=leaf)
    configured = rf.Model.xs(d_model=8, mask=owner == "root", items=branch)
    source = pa.Table.from_pylist([{"items": [{"amount": 2.0}, {"amount": None}, {"amount": 8.0}]}])
    field = configured.encode(source, strata=rf.Strata.train)[address]

    assert not field.present.any()
    assert field.trainable.all()
    assert not field.raw.any()
    assert field.targets[TensorKey.state].flatten().tolist() == [Tokens.valued, Tokens.null, Tokens.valued]
    torch.testing.assert_close(
        field.targets["raw"].view(torch.float64).flatten(), torch.tensor([2.0, 0.0, 8.0], dtype=torch.float64)
    )
    assert rf.Quantile.normalization(configured, address)["count"] == 2


@pytest.mark.parametrize("strata", [rf.Strata.validate, rf.Strata.test, rf.Strata.predict])
def test_quantile_evaluation_reuses_state(strata):
    configured = model()
    configured.encode(values(1, 2, 3), strata=rf.Strata.train)
    before = normalization(configured)

    field = configured.encode(values(-100, 100), strata=strata)[ADDRESS]

    torch.testing.assert_close(field.content[..., 0].flatten(), torch.tensor([0.0, 1.0]))
    assert normalization(configured) == before


@pytest.mark.parametrize("workers", [0, 1])
def test_quantile_only_consumed_prefetched_batches_update_state(workers):
    configured = model(mask=True)
    data = rf.ArrowDataModule(
        configured,
        train=values(1, 2, 3, 4, 5, 6),
        shuffle=False,
        num_workers=workers,
        persistent_workers=False,
        pin_memory=False,
    )
    batches = list(data.train_dataloader())
    assert normalization(configured)["count"] == 0

    consumed = bind(configured, batches[0], rf.Strata.train)
    assert not consumed.bindings
    assert normalization(configured)["count"] == 2
    bind(configured, consumed, rf.Strata.train)
    assert normalization(configured)["count"] == 2
    pending = batches[1].tensors[ADDRESS].targets[TensorKey.content].clone()
    second = bind(configured, batches[1], rf.Strata.train)
    assert normalization(configured)["count"] == 4
    assert not torch.equal(second.tensors[ADDRESS].targets[TensorKey.content], pending)


def test_quantile_encoding_context_reads_latest_snapshot():
    configured = model()
    context = configured.interprocess_encoding_context
    before = rf.Quantile.normalization(context, ADDRESS)

    configured.encode(values(10, 20, 30), strata=rf.Strata.train)

    assert before["count"] == 0
    assert rf.Quantile.normalization(context, ADDRESS) == normalization(configured)
    assert normalization(configured)["median"] == pytest.approx(20)


def test_quantile_normalization_rejects_incorrect_source_and_address():
    configured = model()
    with pytest.raises(KeyError, match="missing"):
        rf.Quantile.normalization(configured, rf.Address("missing"))
    with pytest.raises(TypeError, match="Quantile"):
        rf.Quantile.normalization({ADDRESS: object()}, ADDRESS)
    with pytest.raises(TypeError, match="Model or encoding context"):
        rf.Quantile.normalization(object(), ADDRESS)


def test_quantile_embedding_retains_percentile_lane_without_learning():
    configured = model()
    field = configured.encode(values(1, 2, 100), strata=rf.Strata.train)[ADDRESS]
    before = normalization(configured)

    embedded = configured.nodes[ADDRESS].embedder.embed(field)

    torch.testing.assert_close(embedded.payload[..., -1], field.content[..., 0])
    assert torch.isfinite(embedded.payload).all()
    assert normalization(configured) == before


def test_quantile_checkpoint_rebuild_and_reset_preserve_state_contract(tmp_path: Path):
    configured = model(compression=150)
    configured.encode(values(10, 20, 30), strata=rf.Strata.train)
    before = normalization(configured)
    expected = configured.encode(values(15, 25))[ADDRESS].content.clone()
    pathname = tmp_path / "quantile.ckpt"
    configured.save(pathname)
    restored = rf.Model.load(pathname)

    assert normalization(restored) == before
    torch.testing.assert_close(restored.encode(values(15, 25))[ADDRESS].content, expected)
    restored.extend(other=rf.Number)
    assert normalization(restored) == before
    restored.update(rf.where("address") == ADDRESS, n_bands=4)
    assert normalization(restored) == before
    restored.encode(pa.table({"amount": [40.0], "other": [0.0]}), strata=rf.Strata.train)
    assert normalization(restored)["count"] == 4
    restored.reset(rf.where("address") == ADDRESS)
    assert normalization(restored)["count"] == 0
    assert normalization(restored)["compression"] == 150


@pytest.mark.parametrize(("objective", "expected"), [("mae", 0.25), ("mse", 0.0625), ("huber", 0.03125)])
def test_quantile_loss_uses_percentile_units_and_excludes_null_targets(monkeypatch, objective, expected):
    configured = model(mask=True, objective=objective)
    field = configured.encode(values(50, None), strata=rf.Strata.train)[ADDRESS]
    content = torch.tensor([[0.75], [1000.0]], requires_grad=True)
    tracked = {}

    def track(names, value):
        tracked[names[-2:]] = value.detach().clone()
        return value

    monkeypatch.setattr(configured, "track", track)
    objective_loss = rf.TENSORFIELDS["quantile"].loss(configured, prediction(field, content), field, rf.Strata.train)
    objective_loss.backward()

    assert tracked[Metric.loss, TensorKey.content].item() == pytest.approx(expected)
    assert content.grad[0].abs().item() > 0
    assert content.grad[1].item() == 0
    assert normalization(configured)["count"] == 1


def test_quantile_all_null_targets_have_finite_state_only_loss(monkeypatch):
    configured = model(mask=True)
    field = configured.encode(values(None, None), strata=rf.Strata.train)[ADDRESS]
    content = torch.zeros_like(field.content[..., 0], requires_grad=True)
    monkeypatch.setattr(configured, "track", lambda names, value: value)

    result = rf.TENSORFIELDS["quantile"].loss(configured, prediction(field, content), field, rf.Strata.train)

    assert torch.isfinite(result)
    assert normalization(configured)["count"] == 0


def test_quantile_output_and_metrics_return_to_source_units(monkeypatch):
    results = []
    for scale in (1.0, 1000.0):
        configured = model(mask=True)
        configured.encode(values(*(value * scale for value in range(0, 101, 10))), strata=rf.Strata.train)
        field = configured.encode(values(50 * scale), strata=rf.Strata.train)[ADDRESS]
        predicted = decoded(configured, field, 0.75)
        tracked = {}

        def track(names, value):
            tracked[names[-2:]] = float(value.detach())
            return value

        monkeypatch.setattr(configured, "track", track)
        rf.TENSORFIELDS["quantile"].loss(configured, predicted, field, rf.Strata.train)
        output = write(configured, predicted)
        assert output.type.field("content").type == pa.float64()
        assert output.type.field("content").nullable
        raw = output.field("content")[0].as_py()
        assert tracked[Metric.mae, TensorKey.content] == pytest.approx(abs(raw - 50 * scale))
        results.append((raw, tracked))

    assert results[1][0] == pytest.approx(results[0][0] * 1000)
    assert results[1][1][Metric.loss, TensorKey.content] == pytest.approx(results[0][1][Metric.loss, TensorKey.content])
    assert results[1][1][Metric.mae, TensorKey.content] == pytest.approx(
        results[0][1][Metric.mae, TensorKey.content] * 1000
    )


def test_quantile_decoded_prediction_keeps_its_distribution_snapshot():
    configured = model(mask=True)
    field = configured.encode(values(10, 20, 30), strata=rf.Strata.train)[ADDRESS]
    predicted = decoded(configured, field)
    expected = write(configured, predicted)

    configured.encode(values(1000, 2000, 3000), strata=rf.Strata.train)

    assert write(configured, predicted).equals(expected)


@pytest.mark.parametrize(("percentile", "expected"), [(0.0, 10.0), (1.0, 30.0)])
def test_quantile_decoder_endpoints_return_learned_extrema(percentile, expected):
    configured = model(mask=True)
    field = configured.encode(values(10, 20, 30), strata=rf.Strata.train)[ADDRESS]

    output = write(configured, decoded(configured, field, percentile))

    assert output.field("content").to_pylist() == [expected] * 3


def test_quantile_nullable_false_and_typed_empty_predictions():
    configured = model(mask=True, nullable=False)
    with pytest.raises(ValueError, match="nullable=False"):
        configured.encode(values(None))

    output = configured.predict(values())
    assert output.num_rows == 0
    content = output.schema.field("predictions").type.field(str(ADDRESS).removeprefix("/")).type.field("content")
    assert content.type == pa.float64()
    assert content.nullable


def test_quantile_worker_encoding_does_not_mutate_live_distribution():
    configured = model(mask=True)
    encoded = encode(values(1, 2), configured.schema, rf.Strata.train, configured.interprocess_encoding_context)

    assert encoded.bindings
    assert normalization(configured)["count"] == 0
    bound = bind(configured, encoded, rf.Strata.train)
    assert normalization(configured)["count"] == 2
    assert torch.isfinite(bound.tensors[ADDRESS].targets[TensorKey.content]).all()


def test_quantile_lightning_training_learns_only_consumed_values():
    torch.manual_seed(811)
    configured = rf.Model.xs(
        d_model=16,
        batch_size=64,
        x=rf.Quantile,
        y=rf.Quantile(mask=True),
    )
    x = torch.linspace(0.0, 1.0, 64, dtype=torch.float64)
    train = pa.table({"x": x.numpy(), "y": (100 + 900 * x.square()).numpy()})
    validate = pa.table({"x": (10 + x).numpy(), "y": (1_000_000 + x).numpy()})
    target = rf.Address("y")
    before = configured.nodes[target].decoder.regression.weight.detach().clone()
    losses: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    class Audit(lit.Callback):
        def on_after_backward(self, trainer, pl_module):
            gradient = pl_module.nodes[target].decoder.regression.weight.grad
            assert gradient is not None
            gradients.append(gradient.detach().clone())

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            losses.append(outputs["loss"].detach().clone())
            assert not batch.bindings

    data = rf.ArrowDataModule(
        model=configured,
        train=train,
        validate=validate,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    configured.fit(
        data,
        learning_rate=0.01,
        weight_decay=0.0,
        fused=False,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[Audit()],
    )

    assert len(losses) == len(gradients) == 1
    assert torch.isfinite(losses[0])
    assert torch.isfinite(gradients[0]).all()
    assert gradients[0].abs().sum() > 0
    assert not torch.equal(configured.nodes[target].decoder.regression.weight, before)
    for address, minimum, maximum in ((rf.Address("x"), 0.0, 1.0), (target, 100.0, 1000.0)):
        state = rf.Quantile.normalization(configured, address)
        assert state["count"] == 64
        assert state["minimum"] == minimum
        assert state["maximum"] == maximum
        distribution = configured.nodes[address].embedder.distribution
        assert distribution.state.manager is None
        assert isinstance(distribution.state.master, list)

    predictions = (
        configured.predict(validate)["predictions"]
        .combine_chunks()
        .field(str(target).removeprefix("/"))
        .field("content")
    )
    assert predictions.null_count == 0
    content = torch.from_numpy(predictions.to_numpy().copy())
    assert torch.isfinite(content).all()
    assert content.ge(100).all() and content.le(1000).all()
    assert rf.Quantile.normalization(configured, target)["count"] == 64


@pytest.mark.parametrize("operation", ["restore", "rebuild"])
def test_quantile_forward_reuses_encoded_inputs_after_restoring_resources(tmp_path: Path, operation):
    torch.manual_seed(811)
    configured = rf.Model.xs(
        d_model=8,
        x=rf.Quantile,
        y=rf.Quantile(mask=True),
    )
    configured.encode(pa.table({"x": [1.0, 2.0, 3.0], "y": [100.0, 200.0, 300.0]}), strata=rf.Strata.train)
    encoded = configured.encode(pa.table({"x": [2.0], "y": [None]}))
    configured.eval()
    with torch.inference_mode():
        previous = next(value for value in configured(encoded, strata=rf.Strata.predict) if value.address == "/y")
    expected = previous.payload[TensorKey.content].clone()
    assert torch.isfinite(expected).all()

    if operation == "restore":
        pathname = tmp_path / "quantile-forward.ckpt"
        configured.save(pathname)
        configured = rf.Model.load(pathname).eval()
    else:
        configured.update(rf.where("address") == "/x", description="Visible numeric context")

    # Rebinding would hide an independent, empty decoder resource after rebuild.
    with torch.inference_mode():
        current = next(value for value in configured(encoded, strata=rf.Strata.predict) if value.address == "/y")

    assert torch.isfinite(current.payload[TensorKey.content]).all()
    torch.testing.assert_close(current.payload[TensorKey.content], expected, rtol=0, atol=0)
    assert rf.Quantile.normalization(configured, rf.Address("y"))["count"] == 3


def test_quantile_cold_validation_registers_undefined_source_unit_metrics(monkeypatch):
    configured = model(mask=True)
    field = configured.encode(values(10, 20), strata=rf.Strata.validate)[ADDRESS]
    predicted = decoded(configured, field)
    tracked = {}

    def track(names, value):
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()
        tracked[names[-2:]] = value
        return value

    monkeypatch.setattr(configured, "track", track)

    result = rf.TENSORFIELDS["quantile"].loss(configured, predicted, field, rf.Strata.validate)

    assert torch.isfinite(result)
    assert (Metric.loss, TensorKey.state) in tracked
    assert (Metric.loss, TensorKey.content) in tracked
    # Every rank accumulates the source-unit totals for the epoch; without an
    # inverse their value is undefined rather than fabricated from nothing.
    assert (Metric.mae, TensorKey.content) not in tracked
    metrics = configured.nodes[ADDRESS].decoder.metrics.compute(rf.Strata.validate)
    for name in ("mae.content", "rmse.content"):
        assert torch.isnan(metrics[name])
    assert normalization(configured)["count"] == 0


@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_quantile_evaluation_metrics_count_valued_targets_across_unequal_batches(batch_size, tmp_path):
    configured = model(mask=True)
    configured.encode(values(*range(0, 101, 10)), strata=rf.Strata.train)
    decoder = configured.nodes[ADDRESS].decoder
    with torch.no_grad():
        decoder.regression.weight.zero_()
        decoder.regression.bias.fill_(0.5)
    data = rf.ArrowDataModule(configured, validate=values(30.0, 40.0, None))
    results = configured.validate(
        data,
        batch_size=batch_size,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_progress_bar=False,
        default_root_dir=tmp_path,
    )[0]
    # The median readout predicts 50 for every row: errors 20 and 10 over two valued rows.
    assert results[".amount/validate.mae.content"] == pytest.approx(15.0)
    assert results[".amount/validate.rmse.content"] == pytest.approx((500 / 2) ** 0.5)


def test_quantile_readout_stays_wide_under_autocast():
    configured = model(mask=True)
    decoder = configured.nodes[ADDRESS].decoder

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        payload = decoder.decode(torch.zeros(3, configured.schema.d_model))

    assert payload["percentile"].dtype == torch.float32
    assert payload[TensorKey.content].dtype == torch.float64


def test_quantile_spread_lane_keeps_growing_beyond_learned_support():
    configured = model()
    configured.encode(values(*range(0, 101, 10)), strata=rf.Strata.train)

    field = configured.encode(values(200, 1_000, 10_000, None))[ADDRESS]

    ranks, spread = field.content[..., 0].flatten(), field.content[..., 1].flatten()
    assert torch.equal(ranks[:3], torch.ones(3))
    assert torch.all(spread[:3].diff() > 0)
    assert spread.max() <= 64.0
    assert spread[3] == 0.0


@pytest.mark.parametrize(("score", "low", "high"), [(-0.5, -30.0, -20.0), (1.5, 120.0, 130.0)])
def test_quantile_decoder_extrapolates_one_interquartile_range_per_unit_score(score, low, high):
    configured = model(mask=True)
    field = configured.encode(values(*range(0, 101)), strata=rf.Strata.train)[ADDRESS]

    output = write(configured, decoded(configured, field, score)).field("content").to_pylist()

    assert all(low <= value <= high for value in output)


def test_quantile_heavy_tail_predictions_stay_near_the_bulk():
    from relflow.tensorfields.extensions.quantile import SPREAD_LIMIT, inverse, spread
    from relflow.tensorfields.shared.digest import Digest

    sample = np.random.default_rng(0).lognormal(0.0, 2.0, 200_000)
    digest = Digest(100)
    digest.update(pa.array(sample))
    median, scale = spread(digest)
    p99 = float(np.quantile(sample, 0.99))

    scores = torch.tensor([-0.1, 0.0, 0.5, 0.9, 1.0, 1.01, 1.1, 1.5], dtype=torch.float32)
    content = inverse(digest, scores).tolist()

    # The training maximum is hundreds of times the 99th percentile; the
    # inverse no longer continues from it, so overshoot stays near the bulk.
    assert float(np.max(sample)) > 100 * p99
    assert max(content) <= median + SPREAD_LIMIT * scale < 5 * p99
    assert min(content) >= median - SPREAD_LIMIT * scale
    assert content[2] == pytest.approx(median, rel=0.05)
    assert all(earlier <= later for earlier, later in zip(content[:-1], content[1:], strict=True))
