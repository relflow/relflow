"""Coordinates input, geographic loss, and model integration."""

from types import SimpleNamespace

import polars as pl
import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.query import Projection
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions import coordinates, vector


def model(field=rf.Coordinates):
    return rf.Model(preset="small", d_model=8, point=field)


def encoded(field, data):
    return model(field).encode(data)["/point"]


def test_coordinates_is_an_inherited_public_type():
    assert rf.Coordinates is rf.TENSORFIELDS["coordinates"].Request
    assert issubclass(rf.Coordinates, rf.Vector)
    assert coordinates.coordinates.callback_factories == vector.vector.callback_factories
    assert issubclass(coordinates.TensorField, vector.TensorField)
    assert issubclass(coordinates.Decoder, vector.Decoder)
    assert not hasattr(rf.Component, "derive")


def test_one_pair_and_two_queries_produce_the_same_double_precision_values():
    pair = encoded(rf.Coordinates, pl.DataFrame({"point": [[40.0, -73.0], [40.000001, -73.0], None]}))
    separate = encoded(
        rf.Coordinates(query=("lat", "lon")),
        pl.DataFrame({"lat": [40.0, 40.000001, None], "lon": [-73.0, -73.0, None]}),
    )
    assert pair.content.dtype == torch.float64
    assert torch.equal(pair.content, separate.content)
    assert torch.equal(pair.state, separate.state)
    assert pair.content.reshape(-1, 2).tolist() == [[40.0, -73.0], [40.000001, -73.0], [0.0, 0.0]]
    assert pair.state.reshape(-1).tolist() == [Tokens.valued, Tokens.valued, Tokens.null]


@pytest.mark.parametrize("query", [(), ("a", "b", "c")])
def test_request_rejects_wrong_query_count(query):
    with pytest.raises(ValueError, match="quer"):
        model(rf.Coordinates(query=query))


@pytest.mark.parametrize("options", [{"n_dim": 3}, {"objective": "l1"}])
def test_request_rejects_non_geographic_vector_options(options):
    with pytest.raises(ValueError):
        rf.Coordinates(**options)


@pytest.mark.parametrize(
    ("point", "message"),
    [([1.0], "pairs"), ([1.0, 2.0, 3.0], "pairs"), ([91.0, 0.0], "latitude"), ([0.0, -181.0], "longitude")],
)
def test_invalid_coordinates_are_rejected_even_when_masked(point, message):
    with pytest.raises(ValueError, match=message):
        encoded(rf.Coordinates(mask=True), pl.DataFrame({"point": [point]}))


def test_partial_null_and_nonfinite_points_are_not_targets():
    source = pa.table(
        {
            "point": pa.array(
                [[1.0, 2.0], [None, 2.0], [3.0, None], None, [float("nan"), 2.0], [3.0, float("inf")]],
                type=pa.list_(pa.float64()),
            )
        }
    )
    field = model(rf.Coordinates(mask=True)).encode(source, strata="train")["/point"]
    assert field.targets[TensorKey.state].reshape(-1).tolist() == [
        Tokens.valued,
        Tokens.null,
        Tokens.null,
        Tokens.null,
        Tokens.other,
        Tokens.other,
    ]


def test_structural_absence_hides_invalid_values_from_the_other_source():
    latitude = Projection(pa.array([40.0, 900.0]), pa.array([True, True]))
    longitude = Projection(pa.array([-73.0, 0.0]), pa.array([True, False]))
    combined = coordinates.values((latitude, longitude), rows=2, address=rf.Address("point"))
    assert combined.values.to_pylist() == [[40.0, -73.0], None]
    assert combined.present.to_pylist() == [True, False]


def test_training_center_uses_hidden_valued_targets_and_is_not_updated_by_validation(tmp_path):
    fitted = model(rf.Coordinates(mask=True))
    frame = pa.table(
        {"point": pa.array([[0.0, 179.9], [0.0, -179.9], None, [float("nan"), 0.0]], type=pa.list_(pa.float64()))}
    )
    field = fitted.encode(frame, strata="train")["/point"]
    decoder = fitted.nodes["/point"].decoder
    assert all(item.count.item() == 2 for item in decoder.normalizers)
    center = torch.cat([item.mean for item in decoder.normalizers])
    assert center[0] < -0.99
    assert abs(center[1]) < 1e-12
    assert torch.count_nonzero(field.content) == 0

    fitted.encode(pa.table({"point": [[30.0, 0.0]]}), strata="validate")
    assert all(item.count.item() == 2 for item in decoder.normalizers)
    assert torch.equal(torch.cat([item.mean for item in decoder.normalizers]), center)

    restored = rf.Model.load(fitted.save(tmp_path / "coordinates.ckpt"))
    assert isinstance(restored.schema.requests["/point"], rf.Coordinates)
    assert torch.equal(torch.cat([item.mean for item in restored.nodes["/point"].decoder.normalizers]), center)


def test_polar_training_targets_have_finite_gradients():
    fitted = model(rf.Coordinates(mask=True))
    fitted.encode(pa.table({"point": [[90.0, 0.0], [90.0, 180.0]]}), strata="train")
    decoder = fitted.nodes["/point"].decoder
    pooled = torch.zeros(2, 8, requires_grad=True)
    predicted = decoder.decode(pooled)[TensorKey.content]
    loss = coordinates.angle(predicted, torch.tensor([[90.0, 0.0], [90.0, 180.0]], dtype=torch.float64)).sum()
    loss.backward()
    assert torch.isfinite(predicted).all()
    assert torch.isfinite(pooled.grad).all()


def test_relative_and_absolute_ancestor_queries_broadcast_to_retained_records():
    record = pa.struct([("lon", pa.float64())])
    source = pa.table(
        {
            "lat": [45.0, -12.0],
            "visits": pa.array([[{"lon": 11.0}, None, {"lon": 12.0}], [{"lon": 20.0}]], pa.list_(record)),
        }
    )
    for path in ("../lat", "/lat"):
        fitted = rf.Model(
            preset="small",
            d_model=8,
            visits=rf.Branch(length=2, overflow="tail", point=rf.Coordinates(query=(path, "lon"))),
        )
        field = fitted.encode(source)["/visits/point"]
        assert field.state.reshape(2, -1).tolist() == [[Tokens.padded, Tokens.valued], [Tokens.valued, Tokens.padded]]
        assert field.content.reshape(2, -1, 2).tolist() == [[[0.0, 0.0], [45.0, 12.0]], [[-12.0, 20.0], [0.0, 0.0]]]


@pytest.mark.parametrize(
    ("first", "second", "meters"),
    [
        ([0.0, 0.0], [0.0, 0.0], 0.0),
        ([0.0, 0.0], [1.0, 0.0], coordinates.EARTH_RADIUS * torch.pi / 180),
        ([0.0, 179.999], [0.0, -179.999], coordinates.EARTH_RADIUS * torch.pi / 180 * 0.002),
        ([90.0, 0.0], [90.0, 180.0], 0.0),
        ([0.0, 0.0], [0.0, 180.0], coordinates.EARTH_RADIUS * torch.pi),
        ([40.0, -73.0], [40.000001, -73.0], coordinates.EARTH_RADIUS * torch.pi / 180 * 0.000001),
    ],
)
def test_distance_and_gradients_are_well_behaved(first, second, meters):
    prediction = torch.tensor([first], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([second], dtype=torch.float64)
    distance = coordinates.angle(prediction, target) * coordinates.EARTH_RADIUS
    assert distance.item() == pytest.approx(meters, abs=1e-6, rel=1e-8)
    distance.sum().backward()
    assert torch.isfinite(prediction.grad).all()


def test_embedder_represents_the_two_sides_of_the_date_line_together():
    near = torch.tensor([[0.0, 180.0], [0.0, -180.0]], dtype=torch.float64)
    assert torch.allclose(coordinates.unit(near)[0], coordinates.unit(near)[1], atol=1e-15)


def test_decoder_readout_keeps_source_precision_under_mixed_precision():
    fitted = model(rf.Coordinates(mask=True))
    decoder = fitted.nodes["/point"].decoder
    pooled = torch.randn(4, 8).bfloat16()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = decoder.decode(pooled)
    assert output[TensorKey.content].dtype == torch.float64
    assert output[TensorKey.content].abs().le(torch.tensor([90.0, 180.0])).all()


def test_loss_uses_spherical_distance_and_metrics_count_positions():
    fitted = model(rf.Coordinates(mask=True))
    field = fitted.encode(pl.DataFrame({"point": [[0.0, 1.0], [0.0, 2.0], None]}), strata="train")["/point"]
    state = torch.full((*field.state.shape, len(Tokens)), -50.0)
    state.scatter_(-1, field.targets[TensorKey.state].unsqueeze(-1), 50.0)
    content = torch.zeros((*field.state.shape, 2), requires_grad=True)
    prediction = Prediction(address="/point", payload=TensorDict({TensorKey.state: state, TensorKey.content: content}))
    logged = {}
    module = SimpleNamespace(
        schema=fitted.schema, nodes=fitted.nodes, track=lambda names, value: logged.setdefault(names, value)
    )
    loss = coordinates.loss(module, prediction, field, Strata.validate)
    assert loss.item() == pytest.approx((1.5 * torch.pi / 180) / fitted.nodes["/point"].decoder.spread.item())
    metrics = fitted.nodes["/point"].decoder.metrics.compute(Strata.validate)
    degree = coordinates.EARTH_RADIUS * torch.pi / 180
    assert metrics["mae.content"].item() == pytest.approx(1.5 * degree, rel=1e-6)
    assert metrics["rmse.content"].item() == pytest.approx((2.5**0.5) * degree, rel=1e-6)
    loss.backward()
    assert torch.isfinite(content.grad).all()


def test_null_only_targets_have_finite_loss():
    fitted = model(rf.Coordinates(mask=True))
    field = fitted.encode(pa.table({"point": pa.array([None], type=pa.list_(pa.float64()))}), strata="train")["/point"]
    state = torch.zeros((*field.state.shape, len(Tokens)), requires_grad=True)
    content = torch.zeros((*field.state.shape, 2), requires_grad=True)
    prediction = Prediction(address="/point", payload=TensorDict({TensorKey.state: state, TensorKey.content: content}))
    module = SimpleNamespace(schema=fitted.schema, nodes=fitted.nodes, track=lambda names, value: value)
    loss = coordinates.loss(module, prediction, field, Strata.validate)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(content.grad).all()


def test_schema_roundtrip_preserves_coordinates():
    fitted = model(rf.Coordinates(query=("lat", "lon"), mask=True))
    schema = rf.Schema.model_validate_json(fitted.schema.model_dump_json())
    request = schema.requests["/point"]
    assert isinstance(request, rf.Coordinates)
    assert request.query == ("lat", "lon")


@pytest.mark.integration
def test_coordinates_fit_validate_and_predict(tmp_path):
    source = pl.DataFrame(
        {
            "lat": [40.0, 41.0, 42.0, 43.0],
            "lon": [-73.0, -74.0, -75.0, -76.0],
            "input": [1.0, 2.0, 3.0, 4.0],
            "current": [[39.0, -72.0], [40.0, -73.0], [41.0, -74.0], [42.0, -75.0]],
        }
    )
    fitted = rf.Model(
        preset="small",
        d_model=8,
        input=rf.Number,
        current=rf.Coordinates,
        point=rf.Coordinates(query=("lat", "lon"), mask=True),
    )
    data = rf.PolarsDataModule(train=source, validate=source, num_workers=0, persistent_workers=False, pin_memory=False)
    fitted.fit(
        data,
        batch_size=2,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        rollback=False,
        early_stopping=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    metrics = fitted.validate(
        data, batch_size=2, accelerator="cpu", devices=1, precision="32-true", enable_progress_bar=False
    )[0]
    assert 0 <= metrics[".point/validate.mae.content"] <= coordinates.EARTH_RADIUS * torch.pi
    result = fitted.predict(source.drop("lat", "lon"), batch_size=2).to_dicts()
    assert len(result) == 4
    for row in result:
        lat, lon = row["predictions"]["point"]["content"]
        assert -90 <= lat <= 90
        assert -180 <= lon <= 180
