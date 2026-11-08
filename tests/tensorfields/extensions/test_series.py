import ast
import warnings
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pydantic
import pytest
import torch
import yaml
from lightning.pytorch import seed_everything
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Metric, Overflow, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.series import Block, Decoder, Embedder, TensorField, encode, loss, write
from relflow.tensorfields.extensions.series import output as output_type
from relflow.tensorfields.extensions.series import series as extension
from relflow.tensorfields.shared.jitter import Jitter
from relflow.tensorfields.shared.normalizer import GlobalOnlineNormalizer, moments
from tests.arrow import table as arrow_table
from tests.tensorfields.helpers import tensorize

SEED = 0


def randn(*shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(SEED))


NAN = float("nan")
INF = float("inf")

ADDRESS = "/items/signal"


def build_schema(**options) -> Schema:
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        items=rf.Branch(length=2, signal=rf.Series(**options)),
    )


def test_series_request_defaults():
    request = build_schema(channels=2).requests[ADDRESS]
    assert (request.type, request.length, request.overflow, request.horizon, request.n_blocks) == (
        "series",
        None,
        None,
        None,
        6,
    )


def test_series_capped_length_defaults_overflow_to_head():
    assert build_schema(channels=1, length=8).requests[ADDRESS].overflow == Overflow.head


def test_series_uncapped_rejects_overflow():
    with pytest.raises(ValueError, match="cannot configure overflow"):
        rf.Series(channels=1, overflow="tail")


@pytest.mark.parametrize(
    "option",
    [{"channels": 0}, {"channels": 1, "length": 0}, {"channels": 1, "horizon": 0}, {"channels": 1, "n_blocks": 0}],
)
def test_series_rejects_non_positive_sizes(option):
    with pytest.raises(pydantic.ValidationError):
        rf.Series(**option)


def test_series_is_exported():
    assert rf.Series is TENSORFIELDS["series"].Request


def test_named_channels_set_width():
    request = build_schema(channels=("temp", "hum")).requests[ADDRESS]
    assert request.channels == ("temp", "hum")
    assert (request.layout.width, request.layout.names) == (2, ("temp", "hum"))


def test_integer_channels_width():
    layout = build_schema(channels=3).requests[ADDRESS].layout
    assert (layout.width, layout.names) == (3, None)


@pytest.mark.parametrize("channels", [(), ("temp", "temp"), ("",)])
def test_named_channels_reject_empty_duplicate_or_blank(channels):
    with pytest.raises(pydantic.ValidationError):
        rf.Series(channels=channels)


def test_named_channels_survive_schema_yaml_and_checkpoint(tmp_path):
    seed_everything(SEED, workers=True)
    model = rf.Model.xs(d_model=8, signal=rf.Series(channels=("temp", "hum")))
    path = tmp_path / "schema.yaml"
    path.write_text(yaml.safe_dump(model.schema.model_dump(mode="json")), encoding="utf-8")
    reloaded = Schema.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    assert reloaded.requests["/signal"].channels == ("temp", "hum")
    loaded = rf.Model.load(model.save(tmp_path / "model.ckpt"))
    assert loaded.schema.requests["/signal"].channels == ("temp", "hum")


def test_named_channels_round_trip_yaml(tmp_path):
    path = tmp_path / "model.yaml"
    path.write_text(
        "d_model: 8\nn_layers: 1\nn_heads: 4\nfields:\n  signal:\n    type: series\n    channels: [temp, hum]\n",
        encoding="utf-8",
    )
    loaded = rf.Model.from_yaml(path)
    assert loaded.schema.requests["/signal"].channels == ("temp", "hum")
    assert loaded.schema.model_dump(mode="json")["fields"]["fields"][0]["channels"] == ["temp", "hum"]


STEP = pa.struct([("temp", pa.float64()), ("hum", pa.int64())])
COLUMNS = pa.struct([("temp", pa.list_(pa.float64())), ("hum", pa.large_list(pa.float32()))])


@pytest.mark.parametrize("datatype", [pa.list_(STEP), pa.large_list(STEP), COLUMNS])
def test_extension_accepts_struct_channels(datatype):
    assert extension.accepts(datatype)


@pytest.mark.parametrize(
    "datatype",
    [
        pa.list_(pa.struct([("temp", pa.float64()), ("unit", pa.string())])),
        pa.struct([("temp", pa.list_(pa.float64())), ("hum", pa.float64())]),
        pa.list_(pa.struct([])),
        pa.list_(pa.string()),
    ],
)
def test_extension_rejects_mixed_or_non_numeric_structs(datatype):
    assert not extension.accepts(datatype)


def projection(rows: list, schema: Schema, strata: Strata, datatype: pa.DataType | None = None):
    arrow = None
    if datatype is not None:
        arrow = pa.schema([("items", pa.list_(pa.struct([("signal", datatype)])))])
    batch = arrow_table([{"items": [{"signal": row}]} for row in rows], arrow)
    return coalesce(batch, schema=schema, strata=strata).fields[ADDRESS]


def tensorfield(
    rows: list,
    schema: Schema,
    strata: Strata = Strata.train,
    *,
    salt: int = 0,
    datatype: pa.DataType | None = None,
) -> TensorField:
    return tensorize(
        TensorField,
        projection(rows, schema, strata, datatype),
        TENSORFIELDS["series"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
        context=Context(salt=salt),
    )


def steps(field: TensorField) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Values, observed and state of the first item of every record."""
    return field.content["values"][:, 0, 0], field.content["observed"][:, 0, 0], field.state[:, 0, 0]


LISTS = [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]
A_ROW = [{"temp": 1.0, "hum": 10.0}, {"temp": 2.0, "hum": 20.0}, {"temp": 3.0, "hum": 30.0}]
B_ROW = {"temp": [1.0, 2.0, 3.0], "hum": [10.0, 20.0, 30.0]}
A_TYPE = pa.list_(pa.struct([("temp", pa.float64()), ("hum", pa.float64())]))
B_TYPE = pa.struct([("temp", pa.list_(pa.float64())), ("hum", pa.list_(pa.float64()))])


@pytest.mark.parametrize("row, datatype", [(A_ROW, A_TYPE), (B_ROW, B_TYPE)])
def test_struct_shapes_match_list_of_lists(row, datatype):
    expected, _, _ = steps(tensorfield([LISTS], build_schema(channels=2)))
    values, observed, state = steps(tensorfield([row], build_schema(channels=2), datatype=datatype))
    torch.testing.assert_close(values, expected)
    assert observed.all() and state.tolist() == [Tokens.valued.value]


@pytest.mark.parametrize("row, datatype", [(A_ROW, A_TYPE), (B_ROW, B_TYPE)])
def test_named_channels_set_order(row, datatype):
    values, _, _ = steps(tensorfield([row], build_schema(channels=("hum", "temp")), datatype=datatype))
    assert values[0, :, 0].tolist() == [10.0, 20.0, 30.0]


def test_named_channels_select_a_subset():
    row = [{"temp": 1.0, "hum": 10.0, "wind": 5.0}]
    datatype = pa.list_(pa.struct([("temp", pa.float64()), ("hum", pa.float64()), ("wind", pa.float64())]))
    values, _, _ = steps(tensorfield([row], build_schema(channels=("wind",)), datatype=datatype))
    assert values.shape[-1] == 1 and values[0, 0, 0].item() == 5.0


def test_missing_named_member_names_address_and_members():
    with pytest.raises(ValueError, match=r"/items/signal.*rain.*preprocessor"):
        tensorfield([A_ROW], build_schema(channels=("temp", "rain")), datatype=A_TYPE)


def test_integer_channels_must_match_member_count():
    with pytest.raises(ValueError, match=r"/items/signal.*channels=3"):
        tensorfield([A_ROW], build_schema(channels=3), datatype=A_TYPE)


def test_named_channels_reject_list_input():
    with pytest.raises(ValueError, match=r"/items/signal.*struct"):
        tensorfield([LISTS], build_schema(channels=("temp", "hum")))


def test_columns_with_different_step_counts_raise():
    row = {"temp": [1.0, 2.0], "hum": [10.0]}
    with pytest.raises(ValueError, match=r"/items/signal.*step count.*pad"):
        tensorfield([row], build_schema(channels=2), datatype=B_TYPE)


def test_null_column_warns_and_marks_its_steps_unobserved():
    row = {"temp": [1.0, 2.0], "hum": None}
    with pytest.warns(UserWarning, match=r"/items/signal.*hum"):
        _, observed, state = steps(tensorfield([row], build_schema(channels=2), datatype=B_TYPE))
    assert state.tolist() == [Tokens.other.value]
    assert not observed.any()


def test_null_column_warning_shows_once_per_field_and_member():
    schema = build_schema(channels=2)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter("default")
        for count in (1, 2):
            tensorfield([{"temp": [1.0, 2.0], "hum": None}] * count, schema, datatype=B_TYPE)
    assert len([w for w in shown if "null channel list" in str(w.message)]) == 1


def test_all_null_columns_warn_and_give_zero_steps():
    row = {"temp": None, "hum": None}
    with pytest.warns(UserWarning, match=r"/items/signal"):
        _, observed, state = steps(tensorfield([row], build_schema(channels=2), datatype=B_TYPE))
    assert state.tolist() == [Tokens.valued.value] and not observed.any()


def test_null_value_inside_column_is_an_unobserved_step_without_warning(recwarn):
    row = {"temp": [1.0, None], "hum": [10.0, 20.0]}
    _, observed, _ = steps(tensorfield([row], build_schema(channels=2), datatype=B_TYPE))
    assert observed[0, :2].tolist() == [True, False]
    assert not [w for w in recwarn if issubclass(w.category, UserWarning) and "/items/signal" in str(w.message)]


def test_null_struct_step_and_null_member_are_unobserved():
    row = [{"temp": 1.0, "hum": 10.0}, None, {"temp": 3.0, "hum": None}]
    _, observed, _ = steps(tensorfield([row], build_schema(channels=2), datatype=A_TYPE))
    assert observed[0, :3].tolist() == [True, False, False]


def test_integer_struct_members_cast_to_float64():
    datatype = pa.list_(pa.struct([("temp", pa.int64()), ("hum", pa.int64())]))
    values, _, _ = steps(tensorfield([[{"temp": 1, "hum": 2}]], build_schema(channels=2), datatype=datatype))
    assert values[0, 0].tolist() == [1.0, 2.0]


@pytest.mark.parametrize(
    "row, datatype",
    [
        (A_ROW, pa.large_list(pa.struct([("temp", pa.float64()), ("hum", pa.float64())]))),
        (B_ROW, pa.struct([("temp", pa.large_list(pa.float64())), ("hum", pa.list_(pa.float64(), 3))])),
    ],
)
def test_struct_shapes_large_and_fixed_size_variants(row, datatype):
    values, _, _ = steps(tensorfield([row], build_schema(channels=2), datatype=datatype))
    assert values[0, :3, 0].tolist() == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "datatype, row",
    [
        (pa.list_(pa.float64()), [1.0, 2.0, 3.0]),
        (pa.list_(pa.list_(pa.float64())), [[1.0], [2.0], [3.0]]),
        (pa.list_(pa.list_(pa.float64(), 1)), [[1.0], [2.0], [3.0]]),
    ],
)
def test_univariate_input_types(datatype, row):
    values, observed, state = steps(tensorfield([row], build_schema(channels=1), datatype=datatype))
    assert values[0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert observed[0].tolist() == [True, True, True]
    assert state.tolist() == [Tokens.valued]


@pytest.mark.parametrize(
    "datatype, row",
    [
        (pa.large_list(pa.float64()), [1.0, 2.0, 3.0]),
        (pa.large_list(pa.large_list(pa.float64())), [[1.0], [2.0], [3.0]]),
        (pa.large_list(pa.list_(pa.float64())), [[1.0], [2.0], [3.0]]),
    ],
)
def test_large_list_variants_univariate(datatype, row):
    values, observed, state = steps(tensorfield([row], build_schema(channels=1), datatype=datatype))
    assert values[0, :, 0].tolist() == [1.0, 2.0, 3.0]
    assert observed[0].tolist() == [True, True, True]
    assert state.tolist() == [Tokens.valued]


@pytest.mark.parametrize(
    "datatype",
    [
        pa.large_list(pa.large_list(pa.float64())),
        pa.large_list(pa.list_(pa.float64())),
        pa.list_(pa.list_(pa.float64(), 2)),
    ],
)
def test_multivariate_list_variants_with_null_step(datatype):
    row = [[1.0, 2.0], None, [5.0, 6.0]]
    values, observed, state = steps(tensorfield([row], build_schema(channels=2), datatype=datatype))
    assert observed[0].tolist() == [True, False, True]
    assert values[0, [0, 2]].tolist() == [[1.0, 2.0], [5.0, 6.0]]
    assert state.tolist() == [Tokens.valued]


def test_multivariate_wrong_inner_width_names_address():
    with pytest.raises(ValueError, match=r"/items/signal.*channels=2.*3"):
        tensorfield([[[1.0, 2.0, 3.0]]], build_schema(channels=2))


def test_integer_elements_cast_to_float64():
    values, _, _ = steps(tensorfield([[1, 2, 3]], build_schema(channels=1)))
    assert values.dtype == torch.float64
    assert values[0, :, 0].tolist() == [1.0, 2.0, 3.0]


def test_integers_beyond_float64_precision_name_address():
    with pytest.raises(ValueError, match="/items/signal"):
        tensorfield([[2**53 + 1]], build_schema(channels=1))


def test_missing_steps_are_unobserved_not_whole_record():
    row = [[1.0, 2.0], [NAN, 2.0], [1.0, INF], [None, 1.0], [3.0, 4.0]]
    values, observed, state = steps(tensorfield([row], build_schema(channels=2)))
    assert observed[0].tolist() == [True, False, False, False, True]
    assert state.tolist() == [Tokens.valued]
    assert torch.isfinite(values).all()
    assert values[0, 1].tolist() == [0.0, 0.0]


def test_empty_series_is_valued_and_all_nan_is_other():
    _, _, state = steps(tensorfield([[], [[NAN]], None], build_schema(channels=1)))
    assert state.tolist() == [Tokens.valued, Tokens.other, Tokens.null]


def test_batch_pads_to_longest_and_never_to_zero_width():
    schema = build_schema(channels=1)
    values, observed, _ = steps(tensorfield([[1.0], [1.0, 2.0, 3.0]], schema))
    assert values.shape == (2, 3, 1)
    assert observed.tolist() == [[True, False, False], [True, True, True]]

    values, observed, state = steps(tensorfield([[], None], schema, datatype=pa.list_(pa.float64())))
    assert values.shape == (2, 1, 1)
    assert not observed.any()
    assert state.tolist() == [Tokens.valued, Tokens.null]


@pytest.mark.parametrize("overflow, expected", [("head", [1.0, 2.0]), ("tail", [4.0, 5.0])])
def test_overflow_head_tail(overflow, expected):
    schema = build_schema(channels=1, length=2, overflow=overflow)
    values, observed, _ = steps(tensorfield([[1.0, 2.0, 3.0, 4.0, 5.0]], schema))
    assert values[0, :, 0].tolist() == expected
    assert observed.all()


def test_overflow_error_names_cap_and_length():
    schema = build_schema(channels=1, length=2, overflow="error")
    with pytest.raises(ValueError, match=r"/items/signal.*3 steps.*length=2"):
        tensorfield([[1.0, 2.0, 3.0]], schema)


def test_random_overflow_repeats_for_salt_and_keeps_tail_outside_training():
    schema = build_schema(channels=1, length=2, overflow="random")
    row = [[float(step) for step in range(1, 9)]] * 6
    first = steps(tensorfield(row, schema, salt=7))[0][:, :, 0]
    again = steps(tensorfield(row, schema, salt=7))[0][:, :, 0]
    other = steps(tensorfield(row, schema, salt=8))[0][:, :, 0]
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert (first[:, 1] - first[:, 0] == 1).all()

    validated = steps(tensorfield(row, schema, Strata.validate, salt=7))[0][:, :, 0]
    assert validated.tolist() == [[7.0, 8.0]] * 6


def encode_target(rows: list, *, horizon: int) -> tuple[TensorDict, torch.Tensor]:
    """Encode rows as targets; the mask-free schema keeps this independent of the decoder."""
    schema = build_schema(channels=1, horizon=horizon)
    pristine = projection(rows, schema, Strata.train).pristine
    field = replace(pristine, values=TENSORFIELDS["series"].prepare(pristine.values, address=Address(ADDRESS)))
    return encode(
        field, request=schema.requests[ADDRESS], address=Address(ADDRESS), strata=Strata.train, salt=0, target=True
    )


def test_target_longer_than_horizon_raises():
    with pytest.raises(ValueError, match=r"/items/signal.*3 steps.*horizon=2"):
        encode_target([[1.0, 2.0, 3.0]], horizon=2)


def test_shorter_target_pads_to_horizon_unobserved():
    content, state = encode_target([[7.0]], horizon=3)
    assert content["values"].shape[-2:] == (3, 1)
    assert content["observed"][0, 0, 0].tolist() == [True, False, False]
    assert state[0, 0, 0] == Tokens.valued


def test_observe_ignores_nan_records_and_unobserved_steps():
    schema = build_schema(channels=2)
    pristine = projection([[[1.0, 10.0], [3.0, 30.0]], [[NAN, NAN]]], schema, Strata.train).pristine
    observation = TENSORFIELDS["series"].observe(field=pristine, address=ADDRESS, schema=schema, state=None, learn=True)
    stats = observation[TensorKey.content]
    assert stats["count"].tolist() == [2, 2]
    torch.testing.assert_close(stats["mean"], torch.tensor([2.0, 20.0], dtype=torch.float64))
    assert (
        TENSORFIELDS["series"].observe(field=pristine, address=ADDRESS, schema=schema, state=None, learn=False) is None
    )


def embed(rows: list, *, channels: int = 1, datatype: pa.DataType | None = None, **options) -> torch.Tensor:
    schema = build_schema(channels=channels, **options)
    field = tensorfield(rows, schema, datatype=datatype)
    return Embedder(schema=schema, address=ADDRESS).eval().embed(field).payload


def test_block_keeps_shape_and_zeroes_unobserved_steps():
    seed_everything(SEED, workers=True)
    observed = torch.tensor([[True, True, False, False, True]] * 2)
    out = Block(width=8, dilation=2)(randn(2, 5, 8), observed=observed)
    assert out.shape == (2, 5, 8)
    assert not out[:, 2:4].any()


def test_embedder_payload_shape():
    assert embed([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]], channels=2).shape == (2, 1, 1, 16)


def test_unobserved_and_padded_values_do_not_change_embedding():
    schema = build_schema(channels=2)
    rows = [[[1.0, 2.0], [NAN, 2.0], [3.0, 4.0]], [[5.0, 6.0]]]
    field = tensorfield(rows, schema)
    embedder = Embedder(schema=schema, address=ADDRESS).eval()
    expected = embedder.embed(field).payload

    values = field.content["values"].clone()
    values[~field.content["observed"]] = 1e6
    field.content["values"] = values
    assert torch.equal(embedder.embed(field).payload, expected)


def test_embedding_does_not_depend_on_batch_width():
    schema = build_schema(channels=1)
    embedder = Embedder(schema=schema, address=ADDRESS).eval()
    short = [[1.0, 2.0, 3.0]]
    alone = embedder.embed(tensorfield(short, schema)).payload
    padded = embedder.embed(tensorfield(short + [[float(i) for i in range(40)]], schema)).payload
    torch.testing.assert_close(alone[0], padded[0], atol=1e-5, rtol=0)


def test_single_step_series_embeds():
    payload = embed([[[1.0, 2.0]]], channels=2)
    assert payload.shape == (1, 1, 1, 16) and torch.isfinite(payload).all()


def test_all_empty_batch_embeds_finite():
    payload = embed([[], None], datatype=pa.list_(pa.float64()))
    assert payload.shape == (2, 1, 1, 16) and torch.isfinite(payload).all()


def test_horizon_rejected_on_inputs():
    with pytest.raises(ValueError, match="horizon applies only to reconstruction targets"):
        rf.Model.xs(d_model=8, signal=rf.Series(channels=1, horizon=4), label=rf.Boolean(mask=True))


def test_learn_updates_per_channel_statistics():
    model = rf.Model.xs(d_model=8, signal=rf.Series(channels=2), label=rf.Boolean(mask=True))
    rows = pa.table({"signal": [[[1.0, 10.0], [3.0, 30.0]], [[5.0, 50.0]]], "label": [True, False]})
    model.encode(rows, strata=Strata.train)
    normalizer = model.nodes["/signal"].embedder.normalizer
    assert normalizer.count.tolist() == [3, 3]
    torch.testing.assert_close(normalizer.mean, torch.tensor([3.0, 30.0], dtype=torch.float64))


def test_jitter_runs_only_in_training(monkeypatch: pytest.MonkeyPatch):
    schema = build_schema(channels=1, jitter=Jitter(add=0.5))
    embedder = Embedder(schema=schema, address=ADDRESS)
    field = tensorfield([[1.0, 2.0]], schema)
    calls: list[torch.Tensor] = []

    def capture(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        calls.append(mask.clone())
        return inputs

    monkeypatch.setattr(Jitter, "apply", capture)
    embedder.train()
    embedder.embed(field)
    embedder.eval()
    embedder.embed(field)

    assert len(calls) == 1
    assert calls[0].reshape(-1).tolist() == [True, True]


def forecast_schema(*, channels: int = 1, horizon: int = 3, **options) -> Schema:
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        items=rf.Branch(length=2, signal=rf.Series(channels=channels, horizon=horizon, mask=True, **options)),
    )


class DummyModule:
    def __init__(self, schema: Schema):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(decoder=Decoder(schema=schema, address=ADDRESS))}
        self.logged: list[tuple[tuple, float]] = []

    def track(self, names: tuple, value: torch.Tensor) -> torch.Tensor:
        self.logged.append((names, float(value.detach().cpu())))
        return value

    def content_losses(self) -> list[float]:
        return [value for names, value in self.logged if names[-2:] == (Metric.loss, TensorKey.content)]


def forecast(field: TensorField, module: DummyModule, scaled: torch.Tensor) -> Prediction:
    """Prediction that classifies every target state correctly and forecasts ``scaled`` deviations."""
    decoder = module.nodes[ADDRESS].decoder
    logits = torch.full((*field.state.shape, len(Tokens)), -50.0)
    logits.scatter_(-1, field.targets[TensorKey.state].unsqueeze(-1), 50.0)
    content = decoder.normalizer.mean + decoder.normalizer.scale * scaled
    return Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: logits, TensorKey.content: content, "scaled": scaled},
            batch_size=field.state.shape[:1],
        ),
    )


def test_horizon_required_on_targets():
    with pytest.raises(ValueError, match="requires horizon"):
        rf.Model.xs(d_model=8, signal=rf.Series(channels=1, mask=True))


def test_horizon_allowed_when_ancestor_reconstructs():
    rf.Model.xs(
        d_model=8,
        items=rf.Branch(length=2, mask=rf.Mask(reconstruct=True), signal=rf.Series(channels=1, horizon=2)),
    )


def test_target_tensorfield_rejects_longer_than_horizon_and_pads_shorter():
    with pytest.raises(ValueError, match=r"/items/signal.*3 steps.*horizon=2"):
        tensorfield([[1.0, 2.0, 3.0]], forecast_schema(horizon=2))

    target = tensorfield([[7.0]], forecast_schema(horizon=3)).targets[TensorKey.content]
    assert target["values"].shape[-2:] == (3, 1)
    assert target["observed"][0, 0, 0].tolist() == [True, False, False]


def test_decoder_shapes_for_target():
    decoder = Decoder(schema=forecast_schema(channels=2, horizon=3), address=ADDRESS)
    payload = decoder.decode(randn(5, 16))
    assert payload["scaled"].shape == (5, 3, 2)
    assert payload[TensorKey.content].shape == (5, 3, 2)
    assert payload[TensorKey.state].shape == (5, len(Tokens))


def test_decoder_content_is_source_units():
    decoder = Decoder(schema=forecast_schema(channels=2, horizon=1), address=ADDRESS)
    decoder.normalizer.mean.copy_(torch.tensor([100.0, -5.0]))
    decoder.normalizer.var.copy_(torch.tensor([4.0, 1.0]))
    with torch.no_grad():
        decoder.regression.weight.zero_()
        decoder.regression.bias.copy_(torch.tensor([1.0, 3.0]))
    content = decoder.decode(randn(2, 16))[TensorKey.content]
    torch.testing.assert_close(content[0, 0], torch.tensor([102.0, -2.0], dtype=torch.float64))


def test_loss_counts_only_trainable_valued_observed_entries():
    schema = forecast_schema(horizon=3)
    field = tensorfield([[1.0, NAN, 3.0], None], schema, datatype=pa.list_(pa.float64()))
    module = DummyModule(schema)
    base = torch.zeros(*field.state.shape, 3, 1, dtype=torch.float64)
    loss(module=module, prediction=forecast(field, module, base), batch=field, strata=Strata.train)
    (expected,) = module.content_losses()
    assert expected > 0

    moved = base.clone()
    moved[0, 0, 0, 1] = 50.0  # unobserved step of a valued record
    moved[1:] = 50.0  # null record
    loss(module=module, prediction=forecast(field, module, moved), batch=field, strata=Strata.train)
    assert module.content_losses() == [expected, expected]

    moved[0, 0, 0, 0] = 50.0  # observed step
    loss(module=module, prediction=forecast(field, module, moved), batch=field, strata=Strata.train)
    assert module.content_losses()[-1] != expected


def test_loss_metrics_report_source_units():
    schema = forecast_schema(horizon=1)
    field = tensorfield([[103.0]], schema, Strata.validate)
    module = DummyModule(schema)
    decoder = module.nodes[ADDRESS].decoder
    decoder.normalizer.mean.fill_(100.0)
    decoder.normalizer.var.fill_(4.0)
    loss(
        module=module,
        prediction=forecast(field, module, torch.zeros(*field.state.shape, 1, 1, dtype=torch.float64)),
        batch=field,
        strata=Strata.validate,
    )
    scores = decoder.metrics.compute(Strata.validate)
    assert scores["mae.content"].item() == pytest.approx(3.0)
    assert scores["rmse.content"].item() == pytest.approx(3.0)


def test_large_offset_series_keeps_precision():
    rows = [[1e9 + 0.01, 1e9 + 0.02, 1e9 + 0.03]]
    values = tensorfield(rows, build_schema(channels=1))
    steps_ = values.content["values"][0, 0, 0, :, 0]
    assert steps_.dtype == torch.float64
    assert (steps_[1:] - steps_[:-1]).tolist() == pytest.approx([0.01, 0.01], abs=1e-6)

    normalizer = GlobalOnlineNormalizer(width=1)
    normalizer.learn(moments(steps_))
    assert normalizer.var.item() == pytest.approx(2 / 3 * 1e-4, rel=1e-3)

    decoder = Decoder(schema=forecast_schema(horizon=1), address=ADDRESS)
    decoder.normalizer.learn(moments(steps_))
    with torch.no_grad():
        decoder.regression.weight.zero_()
        decoder.regression.bias.zero_()
    content = decoder.decode(torch.zeros(1, 16))[TensorKey.content]
    assert content.item() == pytest.approx(1e9 + 0.02, abs=1e-6)


def test_embed_only_decoder_has_no_regression():
    model = rf.Model.xs(d_model=8, signal=rf.Series(channels=1, embed=True), label=rf.Boolean(mask=True))
    decoder = model.nodes["/signal"].decoder
    assert not hasattr(decoder, "regression")
    assert decoder.decode(randn(2, 8)).keys() == {TensorKey.state}


def test_learn_updates_decoder_normalizer_for_targets():
    model = rf.Model.xs(d_model=8, past=rf.Series(channels=1), future=rf.Series(channels=2, horizon=2, mask=True))
    rows = pa.table({"past": [[1.0, 2.0]] * 2, "future": [[[10.0, 100.0], [20.0, 200.0]]] * 2})
    model.encode(rows, strata=Strata.train)
    normalizer = model.nodes["/future"].decoder.normalizer
    assert normalizer.count.tolist() == [4, 4]
    torch.testing.assert_close(normalizer.mean, torch.tensor([15.0, 150.0], dtype=torch.float64))


def test_output_schema():
    module = DummyModule(forecast_schema())
    element = pa.field("element", pa.list_(pa.field("element", pa.float64())))
    assert output_type(module, ADDRESS) == pa.struct([pa.field("content", pa.list_(element), nullable=False)])


def test_write_outputs_horizon_steps_even_for_null_state():
    module = DummyModule(forecast_schema(channels=3, horizon=2))
    logits = torch.full((2, 2, len(Tokens)), -50.0)
    logits[0, :, Tokens.valued.value] = 50.0
    logits[1, :, Tokens.null.value] = 50.0
    content = torch.arange(24, dtype=torch.float64).reshape(2, 2, 2, 3)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict({TensorKey.state: logits, TensorKey.content: content}, batch_size=[2]),
    )
    written = write(module=module, prediction=prediction, datatype=output_type(module, ADDRESS))
    rows = written.field("content").to_pylist()
    assert len(rows) == 4
    assert all(len(row) == 2 and all(len(step) == 3 for step in row) for row in rows)
    assert rows[2] == [[12.0, 13.0, 14.0], [15.0, 16.0, 17.0]]


def test_predict_inside_branch_with_null_items_round_trips_parquet(tmp_path):
    model = rf.Model.xs(
        d_model=8,
        items=rf.Branch(length=3, future=rf.Series(channels=2, horizon=2, mask=True), x=rf.Number),
    )
    rows = pa.table({"items": [[{"future": [[1.0, 2.0]], "x": 1.0}, None], None]})
    table = model.predict(rows)
    pq.write_table(table, tmp_path / "p.parquet")
    assert pq.read_table(tmp_path / "p.parquet").equals(table)


def values_of(table: pa.Table, *path: str) -> list:
    return table.to_pylist()[0]["predictions"][path[0]][path[1]] if len(path) == 2 else None


def test_classification_with_series_input():
    model = rf.Model.xs(d_model=8, signal=rf.Series(channels=2), label=rf.Category(mask=True))
    rows = pa.table({"signal": [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0]]], "label": ["a", "b"]})
    result = model.training_step(model.encode(rows, strata=Strata.train), 0)
    assert torch.isfinite(result["loss"])
    assert values_of(model.predict(rows), "label", "inferred") is True


def test_embed_output_width():
    model = rf.Model.xs(d_model=8, signal=rf.Series(channels=1, embed=True), label=rf.Boolean(mask=True))
    table = model.predict(pa.table({"signal": [[1.0, 2.0], [3.0]], "label": [True, False]}))
    embedding = table.to_pylist()[0]["representations"][0]["embedding"]
    assert len(embedding) == 8


def test_predict_without_target_column_outputs_horizon_steps():
    model = rf.Model.xs(d_model=8, history=rf.Series(channels=1), future=rf.Series(channels=2, horizon=3, mask=True))
    table = model.predict(pa.table({"history": [[1.0, 2.0], [3.0]]}))
    for row in table.to_pylist():
        content = row["predictions"]["future"]["content"]
        assert len(content) == 3 and all(len(step) == 2 for step in content)


def test_rebuild_keeps_weights_across_length_and_n_blocks_and_resets_only_regression_on_horizon():
    model = rf.Model.xs(
        d_model=8,
        past=rf.Series(channels=1, length=8, n_blocks=2),
        future=rf.Series(channels=1, horizon=3, mask=True),
    )

    def weights(address: str) -> dict[str, torch.Tensor]:
        node = model.nodes[address]
        decoder = getattr(node, "decoder", None)
        state = {f"embedder.{name}": value for name, value in node.embedder.state_dict().items()}
        state |= {f"decoder.{name}": value for name, value in (decoder.state_dict() if decoder else {}).items()}
        return deepcopy(state)

    def unchanged(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> set[str]:
        return {
            name
            for name in before.keys() & after.keys()
            if before[name].shape == after[name].shape and torch.equal(before[name], after[name])
        }

    before = weights("/past")
    model.update(rf.where("address") == "/past", length=16)
    assert model.schema.requests["/past"].length == 16
    after = weights("/past")
    assert before.keys() == after.keys() and unchanged(before, after) == set(before)

    model.update(rf.where("address") == "/past", n_blocks=3)
    grown = weights("/past")
    assert not any("blocks.2." in name for name in before) and any("blocks.2." in name for name in grown)
    assert unchanged(before, grown) == set(before)

    first = weights("/future")
    model.update(rf.where("address") == "/future", horizon=4)
    second = weights("/future")
    assert second["decoder.regression.weight"].shape[0] == 4
    assert {name for name in first if name not in unchanged(first, second)} == {
        "decoder.regression.weight",
        "decoder.regression.bias",
    }


def test_shared_modules_do_not_reference_series():
    root = Path(rf.__file__).parent
    offenders = []
    for package in ("structs", "data", "architecture"):
        for path in sorted((root / package).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("extensions.series"):
                    offenders.append(f"{path.name}:{node.lineno} imports series")
                if isinstance(node, ast.Import) and any(
                    alias.name.endswith("extensions.series") for alias in node.names
                ):
                    offenders.append(f"{path.name}:{node.lineno} imports series")
                if isinstance(node, ast.Constant) and node.value == "series":
                    offenders.append(f"{path.name}:{node.lineno} names series")
    assert offenders == []


@pytest.mark.integration
@pytest.mark.parametrize("nested", [False, True])
def test_forecast_smoke(tmp_path, nested):
    seed_everything(SEED, workers=True)
    history, future = [1.0, 2.0, 3.0], [4.0, 5.0]
    if nested:
        model = rf.Model.xs(
            d_model=8,
            items=rf.Branch(
                length=2, history=rf.Series(channels=1), future=rf.Series(channels=1, horizon=2, mask=True)
            ),
        )
        rows = pa.table(
            {"items": [[{"history": history, "future": future}, {"history": history[:2], "future": future}]] * 4}
        )
    else:
        model = rf.Model.xs(
            d_model=8, history=rf.Series(channels=1), future=rf.Series(channels=1, horizon=2, mask=True)
        )
        rows = pa.table({"history": [history] * 4, "future": [future] * 4})
    data = rf.ArrowDataModule(model, train=rows, validate=rows, shuffle=False)
    model.fit(
        data,
        batch_size=4,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        early_stopping=False,
        rollback=False,
        enable_progress_bar=False,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=tmp_path,
    )
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.parametrize("option", [{"channels": True}, {"channels": 2.0}, {"channels": 1, "n_blocks": "2"}])
def test_series_rejects_lax_integer_sizes(option):
    with pytest.raises(pydantic.ValidationError):
        rf.Series(**option)


def named_forecast_model(*, channels: int | tuple[str, ...], horizon: int) -> rf.Model:
    return rf.Model.xs(
        d_model=8,
        history=rf.Series(channels=1),
        future=rf.Series(channels=channels, horizon=horizon, mask=True),
    )


def test_named_channel_output_type():
    model = named_forecast_model(channels=("temp", "hum"), horizon=2)
    step = output_type(model, Address("/future")).field("content").type.value_type
    assert pa.types.is_struct(step)
    assert [field.name for field in step] == ["temp", "hum"]
    assert all(field.type == pa.float64() for field in step)


def test_integer_channel_output_type_unchanged():
    model = named_forecast_model(channels=2, horizon=2)
    step = output_type(model, Address("/future")).field("content").type.value_type
    assert pa.types.is_list(step) and step.value_type == pa.float64()


def test_predict_writes_named_steps_and_round_trips_parquet(tmp_path):
    seed_everything(SEED, workers=True)
    model = named_forecast_model(channels=("temp", "hum"), horizon=3)
    table = model.predict(pa.table({"history": [[1.0, 2.0], [3.0]]}))
    for row in table.to_pylist():
        content = row["predictions"]["future"]["content"]
        assert len(content) == 3 and all(list(step) == ["temp", "hum"] for step in content)
    path = tmp_path / "out.parquet"
    pq.write_table(table, path)
    assert pq.read_table(path).equals(table)


def test_fixed_size_outer_list_of_structs_matches_list_of_lists():
    datatype = pa.list_(pa.struct([("temp", pa.float64()), ("hum", pa.float64())]), 3)
    expected, _, _ = steps(tensorfield([LISTS], build_schema(channels=2)))
    values, observed, _ = steps(tensorfield([A_ROW], build_schema(channels=2), datatype=datatype))
    torch.testing.assert_close(values, expected)
    assert observed.all()


@pytest.mark.parametrize(
    "column",
    [
        pa.array([[{"temp": 1.0, "hum": 2.0}, {"temp": 3.0, "hum": 4.0}]]),
        pa.array([{"temp": [1.0, 3.0], "hum": [2.0, 4.0]}]),
    ],
)
def test_predict_reads_struct_shapes(column):
    seed_everything(SEED, workers=True)
    model = rf.Model.xs(
        d_model=8,
        history=rf.Series(channels=("temp", "hum")),
        future=rf.Series(channels=2, horizon=3, mask=True),
    )
    table = model.predict(pa.table({"history": column}))
    content = table.to_pylist()[0]["predictions"]["future"]["content"]
    assert len(content) == 3 and all(len(step) == 2 for step in content)


@pytest.mark.parametrize(
    "column",
    [
        pa.array([[{"temp": 1.0, "hum": 2.0}, {"temp": 3.0, "hum": 4.0}]]),
        pa.array([{"temp": [1.0, 3.0], "hum": [2.0, 4.0]}]),
    ],
)
def test_predict_int_channels_writes_list_of_lists(column):
    seed_everything(SEED, workers=True)
    model = rf.Model.xs(
        d_model=8,
        history=rf.Series(channels=2),
        future=rf.Series(channels=2, horizon=3, mask=True),
    )
    table = model.predict(pa.table({"history": column}))
    content = table["predictions"].type.field("future").type.field("content")
    assert content.type == pa.list_(pa.field("element", pa.list_(pa.field("element", pa.float64()))))
