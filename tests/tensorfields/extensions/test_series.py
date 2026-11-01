from dataclasses import replace

import pyarrow as pa
import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Overflow, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.series import Block, Embedder, TensorField, encode
from relflow.tensorfields.shared.jitter import Jitter
from tests.arrow import table as arrow_table
from tests.tensorfields.helpers import tensorize

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
    observed = torch.tensor([[True, True, False, False, True]] * 2)
    out = Block(width=8, dilation=2)(torch.randn(2, 5, 8), observed)
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
