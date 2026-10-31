import math
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pydantic
import pytest
import torch
import torch.nn.functional as F
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.group import Decoder, Embedder, TensorField, bind, loss, output, write
from relflow.tensorfields.shared.resize import Resize
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyBatch, VocabularyState
from tests.arrow import table as arrow_table
from tests.tensorfields.helpers import tensorize

ADDRESS = "/items/group"


def build_schema(*, n_groups: int = 4, p_unavailable: float = 0.0, mask: bool | rf.Mask = False) -> Schema:
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        items=rf.Branch(
            length=2,
            group=rf.Group(n_groups=n_groups, p_unavailable=p_unavailable, mask=mask),
        ),
    )


def build_embedder(schema: Schema) -> Embedder:
    embedder = Embedder(schema=schema, address=ADDRESS)
    resize = Resize()
    embedder.resize(8, resize)
    resize.commit()
    return embedder


def tensorfield(
    rows: list[list[Any]],
    *,
    schema: Schema,
    strata: Strata,
    state: VocabularyState,
) -> TensorField:
    batch = arrow_table([{"items": [{"group": value} for value in row]} for row in rows])
    projection = coalesce(batch, schema=schema, strata=strata).fields[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["group"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
        context=Context(state=state),
    )


def input_field(state: torch.Tensor, content: torch.Tensor) -> TensorField:
    return TensorField(
        state=state,
        content=content,
        present=torch.ones_like(state, dtype=torch.bool),
        trainable=torch.zeros_like(state, dtype=torch.bool),
        inferred=torch.zeros_like(state, dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=state.shape[0],
    )


@pytest.mark.parametrize("count", [2, 4, 8])
def test_group_count_is_a_fixed_positive_integer(count):
    request = rf.Group(n_groups=count)

    assert request.n_groups == count
    assert request.model_dump()["n_groups"] == count
    assert rf.Group.model_validate(request.model_dump()).n_groups == count


@pytest.mark.parametrize("count", [0, -1, None, (2, 4), [2, 4], True])
def test_group_count_rejects_nonpositive_or_noninteger_values(count):
    with pytest.raises(pydantic.ValidationError, match="n_groups"):
        rf.Group(n_groups=count)


def test_group_requires_a_count_and_rejects_adaptive_options():
    with pytest.raises(pydantic.ValidationError, match="n_groups"):
        rf.Group()
    for option in ("bounds", "revive_temperature", "ema_decay"):
        with pytest.raises(pydantic.ValidationError) as error:
            rf.Group(n_groups=4, **{option: 1})
        assert any(item["loc"] == (option,) for item in error.value.errors())
        assert option not in rf.Group.model_json_schema()["properties"]


def test_group_tensorfield_routes_oov_to_sentinel_at_validate():
    schema = build_schema()
    state = OnlineVocabularyModel(size=8).state
    tensorfield([["ALPHA"]], schema=schema, strata=Strata.train, state=state)

    field = tensorfield([["OMEGA"]], schema=schema, strata=Strata.validate, state=state)

    assert field.state.tolist() == [[[Tokens.valued.value]]]
    assert field.content.tolist() == [[[-1]]]


def test_group_tensorfield_simulates_unavailable_during_training():
    schema = build_schema(p_unavailable=1.0)
    state = OnlineVocabularyModel(size=8).state

    field = tensorfield([["ALPHA", None], ["BETA"]], schema=schema, strata=Strata.train, state=state)

    assert field.content.tolist() == [[[-1, 0]], [[-1, 0]]]


def test_group_embedder_and_decoder_use_scalar_angles_and_fixed_group_count():
    schema = build_schema(n_groups=4)
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)

    assert embedder.size == 4
    assert embedder.embeddings["group"].weight.shape == (9, 1)
    assert embedder.embeddings["content"].weight.shape == (16, 4)
    assert decoder.linears["group"].weight.shape == (1, 16)
    assert decoder.decode(torch.randn(3, 16))[TensorKey.group].shape == (3, 1)


def test_group_embedder_routes_angles_to_fixed_groups():
    embedder = build_embedder(build_schema())
    embedder.eval()
    with torch.no_grad():
        embedder.embeddings["group"].weight[:4, 0] = torch.tensor([0.0, math.pi / 2, math.pi, 3 * math.pi / 2])
    state = torch.full((4, 1), Tokens.valued.value, dtype=torch.int64)
    field = input_field(state, torch.arange(4).reshape(4, 1))

    actual = embedder.embed(field).payload
    expected = embedder.embeddings["state"](state)
    expected += embedder.embeddings["content"].weight.T.reshape(4, 1, -1)

    torch.testing.assert_close(actual, expected)


def test_group_embedder_routes_angles_across_the_circle_boundary():
    embedder = build_embedder(build_schema())
    embedder.eval()
    with torch.no_grad():
        embedder.embeddings["group"].weight[:4, 0] = torch.tensor([-0.1, math.tau - 0.1, 0.1, math.tau + 0.1])
    state = torch.full((4, 1), Tokens.valued.value, dtype=torch.int64)
    field = input_field(state, torch.arange(4).reshape(4, 1))

    actual = embedder.embed(field).payload
    expected = embedder.embeddings["state"](state) + embedder.embeddings["content"].weight[:, 0]

    torch.testing.assert_close(actual, expected)


def test_group_embedder_training_forward_is_deterministic():
    embedder = build_embedder(build_schema())
    embedder.train()
    field = input_field(torch.tensor([[Tokens.valued.value]]), torch.tensor([[3]]))
    with torch.no_grad():
        embedder.embeddings["group"].weight[3, 0] = 0.3

    torch.manual_seed(1)
    first = embedder.embed(field).payload
    torch.manual_seed(2)
    second = embedder.embed(field).payload

    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_group_embedder_straight_through_routes_gradient_to_assignment():
    embedder = build_embedder(build_schema())
    embedder.train()
    with torch.no_grad():
        embedder.embeddings["group"].weight[0, 0] = 0.2
        embedder.embeddings["content"].weight.zero_()
        embedder.embeddings["content"].weight[0] = torch.tensor([0.0, 1.0, 3.0, 7.0])
    field = input_field(torch.tensor([[Tokens.valued.value]]), torch.tensor([[0]]))

    embedder.embed(field).payload[..., 0].sum().backward()

    gradient = embedder.embeddings["group"].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient[0].abs().sum() > 0


def test_group_assignments_learn_from_downstream_supervision():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        model = rf.Model.xs(
            d_model=8,
            batch_size=4,
            value=rf.Group(n_groups=3, p_unavailable=0.0),
            target=rf.Number(mask=True),
        )
        model.track = lambda names, value, **kwargs: value
        inputs = model.encode(pa.table({"value": ["a", "b", "a", "b"], "target": [1.0, 2.0, 3.0, 4.0]}), strata="train")
        assert not inputs["/value"].trainable.any()
        output = model.training_step(inputs, batch_idx=0)
        assert output is not None and torch.isfinite(output["loss"])
        output["loss"].backward()

        gradient = model.nodes["/value"].embedder.embeddings["group"].weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient[:2].count_nonzero() > 0
        assert not gradient[2:].any()


def test_group_embedder_routes_oov_to_the_sentinel_row():
    embedder = build_embedder(build_schema())
    embedder.eval()
    with torch.no_grad():
        embedder.embeddings["group"].weight[embedder.capacity, 0] = math.pi
    state = torch.tensor([[Tokens.valued.value]])
    field = input_field(state, torch.tensor([[-1]]))

    actual = embedder.embed(field).payload
    expected = embedder.embeddings["state"](state) + embedder.embeddings["content"].weight[:, 2]

    torch.testing.assert_close(actual, expected)


def test_group_embedder_ignores_content_for_nonvalued_states():
    embedder = build_embedder(build_schema())
    embedder.eval()
    state = torch.tensor([[Tokens.padded.value], [Tokens.null.value], [Tokens.masked.value]])
    field = input_field(state, torch.tensor([[2], [2], [2]]))

    actual = embedder.embed(field).payload

    torch.testing.assert_close(actual, embedder.embeddings["state"](state))


def test_group_embedder_preserves_repeated_branch_geometry():
    embedder = build_embedder(build_schema())
    state = torch.full((2, 1, 2), Tokens.valued.value, dtype=torch.int64)
    field = input_field(state, torch.tensor([[[0, 1]], [[2, 3]]]))

    actual = embedder.embed(field).payload

    assert actual.shape == (2, 1, 2, 16)
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_group_embedder_uses_the_input_device(device):
    embedder = build_embedder(build_schema()).to(device)
    field = input_field(torch.tensor([[Tokens.valued.value]]), torch.tensor([[0]])).to(device)

    actual = embedder.embed(field).payload

    assert actual.device.type == device
    assert torch.isfinite(actual).all()


def test_group_bind_rejects_worker_ids_outside_the_batch():
    model = rf.Model.xs(d_model=8, batch_size=2, group=rf.Group(n_groups=4, p_unavailable=0.0))
    address = rf.Address("group")
    field = input_field(torch.tensor([Tokens.valued.value]), torch.tensor([9]))
    binding = VocabularyBatch(labels=pa.array(["ALPHA", "BETA"]), size=2)

    with pytest.raises(ValueError, match=r"outside \[-1, 2\)"):
        bind(model, field, None, binding, address=address, strata=Strata.validate, resize=Resize())


def test_group_float_nan_labels_take_the_other_state(tmp_path):
    model = rf.Model.xs(d_model=8, label=rf.Group(n_groups=2, mask=True), signal=rf.Number)
    rows = pa.table({"label": [1.0, 2.0, float("nan"), 0.0], "signal": [1.0, 2.0, 3.0, 4.0]})
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

    assert rf.Group.vocabulary(model, "/label") == (1.0, 2.0, 0.0)
    expected = [Tokens.valued.value, Tokens.valued.value, Tokens.other.value, Tokens.valued.value]
    targets = model.encode(rows, strata="train")[rf.Address("label")].targets
    assert targets[TensorKey.state].flatten().tolist() == expected
    assert targets[TensorKey.content].flatten()[2] < 0
    embedded = rf.Model.xs(d_model=8, label=rf.Group(n_groups=2), signal=rf.Number)
    for strata in ("train", "predict"):
        field = embedded.encode(rows, strata=strata)[rf.Address("label")]
        assert field.state.flatten().tolist() == expected
        assert field.content.flatten()[2] < 0
    assert rf.Group.vocabulary(embedded, "/label") == (1.0, 2.0, 0.0)
    assert model.predict(rows).num_rows == 4


class Vocabulary:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    def labels(self) -> pa.Array:
        return pa.array(self.tokens, type=pa.large_string())


def write_module(tokens: list[str], angles: list[float], count: int):
    embedder = SimpleNamespace(
        vocab=Vocabulary(tokens),
        capacity=len(angles) - 1,
        size=count,
        embeddings={"group": SimpleNamespace(weight=torch.tensor(angles).reshape(-1, 1))},
    )
    return SimpleNamespace(nodes={ADDRESS: SimpleNamespace(embedder=embedder)})


def group_prediction(angles: list[float], *, dtype: torch.dtype = torch.float32) -> Prediction:
    values = torch.tensor(angles, dtype=dtype).reshape(len(angles), 1, 1)
    return Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: torch.zeros(len(angles), 1, len(Tokens)), TensorKey.group: values},
            batch_size=[len(angles)],
        ),
    )


def test_group_write_emits_arrow_group_and_content_payloads():
    module = write_module(
        ["ALPHA", "BETA", "GAMMA", "DELTA"],
        [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 0.0],
        count=4,
    )
    datatype = output(module, ADDRESS)

    result = write(module=module, prediction=group_prediction([math.pi / 2, math.pi]), datatype=datatype)

    assert result.type == datatype
    assert result.type.names == ["group", "content"]
    groups = result.field("group")
    assert groups.field("value").to_pylist() == [1, 2]
    expected = 1 / (1 + 2 * math.exp(-math.pi / 2) + math.exp(-math.pi))
    assert groups.field("probability").to_pylist() == pytest.approx([expected, expected])
    content = result.field("content")
    assert content.field("value").to_pylist() == ["BETA", "GAMMA"]
    assert content.field("probability").to_pylist() == pytest.approx([expected, expected])


def test_group_write_does_not_predict_the_unavailable_row():
    module = write_module(["A", "B"], [2 * math.pi / 3, 4 * math.pi / 3, 0.0, 0.0, 0.0], count=3)

    result = write(module=module, prediction=group_prediction([0.0]), datatype=output(module, ADDRESS))

    content = result.field("content")
    assert content.field("value").to_pylist()[0] in ("A", "B")
    assert content.field("probability").to_pylist() == pytest.approx([0.5])


def test_group_write_returns_none_labels_when_vocabulary_is_empty():
    module = write_module([], [0.0], count=4)

    result = write(module=module, prediction=group_prediction([0.0]), datatype=output(module, ADDRESS))

    content = result.field("content")
    assert content.field("value").to_pylist() == [None]
    assert content.field("probability").to_pylist() == [0.0]


@pytest.mark.parametrize("count", [1, 2, 3, 7])
def test_group_write_uses_the_configured_number_of_centroids(count):
    angles = [math.tau * index / count for index in range(count)]
    tokens = [f"label-{index}" for index in range(count)]
    module = write_module(tokens, [*angles, 0.0], count=count)

    result = write(module=module, prediction=group_prediction(angles), datatype=output(module, ADDRESS))

    assert result.field("group").field("value").to_pylist() == list(range(count))
    assert result.field("content").field("value").to_pylist() == tokens


def test_group_write_wraps_angles_before_scoring():
    module = write_module(["A", "B"], [0.0, math.pi, 0.0], count=4)
    prediction = group_prediction([-0.1, math.tau - 0.1, 0.1, math.tau + 0.1])

    result = write(module=module, prediction=prediction, datatype=output(module, ADDRESS))

    groups = result.field("group")
    assert groups.field("value").to_pylist() == [0, 0, 0, 0]
    probabilities = groups.field("probability").to_pylist()
    assert probabilities[0] == pytest.approx(probabilities[1])
    assert probabilities[2] == pytest.approx(probabilities[3])
    assert result.field("content").field("value").to_pylist() == ["A"] * 4


def test_group_write_distinguishes_known_angles_within_one_group():
    module = write_module(["A", "B"], [0.0, 0.3, 0.0], count=2)

    result = write(module=module, prediction=group_prediction([0.28]), datatype=output(module, ADDRESS))

    assert result.field("group").field("value").to_pylist() == [0]
    assert result.field("content").field("value").to_pylist() == ["B"]


def test_group_write_flattens_repeated_coordinates_in_row_order():
    module = write_module(
        ["A", "B", "C", "D"],
        [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 0.0],
        count=4,
    )
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(2, 2, len(Tokens)),
                TensorKey.group: torch.tensor([[[0.0], [math.pi]], [[math.pi / 2], [3 * math.pi / 2]]]),
            },
            batch_size=[2],
        ),
    )

    result = write(module=module, prediction=prediction, datatype=output(module, ADDRESS))

    assert result.field("group").field("value").to_pylist() == [0, 2, 1, 3]
    assert result.field("content").field("value").to_pylist() == ["A", "C", "B", "D"]


def test_group_write_preserves_assignment_precision():
    module = write_module(["A", "B"], [0.0, 2.0, 0.0], count=2)
    module.nodes[ADDRESS].embedder.embeddings["group"].weight = torch.tensor([[0.0], [2.0], [0.0]], dtype=torch.float64)

    result = write(
        module=module, prediction=group_prediction([1.0 + 1e-8], dtype=torch.float64), datatype=output(module, ADDRESS)
    )

    assert result.field("content").field("value").to_pylist() == ["B"]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_group_write_supports_low_precision_angles(dtype):
    module = write_module(["A", "B"], [0.0, math.pi, 0.0], count=4)
    embedding = module.nodes[ADDRESS].embedder.embeddings["group"]
    embedding.weight = embedding.weight.to(dtype=dtype)

    result = write(module=module, prediction=group_prediction([math.pi], dtype=dtype), datatype=output(module, ADDRESS))

    assert result.field("group").field("value").to_pylist() == [2]
    assert result.field("content").field("value").to_pylist() == ["B"]
    assert math.isfinite(result.field("content").field("probability").to_pylist()[0])


def test_group_write_supports_empty_predictions():
    module = write_module(["A"], [0.0, 0.0], count=3)
    datatype = output(module, ADDRESS)

    result = write(module=module, prediction=group_prediction([]), datatype=datatype)

    assert result.type == datatype
    assert len(result) == 0


def test_group_write_rejects_non_scalar_predictions():
    module = write_module(["A"], [0.0, 0.0], count=4)
    prediction = group_prediction([0.0])
    prediction.payload[TensorKey.group] = torch.zeros(1, 1, 4)

    with pytest.raises(ValueError, match="one angle per coordinate"):
        write(module=module, prediction=prediction, datatype=output(module, ADDRESS))


def test_group_write_rejects_vocabulary_beyond_assignment_capacity():
    module = write_module(["A", "B"], [0.0, math.pi], count=2)

    with pytest.raises(ValueError, match="2 values but assignment capacity is 1"):
        write(module=module, prediction=group_prediction([0.0]), datatype=output(module, ADDRESS))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_group_write_uses_the_prediction_device(device):
    module = write_module(["A", "B"], [0.0, math.pi, 0.0], count=4)
    embedding = module.nodes[ADDRESS].embedder.embeddings["group"]
    embedding.weight = embedding.weight.to(device)
    prediction = group_prediction([math.pi]).to(device)

    result = write(module=module, prediction=prediction, datatype=output(module, ADDRESS))

    assert result.field("group").field("value").to_pylist() == [2]
    assert result.field("content").field("value").to_pylist() == ["B"]


@pytest.mark.parametrize("known", [False, True], ids=["empty-vocabulary", "known-labels"])
def test_group_predict_writes_scalar_angles(known):
    model = rf.Model.xs(d_model=8, label=rf.Group(n_groups=4, mask=True), signal=rf.Number)
    if known:
        rf.Group.assign(model, "/label", "A", 0)
        rf.Group.assign(model, "/label", "B", 2)
    node = model.nodes["/label"]
    with torch.no_grad():
        node.decoder.linears["group"].weight.zero_()

    result = model.predict(pa.table({"signal": [1.0]}))

    label = result["predictions"][0].as_py()["label"]
    expected_group = 1 / (1 + 2 * math.exp(-math.pi / 2) + math.exp(-math.pi))
    assert label["group"]["value"] == 0
    assert label["group"]["probability"] == pytest.approx(expected_group)
    assert label["content"]["value"] == ("A" if known else None)
    expected_content = 1 / (1 + math.exp(-math.pi)) if known else 0
    assert label["content"]["probability"] == pytest.approx(expected_content)


class TrackingModule:
    def __init__(self, embedder: Embedder):
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder)}
        self.tracked: dict[tuple, torch.Tensor] = {}

    def track(self, names: tuple, value: torch.Tensor) -> torch.Tensor:
        self.tracked[names] = value
        return value


@pytest.mark.parametrize("strata", [Strata.train, Strata.validate, Strata.test])
@pytest.mark.parametrize(
    "trainable",
    [
        pytest.param([True, True, False], id="some"),
        pytest.param([False, False, False], id="none"),
        pytest.param([], id="empty"),
    ],
)
@pytest.mark.parametrize("nonfinite", [False, True], ids=["finite", "untrainable-nan"])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_group_state_loss_respects_trainable_coordinates(trainable, strata, nonfinite, device):
    embedder = build_embedder(build_schema())
    module = TrackingModule(embedder)
    state = torch.tensor(
        [Tokens.valued.value, Tokens.null.value, Tokens.padded.value][: len(trainable)], dtype=torch.long, device=device
    ).reshape(-1, 1)
    targets = TensorDict({TensorKey.state: state, TensorKey.content: torch.zeros_like(state)}, batch_size=state.shape)
    field = input_field(state, torch.zeros_like(state))
    field.targets = targets
    mask = torch.tensor(trainable, dtype=torch.bool, device=device)
    field.trainable = mask.reshape(-1, 1)
    state_logits = torch.randn(len(trainable), 1, len(Tokens), device=device)
    if nonfinite:
        state_logits[~mask] = float("nan")
    state_logits.requires_grad_()
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.group: torch.zeros(len(trainable), 1, 1, device=device)},
            batch_size=[len(trainable)],
        ),
    )

    objective = loss(module, prediction, field, strata)

    assert torch.isfinite(objective)
    assert objective.requires_grad
    state_loss = module.tracked[(ADDRESS, strata, Metric.loss, TensorKey.state)]
    accuracy = (ADDRESS, strata, Metric.accuracy, TensorKey.state)
    if any(trainable):
        selected_logits = state_logits.reshape(-1, len(Tokens))[mask]
        selected_targets = state.reshape(-1)[mask]
        expected = F.cross_entropy(selected_logits, selected_targets)
        torch.testing.assert_close(state_loss, expected)
        expected_accuracy = selected_logits.argmax(dim=1).eq(selected_targets).float().mean()
        torch.testing.assert_close(module.tracked[accuracy], expected_accuracy)
        expected_gradient = torch.autograd.grad(expected, state_logits)[0]
    else:
        assert objective.item() == 0.0
        assert state_loss.item() == 0.0
        assert accuracy not in module.tracked
        expected_gradient = torch.zeros_like(state_logits)
    objective.backward()
    assert state_logits.grad is not None
    torch.testing.assert_close(state_logits.grad, expected_gradient)


def test_group_state_loss_logs_zero_without_accuracy(monkeypatch: pytest.MonkeyPatch):
    model = rf.Model.xs(
        d_model=8,
        label=rf.Group(n_groups=4, mask=rf.Mask(query="selected", dropout=False, reconstruct=True)),
    )
    field = model.encode(pa.table({"label": ["a", "b"], "selected": [False, False]}), strata="train")["/label"]
    assert not field.trainable.any()
    state_logits = torch.randn(2, 1, len(Tokens), requires_grad=True)
    prediction = Prediction(
        address="/label",
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.group: torch.zeros(2, 1, 1)},
            batch_size=[2],
        ),
    )
    entries = []
    monkeypatch.setattr(model, "log", lambda **options: entries.append(options))

    objective = loss(model, prediction, field, Strata.train)

    logged = {entry["name"]: entry["value"] for entry in entries}
    assert objective.item() == 0.0
    assert logged[".label/train.loss.state"].item() == 0.0
    assert ".label/train.accuracy.state" not in logged
    objective.backward()
    assert state_logits.grad is not None
    torch.testing.assert_close(state_logits.grad, torch.zeros_like(state_logits))
