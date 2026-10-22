from types import SimpleNamespace

import pyarrow as pa
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.set import Decoder, Embedder, TensorField, loss, write
from relflow.tensorfields.extensions.set import output as output_type
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel
from tests.arrow import table as arrow_table
from tests.tensorfields.helpers import tensorize

ADDRESS = "/items/tags"


def build_schema(
    *,
    p_unavailable: float | None = None,
    threshold: float | None = None,
    mask: bool | Mask = False,
) -> Schema:
    options = {}
    if p_unavailable is not None:
        options["p_unavailable"] = p_unavailable
    if threshold is not None:
        options["threshold"] = threshold
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        items=rf.Branch(length=2, tags=rf.Set(mask=mask, **options)),
    )


def vocabulary_state(size: int = 8):
    return OnlineVocabularyModel(size=size).state


def tensorfield(
    *,
    values: list,
    schema: Schema,
    strata: Strata,
    state,
) -> TensorField:
    batch = arrow_table([{"items": [{"tags": value} for value in root]} for (root,) in values])
    projection = coalesce(batch, schema=schema, strata=strata).fields[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["set"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
        context=Context(state=state),
    )


def test_set_request_is_available_in_schema():
    schema = build_schema()
    request = schema.requests[ADDRESS]

    assert request.type == "set"
    assert "size" not in request.model_dump()
    assert request.threshold is None


def test_set_request_accepts_threshold():
    schema = build_schema(threshold=0.8)
    assert schema.requests[ADDRESS].threshold == 0.8


def test_set_tensorfield_encodes_compact_member_ids():
    schema = build_schema(p_unavailable=0.0)
    state = vocabulary_state()

    field = tensorfield(
        values=[[[["ALPHA", "BETA"], []]], [[["BETA"]]]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )

    assert torch.equal(
        field.state,
        torch.tensor(
            [
                [[Tokens.valued.value, Tokens.valued.value]],
                [[Tokens.valued.value, Tokens.padded.value]],
            ],
            dtype=torch.int64,
        ),
    )
    assert field.content["indices"].dtype == torch.int64
    assert field.content["indices"].tolist() == [
        [[[0, 1], [-1, -1]]],
        [[[1, -1], [-1, -1]]],
    ]


def test_set_nested_mask_string_is_an_ordinary_label():
    schema = build_schema(p_unavailable=0.0)
    state = vocabulary_state()

    field = tensorfield(
        values=[[[["<MASK>", "ALPHA"]]]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )

    assert state.vocab == ["<MASK>", "ALPHA"]
    assert field.state.tolist() == [[[Tokens.valued.value]]]
    assert field.content["indices"][0, 0, 0].tolist() == [0, 1]


def test_set_tensorfield_reserves_real_vocabulary_in_batch():
    schema = build_schema(p_unavailable=0.0)
    vocabulary = OnlineVocabularyModel()

    tensorfield(
        values=[[[["ALPHA", "BETA"], ["ALPHA"]]], [[["BETA"]]]],
        schema=schema,
        strata=Strata.train,
        state=vocabulary.state,
    )

    assert vocabulary.snapshot() == ["ALPHA", "BETA"]


def test_set_tensorfield_zeros_oov_content_without_changing_state():
    schema = build_schema(p_unavailable=0.0)
    state = vocabulary_state()

    tensorfield(
        values=[[[["ALPHA"]]]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )

    field = tensorfield(
        values=[[[["OMEGA"]]]],
        schema=schema,
        strata=Strata.validate,
        state=state,
    )

    assert field.state[0, 0, 0] == Tokens.valued.value
    assert field.content["indices"].shape == (*field.state.shape, 0)
    assert field.content["unavailable"][0, 0, 0] == 1


def test_set_tensorfield_simulated_unavailable_zeros_content():
    schema = build_schema(p_unavailable=1.0)
    state = vocabulary_state()

    field = tensorfield(
        values=[[[["ALPHA", "BETA"]]]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )

    assert field.content["indices"].shape[-1] == 2
    assert field.content["indices"][0, 0, 0].tolist() == [-1, -1]
    assert field.content["unavailable"][0, 0, 0] == 2


class _DummyVocab:
    def labels(self) -> pa.Array:
        return pa.array(["ALPHA", "BETA"], type=pa.large_string())


class _DummyEmbedder:
    def __init__(self):
        self.vocab = _DummyVocab()


class _DummyNode:
    def __init__(self, embedder=None, decoder: Decoder | None = None):
        self.embedder = embedder or _DummyEmbedder()
        self.decoder = decoder


class DummyModule:
    def __init__(self, schema=None, embedder=None, decoder: Decoder | None = None):
        self.nodes = {ADDRESS: _DummyNode(embedder=embedder, decoder=decoder)}
        self.schema = schema

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        return value


def test_set_write_emits_candidates_for_each_known_vocab_item():
    module = DummyModule(schema=build_schema())
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.padded.value] = 10.0
    content_logits = torch.tensor(
        [
            [[0.0, 2.0, -2.0]],
            [[1.0, -1.0, 3.0]],
        ]
    )
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: content_logits,
            },
            batch_size=[2],
        ),
    )

    datatype = output_type(module, ADDRESS)
    output = write(module=module, prediction=prediction, datatype=datatype)
    content = output.field(TensorKey.content.name).to_pylist()

    assert output.type == datatype
    assert [[candidate[TensorKey.value.name] for candidate in row] for row in content] == [
        ["ALPHA", "BETA"],
        ["ALPHA", "BETA"],
    ]
    assert content[0][1][TensorKey.probability.name] > content[0][0][TensorKey.probability.name]


def test_set_write_filters_content_when_threshold_is_configured():
    module = DummyModule(schema=build_schema(threshold=0.75))
    state_logits = torch.zeros(2, 1, len(Tokens))
    content_logits = torch.tensor(
        [
            [[0.0, 2.0, -2.0]],
            [[2.0, -2.0, 0.0]],
        ]
    )
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: content_logits,
            },
            batch_size=[2],
        ),
    )

    output = write(module=module, prediction=prediction, datatype=output_type(module, ADDRESS))
    expected_probability = torch.sigmoid(torch.tensor(2.0)).item()
    content = output.field(TensorKey.content.name).to_pylist()

    assert content == [
        [{TensorKey.value.name: "BETA", TensorKey.probability.name: expected_probability}],
        [{TensorKey.value.name: "ALPHA", TensorKey.probability.name: expected_probability}],
    ]


def test_set_empty_vocabulary_keeps_declared_schema():
    schema = build_schema()
    module = DummyModule(schema=schema, embedder=SimpleNamespace(vocab=OnlineVocabularyModel(size=8)))
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(1, 1, len(Tokens)),
                TensorKey.content: torch.zeros(1, 1, 8),
            },
            batch_size=[1],
        ),
    )

    datatype = output_type(module, ADDRESS)
    written = write(module=module, prediction=prediction, datatype=datatype)

    assert written.type == datatype
    assert written.field(TensorKey.content.name).to_pylist() == [[]]


def test_set_loss_does_not_mutate_counter():
    schema = build_schema(p_unavailable=0.0, mask=Mask(reconstruct=True))
    state = vocabulary_state()
    field = tensorfield(
        values=[[[["ALPHA", "BETA"], []]], [[["BETA"]]]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )
    embedder = Embedder(schema=schema, address=ADDRESS)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = DummyModule(schema=schema, embedder=embedder, decoder=decoder)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.state.shape, state.size),
            },
            batch_size=field.batch_size,
        ),
    )

    result = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    expected_state_counts = torch.ones(len(Tokens), dtype=torch.int64)
    assert torch.equal(embedder.counters[TensorKey.state.name].counts, expected_state_counts)
    assert torch.isfinite(result)
