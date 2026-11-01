from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.category import (
    Decoder,
    Embedder,
    TensorField,
    loss,
    write,
)
from relflow.tensorfields.extensions.category import (
    output as output_type,
)
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyState
from tests.arrow import table as arrow_table
from tests.tensorfields.helpers import tensorize

ADDRESS = "/items/category"


def build_schema(
    *,
    topk: list[int] | None = None,
    p_unavailable: float | None = None,
    mask: bool | Mask = False,
) -> Schema:
    options = {}
    if topk is not None:
        options["topk"] = topk
    if p_unavailable is not None:
        options["p_unavailable"] = p_unavailable
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        items=rf.Branch(length=2, category=rf.Category(mask=mask, **options)),
    )


def vocabulary_state(size: int = 8):
    return OnlineVocabularyModel(size=size).state


def tensorfield(
    rows: list[list[Any]],
    *,
    schema: Schema,
    strata: Strata,
    state: VocabularyState,
) -> TensorField:
    batch = arrow_table([{"items": [{"category": value} for value in row]} for row in rows])
    projection = coalesce(batch, schema=schema, strata=strata).fields[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["category"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
        context=Context(state=state),
    )


def test_category_vocabulary_refreshes_stale_validation_snapshot():
    vocabulary = OnlineVocabularyModel(size=8)
    validation_state = vocabulary.state
    training_state = vocabulary.state

    training_state.reserve("ALPHA", learn=True)
    validation_state.reserve("ALPHA", learn=False)

    assert training_state.encode("ALPHA") == 0
    assert validation_state.encode("ALPHA") == 0
    assert len(validation_state) == 1


def test_category_vocabulary_batch_dictionary_does_not_publish_unseen_tokens():
    vocabulary = OnlineVocabularyModel(size=1)
    state, binding = vocabulary.state.batch(pa.array(["ALPHA", "BETA"]))

    assert state.encode("ALPHA") == 0
    assert state.encode("BETA") == 1
    assert binding.labels.to_pylist() == ["ALPHA", "BETA"]
    assert vocabulary.snapshot() == []


def test_category_vocabulary_reserves_nested_tokens_in_batch():
    vocabulary = OnlineVocabularyModel(size=8)
    state = vocabulary.state

    state.reserve([[["ALPHA", None, "BETA"], ["ALPHA"]]], learn=True)

    assert vocabulary.snapshot() == ["ALPHA", "BETA"]
    assert state.encode("ALPHA") == 0
    assert state.encode("BETA") == 1


def test_category_vocabulary_batch_dictionary_is_unique_per_call():
    vocabulary = OnlineVocabularyModel(size=8)
    state, binding = vocabulary.state.batch(pa.array([["ALPHA", "ALPHA"], ["BETA"]]))

    assert state.encode("ALPHA") == 0
    assert state.encode("BETA") == 1
    assert binding.labels.to_pylist() == ["ALPHA", "BETA"]
    assert vocabulary.snapshot() == []


def test_category_tensorfield_separates_state_and_content():
    schema = build_schema(p_unavailable=0.0)
    state = vocabulary_state()

    field = tensorfield(
        rows=[["ALPHA", None], ["BETA"]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )

    assert torch.equal(
        field.state,
        torch.tensor(
            [
                [[Tokens.valued.value, Tokens.null.value]],
                [[Tokens.valued.value, Tokens.padded.value]],
            ],
            dtype=torch.int64,
        ),
    )
    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [[0, 0]],
                [[1, 0]],
            ],
            dtype=torch.int64,
        ),
    )


def test_category_tensorfield_marks_oov_as_unavailable_without_changing_state():
    schema = build_schema(p_unavailable=0.0)
    state = vocabulary_state()

    tensorfield(
        rows=[["ALPHA"]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )

    field = tensorfield(
        rows=[["OMEGA"]],
        schema=schema,
        strata=Strata.validate,
        state=state,
    )

    assert torch.equal(
        field.state,
        torch.tensor([[[Tokens.valued.value]]], dtype=torch.int64),
    )
    assert torch.equal(
        field.content,
        torch.tensor([[[-1]]], dtype=torch.int64),
    )


def test_category_tensorfield_can_simulate_unavailable_during_training():
    schema = build_schema(p_unavailable=1.0)
    state = vocabulary_state()

    field = tensorfield(
        rows=[["ALPHA", None], ["BETA"]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )

    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [[-1, 0]],
                [[-1, 0]],
            ],
            dtype=torch.int64,
        ),
    )


def test_category_embedder_and_decoder_use_real_vocab_width():
    schema = build_schema(p_unavailable=0.0)
    embedder = Embedder(schema=schema, address=ADDRESS)
    decoder = Decoder(schema=schema, address=ADDRESS)

    assert embedder.embeddings[TensorKey.content.name].num_embeddings == embedder.vocab.size
    assert embedder.counters[TensorKey.content.name].size == embedder.vocab.size
    assert decoder.linears[TensorKey.content.name].out_features == embedder.vocab.size


def test_category_embedder_zeroes_unavailable_and_non_valued_content_contributions():
    schema = build_schema(p_unavailable=0.0)
    embedder = Embedder(schema=schema, address=ADDRESS)
    unavailable = -1
    field = TensorField(
        state=torch.tensor(
            [
                [
                    Tokens.valued.value,
                    Tokens.valued.value,
                    Tokens.null.value,
                    Tokens.padded.value,
                    Tokens.masked.value,
                    Tokens.other.value,
                ]
            ],
            dtype=torch.int64,
        ),
        content=torch.tensor([[0, unavailable, 0, 0, 0, 0]], dtype=torch.int64),
        present=torch.ones((1, 6), dtype=torch.bool),
        trainable=torch.zeros((1, 6), dtype=torch.bool),
        inferred=torch.zeros((1, 6), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )

    output = embedder.embed(field).payload
    expected = embedder.embeddings[TensorKey.state.name](field.state)
    expected[:, 0] += embedder.embeddings[TensorKey.content.name](field.content[:, 0])

    assert torch.allclose(output, expected)


def writer_module(labels=("ALPHA", "BETA", "GAMMA", "DELTA", "EPS")):
    vocabulary = OnlineVocabularyModel(size=8)
    vocabulary.load_snapshot(labels)
    return SimpleNamespace(
        nodes={ADDRESS: SimpleNamespace(embedder=SimpleNamespace(vocab=vocabulary))},
        schema=build_schema(topk=[2, 3, 5, 10]),
    )


def test_category_write_emits_arrow_content_and_candidates():
    module = writer_module()
    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.padded.value] = 10.0
    content_logits = torch.tensor(
        [
            [[0.1, 0.9, 0.2, 0.3, 0.4, 0.0, 0.0, 0.0]],
            [[0.1, 0.2, 0.8, 0.3, 0.4, 0.0, 0.0, 0.0]],
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
    content = output.field(TensorKey.content.name)
    topk = content.field(TensorKey.topk.name).to_pylist()

    assert output.type == datatype
    assert len(output) == 2
    assert content.field(TensorKey.value.name).to_pylist() == ["BETA", "GAMMA"]
    assert len(topk[0]) == 5
    assert len(topk[1]) == 5
    assert set(topk[0][0]) == {TensorKey.value.name, TensorKey.probability.name}


def test_category_write_ignores_logits_beyond_vocabulary_snapshot():
    module = writer_module()
    state_logits = torch.zeros(1, 1, len(Tokens))
    content_logits = torch.tensor([[[0.1, 0.2, 0.3, 0.4, 0.5, 0.0, 0.0, 100.0]]])
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: content_logits,
            },
            batch_size=[1],
        ),
    )

    output = write(module=module, prediction=prediction, datatype=output_type(module, ADDRESS))
    content = output.field(TensorKey.content.name)

    assert content.field(TensorKey.value.name).to_pylist() == ["EPS"]


def test_category_empty_vocabulary_keeps_declared_schema():
    module = writer_module(labels=())
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
    content = written.field(TensorKey.content.name)

    assert written.type == datatype
    assert content.field(TensorKey.value.name).to_pylist() == [None]
    assert content.field(TensorKey.probability.name).to_pylist() == [0.0]
    assert content.field(TensorKey.topk.name).to_pylist() == [[]]


class TrackingModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}
        self.tracked = {}

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.tracked[names] = value
        return value


def test_category_loss_does_not_mutate_counters():
    schema = build_schema(p_unavailable=0.0, mask=Mask(reconstruct=True))
    state = vocabulary_state()

    field = tensorfield(
        rows=[["ALPHA", None], ["BETA"]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )
    embedder = Embedder(schema=schema, address=ADDRESS)
    embedder.vocab.load_snapshot(state.vocab)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = TrackingModule(schema=schema, embedder=embedder, decoder=decoder)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(
                    *field.content.shape,
                    state.size,
                ),
            },
            batch_size=field.batch_size,
        ),
    )

    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    expected_state_counts = torch.ones(len(Tokens), dtype=torch.int64)
    assert torch.equal(embedder.counters[TensorKey.state.name].counts, expected_state_counts)

    expected_content_counts = torch.ones_like(embedder.counters[TensorKey.content.name].counts)
    assert torch.equal(embedder.counters[TensorKey.content.name].counts, expected_content_counts)


def test_category_loss_retains_state_supervision_for_unavailable_content():
    schema = build_schema(p_unavailable=0.0, mask=True)
    state = vocabulary_state()

    tensorfield(
        rows=[["ALPHA"]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )
    field = tensorfield(
        rows=[["OMEGA"]],
        schema=schema,
        strata=Strata.validate,
        state=state,
    )
    embedder = Embedder(schema=schema, address=ADDRESS)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = TrackingModule(schema=schema, embedder=embedder, decoder=decoder)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.content: torch.zeros(*field.content.shape, state.size),
            },
            batch_size=field.batch_size,
        ),
    )

    result = loss(module=module, prediction=prediction, batch=field, strata=Strata.validate)

    assert torch.isfinite(result)
    assert torch.allclose(
        module.tracked[(ADDRESS, Strata.validate, "vocabulary", "size")],
        torch.tensor(0.0),
    )


@pytest.mark.integration
def test_category_float_nan_labels_take_the_other_state_and_never_enter_the_vocabulary(tmp_path):
    model = rf.Model.xs(d_model=8, code=rf.Category, label=rf.Boolean(mask=True))
    rows = pa.table({"code": pa.array([1.0, float("nan"), 2.0, None, 1.0]), "label": [True, False, True, False, True]})
    data = rf.ArrowDataModule(model, train=rows, validate=rows, shuffle=False)
    model.fit(
        data,
        batch_size=5,
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

    assert rf.Category.vocabulary(model, "/code") == (1.0, 2.0)
    field = model.encode(rows, strata="predict")[rf.Address("code")]
    assert field.state.reshape(-1).tolist() == [Tokens.valued, Tokens.other, Tokens.valued, Tokens.null, Tokens.valued]
    assert field.content.reshape(-1)[1].item() == -1
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())


@pytest.mark.integration
def test_category_strip_and_cased_fold_look_alike_labels(tmp_path):
    rows = pa.table({"code": pa.array(["Foo", "foo ", " FOO", "bar"]), "label": [True, True, True, False]})

    folded = rf.Model.xs(d_model=8, code=rf.Category(strip=True, cased=False), label=rf.Boolean(mask=True))
    data = rf.ArrowDataModule(folded, train=rows, validate=rows, shuffle=False)
    folded.fit(
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
    assert rf.Category.vocabulary(folded, "/code") == ("foo", "bar")
    field = folded.encode(rows, strata="predict")[rf.Address("code")]
    assert field.content.reshape(-1).tolist() == [0, 0, 0, 1]

    exact = rf.Model.xs(d_model=8, code=rf.Category, label=rf.Boolean(mask=True))
    data = rf.ArrowDataModule(exact, train=rows, validate=rows, shuffle=False)
    exact.fit(
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
    assert len(rf.Category.vocabulary(exact, "/code")) == 4


def test_category_folding_options_round_trip_through_the_schema():
    schema = rf.Schema.from_tree(d_model=8, n_heads=2, n_layers=1, code=rf.Category(strip=True, cased=False))
    dumped = schema.model_dump(mode="json")
    restored = rf.Schema.model_validate(dumped).requests[rf.Address("code")]
    assert restored.strip is True and restored.cased is False
    assert rf.Category().strip is False and rf.Category().cased is True
