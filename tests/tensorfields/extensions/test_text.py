import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import RaggedField, coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.text import (
    ATTENTION_MASK,
    DEFAULT_TEXT_MODEL,
    INPUT_IDS,
    CachedModel,
    Decoder,
    Embedder,
    TensorField,
    loss,
)
from relflow.tensorfields.shared.jitter import Jitter
from tests.arrow import table as arrow_table
from tests.tensorfields.helpers import tensorize

ADDRESS = "/items/body"


def build_schema(
    *,
    d_model: int = 16,
    n_heads: int = 4,
    model: str = "bert-base-uncased",
    max_length: int = 4,
    objective: str = "l2",
    encoder_pooling: str = "cls",
    encoder_batch_size: int = 2,
    tokenizer_batch_size: int | None = None,
    mask: bool | Mask = False,
    jitter: Jitter | dict[str, object] | None = None,
) -> Schema:
    options = {}
    if tokenizer_batch_size is not None:
        options["tokenizer_batch_size"] = tokenizer_batch_size
    if jitter is not None:
        options["jitter"] = jitter
    return rf.Schema.from_tree(
        d_model=d_model,
        n_layers=1,
        n_heads=n_heads,
        dropout=0.1,
        items=rf.Branch(
            length=2,
            n_heads=n_heads,
            body=rf.Text(
                model=model,
                max_length=max_length,
                encoder_batch_size=encoder_batch_size,
                encoder_pooling=encoder_pooling,
                objective=objective,
                mask=mask,
                **options,
            ),
        ),
    )


def values() -> list:
    return [
        [["alpha", "beta"]],
        [["gamma", "delta"]],
    ]


def tensorfield(*, values: list, schema: Schema, strata: Strata) -> TensorField:
    batch = arrow_table([{"items": [{"body": value} for value in root]} for (root,) in values])
    projection = coalesce(batch, schema=schema, strata=strata).fields[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["text"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
    )


class FakeTokenizer:
    pad_token_id = 0
    eos_token = "[EOS]"
    sep_token = "[SEP]"
    padding_side = "right"

    def __init__(self):
        self.batches: list[list[str]] = []

    def __call__(self, texts, *, padding, truncation, max_length, return_tensors):
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "np"
        self.batches.append(texts)

        token_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []

        for text in texts:
            used = min(max_length, max(1, min(2, len(text))))
            tokens = list(range(len(text), len(text) + used))
            padding = [0] * (max_length - used)
            token_rows.append(padding + tokens if self.padding_side == "left" else tokens + padding)
            mask_rows.append(padding + [1] * used if self.padding_side == "left" else [1] * used + padding)

        return {
            INPUT_IDS: np.asarray(token_rows, dtype=np.int64),
            ATTENTION_MASK: np.asarray(mask_rows, dtype=np.int64),
        }


class FakeHFModel:
    def __init__(self, hidden_size: int = 4):
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.device = torch.device("cpu")
        self.calls = 0
        self.batches: list[tuple[torch.Tensor, torch.Tensor]] = []

    def eval(self):
        return self

    def requires_grad_(self, flag: bool):
        return self

    def to(self, device):
        self.device = torch.device(device)
        return self

    def __call__(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        self.calls += 1
        self.batches.append((input_ids.detach().cpu().clone(), attention_mask.detach().cpu().clone()))
        input_ids = input_ids.to(dtype=torch.float32)
        attention_mask = attention_mask.to(dtype=torch.float32)
        hidden = torch.stack(
            [
                input_ids,
                input_ids + 1.0,
                attention_mask,
                input_ids * attention_mask,
            ],
            dim=-1,
        )
        return SimpleNamespace(
            last_hidden_state=hidden,
            pooler_output=hidden[:, 0],
        )


def patch_hf(monkeypatch: pytest.MonkeyPatch, *, hidden_size: int = 4) -> FakeHFModel:
    fake_model = FakeHFModel(hidden_size=hidden_size)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model: str):
            return fake_model

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model: str):
            return FakeTokenizer()

    monkeypatch.setattr(CachedModel, "_models", {})
    monkeypatch.setattr(CachedModel, "_tokenizers", {})
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModel=FakeAutoModel, AutoTokenizer=FakeAutoTokenizer),
    )
    return fake_model


def test_text_request_is_available_in_structure():
    schema = build_schema()
    request = schema.requests[ADDRESS]
    assert request.type == "text"
    assert request.model == "bert-base-uncased"
    assert request.max_length == 4
    assert request.tokenizer_batch_size == 4096


def test_text_request_uses_default_model():
    request = rf.Text()

    assert request.model == DEFAULT_TEXT_MODEL


def test_text_request_rejects_blank_model():
    with pytest.raises(ValueError, match="at least 1 character"):
        rf.Text(model="   ")


def test_text_request_strips_model():
    request = rf.Text(model="  bert-base-uncased  ")

    assert request.model == "bert-base-uncased"


@pytest.mark.parametrize("tokenizer_batch_size", [0, -1])
def test_text_request_rejects_nonpositive_tokenizer_batch_size(tokenizer_batch_size: int):
    with pytest.raises(ValueError, match="greater than 0"):
        build_schema(tokenizer_batch_size=tokenizer_batch_size)


def test_text_request_hydrates_jitter_from_mapping():
    schema = build_schema(jitter={"add": 0.2, "multiply": 0.1})

    assert schema.requests[ADDRESS].jitter == Jitter(add=0.2, multiply=0.1)


@pytest.mark.parametrize("normalize", [True, False])
def test_text_accepts_jitter_normalization_modes(normalize: bool):
    schema = build_schema(jitter=Jitter(add=0.2, normalize=normalize))

    assert schema.requests[ADDRESS].jitter == Jitter(add=0.2, normalize=normalize)


def test_text_raises_when_transformers_is_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    find_spec = importlib.util.find_spec

    def missing_transformers(name, *args, **kwargs):
        return None if name == "transformers" else find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", missing_transformers)

    with pytest.raises(ModuleNotFoundError, match="relflow\\[text\\]"):
        rf.Model.xs(body=rf.Text, d_model=8)

    assert TENSORFIELDS["text"].requires == {"transformers": "relflow[text]"}


def test_text_cached_model_reuses_same_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(CachedModel, "_models", {})
    loaded: list[str] = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model: str):
            loaded.append(model)
            return FakeHFModel()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=FakeAutoModel))

    first = CachedModel.get_model("bert-base-uncased")
    second = CachedModel.get_model("bert-base-uncased")

    assert first is second
    assert first.key == "bert-base-uncased"
    assert first.model is second.model
    assert loaded == ["bert-base-uncased"]


def test_text_cached_model_reuses_same_tokenizer(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(CachedModel, "_tokenizers", {})
    loaded: list[str] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model: str):
            loaded.append(model)
            return FakeTokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=FakeAutoTokenizer))

    first = CachedModel.get_tokenizer("bert-base-uncased")
    second = CachedModel.get_tokenizer("bert-base-uncased")

    assert first is second
    assert loaded == ["bert-base-uncased"]


def test_text_embedders_share_frozen_model_resource(monkeypatch: pytest.MonkeyPatch):
    patch_hf(monkeypatch)

    schema = build_schema()
    first = Embedder(schema=schema, address=ADDRESS)
    second = Embedder(schema=schema, address=ADDRESS)

    assert first.__dict__["_cached_model"] is second.__dict__["_cached_model"]
    assert first.__dict__["_cached_model"].model is second.__dict__["_cached_model"].model


def test_text_shared_model_moves_with_shared_device_state(monkeypatch: pytest.MonkeyPatch):
    fake_model = patch_hf(monkeypatch)

    schema = build_schema()
    first = Embedder(schema=schema, address=ADDRESS)
    second = Embedder(schema=schema, address=ADDRESS)
    cached_model = first.__dict__["_cached_model"]

    assert cached_model.module_for(torch.device("cpu")) is fake_model
    assert cached_model.device == torch.device("cpu")
    assert second.__dict__["_cached_model"].device == torch.device("cpu")
    assert fake_model.device == torch.device("cpu")


def test_text_tensorfield_tokenizes_strings(monkeypatch: pytest.MonkeyPatch):
    patch_hf(monkeypatch)

    schema = build_schema()
    field = tensorfield(
        values=values(),
        schema=schema,
        strata=Strata.train,
    )

    assert field.content[INPUT_IDS].shape == (2, 1, 2, 4)
    assert field.content[ATTENTION_MASK].shape == (2, 1, 2, 4)
    assert field.content[INPUT_IDS][0, 0, 0].tolist() == [5, 6, 0, 0]
    assert field.content[ATTENTION_MASK][0, 0, 0].tolist() == [1, 1, 0, 0]


def test_text_tensorfield_tokenizes_bounded_numpy_batches(monkeypatch: pytest.MonkeyPatch):
    patch_hf(monkeypatch)

    schema = build_schema(tokenizer_batch_size=2)
    field = tensorfield(
        values=[[["a", "bb"]], [["ccc", "dddd"]], [["eeeee", None]]],
        schema=schema,
        strata=Strata.train,
    )
    tokenizer = CachedModel.get_tokenizer("bert-base-uncased")

    assert tokenizer.batches == [["a", "bb"], ["ccc", "dddd"], ["eeeee"]]
    assert field.content[INPUT_IDS].shape == (3, 1, 2, 4)
    assert torch.count_nonzero(field.content[INPUT_IDS][2, 0, 1]) == 0


@pytest.mark.parametrize("storage", ["string", "large_string", "chunked", "dictionary", "chunked_dictionary"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("max_length", [1, 4])
@pytest.mark.parametrize("objective", [False, True])
def test_text_tokenizes_unique_strings_and_restores_sparse_coordinates(
    monkeypatch: pytest.MonkeyPatch, storage: str, padding_side: str, max_length: int, objective: bool
):
    patch_hf(monkeypatch)
    tokenizer = CachedModel.get_tokenizer("bert-base-uncased")
    tokenizer.padding_side = padding_side
    texts = ["alpha", "", "alpha", "bb", "bb"]
    values = pa.array(texts, type=pa.large_string() if storage == "large_string" else pa.string())
    if storage == "chunked":
        values = pa.chunked_array([values.slice(0, 3), values.slice(3)])
    elif storage == "dictionary":
        values = pa.DictionaryArray.from_arrays(
            pa.array([1, 0, 1, 2, 2], type=pa.int32()), pa.array(["", "alpha", "bb", "unobserved"])
        )
    elif storage == "chunked_dictionary":
        values = pa.chunked_array([pc.dictionary_encode(values.slice(0, 3)), pc.dictionary_encode(values.slice(3))])
    positions = np.array([0, 1, 3, 5, 7], dtype=np.int64)
    states = np.array(
        [
            Tokens.valued,
            Tokens.valued,
            Tokens.null,
            Tokens.valued,
            Tokens.null,
            Tokens.valued,
            Tokens.padded,
            Tokens.valued,
        ],
        dtype=np.int8,
    )
    target = RaggedField(values, pa.array(states), pa.array(positions), (4, 1, 2))
    input = (
        RaggedField(
            pa.array([], type=pa.string()),
            pa.array(np.where(states == Tokens.valued, Tokens.masked, states), type=pa.int8()),
            pa.array([], type=pa.int64()),
            target.shape,
        )
        if objective
        else target
    )
    schema = build_schema(max_length=max_length, tokenizer_batch_size=2, mask=objective)
    present = torch.from_numpy(states != Tokens.padded).reshape(target.shape)
    field = TensorField.new(
        input=input,
        target=target,
        present=present,
        trainable=present.clone() if objective else torch.zeros_like(present),
        inferred=torch.zeros_like(present),
        address=ADDRESS,
        schema=schema,
        strata=Strata.train,
        context=Context(),
    )

    reference = FakeTokenizer()
    reference.padding_side = padding_side
    expected = reference(texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="np")
    content = field.targets[TensorKey.content] if objective else field.content
    for name in (INPUT_IDS, ATTENTION_MASK):
        dense = np.zeros((8, max_length), dtype=np.int64)
        dense[positions] = expected[name]
        torch.testing.assert_close(content[name], torch.from_numpy(dense.reshape(4, 1, 2, max_length)))
    assert tokenizer.batches == [["alpha", ""], ["bb"]]
    assert torch.equal(field.state, torch.from_numpy(input.dense))
    assert torch.equal(field.present, present)
    if objective:
        assert not field.content[INPUT_IDS].any()
        assert not field.content[ATTENTION_MASK].any()


def test_text_embedder_and_decoder_shapes(monkeypatch: pytest.MonkeyPatch):
    fake_model = patch_hf(monkeypatch)

    schema = build_schema(encoder_batch_size=1)
    field = tensorfield(
        values=values(),
        schema=schema,
        strata=Strata.train,
    )

    embedder = Embedder(schema=schema, address=ADDRESS)
    parcel = embedder.embed(field)
    assert parcel.payload.shape == (2, 1, 2, 16)
    assert fake_model.calls == 4

    decoder = Decoder(schema=schema, address=ADDRESS)
    prediction = decoder(
        [parcel],
        batch_size=field.state.shape[0],
        device=parcel.payload.device,
    )
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 4)


@pytest.mark.parametrize("normalize", [True, False])
def test_text_jitters_only_training_valued_finite_embeddings(
    monkeypatch: pytest.MonkeyPatch,
    normalize: bool,
):
    patch_hf(monkeypatch)

    schema = build_schema(d_model=4, n_heads=2, jitter=Jitter(add=0.5, normalize=normalize))
    field = tensorfield(
        values=[[["alpha", None]], [["gamma", "delta"]]],
        schema=schema,
        strata=Strata.train,
    )
    embedder = Embedder(schema=schema, address=ADDRESS)
    embedder.linear = torch.nn.Identity()
    with torch.no_grad():
        embedder.embeddings.weight.zero_()

    encoded = torch.arange(field.state.numel() * embedder.hidden_size, dtype=torch.float32).reshape(
        *field.state.shape, embedder.hidden_size
    )
    encoded[0, 0, 0, 2] = torch.nan
    monkeypatch.setattr(embedder, "encode", lambda content, state: encoded.clone())

    draws = iter((torch.ones(11), torch.zeros(11)))
    monkeypatch.setattr(torch, "rand_like", lambda values: next(draws).to(values))

    embedder.train()
    training = embedder.embed(field).payload
    expected = encoded.clone()
    valued = field.state.eq(Tokens.valued.value).unsqueeze(-1).expand_as(expected)
    expected[valued & torch.isfinite(expected)] += 0.5
    expected[~valued] = 0.0

    assert torch.equal(torch.isnan(training), torch.isnan(expected))
    assert torch.allclose(torch.nan_to_num(training), torch.nan_to_num(expected))

    embedder.eval()
    evaluation = embedder.embed(field).payload
    pristine = encoded.clone()
    pristine[~valued] = 0.0
    assert torch.equal(torch.isnan(evaluation), torch.isnan(pristine))
    assert torch.allclose(torch.nan_to_num(evaluation), torch.nan_to_num(pristine))


def test_text_forward_never_reads_reconstruction_targets(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = patch_hf(monkeypatch)

    schema = build_schema(
        encoder_batch_size=2,
        mask=Mask(reconstruct=True),
    )
    field = tensorfield(
        values=values(),
        schema=schema,
        strata=Strata.train,
    )

    embedder = Embedder(schema=schema, address=ADDRESS)
    parcel = embedder.embed(field)

    assert fake_model.calls == 0
    assert torch.all(field.state == Tokens.masked.value)
    assert torch.count_nonzero(field.content[INPUT_IDS]) == 0
    assert torch.count_nonzero(field.targets[TensorKey.content][INPUT_IDS]) > 0
    assert torch.allclose(parcel.payload, embedder.embeddings(field.state))

    state_logits = torch.full((*field.targets[TensorKey.state].shape, len(Tokens)), -50.0)
    state_logits[..., Tokens.valued.value] = 50.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: torch.zeros(*field.state.shape, embedder.hidden_size),
            },
            batch_size=[2],
        ),
    )
    module = DummyModule(schema, embedder, Decoder(schema=schema, address=ADDRESS))

    output = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    assert torch.isfinite(output)
    assert fake_model.calls == 2


@pytest.mark.parametrize("encoder_pooling", ["cls", "mean", "pooler"])
def test_text_encoder_buckets_by_length_trims_batches_and_restores_order(
    monkeypatch: pytest.MonkeyPatch,
    encoder_pooling: str,
):
    fake_model = patch_hf(monkeypatch)
    schema = build_schema(encoder_pooling=encoder_pooling, max_length=5)
    field = tensorfield(
        values=[
            [["a", "abcde"]],
            [["ab", "abcd"]],
            [["abc", None]],
        ],
        schema=schema,
        strata=Strata.train,
    )

    # Five valued rows have deliberately shuffled sequence lengths, including
    # a tie. The sixth row is missing and must remain a zero embedding.
    input_ids = torch.tensor(
        [
            [10, 0, 0, 0, 0],
            [20, 21, 22, 23, 24],
            [30, 31, 0, 0, 0],
            [40, 41, 42, 43, 0],
            [50, 51, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    field.content[INPUT_IDS] = input_ids.reshape(3, 1, 2, 5)
    field.content[ATTENTION_MASK] = attention_mask.reshape(3, 1, 2, 5)

    embedder = Embedder(schema=schema, address=ADDRESS)
    encoded = embedder.encode(content=field.content, state=field.state).reshape(6, 4)

    # Stable ascending bucketing yields [1, 2], [2, 4], [5], leaving the
    # longest row in the uneven final batch. Each batch is cropped to its
    # longest member.
    assert [tuple(ids.shape) for ids, _ in fake_model.batches] == [(2, 2), (2, 4), (1, 5)]
    assert [ids[:, 0].tolist() for ids, _ in fake_model.batches] == [[10, 30], [50, 40], [20]]
    assert [mask.tolist() for _, mask in fake_model.batches] == [
        [[1, 0], [1, 1]],
        [[1, 1, 0, 0], [1, 1, 1, 1]],
        [[1, 1, 1, 1, 1]],
    ]

    first_channel = (
        torch.tensor([10.0, 22.0, 30.5, 41.5, 50.5])
        if encoder_pooling == "mean"
        else torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
    )
    expected = torch.stack(
        [first_channel, first_channel + 1.0, torch.ones(5), first_channel],
        dim=1,
    )
    expected = torch.cat([expected, torch.zeros(1, 4)])
    assert torch.equal(encoded, expected)


def test_text_encoder_keeps_full_width_for_left_padding(monkeypatch: pytest.MonkeyPatch):
    fake_model = patch_hf(monkeypatch)
    schema = build_schema(encoder_pooling="mean", max_length=5)
    field = tensorfield(
        values=values(),
        schema=schema,
        strata=Strata.train,
    )
    tokenizer = CachedModel.get_tokenizer("bert-base-uncased")
    tokenizer.padding_side = "left"
    input_ids = torch.tensor(
        [
            [0, 0, 0, 0, 10],
            [0, 20, 21, 22, 23],
            [0, 0, 0, 30, 31],
            [0, 0, 40, 41, 42],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
        ]
    )
    field.content[INPUT_IDS] = input_ids.reshape(2, 1, 2, 5)
    field.content[ATTENTION_MASK] = attention_mask.reshape(2, 1, 2, 5)

    embedder = Embedder(schema=schema, address=ADDRESS)
    encoded = embedder.encode(content=field.content, state=field.state).reshape(4, 4)

    # Prefix slicing is only valid for right padding. Retaining full width for
    # left padding prevents the content tokens at the end from being dropped.
    assert [tuple(ids.shape) for ids, _ in fake_model.batches] == [(2, 5), (2, 5)]
    first_channel = torch.tensor([10.0, 21.5, 30.5, 41.0])
    expected = torch.stack(
        [first_channel, first_channel + 1.0, torch.ones(4), first_channel],
        dim=1,
    )
    assert torch.equal(encoded, expected)


def test_text_encoder_keeps_full_width_for_valued_row_without_attended_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_model = patch_hf(monkeypatch)
    schema = build_schema()
    field = tensorfield(
        values=values(),
        schema=schema,
        strata=Strata.train,
    )
    field.content[INPUT_IDS][0, 0, 0].zero_()
    field.content[ATTENTION_MASK][0, 0, 0].zero_()

    embedder = Embedder(schema=schema, address=ADDRESS)
    encoded = embedder.encode(content=field.content, state=field.state).reshape(4, 4)

    # Empty strings can be valued while producing no attended tokens for
    # tokenizers that add no special tokens. Preserve the previous cls/pooler
    # behavior by retaining the original width for the containing microbatch.
    assert [tuple(ids.shape) for ids, _ in fake_model.batches] == [(2, 4), (2, 2)]
    assert torch.equal(fake_model.batches[0][1][0], torch.zeros(4, dtype=torch.int64))
    assert torch.equal(encoded[0], torch.tensor([0.0, 1.0, 0.0, 0.0]))


def test_text_frozen_hf_model_is_not_registered_in_state_dict(monkeypatch: pytest.MonkeyPatch):
    patch_hf(monkeypatch)

    schema = build_schema()
    embedder = Embedder(schema=schema, address=ADDRESS)

    state_keys = set(embedder.state_dict())
    assert not any("_cached_model" in key or "hf_model" in key for key in state_keys)
    assert "linear.0.weight" in state_keys


class DummyModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder | None):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}
        self.logged: list[tuple[tuple[str, ...], object]] = []

    def track(self, names: tuple[str, ...], value: torch.Tensor) -> torch.Tensor:
        self.logged.append((names, float(value.detach().cpu()) if torch.is_tensor(value) else value))
        return value


def test_text_loss_encodes_pristine_targets_without_jitter(monkeypatch: pytest.MonkeyPatch):
    patch_hf(monkeypatch)

    schema = build_schema(
        jitter=Jitter(add=0.5),
        mask=Mask(reconstruct=True),
    )
    field = tensorfield(values=values(), schema=schema, strata=Strata.train)
    embedder = Embedder(schema=schema, address=ADDRESS)
    pristine = torch.arange(field.state.numel() * embedder.hidden_size, dtype=torch.float32).reshape(
        *field.state.shape,
        embedder.hidden_size,
    )

    monkeypatch.setattr(embedder, "encode", lambda content, state: pristine.clone())

    def reject(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pytest.fail("Text reconstruction targets must bypass Jitter.apply")

    monkeypatch.setattr(Jitter, "apply", reject)

    state_logits = torch.full((*field.targets[TensorKey.state].shape, len(Tokens)), -50.0)
    state_logits[..., Tokens.valued.value] = 50.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: torch.zeros_like(pristine),
            },
            batch_size=[field.state.shape[0]],
        ),
    )
    module = DummyModule(schema, embedder, Decoder(schema=schema, address=ADDRESS))

    output = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    assert torch.isfinite(output)
    assert torch.equal(field.targets[TensorKey.embedding], pristine)


@pytest.mark.parametrize(("objective", "expected"), [("l1", 2.0), ("l2", 4.0)])
def test_text_loss_reconstructs_frozen_embedding(monkeypatch: pytest.MonkeyPatch, objective: str, expected: float):
    patch_hf(monkeypatch)

    schema = build_schema(objective=objective, mask=Mask(reconstruct=True))
    field = tensorfield(
        values=values(),
        schema=schema,
        strata=Strata.train,
    )
    embedder = Embedder(schema=schema, address=ADDRESS)
    decoder = Decoder(schema=schema, address=ADDRESS)
    targets = embedder.encode(
        content=field.targets[TensorKey.content],
        state=field.targets[TensorKey.state],
    )
    field.targets[TensorKey.embedding] = targets
    state_logits = torch.full((*field.targets[TensorKey.state].shape, len(Tokens)), -50.0)
    state_logits[..., Tokens.valued.value] = 50.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: state_logits,
                TensorKey.content: targets + 2.0,
            },
            batch_size=[2],
        ),
    )

    module = DummyModule(schema, embedder, decoder)
    output = loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    assert torch.isclose(output, torch.tensor(expected, dtype=output.dtype), atol=1e-3)


def test_text_evaluation_metrics_accumulate_across_unequal_batches(monkeypatch: pytest.MonkeyPatch):
    patch_hf(monkeypatch)

    schema = build_schema(mask=Mask(reconstruct=True))
    embedder = Embedder(schema=schema, address=ADDRESS)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = DummyModule(schema, embedder, decoder)
    errors = []
    for rows, offset in ((values(), 1.0), ([[["solo", None]]], 4.0)):
        field = tensorfield(values=rows, schema=schema, strata=Strata.validate)
        targets = embedder.encode(content=field.targets[TensorKey.content], state=field.targets[TensorKey.state])
        state_logits = torch.full((*field.targets[TensorKey.state].shape, len(Tokens)), -50.0)
        state_logits.scatter_(-1, field.targets[TensorKey.state].unsqueeze(-1), 50.0)
        prediction = Prediction(
            address=ADDRESS,
            payload=TensorDict(
                {TensorKey.state: state_logits, TensorKey.content: targets + offset}, batch_size=field.state.shape[:1]
            ),
        )
        loss(module=module, prediction=prediction, batch=field, strata=Strata.validate)
        valued = field.targets[TensorKey.state].reshape(-1).eq(Tokens.valued.value).sum().item()
        errors.extend([offset] * int(valued) * embedder.hidden_size)

    # Four valued rows at error 1 and one at error 4: the global mean is not the mean of batch means.
    metrics = decoder.metrics.compute(Strata.validate)
    assert metrics["mae.content"].item() == pytest.approx(sum(errors) / len(errors))
    assert metrics["rmse.content"].item() == pytest.approx((sum(e * e for e in errors) / len(errors)) ** 0.5)
    assert sum(errors) / len(errors) != pytest.approx((1.0 + 4.0) / 2)
