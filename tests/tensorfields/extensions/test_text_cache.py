import pickle
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address, Mask
from relflow.tensorfields.base import TENSORFIELDS
from relflow.tensorfields.extensions.text import (
    ATTENTION_MASK,
    INPUT_IDS,
    CachedModel,
    Decoder,
    Embedder,
    TensorField,
    loss,
)
from relflow.tensorfields.shared.jitter import Jitter
from tests.arrow import table as arrow_table
from tests.tensorfields.extensions.test_text import FakeHFModel, FakeTokenizer
from tests.tensorfields.helpers import tensorize

ADDRESS = Address("body")
MODEL = "test-text-cache-encoder"


@pytest.fixture
def encoder(monkeypatch: pytest.MonkeyPatch) -> FakeHFModel:
    model = FakeHFModel()
    monkeypatch.setattr(
        CachedModel,
        "_models",
        {MODEL: CachedModel(key=MODEL, model=model, hidden_size=4)},
    )
    monkeypatch.setattr(CachedModel, "_tokenizers", {MODEL: FakeTokenizer()})
    return model


def schema(
    *,
    cache: bool | int = False,
    pooling: str = "mean",
    jitter: Jitter | None = None,
    mask: bool | Mask = False,
) -> Schema:
    return rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=4,
        body=rf.Text(
            model=MODEL,
            max_length=4,
            encoder_batch_size=2,
            encoder_pooling=pooling,
            cache=cache,
            jitter=jitter or Jitter(),
            mask=mask,
        ),
    )


def field(structure: Schema, texts: list[str | None]) -> TensorField:
    batch = arrow_table([{"body": text} for text in texts])
    projection = coalesce(batch, schema=structure, strata=Strata.train).fields[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["text"],
        address=ADDRESS,
        schema=structure,
        strata=Strata.train,
    )


def encode(embedder: Embedder, texts: list[str | None]) -> torch.Tensor:
    batch = field(schema(), texts)
    return embedder.encode(content=batch.content, state=batch.state)


def test_text_cache_defaults_to_disabled(encoder: FakeHFModel):
    assert rf.Text().cache is False
    embedder = Embedder(schema=schema(), address=ADDRESS)
    assert embedder.cache is None

    first = encode(embedder, ["alpha"])
    second = encode(embedder, ["alpha"])

    assert torch.equal(first, second)
    assert encoder.calls == 2


@pytest.mark.parametrize("cache", [False, True, 1, 128])
def test_text_cache_configuration_roundtrips(cache: bool | int):
    request = rf.Text(cache=cache)
    assert request.cache == cache
    assert type(request.cache) is type(cache)
    restored = Schema.model_validate(schema(cache=cache).model_dump(mode="json"))
    assert restored.requests[ADDRESS].cache == cache
    assert type(restored.requests[ADDRESS].cache) is type(cache)


@pytest.mark.parametrize("cache", [0, -1, 1.0, 1.5, "2", "true", None])
def test_text_cache_rejects_invalid_configuration(cache: object):
    with pytest.raises(ValueError):
        rf.Text(cache=cache)


@pytest.mark.parametrize("pooling", ["cls", "mean", "pooler"])
def test_text_cache_reuses_unique_rows_across_fresh_batches_in_order(
    encoder: FakeHFModel,
    pooling: str,
):
    texts = ["a", "longer", None, "a", "bbb", "longer"]
    reference = Embedder(schema=schema(pooling=pooling), address=ADDRESS)
    expected = encode(reference, texts)
    encoder.calls = 0
    encoder.batches.clear()
    structure = schema(cache=True, pooling=pooling)
    cached = Embedder(schema=structure, address=ADDRESS)

    first = encode(cached, texts)
    encoded_rows = sum(ids.shape[0] for ids, _ in encoder.batches)
    calls = encoder.calls
    second = encode(cached, texts)

    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert torch.count_nonzero(first[2]) == 0
    assert encoded_rows == 3
    assert encoder.calls == calls
    assert cached.cache is not None
    assert len(cached.cache) == 3
    tokenizer = CachedModel.get_tokenizer(MODEL)
    assert len(tokenizer.batches) == 3


def test_text_cache_distinguishes_attention_masks(encoder: FakeHFModel):
    embedder = Embedder(schema=schema(cache=True), address=ADDRESS)
    state = torch.tensor([[Tokens.valued.value]])
    content = TensorDict(
        {
            INPUT_IDS: torch.tensor([[[5, 7, 0, 0]]]),
            ATTENTION_MASK: torch.tensor([[[1, 1, 0, 0]]]),
        },
        batch_size=[1, 1],
    )

    first = embedder.encode(content=content, state=state)
    content[ATTENTION_MASK][0, 0, 1] = 0
    second = embedder.encode(content=content, state=state)
    repeated = embedder.encode(content=content, state=state)

    assert not torch.equal(first, second)
    assert torch.equal(second, repeated)
    assert encoder.calls == 2
    assert embedder.cache is not None
    assert len(embedder.cache) == 2


@pytest.mark.parametrize("pooling", ["cls", "mean", "pooler"])
def test_text_cache_preserves_left_padding_and_valued_empty_tokens(
    encoder: FakeHFModel,
    pooling: str,
):
    CachedModel.get_tokenizer(MODEL).padding_side = "left"
    content = TensorDict(
        {
            INPUT_IDS: torch.tensor([[[0, 0, 0, 5]], [[0, 0, 7, 9]], [[0, 0, 0, 0]], [[0, 0, 0, 0]]]),
            ATTENTION_MASK: torch.tensor([[[0, 0, 0, 1]], [[0, 0, 1, 1]], [[0, 0, 0, 0]], [[0, 0, 0, 0]]]),
        },
        batch_size=[4, 1],
    )
    state = torch.tensor([[Tokens.valued], [Tokens.valued], [Tokens.valued], [Tokens.null]])
    reference = Embedder(schema=schema(pooling=pooling), address=ADDRESS)
    expected = reference.encode(content=content, state=state)
    encoder.calls = 0
    encoder.batches.clear()
    cached = Embedder(schema=schema(cache=True, pooling=pooling), address=ADDRESS)

    first = cached.encode(content=content, state=state)
    calls = encoder.calls
    second = cached.encode(content=content, state=state)

    assert torch.equal(first, expected)
    assert torch.equal(second, expected)
    assert torch.count_nonzero(first[3]) == 0
    assert encoder.calls == calls
    assert all(ids.shape[-1] == 4 for ids, _ in encoder.batches)
    assert sum(ids.shape[0] for ids, _ in encoder.batches) == 3


def test_text_cache_retains_hot_entries_with_a_bounded_capacity(encoder: FakeHFModel):
    embedder = Embedder(schema=schema(cache=2), address=ADDRESS)
    encode(embedder, ["a", "bb"])
    for _ in range(3):
        encode(embedder, ["bb"])
    assert encoder.calls == 1

    for text in ["ccc", "dddd", "eeeee", "ffffff"]:
        encode(embedder, [text])
        calls = encoder.calls
        encode(embedder, ["bb"])
        assert encoder.calls == calls
        assert embedder.cache is not None
        assert len(embedder.cache) == 2
        assert len(embedder.cache.frequencies) == 2
        assert sum(len(bucket) for bucket in embedder.cache.buckets.values()) == 2


def test_text_cache_counts_duplicate_exposure_within_a_batch(encoder: FakeHFModel):
    embedder = Embedder(schema=schema(cache=2), address=ADDRESS)
    encode(embedder, ["a", "a", "a", "bb"])
    assert encoder.calls == 1
    encode(embedder, ["ccc"])
    calls = encoder.calls

    encode(embedder, ["a"])
    assert encoder.calls == calls
    encode(embedder, ["bb"])
    assert encoder.calls == calls + 1


def test_text_cache_breaks_frequency_ties_by_recency(encoder: FakeHFModel):
    embedder = Embedder(schema=schema(cache=2), address=ADDRESS)
    encode(embedder, ["a", "bb"])
    encode(embedder, ["a"])
    encode(embedder, ["bb"])
    encode(embedder, ["ccc"])
    calls = encoder.calls

    encode(embedder, ["bb"])
    assert encoder.calls == calls
    encode(embedder, ["a"])
    assert encoder.calls == calls + 1


def test_text_cache_keeps_autocast_contexts_separate(encoder: FakeHFModel):
    embedder = Embedder(schema=schema(cache=True), address=ADDRESS)
    original = encode(embedder, ["alpha"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encode(embedder, ["alpha"])
        encode(embedder, ["alpha"])

    repeated = encode(embedder, ["alpha"])

    assert torch.equal(original, repeated)
    assert encoder.calls == 2


def test_text_caches_are_owned_by_each_field(encoder: FakeHFModel):
    structure = schema(cache=True)
    first = Embedder(schema=structure, address=ADDRESS)
    second = Embedder(schema=structure, address=ADDRESS)

    encode(first, ["alpha"])
    encode(second, ["alpha"])

    assert encoder.calls == 2
    assert first.cache is not second.cache


def test_text_cache_keeps_jitter_and_projection_trainable(
    encoder: FakeHFModel,
    monkeypatch: pytest.MonkeyPatch,
):
    structure = schema(cache=True, jitter=Jitter(add=0.5))
    embedder = Embedder(schema=structure, address=ADDRESS)
    batch = field(structure, ["alpha"])
    pristine = embedder.encode(content=batch.content, state=batch.state).clone()
    perturbations = iter((1.0, 2.0))

    def perturb(self: Jitter, inputs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        result = inputs.clone()
        result[mask] += next(perturbations)
        return result

    monkeypatch.setattr(Jitter, "apply", perturb)
    embedder.train()
    first = embedder.embed(batch).payload
    second = embedder.embed(field(structure, ["alpha"])).payload
    (first.sum() + second.sum()).backward()

    assert encoder.calls == 1
    assert not torch.equal(first, second)
    assert embedder.linear[0].weight.grad is not None
    assert torch.count_nonzero(embedder.linear[0].weight.grad) > 0
    assert torch.equal(embedder.encode(content=batch.content, state=batch.state), pristine)


def test_text_cache_warmed_by_inference_supports_training_gradients(encoder: FakeHFModel):
    structure = schema(cache=True)
    embedder = Embedder(schema=structure, address=ADDRESS)
    embedder.eval()
    with torch.inference_mode():
        embedder.embed(field(structure, ["alpha"]))

    embedder.train()
    output = embedder.embed(field(structure, ["alpha"])).payload
    output.square().sum().backward()

    assert encoder.calls == 1
    assert embedder.linear[0].weight.grad is not None
    assert torch.count_nonzero(embedder.linear[0].weight.grad) > 0
    assert embedder.cache is not None
    for value in embedder.cache.values.values():
        assert value.device.type == "cpu"
        assert value.dtype == torch.float32
        assert value.shape == (embedder.hidden_size,)
        assert not value.requires_grad
        assert value.grad_fn is None


@pytest.mark.integration
@pytest.mark.parametrize("pooling", ["cls", "mean", "pooler"])
@pytest.mark.parametrize("mixed", [False, True])
def test_text_cache_matches_an_offline_frozen_bert(
    encoder: FakeHFModel,
    pooling: str,
    mixed: bool,
):
    transformers = pytest.importorskip("transformers")
    model = transformers.BertModel(
        transformers.BertConfig(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=8,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
    ).eval()
    model.requires_grad_(False)
    CachedModel._models[MODEL] = CachedModel(key=MODEL, model=model, hidden_size=8)
    calls = []
    handle = model.register_forward_pre_hook(lambda module, args: calls.append(None))
    reference = Embedder(schema=schema(pooling=pooling), address=ADDRESS)
    cached = Embedder(schema=schema(cache=True, pooling=pooling), address=ADDRESS)
    texts = ["a", "longer", None, "a", "bbb", "longer"]

    try:
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed):
            expected = encode(reference, texts)
            actual = encode(cached, texts)
            count = len(calls)
            repeated = encode(cached, texts)
    finally:
        handle.remove()

    torch.testing.assert_close(actual, expected, rtol=1e-2 if mixed else 1e-5, atol=5e-3 if mixed else 1e-6)
    assert torch.equal(actual, repeated)
    assert len(calls) == count


def test_text_cache_reuses_reconstruction_targets_without_revealing_hidden_inputs(encoder: FakeHFModel):
    structure = schema(cache=True, mask=Mask(reconstruct=True))
    embedder = Embedder(schema=structure, address=ADDRESS)
    module = SimpleNamespace(
        schema=structure,
        nodes={ADDRESS: SimpleNamespace(embedder=embedder, decoder=Decoder(schema=structure, address=ADDRESS))},
        track=lambda names, value: value,
    )

    for _ in range(2):
        batch = field(structure, ["alpha", "alpha"])
        parcel = embedder.embed(batch)
        assert torch.all(batch.state == Tokens.masked.value)
        assert torch.equal(parcel.payload, embedder.embeddings(batch.state))
        prediction = Prediction(
            address=ADDRESS,
            payload=TensorDict(
                {
                    TensorKey.state: torch.zeros(*batch.state.shape, len(Tokens)),
                    TensorKey.content: torch.zeros(*batch.state.shape, embedder.hidden_size),
                },
                batch_size=batch.batch_size,
            ),
        )
        value = loss(module=module, prediction=prediction, batch=batch, strata=Strata.train)
        assert torch.isfinite(value)
        assert encoder.calls == 1


@pytest.mark.parametrize("operation", ["pickle", "deepcopy"])
def test_text_cache_serializes_cold(encoder: FakeHFModel, operation: str):
    embedder = Embedder(schema=schema(cache=2), address=ADDRESS)
    encode(embedder, ["alpha"])
    assert embedder.cache is not None
    assert len(embedder.cache) == 1

    restored = pickle.loads(pickle.dumps(embedder.cache)) if operation == "pickle" else deepcopy(embedder.cache)

    assert len(restored) == 0
    assert len(embedder.cache) == 1
    assert all("cache" not in name for name in embedder.state_dict())
