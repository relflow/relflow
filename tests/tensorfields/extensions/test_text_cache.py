"""Automatic Text IDs preserve predictions, precision, gradients, and reuse."""

import pickle
from copy import deepcopy

import polars as pl
import pytest
import torch

import relflow as rf
from relflow.tensorfields.extensions.text import IDENTIFIERS, INPUT_IDS, CachedModel
from tests.tensorfields.extensions.test_text import patch_hf

MODEL = "test-automatic-text"
pytestmark = pytest.mark.integration


@pytest.fixture
def encoder(monkeypatch):
    return patch_hf(monkeypatch)


def build(cache=True, **options):
    return rf.Model.xs(attention=None, body=rf.Text(model=MODEL, max_length=4, cache=cache, **options))


def field(model, texts, strata="validate"):
    return model.encode(pl.DataFrame({"body": texts}, schema={"body": pl.String}), strata=strata)["/body"]


def encoded(model, texts):
    batch = field(model, texts)
    return model.nodes["/body"].embedder.encode(content=batch.content, state=batch.state)


def test_default_is_compact_automatic_reuse(encoder):
    model = build()
    assert rf.Text().cache is True
    batch = field(model, ["alpha", "bb", None, "alpha"])
    assert list(batch.content.keys()) == [IDENTIFIERS]
    assert batch.content[IDENTIFIERS].shape == (4, 1, 1)
    first = encoded(model, ["alpha", "bb", None, "alpha"])
    second = encoded(model, ["bb", "alpha", None, "bb"])
    torch.testing.assert_close(first[0], second[1])
    assert not first[2].any()
    assert encoder.calls == 1
    assert CachedModel.get_tokenizer(MODEL).batches == [["alpha", "bb"]]


def test_disabled_cache_retokenizes_and_reencodes(encoder):
    model = build(False)
    batch = field(model, ["alpha"])
    assert INPUT_IDS in batch.content.keys()
    assert model.nodes["/body"].embedder.store is None
    encoded(model, ["alpha"])
    encoded(model, ["alpha"])
    assert encoder.calls == 2
    assert len(CachedModel.get_tokenizer(MODEL).batches) == 3


@pytest.mark.parametrize("cache", [False, True, 1, 128])
def test_cache_options_roundtrip(encoder, cache):
    model = build(cache, memory=2, resident=0)
    schema = rf.Schema.model_validate(model.schema.model_dump(mode="json"))
    request = schema.requests[rf.Address("body")]
    assert request.cache == cache and type(request.cache) is type(cache)
    assert request.memory == 2 and request.resident == 0


@pytest.mark.parametrize("cache", [0, -1, 1.0, 1.5, "2", "true", None])
def test_invalid_cache_options(cache):
    with pytest.raises(ValueError):
        rf.Text(cache=cache)


@pytest.mark.parametrize("pooling", ["cls", "mean", "pooler"])
def test_cached_and_direct_outputs_agree(encoder, pooling):
    texts = ["a", "longer", None, "a", "bbb", "longer"]
    reference = encoded(build(False, encoder_pooling=pooling), texts)
    model = build(encoder_pooling=pooling)
    actual = encoded(model, texts)
    before = encoder.calls
    torch.testing.assert_close(actual, reference)
    assert torch.equal(actual, encoded(model, texts))
    assert encoder.calls == before


def test_equal_token_rows_share_a_stable_id(encoder):
    model = build()
    batch = field(model, ["a", "b", "a"])
    assert batch.content[IDENTIFIERS].unique().numel() == 1
    encoded(model, ["a", "b", "a"])
    assert encoder.calls == 1
    assert model.nodes["/body"].embedder.store.stats()["objects"] == 1


def test_fields_share_frozen_vectors_but_keep_their_projections(encoder):
    model = rf.Model.xs(body=rf.Text(model=MODEL, max_length=4), title=rf.Text(model=MODEL, max_length=4))
    batch = model.encode(pl.DataFrame({"body": ["shared"], "title": ["shared"]}))
    for address in ("/body", "/title"):
        model.nodes[address].embedder.encode(content=batch[address].content, state=batch[address].state)
    assert encoder.calls == 1
    assert model.nodes["/body"].embedder.linear is not model.nodes["/title"].embedder.linear


def test_precision_namespaces_do_not_alias(encoder):
    model = build()
    first = encoded(model, ["alpha"])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        encoded(model, ["alpha"])
        encoded(model, ["alpha"])
    assert torch.equal(first, encoded(model, ["alpha"]))
    assert encoder.calls == 2


@pytest.mark.parametrize("operation", [deepcopy, pickle.loads])
def test_context_copy_reuses_ids_without_copying_device_state(encoder, operation):
    model = build(resident=1)
    encoded(model, ["alpha"])
    store = model.nodes["/body"].embedder.store
    copy = operation(pickle.dumps(store)) if operation is pickle.loads else operation(store)
    assert not copy.views
    assert copy.owner is store.owner
    assert copy.client.endpoint.identity == store.client.endpoint.identity
    assert not any("store" in name for name in model.state_dict())


def test_inference_warmup_keeps_training_projection_gradients(encoder):
    model = build(resident=1)
    embedder = model.nodes["/body"].embedder
    with torch.inference_mode():
        encoded(model, ["alpha"])
    batch = field(model, ["alpha"])
    embedder.train()
    embedder.embed(batch).payload.square().sum().backward()
    assert embedder.linear[0].weight.grad is not None
    assert torch.count_nonzero(embedder.linear[0].weight.grad)
    assert encoder.calls == 1


def test_warm_has_no_required_directory_and_counts_only_new_vectors(encoder):
    model = build()
    embedder = model.nodes["/body"].embedder
    assert embedder.warm(["a", "bbb", None, "a"]) == 2
    assert embedder.warm(["bbb", "ccccc"]) == 1
    before = encoder.calls
    encoded(model, ["a", "bbb", "ccccc"])
    assert encoder.calls == before


def test_empty_collection_and_hidden_targets_keep_geometry(encoder):
    model = rf.Model.xs(events=rf.Branch(length=None, body=rf.Text(model=MODEL, max_length=4, mask=True)))
    source = pl.DataFrame({"events": [[{"body": "a"}, None, {"body": ""}], [], None]})
    batch = model.encode(source, strata="train")["/events/body"]
    embedder = model.nodes["/events/body"].embedder
    assert not batch.content[IDENTIFIERS].any()
    targets = embedder.encode(content=batch.targets["content"], state=batch.targets["state"])
    assert targets.shape[:-1] == batch.state.shape
    assert torch.count_nonzero(targets) > 0
    empty = model.encode(pl.DataFrame({"events": [[], []]}, schema=source.schema), strata="train")["/events/body"]
    assert embedder.encode(content=empty.content, state=empty.state).numel() == 0


def test_compiled_projection_can_reuse_runtime_cache(encoder):
    model = build(resident=1)
    batch = field(model, ["a", "bbb", None])
    embedder = model.nodes["/body"].embedder
    reference = embedder.encode(content=batch.content, state=batch.state)
    compiled = torch.compile(embedder.encode, backend="eager")
    torch.testing.assert_close(compiled(content=batch.content, state=batch.state), reference)
    assert encoder.calls == 1


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_local_encoder_training_and_weight_invalidation(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from transformers import BertConfig, BertModel, BertTokenizerFast

    assets = tmp_path / "encoder"
    assets.mkdir()
    (assets / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\nother\n")
    tokenizer = BertTokenizerFast(vocab_file=str(assets / "vocab.txt"))
    tokenizer.save_pretrained(assets)
    BertModel(
        BertConfig(vocab_size=8, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    ).save_pretrained(assets)
    model = rf.Model.xs(
        attention=None,
        dropout=0.0,
        body=rf.Text(model=str(assets), max_length=8, resident=1, embed=True),
        target=rf.Text(model=str(assets), max_length=8, mask=True),
    ).to(device)
    reference = rf.Model.xs(body=rf.Text(model=str(assets), max_length=8, cache=False)).to(device)
    source = pl.DataFrame({"body": ["hello world", "other", None], "target": ["other", "hello", "world"]})
    actual = model.encode(source)["/body"].to(device)
    raw = field(reference, source["body"].to_list()).to(device)
    embedder = model.nodes["/body"].embedder
    direct = reference.nodes["/body"].embedder.encode(content=raw.content, state=raw.state)
    with torch.inference_mode():
        cached = embedder.encode(content=actual.content, state=actual.state)
    torch.testing.assert_close(cached, direct, rtol=1e-5, atol=1e-6)
    data = rf.PolarsDataModule(train=source, validate=source, num_workers=0, pin_memory=False)
    before = embedder.linear[0].weight.detach().clone()
    model.fit(
        data,
        batch_size=3,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        accelerator=device,
        devices=1,
        # CPU training exercises cache/gradient behavior without requiring BF16 CPU instructions.
        precision="32-true" if device == "cpu" else "bf16-mixed",
        logger=False,
        rollback=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    assert not torch.equal(before.cpu(), embedder.linear[0].weight.detach().cpu())
    frozen = CachedModel.get_model(str(assets))
    assert all(parameter.grad is None for parameter in frozen.model.parameters())
    old = embedder.variant(torch.device(device))
    with torch.no_grad():
        next(frozen.model.parameters()).add_(0.1)
    assert embedder.variant(torch.device(device)) != old
    before = embedder.store.stats()["results"]
    embedder.encode(content=actual.content, state=actual.state)
    assert embedder.store.stats()["results"] > before
