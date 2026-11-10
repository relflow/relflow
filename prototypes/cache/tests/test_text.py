"""Explicit native-process checks; run with the prototype wheel installed."""

import multiprocessing as mp
import pickle
from copy import deepcopy

import polars as pl
import pyarrow as pa
import pytest
import torch
from relflow_cache.core import Owner
from relflow_cache.text import Request

import relflow as rf
from relflow.tensorfields.extensions.text import IDENTIFIERS, CachedModel
from tests.tensorfields.extensions.test_text import FakeTokenizer, patch_hf

pytestmark = pytest.mark.integration
MODEL = "test-native-text"


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("NATIVE_TEXT_MODE", "full")
    monkeypatch.setenv("RELFLOW_CACHE_DIR", str(tmp_path / "cache"))
    yield
    for owner in list(Owner.instances.values()):
        if str(owner.path).startswith(str(tmp_path)):
            owner.close()


def test_native_cold_fill_warm_reuse_precision_and_gradients(monkeypatch):
    fake = patch_hf(monkeypatch)
    model = rf.Model.xs(attention=None, body=Request(model=MODEL, max_length=4, resident=1))
    source = pl.DataFrame({"body": ["a", "bbb", None, "", "a"]})
    batch = model.encode(source)["/body"]
    embedder = model.nodes["/body"].embedder
    with torch.inference_mode():
        first = embedder.encode(content=batch.content, state=batch.state)
    assert fake.calls == 1
    tokenizer = CachedModel.get_tokenizer(MODEL)
    calls = len(tokenizer.batches)
    second = model.encode(source)["/body"]
    assert len(tokenizer.batches) == calls
    torch.testing.assert_close(first, embedder.encode(content=second.content, state=second.state))
    embedder.train()
    embedder.embed(second).payload.square().sum().backward()
    assert torch.count_nonzero(embedder.linear[0].weight.grad)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        embedder.encode(content=second.content, state=second.state)
    assert fake.calls == 2
    assert embedder.store.stats()["claims"] == 0
    for operation in (deepcopy, lambda value: pickle.loads(pickle.dumps(value))):
        copy = operation(embedder.store)
        assert copy.owner is embedder.store.owner
        assert not copy.views
        assert copy.client.endpoint == embedder.store.client.endpoint


def worker(schema, context, table, sending):
    from relflow.data.iterables import encode

    CachedModel._tokenizers[MODEL] = FakeTokenizer()
    try:
        batch = encode(table, schema, rf.Strata.validate, context)
        sending.send((batch.tensors["/body"].content[IDENTIFIERS].tolist(), CachedModel.get_tokenizer(MODEL).batches))
    finally:
        sending.close()


@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_native_worker_uses_same_ids_and_skips_tokenization(monkeypatch, method):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} unavailable")
    patch_hf(monkeypatch)
    model = rf.Model.xs(body=Request(model=MODEL, max_length=4))
    table = pa.table({"body": ["a", "bbb", None]})
    expected = model.encode(table)["/body"].content[IDENTIFIERS].tolist()
    model.nodes["/body"].embedder.context.share()
    context = mp.get_context(method)
    receiving, sending = context.Pipe(False)
    process = context.Process(target=worker, args=(model.schema, model.interprocess_encoding_context, table, sending))
    process.start()
    sending.close()
    try:
        assert receiving.poll(30)
        actual, batches = receiving.recv()
        process.join(30)
        assert process.exitcode == 0
        assert actual == expected
        assert not batches
    finally:
        receiving.close()
        if process.is_alive():
            process.terminate()
            process.join(5)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_encoder_matches_uncached_outputs(tmp_path, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from transformers import BertConfig, BertModel, BertTokenizerFast

    assets = tmp_path / "encoder"
    assets.mkdir()
    (assets / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n")
    BertTokenizerFast(vocab_file=str(assets / "vocab.txt")).save_pretrained(assets)
    BertModel(
        BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16)
    ).save_pretrained(assets)
    source = pl.DataFrame({"body": ["hello world", "world", None, ""]})
    model = rf.Model.xs(body=Request(model=str(assets), max_length=8, encoder_pooling="mean")).to(device)
    direct = rf.Model.xs(body=rf.Text(model=str(assets), max_length=8, encoder_pooling="mean", cache=False)).to(device)
    cached = model.encode(source)["/body"].to(device)
    plain = direct.encode(source)["/body"].to(device)
    embedder = model.nodes["/body"].embedder
    expected = direct.nodes["/body"].embedder.encode(content=plain.content, state=plain.state)
    actual = embedder.encode(content=cached.content, state=cached.state)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    before = embedder.store.stats()["compute_claims"]
    torch.testing.assert_close(actual, embedder.encode(content=cached.content, state=cached.state))
    assert embedder.store.stats()["compute_claims"] == before
