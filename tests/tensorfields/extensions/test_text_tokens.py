"""Token identity, compact transport, and worker ownership boundaries."""

import multiprocessing as mp

import polars as pl
import pytest
import torch

import relflow as rf
from relflow.tensorfields.extensions.text import IDENTIFIERS, CachedModel
from tests.tensorfields.extensions.test_text import FakeTokenizer, patch_hf

MODEL = "test-text-identities"
pytestmark = pytest.mark.integration


@pytest.fixture
def encoder(monkeypatch):
    return patch_hf(monkeypatch)


def build(**options):
    return rf.Model(preset="small", body=rf.Text(model=MODEL, max_length=4, **options))


def test_changed_padding_selects_a_new_identity(encoder):
    model = build()
    first = model.encode(pl.DataFrame({"body": ["ab"]}))["/body"]
    CachedModel.get_tokenizer(MODEL).padding_side = "left"
    second = model.encode(pl.DataFrame({"body": ["ab"]}))["/body"]
    assert not torch.equal(first.content[IDENTIFIERS], second.content[IDENTIFIERS])
    assert len(CachedModel.get_tokenizer(MODEL).batches) == 2


def test_new_tokenizer_object_invalidates_the_vocab_fingerprint(encoder):
    model = build()
    first = model.encode(pl.DataFrame({"body": ["ab"]}))["/body"]

    class Changed(FakeTokenizer):
        def get_vocab(self):
            return {"different": 1}

    CachedModel._tokenizers[MODEL] = Changed()
    second = model.encode(pl.DataFrame({"body": ["ab"]}))["/body"]
    assert not torch.equal(first.content[IDENTIFIERS], second.content[IDENTIFIERS])


def test_nulls_and_empty_strings_are_distinct(encoder):
    model = build()
    field = model.encode(pl.DataFrame({"body": [None, "", None, "a"]}))["/body"]
    ids = field.content[IDENTIFIERS].reshape(-1)
    assert ids[0] == ids[2] == 0
    assert ids[1] > 0 and ids[3] > 0
    assert ids[1] != ids[3]


def worker(context, schema, observations, connection):
    from relflow.data.iterables import encode
    from relflow.structs.enums import Strata

    CachedModel._tokenizers[MODEL] = FakeTokenizer()
    try:
        field = encode(observations, schema, Strata.validate, context).tensors["/body"]
        connection.send((field.content[IDENTIFIERS].tolist(), CachedModel.get_tokenizer(MODEL).batches))
    finally:
        connection.close()


@pytest.mark.integration
@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_workers_share_stable_ids_and_do_not_retokenize(encoder, method):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"{method} unavailable")
    import pyarrow as pa

    model = build()
    observations = pl.DataFrame({"body": ["a", "bb", None]})
    expected = model.encode(observations)["/body"].content[IDENTIFIERS].tolist()
    model.nodes["/body"].embedder.context.share()
    context = mp.get_context(method)
    receiving, sending = context.Pipe(False)
    process = context.Process(
        target=worker,
        args=(
            model.interprocess_encoding_context,
            model.schema,
            pa.Table.from_pydict(observations.to_dict(as_series=False)),
            sending,
        ),
    )
    process.start()
    sending.close()
    try:
        assert receiving.poll(30)
        actual, calls = receiving.recv()
        process.join(30)
        assert process.exitcode == 0
        assert actual == expected and calls == []
    finally:
        receiving.close()
        if process.is_alive():
            process.terminate()
            process.join(5)
