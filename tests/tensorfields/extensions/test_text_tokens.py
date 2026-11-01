"""Persistent token reuse stays within field, tokenizer and process boundaries."""

import multiprocessing as mp
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.data.iterables import encode
from relflow.structs.enums import Strata
from relflow.structs.tree import Address
from relflow.tensorfields.extensions.text import ATTENTION_MASK, INPUT_IDS, CachedModel, TokenCache
from tests.tensorfields.extensions.test_text import FakeTokenizer, patch_hf

MODEL = "test-text-tokens"


def encode_worker(context, schema, observations, connection):
    """Exercise the actual serialized/forked encoding context without model weights."""
    try:
        tokenizer = FakeTokenizer()
        CachedModel._tokenizers[MODEL] = tokenizer
        cache = context[Address("/body")]
        before = len(cache)
        field = encode(observations, schema, Strata.validate, context).tensors["/body"]
        connection.send((before, len(cache), tokenizer.batches, field.content[INPUT_IDS].tolist()))
    finally:
        connection.close()


@pytest.fixture
def resources(monkeypatch):
    return patch_hf(monkeypatch)


def build(cache=True, *, mask=False):
    return rf.Model.xs(
        attention=None,
        body=rf.Text(model=MODEL, cache=cache, max_length=4, mask=mask),
    )


def test_public_encode_reuses_tokens_across_different_batches_and_keeps_one_context(resources):
    model = build()
    cache = model.nodes["/body"].embedder.tokens
    assert model.interprocess_encoding_context[Address("/body")] is cache
    assert model.interprocess_encoding_context[Address("/body")] is cache
    tokenizer = CachedModel.get_tokenizer(MODEL)
    first = model.encode(pa.table({"body": ["a", "bb", None, "a"]}), strata="validate")["/body"]
    second = model.encode(pa.table({"body": ["bb", "", "a", None]}), strata="validate")["/body"]

    assert tokenizer.batches == [["a", "bb"], [""]]
    assert cache.hits == 2 and cache.misses == 3
    torch.testing.assert_close(second.content[INPUT_IDS][0], first.content[INPUT_IDS][1])
    torch.testing.assert_close(second.content[INPUT_IDS][2], first.content[INPUT_IDS][0])
    assert second.state[1].eq(rf.Tokens.valued).all()
    assert not second.content[INPUT_IDS][3].any()


def test_disabled_cache_retains_per_batch_deduplication(resources):
    model = build(cache=False)
    source = pa.table({"body": ["a", "bb", "a"]})
    model.encode(source, strata="validate")
    model.encode(source, strata="validate")
    assert model.nodes["/body"].embedder.tokens is None
    assert Address("/body") not in model.interprocess_encoding_context
    assert CachedModel.get_tokenizer(MODEL).batches == [["a", "bb"], ["a", "bb"]]


def test_token_caches_are_separate_for_each_field_and_model(resources):
    one = rf.Model.xs(
        attention=None,
        body=rf.Text(model=MODEL, cache=True, max_length=4),
        title=rf.Text(model=MODEL, cache=True, max_length=1),
    )
    two = build()
    first = one.encode(pa.table({"body": ["alpha"], "title": ["alpha"]}), strata="validate")
    two.encode(pa.table({"body": ["alpha"]}), strata="validate")
    caches = [one.nodes[address].embedder.tokens for address in ("/body", "/title")]
    assert caches[0] is not caches[1] and caches[0] is not two.nodes["/body"].embedder.tokens
    assert first["/body"].content[INPUT_IDS].shape[-1] == 4
    assert first["/title"].content[INPUT_IDS].shape[-1] == 1
    assert CachedModel.get_tokenizer(MODEL).batches == [["alpha"], ["alpha"], ["alpha"]]


def test_lru_entry_and_payload_limits_release_evicted_rows(resources):
    cache = TokenCache(2)
    tokenizer = CachedModel.get_tokenizer(MODEL)
    cache.encode(["a", "bb"], tokenizer, length=4, batch_size=2)
    cache.encode(["a"], tokenizer, length=4, batch_size=2)
    cache.encode(["ccc"], tokenizer, length=4, batch_size=2)
    assert list(cache.values) == ["a", "ccc"]
    cache.encode(["bb"], tokenizer, length=4, batch_size=2)
    assert list(cache.values) == ["ccc", "bb"] and cache.evictions == 2

    # The payload budget includes retained strings and both integer rows.
    byte_limit = sys.getsizeof("a") + 8 * 4 * 2
    bounded = TokenCache(100, byte_limit=byte_limit)
    bounded.encode(["a", "bb"], tokenizer, length=4, batch_size=2)
    assert bounded.retained_bytes <= byte_limit and len(bounded) == 1
    assert list(bounded.values) == ["a"]  # "bb" is individually too large.
    bounded.encode(["a"], tokenizer, length=4, batch_size=2)
    assert bounded.hits == 1


def test_cached_rows_do_not_alias_tokenizer_results_or_returned_batches(resources):
    cache = TokenCache(2)
    tokenizer = CachedModel.get_tokenizer(MODEL)
    ids, mask = cache.encode(["a", "bb"], tokenizer, length=4, batch_size=2)
    expected_ids, expected_mask = ids.copy(), mask.copy()
    ids.fill(123)
    mask.fill(0)
    repeated = cache.encode(["a", "bb"], tokenizer, length=4, batch_size=2)
    np.testing.assert_array_equal(repeated[0], expected_ids)
    np.testing.assert_array_equal(repeated[1], expected_mask)


@pytest.mark.parametrize("operation", [pickle.loads, deepcopy])
def test_encoding_context_copies_start_cold(resources, operation):
    model = build(cache=2)
    source = pa.table({"body": ["a", "bb"]})
    model.encode(source, strata="validate")
    context = model.interprocess_encoding_context
    restored = operation(pickle.dumps(context)) if operation is pickle.loads else operation(context)
    cache = restored[Address("/body")]
    assert len(cache) == 0 and cache.tokenizer is None
    original = context[Address("/body")]
    assert len(original) == 2 and cache is not original
    assert cache.capacity == original.capacity and cache.byte_limit == original.byte_limit


@pytest.mark.integration
@pytest.mark.parametrize("method", ["spawn", "fork"])
def test_worker_context_starts_cold_and_preserves_parent_state(resources, method):
    if method not in mp.get_all_start_methods():
        pytest.skip(f"multiprocessing {method} is unavailable")
    model = build(cache=2)
    source = pa.table({"body": ["a", "bb"]})
    expected = model.encode(source, strata="validate")["/body"].content[INPUT_IDS].tolist()
    context = model.interprocess_encoding_context
    cache = context[Address("/body")]
    process_context = mp.get_context(method)
    receiving, sending = process_context.Pipe(duplex=False)
    process = process_context.Process(target=encode_worker, args=(context, model.schema, source, sending))
    process.start()
    sending.close()
    try:
        assert receiving.poll(20), f"{method} encoding worker did not return"
        before, after, batches, actual = receiving.recv()
        process.join(20)
        assert process.exitcode == 0
        assert before == 0 and after == 2
        assert batches == [["a", "bb"]] and actual == expected
        assert len(cache) == 2 and cache.hits == 0 and cache.misses == 2
    finally:
        receiving.close()
        if process.is_alive():
            process.terminate()
            process.join(5)


def test_shared_field_cache_serializes_concurrent_misses(resources):
    cache = TokenCache(2)
    tokenizer = CachedModel.get_tokenizer(MODEL)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: cache.encode(["a", "bb"], tokenizer, length=4, batch_size=2), range(16)))
    assert tokenizer.batches == [["a", "bb"]]
    assert cache.misses == 2 and cache.hits == 30 and len(cache) == 2
    for ids, mask in results:
        np.testing.assert_array_equal(ids, results[0][0])
        np.testing.assert_array_equal(mask, results[0][1])


def test_ragged_hidden_targets_reuse_tokens_without_revealing_inputs(resources):
    model = rf.Model.xs(
        attention=None,
        events=rf.Branch(
            length=None,
            attention=None,
            body=rf.Text(model=MODEL, cache=True, max_length=4, mask=True),
        ),
    )
    source = pa.Table.from_pylist([{"events": [{"body": "a"}, None, {"body": ""}]}, {"events": []}, {"events": None}])
    first = model.encode(source, strata="train")["/events/body"]
    second = model.encode(source, strata="train")["/events/body"]
    assert CachedModel.get_tokenizer(MODEL).batches == [["a", ""]]
    for name in (INPUT_IDS, ATTENTION_MASK):
        assert not first.content[name].any() and not second.content[name].any()
        torch.testing.assert_close(first.targets["content"][name], second.targets["content"][name])
    torch.testing.assert_close(first.state, second.state)
    torch.testing.assert_close(first.present, second.present)
    assert model.nodes["/events/body"].embedder.tokens.hits == 2


def test_warm_cached_projection_outputs_and_gradients_match_uncached(resources):
    cached = build()
    reference = build(cache=False)
    reference.load_state_dict(cached.state_dict())
    batches = [pa.table({"body": ["a", "bb", None, "a"]}), pa.table({"body": ["bb", "", "a", None]})]
    for batch in batches:
        cached.encode(batch, strata="validate")
        outcomes, gradients = [], []
        for model in (reference, cached):
            model.zero_grad(set_to_none=True)
            field = model.encode(batch, strata="validate")["/body"]
            output = model.nodes["/body"].embedder.embed(field).payload
            output.square().mean().backward()
            outcomes.append(output.detach())
            gradients.append(
                {
                    name: None if parameter.grad is None else parameter.grad.clone()
                    for name, parameter in model.named_parameters()
                }
            )
        torch.testing.assert_close(outcomes[0], outcomes[1], rtol=0, atol=0)
        for name, expected in gradients[0].items():
            actual = gradients[1][name]
            assert (actual is None) == (expected is None), name
            if expected is not None:
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.fixture
def fast_tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(
            {"[UNK]": 0, "[PAD]": 1, "[CLS]": 2, "[SEP]": 3, "Alpha": 4, "alpha": 5, "beta": 6},
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    backend.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)]
    )
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]"
    )


@pytest.mark.integration
@pytest.mark.parametrize("change", ["padding", "truncation", "length", "normalizer", "pre_tokenizer", "post_processor"])
def test_actual_tokenizer_option_changes_invalidate_cached_rows(fast_tokenizer, change):
    from tokenizers import normalizers, pre_tokenizers, processors

    cache = TokenCache(2)
    texts = ["Alpha beta alpha"]
    cache.encode(texts, fast_tokenizer, length=4, batch_size=2)
    length = 4
    if change == "padding":
        fast_tokenizer.padding_side = "left"
    elif change == "truncation":
        fast_tokenizer.truncation_side = "left"
    elif change == "length":
        length = 7
    elif change == "normalizer":
        fast_tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
    elif change == "pre_tokenizer":
        fast_tokenizer.backend_tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    else:
        fast_tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(single="$A")
    actual = cache.encode(texts, fast_tokenizer, length=length, batch_size=2)
    expected = fast_tokenizer(texts, padding="max_length", truncation=True, max_length=length, return_tensors="np")
    assert cache.misses == 1 and cache.hits == 0
    np.testing.assert_array_equal(actual[0], expected[INPUT_IDS])
    np.testing.assert_array_equal(actual[1], expected[ATTENTION_MASK])


def test_padding_change_after_construction_keeps_frozen_encoder_width(resources):
    model = build()
    tokenizer = CachedModel.get_tokenizer(MODEL)
    tokenizer.padding_side = "left"
    field = model.encode(pa.table({"body": ["a", "bb"]}), strata="validate")["/body"]
    output = model.nodes["/body"].embedder.embed(field).payload
    assert torch.isfinite(output).all()
    assert all(ids.shape[-1] == 4 for ids, _ in resources.batches)
